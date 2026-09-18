"""
MSCI EM stock-weight prediction pipeline.

One function per stage; each stage's output feeds the next.

    Stage 0  config / libraries               (this module header)
    Stage 1  load_and_clean          -> long table: stock x period
    Stage 2  derive_basics           -> price, index total, q/q changes, rel_return
    Stage 3  build_features          -> lagged / rolling features, percentile ranked
    Stage 4  build_target            -> next-period % change in weight
    Stage 5  make_models             -> naive / ridge / xgboost candidates
    Stage 6  predictions_to_weights  -> rescaled predicted weights
    Stage 7  walk_forward            -> per-model MAE table, winner
    Stage 8  simulate_range          -> low / mid / high per stock
    Stage 9  final_forecast          -> exported deliverable

Rules baked in:
    * never shuffle rows: every split is by period
    * every feature uses only data available at that period (positive shift)
    * rescale to 100 % after any prediction or simulation

The source file holds *annual* snapshots (Dec-21 .. Dec-25, Aug-26), so the
word "quarter" in the blueprint maps to "period" here.  The period column is
still called ``quarter`` to stay faithful to the spec.  With 6 periods a 4-lag
feature would leave no training rows, so the default lag set is (1, 2); pass
``lags=(1, 2, 4)`` once quarterly data is available.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------------------------------------------------------------
# Stage 0: setup / config
# --------------------------------------------------------------------------

#: holding types that count as "stocks" for the index.  Everything else
#: (cash, forwards, futures, money-market funds, warrants) is dropped and the
#: remaining weights are rescaled to 100 %.
EQUITY_TYPES = {
    "EQUITY",
    "EQUITY - UNITS",
    "EQUITY - REIT",
    "EQUITY - UNDEFINED",
    "PREFERRED STOCK",
}

#: rows appended to every sheet by the export tool; never real holdings
SUMMARY_ROW_NAMES = {
    "summary statistics", "eightieth percentile", "sixtieth percentile",
    "fortieth percentile", "twentieth percentile", "sum", "average", "count",
    "maximum", "minimum", "median", "standard deviation",
}

RAW_COLUMNS = {
    "Name": "name",
    "Ticker": "ticker",
    "ISIN": "isin",
    "Country": "country",
    "Sector": "sector",
    "Detail Holding Type": "holding_type",
    "Portfolio Weighting %": "weight",
    "Position Market Value": "market_value",
    "Shares": "shares",
}

BASE_COLS = ["stock", "quarter", "weight", "market_value", "shares"]


@dataclass
class Config:
    lags: Tuple[int, ...] = (1, 2)
    vol_window: int = 4
    vol_min_periods: int = 2
    #: set False to drop the volatility feature (it needs vol_min_periods
    #: past returns, which costs one extra period of history)
    use_vol: bool = True
    #: clip the *training* target to these quantiles (fit on training data
    #: only) so a handful of 10x weight jumps don't dominate the fit
    target_clip_q: Optional[float] = 0.01
    ridge_alphas: Tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)
    xgb_grid: Tuple[Dict, ...] = (
        dict(max_depth=2, n_estimators=100),
        dict(max_depth=2, n_estimators=300),
        dict(max_depth=3, n_estimators=200),
    )
    xgb_common: Dict = field(default_factory=lambda: dict(
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        objective="reg:squarederror", random_state=0, n_jobs=4, verbosity=0,
    ))
    min_train_periods: int = 1
    n_sims: int = 1000
    sim_seed: int = 42
    low_q: float = 0.05
    high_q: float = 0.95
    #: how a past period's errors are assigned to stocks in the simulation:
    #:   "size_bucket" - draw from stocks of similar size in that period
    #:   "stock"       - reuse the same stock's own error where available
    #:   "any"         - draw from all of that period's errors
    sim_match: str = "size_bucket"
    sim_n_buckets: int = 10


# --------------------------------------------------------------------------
# Stage 1: load and clean
# --------------------------------------------------------------------------

def _norm_col(c) -> str:
    return re.sub(r"\s+", " ", str(c)).strip()


def _period_sort_key(label: str) -> pd.Timestamp:
    """'Dec-21' -> 2021-12-01 so sheets order chronologically."""
    return pd.to_datetime(label, format="%b-%y")


def _stock_key(row: pd.Series) -> str:
    """Stable identifier across periods: ISIN, else ticker+country, else name."""
    if isinstance(row["isin"], str) and row["isin"].strip():
        return row["isin"].strip()
    if isinstance(row["ticker"], str) and row["ticker"].strip():
        return f"{row['ticker'].strip()}|{row['country']}"
    return f"NAME:{str(row['name']).strip().upper()}"


def read_wide_workbook(path: str) -> pd.DataFrame:
    """Read every sheet (one per period) and stack into one long table."""
    xl = pd.ExcelFile(path)
    frames = []
    for sheet in xl.sheet_names:
        raw = xl.parse(sheet)
        raw.columns = [_norm_col(c) for c in raw.columns]
        missing = [c for c in RAW_COLUMNS if c not in raw.columns]
        if missing:
            raise ValueError(f"sheet {sheet!r} lacks columns {missing}")
        df = raw[list(RAW_COLUMNS)].rename(columns=RAW_COLUMNS)
        df["quarter"] = sheet.strip()
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_and_clean(path: str, verbose: bool = True) -> pd.DataFrame:
    """Stage 1.

    Returns a long table with one row per stock per period, columns
    ``stock, quarter, quarter_idx, weight, market_value, shares`` plus
    descriptive columns and flags ``entered_mid, exited_mid, has_gap``.
    Weights are rescaled to sum to 100 within the equity universe.
    """
    long = read_wide_workbook(path)

    # --- drop the summary block and anything that isn't a stock -------------
    is_summary = long["name"].astype(str).str.strip().str.lower().isin(SUMMARY_ROW_NAMES)
    long = long[~is_summary]
    long = long[long["holding_type"].isin(EQUITY_TYPES)].copy()

    for c in ["weight", "market_value", "shares"]:
        long[c] = pd.to_numeric(long[c], errors="coerce")

    # --- sanity: no zero / negative shares or values ------------------------
    bad = (long["shares"] <= 0) | (long["market_value"] <= 0) | (long["weight"] <= 0) | \
          long[["shares", "market_value", "weight"]].isna().any(axis=1)
    n_bad = int(bad.sum())
    long = long[~bad].copy()

    # --- stable stock id and duplicate handling -----------------------------
    long["stock"] = long.apply(_stock_key, axis=1)
    dup_mask = long.duplicated(["stock", "quarter"], keep=False)
    n_dup = int(dup_mask.sum())
    agg = {
        "weight": "sum", "market_value": "sum", "shares": "sum",
        "name": "first", "ticker": "first", "isin": "first",
        "country": "first", "sector": "first", "holding_type": "first",
    }
    long = long.groupby(["stock", "quarter"], as_index=False).agg(agg)

    # --- period ordering ----------------------------------------------------
    periods = sorted(long["quarter"].unique(), key=_period_sort_key)
    idx_map = {p: i for i, p in enumerate(periods)}
    long["quarter_idx"] = long["quarter"].map(idx_map)
    long = long.sort_values(["stock", "quarter_idx"]).reset_index(drop=True)

    # --- weights sum ~100 per period, then rescale exactly ------------------
    raw_sums = long.groupby("quarter")["weight"].sum().reindex(periods)
    if not ((raw_sums > 95) & (raw_sums < 105)).all():
        raise ValueError(f"weights per period far from 100 %:\n{raw_sums}")
    long["weight_raw"] = long["weight"]
    long["weight"] = long["weight"] / long["quarter_idx"].map(
        long.groupby("quarter_idx")["weight"].sum()) * 100.0

    # --- flag stocks that appear / disappear mid-history --------------------
    g = long.groupby("stock")["quarter_idx"]
    first, last, count = g.transform("min"), g.transform("max"), g.transform("count")
    long["entered_mid"] = first > 0
    long["exited_mid"] = last < len(periods) - 1
    long["has_gap"] = (last - first + 1) != count

    if verbose:
        print("Stage 1: load_and_clean")
        print(f"  periods ({len(periods)}): {periods}")
        print(f"  rows kept: {len(long)}  | dropped zero/neg/NaN rows: {n_bad}"
              f"  | duplicate id rows merged: {n_dup}")
        print("  raw equity weight sum per period (before rescale):")
        print("   ", raw_sums.round(3).to_dict())
        print(f"  stocks: {long['stock'].nunique()}  | entered mid-history: "
              f"{long.loc[long.entered_mid, 'stock'].nunique()}  | exited mid-history: "
              f"{long.loc[long.exited_mid, 'stock'].nunique()}  | with gaps: "
              f"{long.loc[long.has_gap, 'stock'].nunique()}")
    return long


# --------------------------------------------------------------------------
# Stage 2: derive the basics
# --------------------------------------------------------------------------

def _complete_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex to the full stock x period grid so `shift` is period-aligned
    even when a stock is missing for a period (its row is NaN there)."""
    periods = np.sort(df["quarter_idx"].unique())
    full = pd.MultiIndex.from_product(
        [df["stock"].unique(), periods], names=["stock", "quarter_idx"])
    out = df.set_index(["stock", "quarter_idx"]).reindex(full).reset_index()
    label = df.drop_duplicates("quarter_idx").set_index("quarter_idx")["quarter"]
    out["quarter"] = out["quarter_idx"].map(label)
    return out.sort_values(["stock", "quarter_idx"]).reset_index(drop=True)


def derive_basics(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Stage 2: price, index_total, price_chg, share_chg, index_chg, rel_return."""
    out = _complete_panel(df)
    out["price"] = out["market_value"] / out["shares"]
    index_total = out.groupby("quarter_idx")["market_value"].sum()
    out["index_total"] = out["quarter_idx"].map(index_total)

    g = out.groupby("stock")
    out["price_chg"] = out["price"] / g["price"].shift(1) - 1
    out["share_chg"] = out["shares"] / g["shares"].shift(1) - 1
    out["index_chg"] = out["index_total"] / g["index_total"].shift(1) - 1
    out["rel_return"] = out["price_chg"] - out["index_chg"]
    if verbose:
        print("Stage 2: derive_basics")
        print("  index total per period:",
              {q: f"{v/1e9:.1f}bn" for q, v in index_total.items()})
    return out


# --------------------------------------------------------------------------
# Stage 3: features (past data only)
# --------------------------------------------------------------------------

def build_features(df: pd.DataFrame, cfg: Config = Config(),
                   verbose: bool = True) -> Tuple[pd.DataFrame, List[str]]:
    """Stage 3.  Every feature at period t uses data from t and earlier."""
    out = df.copy()
    g = out.groupby("stock")
    raw_feats: List[str] = []

    for k in cfg.lags:
        # cumulative relative return over the last k periods
        idx_ret_k = out["index_total"] / g["index_total"].shift(k) - 1
        px_ret_k = out["price"] / g["price"].shift(k) - 1
        out[f"rel_return_{k}"] = px_ret_k - idx_ret_k
        out[f"share_chg_{k}"] = out["shares"] / g["shares"].shift(k) - 1
        raw_feats += [f"rel_return_{k}", f"share_chg_{k}"]

    if cfg.use_vol:
        out["rel_vol"] = g["rel_return"].transform(
            lambda s: s.rolling(cfg.vol_window, min_periods=cfg.vol_min_periods).std())
        raw_feats.append("rel_vol")

    # size rank within period: 1 = largest market value
    out["size_rank"] = out.groupby("quarter_idx")["market_value"].rank(
        ascending=False, method="min")
    out["size_rank_chg"] = out["size_rank"] - out.groupby("stock")["size_rank"].shift(1)
    raw_feats += ["size_rank", "size_rank_chg"]

    # percentile rank of every feature within its period (NaN stays NaN)
    feat_cols = []
    for f in raw_feats:
        col = f"{f}_pct"
        out[col] = out.groupby("quarter_idx")[f].rank(pct=True)
        feat_cols.append(col)

    if verbose:
        print("Stage 3: build_features")
        print(f"  features ({len(feat_cols)}): {feat_cols}")
    return out, feat_cols


# --------------------------------------------------------------------------
# Stage 4: target
# --------------------------------------------------------------------------

def attach_target(df: pd.DataFrame, feat_cols: Sequence[str]) -> pd.DataFrame:
    """Full panel with ``target``, ``next_weight`` and ``has_features`` added."""
    out = df.copy()
    out["target"] = out.groupby("stock")["weight"].shift(-1) / out["weight"] - 1
    out["next_weight"] = out.groupby("stock")["weight"].shift(-1)
    out["has_features"] = out[list(feat_cols)].notna().all(axis=1)
    return out


def build_target(df: pd.DataFrame, feat_cols: Sequence[str],
                 verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Stage 4.

    target = weight(t+1) / weight(t) - 1 within each stock.  Returns
    ``(train_df, latest_df)``: rows with features+target, and the last period's
    rows (features but no target) kept aside for the final forecast.
    """
    out = attach_target(df, feat_cols)
    has_row = out["weight"].notna()
    has_feats = out["has_features"]
    last_idx = out["quarter_idx"].max()
    train = out[has_row & has_feats & out["target"].notna()].copy()
    # every stock in the last period; those without features are carried at
    # their current weight (naive) in the final forecast
    latest = out[has_row & (out["quarter_idx"] == last_idx)].copy()

    if verbose:
        print("Stage 4: build_target")
        print("  usable training rows per period:",
              train.groupby("quarter")["target"].count().reindex(
                  train.drop_duplicates("quarter_idx").sort_values("quarter_idx")["quarter"]
              ).to_dict())
        print(f"  forecast base rows ({latest['quarter'].iloc[0]}): {len(latest)}"
              f"  | with full features: {int(latest['has_features'].sum())}")
    return train, latest


# --------------------------------------------------------------------------
# Stage 5: models
# --------------------------------------------------------------------------

class NaiveModel:
    """predicted change = 0 (weights carried forward)."""

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X))


def make_models(cfg: Config = Config()) -> Dict[str, Callable[[], object]]:
    """Stage 5: factory per model name so each fold gets a fresh instance."""
    models: Dict[str, Callable[[], object]] = {"naive": NaiveModel}
    for a in cfg.ridge_alphas:
        models[f"ridge_a{a:g}"] = (lambda a=a: Ridge(alpha=a))
    for p in cfg.xgb_grid:
        name = f"xgb_d{p['max_depth']}_n{p['n_estimators']}"
        models[name] = (lambda p=p: XGBRegressor(**p, **cfg.xgb_common))
    return models


def _clip_target(y: pd.Series, q: Optional[float]) -> pd.Series:
    if q is None:
        return y
    lo, hi = y.quantile(q), y.quantile(1 - q)
    return y.clip(lo, hi)


def fit_model(model_factory, X_train, y_train, cfg: Config):
    m = model_factory()
    m.fit(X_train.values, _clip_target(y_train, cfg.target_clip_q).values)
    return m


def fit_predict(model_factory, X_train, y_train, X_test, cfg: Config) -> np.ndarray:
    m = fit_model(model_factory, X_train, y_train, cfg)
    return np.asarray(m.predict(X_test.values), dtype=float)


def model_weights(model, feat_cols: Sequence[str]) -> Optional[pd.DataFrame]:
    """Feature weights of a fitted model, or None for models without any.

    Ridge: coefficient in % of weight change when a stock moves from the
    bottom to the top percentile of that feature (features are 0..1 ranks).
    XGBoost: gain-based feature importance, normalised to sum to 100.
    """
    if isinstance(model, Ridge):
        out = pd.DataFrame({"feature": list(feat_cols), "weight_%": model.coef_ * 100})
        out = pd.concat([out, pd.DataFrame({"feature": ["intercept"],
                                            "weight_%": [model.intercept_ * 100]})])
    elif isinstance(model, XGBRegressor):
        imp = model.get_booster().get_score(importance_type="gain")
        vals = np.array([imp.get(f"f{i}", 0.0) for i in range(len(feat_cols))])
        out = pd.DataFrame({"feature": list(feat_cols), "weight_%": vals / vals.sum() * 100})
    else:
        return None
    out["abs"] = out["weight_%"].abs()
    return (out.sort_values("abs", ascending=False).drop(columns="abs")
            .reset_index(drop=True))


# --------------------------------------------------------------------------
# Stage 6: predictions -> weights
# --------------------------------------------------------------------------

def predictions_to_weights(current_weight: np.ndarray, pred_change: np.ndarray) -> np.ndarray:
    """raw = w * (1 + pred); floor at 0; rescale to sum 100."""
    raw = np.asarray(current_weight, float) * (1.0 + np.asarray(pred_change, float))
    raw = np.clip(raw, 0.0, None)
    s = raw.sum()
    return raw / s * 100.0 if s > 0 else raw


# --------------------------------------------------------------------------
# Stage 7: walk-forward
# --------------------------------------------------------------------------

def period_labels(df: pd.DataFrame) -> Dict[int, str]:
    """quarter_idx -> sheet label, from any frame that carries both columns."""
    return df.drop_duplicates("quarter_idx").set_index("quarter_idx")["quarter"].to_dict()


def walk_forward(train: pd.DataFrame, feat_cols: Sequence[str], cfg: Config = Config(),
                 verbose: bool = True, labels: Optional[Dict[int, str]] = None,
                 ) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """Stage 7.

    For each period t: fit on rows with quarter_idx <= t (their targets are
    realised at t+1), predict the rows at t+1 (whose targets realise at t+2)
    and score the resulting weights against the actual t+2 weights.

    Returns ``(summary, predictions, winner)``.  ``labels`` maps quarter_idx
    to its sheet label for every period in the file (the training frame lacks
    the last one); it is used to name the fold by the period being forecast.
    """
    models = make_models(cfg)
    labels = {**period_labels(train), **(labels or {})}
    periods = np.sort(train["quarter_idx"].unique())
    preds: List[pd.DataFrame] = []

    for i in range(cfg.min_train_periods - 1, len(periods) - 1):
        t, t_next = periods[i], periods[i + 1]
        tr = train[train["quarter_idx"] <= t]
        te = train[train["quarter_idx"] == t_next]
        if te.empty or tr.empty:
            continue
        # actual next-period weights, rescaled over the scored universe
        actual_w = te["next_weight"].values / te["next_weight"].sum() * 100.0
        forecast_q = labels.get(t_next + 1, f"after {labels[t_next]}")
        for name, factory in models.items():
            pc = fit_predict(factory, tr[feat_cols], tr["target"], te[feat_cols], cfg)
            pw = predictions_to_weights(te["weight"].values, pc)
            preds.append(pd.DataFrame({
                "model": name, "train_upto": t, "test_quarter_idx": t_next,
                "test_quarter": te["quarter"].values, "forecast_quarter": forecast_q,
                "stock": te["stock"].values,
                "current_weight": te["weight"].values, "pred_change": pc,
                "pred_weight": pw, "actual_weight": actual_w,
                "abs_err": np.abs(pw - actual_w),
            }))

    predictions = pd.concat(preds, ignore_index=True)
    per_fold = (predictions.groupby(["model", "forecast_quarter"], sort=False)["abs_err"]
                .mean().unstack("forecast_quarter"))
    summary = per_fold.copy()
    summary["mean_mae"] = per_fold.mean(axis=1)
    summary["folds"] = per_fold.notna().sum(axis=1)
    naive_mae = summary.loc["naive", "mean_mae"]
    summary["vs_naive_%"] = (summary["mean_mae"] / naive_mae - 1) * 100
    summary = summary.sort_values("mean_mae")

    best = summary["mean_mae"].idxmin()
    winner = best if summary.loc[best, "mean_mae"] < naive_mae else "naive"

    if verbose:
        print("Stage 7: walk_forward  (columns = period being forecast; MAE in weight points)")
        print(summary.round(5).to_string())
        print(f"  winner: {winner}")
    return summary, predictions, winner


# --------------------------------------------------------------------------
# Stage 8: range via error resampling
# --------------------------------------------------------------------------

def _size_bucket(weights: np.ndarray, n: int) -> np.ndarray:
    """0..n-1 bucket by rank of weight (n-1 = largest)."""
    pct = pd.Series(weights).rank(pct=True, method="first").values
    return np.minimum((pct * n).astype(int), n - 1)


def simulate_range(new_pred: pd.DataFrame, predictions: pd.DataFrame, winner: str,
                   cfg: Config = Config(), verbose: bool = True) -> pd.DataFrame:
    """Stage 8.

    Collect the winner's past errors grouped by test period.  Each simulation
    picks one past period and applies that period's *relative* errors
    (actual / predicted - 1) to the new predicted weights, then rescales to
    100.  How errors are matched to stocks is set by ``cfg.sim_match``; the
    default draws from stocks of similar size within the chosen period, which
    keeps that period's regime and the size-dependence of the errors without
    assuming a stock's own past miss repeats.
    """
    hist = predictions[predictions["model"] == winner].copy()
    hist["rel_err"] = hist["actual_weight"] / hist["pred_weight"].replace(0, np.nan) - 1
    hist = hist.dropna(subset=["rel_err"])
    hist["bucket"] = hist.groupby("test_quarter_idx")["pred_weight"].transform(
        lambda w: _size_bucket(w.values, cfg.sim_n_buckets))

    by_period = {}
    for q, d in hist.groupby("test_quarter_idx"):
        by_period[q] = dict(
            by_stock=d.set_index("stock")["rel_err"],
            all=d["rel_err"].values,
            by_bucket={b: g["rel_err"].values for b, g in d.groupby("bucket")},
        )
    keys = list(by_period)

    rng = np.random.default_rng(cfg.sim_seed)
    base = new_pred["predicted_weight"].values
    stocks = new_pred["stock"].values
    new_bucket = _size_bucket(base, cfg.sim_n_buckets)
    bucket_idx = {b: np.where(new_bucket == b)[0] for b in range(cfg.sim_n_buckets)}

    sims = np.empty((cfg.n_sims, len(base)))
    for s in range(cfg.n_sims):
        pool = by_period[keys[rng.integers(len(keys))]]
        errs = np.empty(len(base))
        if cfg.sim_match == "stock":
            errs = pool["by_stock"].reindex(stocks).to_numpy(dtype=float, copy=True)
            miss = np.isnan(errs)
            errs[miss] = rng.choice(pool["all"], size=miss.sum(), replace=True)
        elif cfg.sim_match == "size_bucket":
            for b, idx in bucket_idx.items():
                src = pool["by_bucket"].get(b, pool["all"])
                errs[idx] = rng.choice(src, size=len(idx), replace=True)
        else:
            errs = rng.choice(pool["all"], size=len(base), replace=True)
        w = np.clip(base * (1 + errs), 0, None)
        sims[s] = w / w.sum() * 100.0

    out = new_pred.copy()
    out["low"] = np.quantile(sims, cfg.low_q, axis=0)
    out["mid"] = np.quantile(sims, 0.5, axis=0)
    out["high"] = np.quantile(sims, cfg.high_q, axis=0)
    if verbose:
        print("Stage 8: simulate_range")
        print(f"  {cfg.n_sims} sims drawn from {len(keys)} past error period(s); "
              f"match={cfg.sim_match}; band = p{int(cfg.low_q*100)} .. p{int(cfg.high_q*100)}")
    return out


# --------------------------------------------------------------------------
# Stage 9: final forecast
# --------------------------------------------------------------------------

def final_forecast(train: pd.DataFrame, latest: pd.DataFrame, feat_cols: Sequence[str],
                   predictions: pd.DataFrame, winner: str, cfg: Config = Config(),
                   verbose: bool = True) -> pd.DataFrame:
    """Stage 9: retrain the winner on every usable period, forecast the next one."""
    models = make_models(cfg)
    covered = latest["has_features"].values
    pc = np.zeros(len(latest))
    fitted = fit_model(models[winner], train[feat_cols], train["target"], cfg)
    if covered.any():
        pc[covered] = np.asarray(fitted.predict(latest.loc[covered, feat_cols].values), float)
    final_forecast.weights = model_weights(fitted, feat_cols)
    pw = predictions_to_weights(latest["weight"].values, pc)
    new_pred = pd.DataFrame({
        "stock": latest["stock"].values,
        "name": latest["name"].values,
        "ticker": latest["ticker"].values,
        "country": latest["country"].values,
        "sector": latest["sector"].values,
        "base_quarter": latest["quarter"].values,
        "current_weight": latest["weight"].values,
        "predicted_change_%": pc * 100,
        "predicted_weight": pw,
        "model": np.where(covered, winner, "naive (no feature history)"),
    })
    out = simulate_range(new_pred, predictions, winner, cfg, verbose)
    out = out.sort_values("predicted_weight", ascending=False).reset_index(drop=True)
    if verbose:
        print("Stage 9: final_forecast")
        print(f"  model: {winner} | base period: {latest['quarter'].iloc[0]} | stocks: {len(out)}"
              f" ({int(covered.sum())} modelled, {int((~covered).sum())} carried at current weight)")
        if final_forecast.weights is not None:
            print("  feature weights (% weight change, bottom -> top percentile):")
            print(final_forecast.weights.round(2).to_string(index=False))
        print(out[["name", "current_weight", "predicted_weight", "low", "high"]].head(10).round(3).to_string())
    return out


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------

def run_pipeline(path: str, cfg: Config = Config(), out_dir: Optional[str] = None,
                 verbose: bool = True) -> Dict[str, object]:
    long = load_and_clean(path, verbose)
    basics = derive_basics(long, verbose)
    feats, feat_cols = build_features(basics, cfg, verbose)
    train, latest = build_target(feats, feat_cols, verbose)
    summary, predictions, winner = walk_forward(train, feat_cols, cfg, verbose,
                                                labels=period_labels(feats))
    forecast = final_forecast(train, latest, feat_cols, predictions, winner, cfg, verbose)

    result = dict(input_long=long, panel=feats, feat_cols=feat_cols, train=train,
                  latest=latest, model_comparison=summary, walk_forward_predictions=predictions,
                  winner=winner, forecast=forecast)

    if out_dir:
        import os
        os.makedirs(out_dir, exist_ok=True)
        long[BASE_COLS + ["quarter_idx", "name", "ticker", "isin", "country", "sector",
                          "holding_type", "weight_raw", "entered_mid", "exited_mid", "has_gap"]
             ].to_csv(os.path.join(out_dir, "input_long.csv"), index=False)
        summary.reset_index().to_csv(os.path.join(out_dir, "model_comparison.csv"), index=False)
        predictions.to_csv(os.path.join(out_dir, "walk_forward_predictions.csv"), index=False)
        deliverable = forecast[["stock", "name", "ticker", "country", "sector", "base_quarter",
                                "current_weight", "predicted_weight", "low", "mid", "high",
                                "predicted_change_%", "model"]]
        deliverable.to_csv(os.path.join(out_dir, "forecast_next_period.csv"), index=False)
        weights = getattr(final_forecast, "weights", None)
        if weights is not None:
            weights.to_csv(os.path.join(out_dir, "model_weights.csv"), index=False)
        with pd.ExcelWriter(os.path.join(out_dir, "forecast_next_period.xlsx")) as xw:
            deliverable.to_excel(xw, sheet_name="forecast", index=False)
            summary.reset_index().to_excel(xw, sheet_name="model_comparison", index=False)
            if weights is not None:
                weights.to_excel(xw, sheet_name="model_weights", index=False)
        if verbose:
            print(f"outputs written to {out_dir}/")
    return result
