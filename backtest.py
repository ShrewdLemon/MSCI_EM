"""
Backtest one period: forecast the weights of a period that is already in the
file, using only information available before it, then compare with what
actually happened.

    python backtest.py --forecast Dec-23
    python backtest.py --forecast Dec-24 --lags 1 --no-vol
    python backtest.py --forecast Dec-23 --in-sample      # fit check, not a backtest

How it works
------------
* base period  = the one just before --forecast (its weights are "current")
* training rows = rows whose *outcome* is known by the base period, i.e.
  feature period <= base - 1.  If there are none (Dec-23 with this file),
  only the naive model can be run and the script says so.
* every model in the pipeline is fitted on those rows and scored on the
  forecast period; the detailed table uses --model (default ridge_a0.1,
  the walk-forward winner).
* --in-sample trains on every usable period, including ones after the
  forecast.  That is a look-ahead: useful to see the fitted tilt, useless
  as evidence of skill.  The output file name carries the flag.
"""
import argparse
import os

import numpy as np
import pandas as pd

from msci_em.pipeline import (Config, attach_target, build_features, derive_basics,
                              fit_model, load_and_clean, make_models, model_weights,
                              period_labels, predictions_to_weights, simulate_range,
                              walk_forward)


def parse_args():
    ap = argparse.ArgumentParser(description="Backtest one period of MSCI EM weights")
    ap.add_argument("--forecast", required=True, help="sheet label to forecast, e.g. Dec-23")
    ap.add_argument("--input", default="data/raw/MSCI_EM_Portfolio_since_Dec-21.xlsx")
    ap.add_argument("--out", default="output")
    ap.add_argument("--model", default="ridge_a0.1")
    ap.add_argument("--lags", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--no-vol", action="store_true",
                    help="drop the volatility feature (needs 2 past returns)")
    ap.add_argument("--in-sample", action="store_true",
                    help="train on all periods, including later ones (look-ahead)")
    ap.add_argument("--n-sims", type=int, default=1000)
    ap.add_argument("--top", type=int, default=15, help="rows to print")
    return ap.parse_args()


def main():
    a = parse_args()
    cfg = Config(lags=tuple(a.lags), n_sims=a.n_sims, use_vol=not a.no_vol)

    # ---- stages 1-4 on the whole file --------------------------------------
    long = load_and_clean(a.input, verbose=False)
    feats, feat_cols = build_features(derive_basics(long, verbose=False), cfg, verbose=False)
    panel = attach_target(feats, feat_cols)
    labels = period_labels(panel)
    idx_of = {v: k for k, v in labels.items()}
    if a.forecast not in idx_of:
        raise SystemExit(f"{a.forecast!r} not in file; periods are {list(idx_of)}")
    f = idx_of[a.forecast]
    if f == 0:
        raise SystemExit(f"{a.forecast} is the first period; nothing to forecast it from")
    b = f - 1
    print(f"Forecasting {labels[f]} from {labels[b]} weights"
          f"{'  [IN-SAMPLE: look-ahead training]' if a.in_sample else ''}")

    # ---- base universe, actual outcome ---------------------------------------
    base = panel[(panel["quarter_idx"] == b) & panel["weight"].notna()].copy()
    actual_all = panel[(panel["quarter_idx"] == f) & panel["weight"].notna()]
    actual_by_stock = actual_all.set_index("stock")["weight"]
    entrants = actual_all[~actual_all["stock"].isin(base["stock"])]
    base["actual_weight"] = actual_by_stock.reindex(base["stock"]).fillna(0.0).values
    base["actual_weight"] *= 100.0 / base["actual_weight"].sum()   # base-universe share
    print(f"  base universe: {len(base)} stocks | deleted by {labels[f]}: "
          f"{int((base['actual_weight'] == 0).sum())} | new entrants (not forecastable): "
          f"{len(entrants)} holding {entrants['weight'].sum():.2f}% of {labels[f]}")

    # ---- training rows -------------------------------------------------------
    usable = panel[panel["has_features"] & panel["target"].notna() & panel["weight"].notna()]
    train = usable if a.in_sample else usable[usable["quarter_idx"] <= b - 1]
    covered = base["has_features"].values
    print(f"  training rows: {len(train)} from periods "
          f"{[labels[i] for i in sorted(train['quarter_idx'].unique())] or 'NONE'}"
          f" | base rows with features: {int(covered.sum())}/{len(base)}")

    # ---- fit every model, score on the forecast period -----------------------
    models = make_models(cfg)
    can_model = len(train) > 0 and covered.any()
    if not can_model:
        print("  -> no training data exists before the base period: only the naive "
              "forecast (carry weights forward) is possible here.")
    rows, changes, fitted = [], {}, {}
    for name, factory in models.items():
        if name != "naive" and not can_model:
            continue
        pc = np.zeros(len(base))
        if name != "naive":
            fitted[name] = fit_model(factory, train[feat_cols], train["target"], cfg)
            pc[covered] = np.asarray(fitted[name].predict(base.loc[covered, feat_cols].values), float)
        pw = predictions_to_weights(base["weight"].values, pc)
        changes[name] = (pc, pw)
        err = pw - base["actual_weight"].values
        rows.append({"model": name, "MAE": np.abs(err).mean(),
                     "RMSE": np.sqrt((err ** 2).mean()),
                     "top50_MAE": np.abs(err)[np.argsort(-base["weight"].values)[:50]].mean()})
    table = pd.DataFrame(rows).set_index("model").sort_values("MAE")
    table["vs_naive_%"] = (table["MAE"] / table.loc["naive", "MAE"] - 1) * 100
    print(f"\nScore on {labels[f]} (weight points, base universe):")
    print(table.round(5).to_string())

    chosen = a.model if a.model in changes else "naive"
    if chosen != a.model:
        print(f"\n  model {a.model!r} unavailable here; detailed table uses naive")
    pc, pw = changes[chosen]
    weights = model_weights(fitted[chosen], feat_cols) if chosen in fitted else None
    if weights is not None:
        print(f"\nFeature weights of {chosen} (% weight change, bottom -> top percentile):")
        print(weights.round(2).to_string(index=False))
    detail = pd.DataFrame({
        "stock": base["stock"].values, "name": base["name"].values,
        "ticker": base["ticker"].values, "country": base["country"].values,
        "base_quarter": labels[b], "forecast_quarter": labels[f],
        "current_weight": base["weight"].values, "predicted_change_%": pc * 100,
        "predicted_weight": pw, "actual_weight": base["actual_weight"].values,
        "model": np.where(covered | (chosen == "naive"), chosen, "naive (no feature history)"),
    })
    detail["error"] = detail["predicted_weight"] - detail["actual_weight"]
    detail["naive_error"] = detail["current_weight"] - detail["actual_weight"]

    # ---- range from past errors (only from folds strictly before the base) ---
    detail["low"] = detail["high"] = np.nan
    if can_model and chosen != "naive" and train["quarter_idx"].nunique() >= 2:
        _, past_preds, _ = walk_forward(train, feat_cols, cfg, verbose=False, labels=labels)
        past_preds = past_preds[past_preds["model"] == chosen]
        if not past_preds.empty:
            band = simulate_range(detail, past_preds, chosen, cfg, verbose=False)
            detail["low"], detail["high"] = band["low"], band["high"]
            inside = ((detail.low <= detail.actual_weight) & (detail.actual_weight <= detail.high)).mean()
            print(f"\n  p5-p95 band from {past_preds['forecast_quarter'].nunique()} earlier fold(s); "
                  f"actual inside band for {inside:.0%} of stocks")
    else:
        print("\n  no earlier out-of-sample folds -> no low/high band for this period")

    # ---- print + export ------------------------------------------------------
    show = ["name", "current_weight", "predicted_weight", "actual_weight", "error"]
    print(f"\nLargest {a.top} stocks ({chosen}):")
    print(detail.sort_values("current_weight", ascending=False).head(a.top)[show].round(3).to_string(index=False))
    print(f"\nBiggest misses ({chosen}):")
    print(detail.reindex(detail["error"].abs().sort_values(ascending=False).index)
          .head(a.top)[show].round(3).to_string(index=False))

    os.makedirs(a.out, exist_ok=True)
    tag = f"backtest_{labels[f]}{'_in_sample' if a.in_sample else ''}"
    detail = detail.sort_values("current_weight", ascending=False)
    detail.to_csv(os.path.join(a.out, f"{tag}.csv"), index=False)
    with pd.ExcelWriter(os.path.join(a.out, f"{tag}.xlsx")) as xw:
        detail.to_excel(xw, sheet_name="stocks", index=False)
        table.reset_index().to_excel(xw, sheet_name="model_scores", index=False)
        if weights is not None:
            weights.to_excel(xw, sheet_name="model_weights", index=False)
    print(f"\nwritten: {a.out}/{tag}.csv and .xlsx")


if __name__ == "__main__":
    main()
