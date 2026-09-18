# MSCI EM stock-weight prediction

Predicts each stock's weight in the MSCI Emerging Markets portfolio one period
ahead, with a low / high range, from the holdings history in
`data/raw/MSCI_EM_Portfolio_since_Dec-21.xlsx`.

```bash
pip install -r requirements.txt
python run_pipeline.py                 # defaults: lags 1 and 2, 1000 sims
python run_pipeline.py --lags 1        # sensitivity: lag-1 features only
python -m pytest -q tests              # no-look-ahead + rescaling checks
```

## Input

The workbook has one sheet per snapshot: **Dec-21, Dec-22, Dec-23, Dec-24,
Dec-25, Aug-26** (annual, not quarterly). Stage 1 stacks the sheets into one
long table (`output/input_long.csv`) with `stock, quarter, weight,
market_value, shares` plus descriptive columns and flags. The `quarter`
column keeps the blueprint's name but holds the sheet label.

Cleaning rules:

* keep only stock-like holding types (equity, units, REIT, undefined equity,
  preferred stock); cash, currency forwards, index futures, money-market
  funds and warrants are dropped, then weights are rescaled to 100 % within
  the remaining universe (raw equity sums were 99.6 to 99.9 %)
* drop the "Summary Statistics" block each sheet ends with
* drop rows with zero weight, zero market value or non-positive shares
  (92 rows across all sheets, all zero-weight residual positions)
* stock id = ISIN, else ticker + country, else name, so renames such as
  "SK Hynix" to "SK hynix" still line up
* stocks that appear or disappear mid-history are flagged
  (`entered_mid`, `exited_mid`, `has_gap`), never deleted

## Pipeline (`msci_em/pipeline.py`)

| Stage | Function | Output |
|---|---|---|
| 1 | `load_and_clean` | long table, one row per stock per period |
| 2 | `derive_basics` | `price`, `index_total`, `price_chg`, `share_chg`, `index_chg`, `rel_return` |
| 3 | `build_features` | lagged relative return and share change, relative-return volatility, size rank and its change, each percentile-ranked within the period |
| 4 | `build_target` | `target` = next-period % change in weight; last period kept aside |
| 5 | `make_models` | naive (0 change), Ridge over several alphas, XGBoost over a small grid |
| 6 | `predictions_to_weights` | `current × (1 + predicted change)`, floored at 0, rescaled to 100 |
| 7 | `walk_forward` | train on periods ≤ t, predict t+1, MAE of predicted vs actual weight per model |
| 8 | `simulate_range` | resample the winner's past errors by period, 1000 draws, p5 / p50 / p95 |
| 9 | `final_forecast` | retrain the winner on everything, forecast the period after Aug-26 |

Design choices worth knowing:

* **Lags.** With six annual snapshots a 4-period lag leaves no training
  rows, so the default lag set is (1, 2). `Config.lags` accepts (1, 2, 4)
  once quarterly data exists. Volatility uses a 4-period window with a
  2-period minimum.
* **Training target clipping.** The training target is winsorised at the
  1st / 99th percentile of the training rows only, so a handful of 10x
  weight jumps do not dominate the fit. Test targets are never clipped.
* **Walk-forward.** Rows are indexed by the feature period; a row's target
  realises one period later. Training on rows ≤ t therefore uses nothing
  after t+1, and the scored rows at t+1 forecast weights at t+2.
* **Coverage.** Stocks without enough history for the features (new
  entrants) are carried forward at their current weight and included in the
  rescale, so the deliverable spans the whole index. The `model` column
  says which rows were modelled.
* **Range.** Each simulation picks one past scored period and applies that
  period's relative errors (actual / predicted − 1) to the new predictions,
  drawn from stocks of similar size within that period, then rescales to
  100. `Config.sim_match` switches to per-stock or pooled matching.

## Results on this file

Walk-forward MAE in weight points, averaged over the two scorable periods
(predicting Dec-24 and Dec-25):

| model | mean MAE | vs naive |
|---|---|---|
| ridge α=0.1 | 0.0341 | −1.3 % |
| ridge α=1 | 0.0342 | −1.2 % |
| naive | 0.0346 | 0 |
| xgb depth 2, 100 trees | 0.0356 | +2.8 % |

Ridge beats naive by about one percent, XGBoost does not. With only two
scorable periods this is a thin margin, so treat the model as a mild tilt
around the carry-forward baseline rather than a strong signal. The lag-1
sensitivity run gives the same ordering.

## Outputs (`output/`)

* `forecast_next_period.csv` / `.xlsx`: `stock, name, ticker, country,
  sector, base_quarter, current_weight, predicted_weight, low, mid, high,
  predicted_change_%, model`. `current_weight` and `predicted_weight` each
  sum to 100; `low` / `high` are the p5 / p95 of the simulated weights.
* `model_comparison.csv`: the walk-forward table above with per-period MAE.
* `walk_forward_predictions.csv`: every out-of-sample prediction per model.
* `input_long.csv`: the cleaned long-format input.
