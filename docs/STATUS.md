# Build status

Project-level state for agents picking up cold. Update this file when a phase completes or a module ships.

## All modules: built and tested

| Module | Status | Notes |
|--------|--------|-------|
| `config.py` | Done | API credentials, `PREFERRED_STATIONS`, `SYDNEY_METRO_POSTCODES` |
| `history.py` | Done | Bulk CSV downloader + transformer |
| `db.py` | Done | SQLite schema; upsert/load helpers; all read helpers |
| `fill.py` | Done | Forward-fill per-station gaps into `daily_prices` |
| `live.py` | Done | FuelCheck OAuth2; all-NSW all-fuel-type snapshots |
| `series.py` | Done | `resolve()`, `resolve_members()`, `enumerate_groups()`, `SeriesError` |
| `cycle.py` | Done | `CycleDetector`; `detect(as_of_date)` → `CycleState`; 21 unit tests |
| `signal.py` | Done | Four-signal port; `combine_signals`; 38 unit + integration tests |
| `compare.py` | Done | Station/LGA/brand/sydney series comparison |
| `inspect.py` | Done | Flask workbench; line/scatter/heatmap charts; URL-driven state |
| `stations.py` | Done | Station lookup CLI |
| `labels.py` | Done | Two-condition BUY label; `lookback_days`, `percentile_pct` params |
| `label_viz.py` | Done | Diagnostic plots for label distributions |
| `label_inspect.py` | Done | Per-station per-day label decomposition table |
| `features.py` | Done | Cycle features + station-vs-aggregate features → model-ready CSV |
| `evaluate.py` | Done | Canonical train/val/test split; `split()`, `log_loss()`, `brier()`, `reliability_table()`, `walk_forward_folds()` |
| `train_logreg.py` | Done | `StandardScaler → LogisticRegression` pipeline; saves `data/models/logreg.joblib` |
| `calibrate.py` | Done | Isotonic calibration wins; saves `data/models/logreg_calibrated.joblib` |
| `score_phase2.py` | Done | Threshold sweep on val → τ=0.40 → test scored once → results.csv |
| `tp_benefit.py` | Done | Empirical TP benefit distribution |
| `fp_cost.py` | Done | Empirical FP cost distribution (bimodal) |
| `fn_cost.py` | Done | Empirical FN cost distribution |
| `backtest.py` | Done | `AlwaysBuyStrategy`, `RuleBasedSignalStrategy`, `ModelStrategy`; CPL table per station |
| `.github/workflows/daily-snapshot.yml` | Done | Daily cron + workflow_dispatch; confirmed working |

## Canonical train/val/test split (fixed — do not adjust after results are in)

| Split | Start | End |
|-------|-------|-----|
| Train | 2016-08-01 | 2025-03-17 |
| Val | 2025-03-25 | 2025-06-23 |
| Test | 2025-07-01 | 2025-12-31 |

7-day buffers between splits prevent label leakage.

## ML Phase results

### Phase 2 locked (2026-05-09)

Logistic regression on cycle features only. τ=0.40 chosen via val sweep + 0.05 adjustment for val/test BUY-rate gap.

| Model | Test logloss | Test brier | vs baseline |
|-------|-------------|------------|-------------|
| Marginal-rate baseline | 0.5821 | 0.1966 | — |
| Logreg cycle features (τ=0.40) | 0.4029 | 0.1346 | −0.179 logloss, −0.062 brier |

At τ=0.40: P=0.618, R=0.581, F1=0.599, BUY%=25% on test.

**Realised spend (preferred stations, test window 2025-07-01 → 2025-12-31):**

| Strategy | CPL (c/L) | vs always-buy |
|----------|-----------|---------------|
| Always-buy baseline | 191.78 | — |
| Logreg τ=0.40 (Phase 2 locked) | 190.35 | +0.74% |

Phase 3 target: beat 190.35 c/L.

## Pending work

### Phase 3: LightGBM with cross-station features
- Station features: `station_code` as categorical, brand, suburb
- Cross-station lead features: same-brand mean, same-LGA mean, leading-indicator LGAs
- Walk-forward CV report CLI (`cv_report.py`) — reads `walk_forward_folds()` output, reports per-fold variance
- Upstream features (TGP first, then MOPS/crude/FX) — Phase 4

### Phase 5 (macro model)
- Separate longer-horizon model (~30/60/90 days)
- Upstream commodity features dominate at this horizon

## Key architectural notes

- `station_code` primary key comes only from FuelCheck API live snapshot — `stations` table cannot be populated from historical CSVs alone.
- `stations.latitude/longitude` always NULL (API returns them but snapshot CSV doesn't include them).
- Storage format: `price_date INTEGER YYYYMMDD` (e.g. 20240101), `price_decicents INTEGER` (e.g. 1619 = 161.9c). Conversion is transparent at db.py boundary.
- `daily_average_e10` queries raw prices (not gap-filled); `sydney_average_series` queries `daily_prices` (gap-filled). Use `sydney_average_series` for cycle detection.
- Stations only upserted when they have ≥1 matching price row — prevents EV chargers and non-petrol venues from causing duplicate normalised-address collisions.
