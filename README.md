# fuel-price-signal

A Python CLI that outputs a one-line buy/don't-buy signal for E10 fuel at preferred stations near postcode 2777 (Springwood/Blue Mountains corridor).

```
BUY  | Day 41/46 of cycle | E10 @ Caltex Springwood: 161.9c
WAIT | Day 12/46 of cycle | E10 @ Caltex Springwood: 179.2c
```

## Setup

```bash
uv sync
```

Create a `.env` file with your FuelCheck API credentials:

```
FUELAPI_API_KEY=your_key_here
FUELAPI_API_SECRET=your_secret_here
```

## Building the database

The signal runs from a local SQLite database (`fuel_signal.db`, gitignored) and a trained model (`data/models/`, gitignored). Everything is rebuilt from committed inputs: historical CSVs downloaded from data.nsw.gov.au plus the daily snapshot CSVs already tracked in `data/snapshots/`.

### Build from scratch (full sequence)

Run this once for a clean rebuild. Each step is explained in the subsections below.

```bash
uv run python -m fuel_signal.history                                    # 1. download + clean historical CSVs
uv run python -m fuel_signal.db                                         # 2. load snapshots + history into SQLite
uv run python -m fuel_signal.fill                                       # 3. forward-fill daily price gaps
uv run python -m fuel_signal.classify --start-date 2016-08-01          # 4. classify stations
uv run python -m fuel_signal.lga_leadership --start-date 2016-08-01    # 5. populate lga_leadership table
uv run python -m fuel_signal.features                                   # 6. assemble ML feature rows
uv run python -m fuel_signal.train_lgbm                                 # 7. train LightGBM (Phase 4 default)
uv run python -m fuel_signal.calibrate --skip-results-csv              # 8. calibrate (lgbm defaults)
uv run python -m fuel_signal.score_phase2                               # 9. final test-set eval (run once; loads calibrated artifact by default)
uv run python -m fuel_signal.shap_report \
    --model data/models/lgbm.joblib \
    --features data/features.csv \
    --output experiments/shap_phase4                                    # 10. SHAP analysis + partner scores
```

You do **not** need to run `fuel_signal.live` first — station reference data comes from the snapshot CSVs committed in `data/snapshots/`. Run `live` only to pull today's prices manually (see below).

### 1. Download and clean historical CSVs

Downloads all bulk price history from data.nsw.gov.au (~2016–present) into `data/raw/`, then cleans into `data/cleaned/`. Both directories are gitignored. Files already present are skipped, so re-running is safe.

```bash
uv run python -m fuel_signal.history
```

Takes a few minutes on first run (100+ files).

### 2. (Optional) Collect a live snapshot

Station reference data (codes, addresses) is already provided by the snapshot CSVs committed in `data/snapshots/`, so a from-scratch build does **not** require this step. Run it only when you want to pull today's prices:

```bash
uv run --env-file .env python -m fuel_signal.live
```

This writes one snapshot CSV to `data/snapshots/YYYY/MM/YYYY-MM-DD.csv` and is also what GitHub Actions runs daily.

### 3. Load everything into SQLite

```bash
uv run python -m fuel_signal.db
```

Loads all snapshot CSVs (from `data/snapshots/`) then all historical cleaned CSVs (from `data/cleaned/`).

### 4. Forward-fill daily price gaps

```bash
uv run python -m fuel_signal.fill
```

Rebuilds the `daily_prices` table by forward-filling gaps between observations. Required after `db` — analysis commands read from `daily_prices`, not from the raw observations.

### 5. Classify stations (required before assembling features)

Classifies each station per date as Competitive, Sticky, or Discount based on its 45-day median price premium relative to LGA peers. Must run after `fill` and before `features` — the LGA/brand mean feature joins rely on `station_class`.

```bash
# Single snapshot (today):
uv run python -m fuel_signal.classify

# Backfill from a start date (use this for first-time setup):
uv run python -m fuel_signal.classify --start-date 2016-08-01

# Backfill up to a specific end date:
uv run python -m fuel_signal.classify --start-date 2016-08-01 --snapshot-date 2026-01-01
```

Writes `station_class` and `classification_summary` tables. Idempotent — re-running a date range is safe. This is step 4 of the [full build sequence](#build-from-scratch-full-sequence) above; the remaining steps (`lga_leadership` → `features` → `train_lgbm` → `calibrate` → `score_phase2`) build the lead-lag table and train and evaluate the model.

## Inspecting the data

Starts a local Flask workbench and opens it in your browser:

```bash
uv run python -m fuel_signal.inspect
# Custom host/port, no auto-open:
uv run python -m fuel_signal.inspect --port 5001 --no-browser
# Point /features at a different SHAP artifact directory:
uv run python -m fuel_signal.inspect --shap-dir experiments/shap_phase4/
```

The workbench is a single GET-driven page — all state lives in the URL query string, so views are bookmarkable and shareable. E10 only.

**Available series types** (select via the controls form or pass as `?series=` params):
- `sydney` — Sydney metro E10 mean
- `lga:Name` — LGA average (e.g. `lga:Penrith`, `lga:Blue Mountains`)
- `brand:Name` — brand average (e.g. `brand:Ampol`)
- `station:CODE` — specific station by numeric code

**Chart types:**
- **Line** — up to 10 series; peak/gap annotations when Sydney avg is selected
- **Scatter** — station-day points coloured by brand; switch to `metric=gradient` for 7-day slope view
- **Gradient heatmap** — LGA × week price-slope table (blue=falling, red=rising)
- **Coverage heatmap** — station × month observation counts

**Cycle state box** is always computed against the Sydney metro average (matches the CLI signal), regardless of what's plotted.

**Group display** toggle (mean / individual stations / both) applies to `lga:` and `brand:` series on line and scatter charts.

**Standalone pages:**
- `/lead-lag` — lead/lag table showing how much earlier or later each series (LGA, brand, or station) reaches the Sydney metro trough, relative to a configurable reference series.
- `/classification-health` — surfaces `classification_summary` per LGA: Competitive/Sticky/Discount counts, ever-zero LGAs (where no Competitive stations were found), and a 90-day competitive-count heatmap.
- `/features` — per-feature SHAP analysis from the artifact emitted by `shap_report.py`. Ranked table (mean|SHAP|, signed r, NaN%) with click-to-drill-down dependence plots. Each row has a **Partners** dropdown (hybrid cutoff: top-6 or all ≥50% of top-1 score) — selecting a partner navigates to `?feature=X&interaction=Y` and generates a dependence plot coloured by that partner (on-demand, disk-cached). The side panel shows the feature's interaction-budget rank and a "Reset to auto" link when a specific interaction is active. A staleness banner fires when `lgbm.joblib` is newer than `shap_values.npy`. Defaults to `experiments/shap_phase4/`; use `--shap-dir` to point at another phase.

## Station lookup

Find station codes by suburb or name — useful when adding entries to `PREFERRED_STATIONS` in `config.py`:

```bash
# Free-text search (matches suburb and name)
uv run python -m fuel_signal.stations blaxland
uv run python -m fuel_signal.stations "emu plains"

# Look up by station code (to find the name for a known ID)
uv run python -m fuel_signal.stations 414

# Field-specific filters
uv run python -m fuel_signal.stations --suburb springwood
uv run python -m fuel_signal.stations --name ampol

# List all stations
uv run python -m fuel_signal.stations
```

Output includes `station_code`, suburb, name, and brand. Use the `station_code` value in `PREFERRED_STATIONS`.

> **Note:** some stations share a name (e.g. two "7-Eleven Emu Plains" in different suburbs). In that case use the station code to refer to a specific one.

## Comparing price series

Compare how often one station or area is cheaper than another:

```bash
# Station vs Sydney metro average
uv run python -m fuel_signal.compare "BP Springwood" sydney

# Station by code vs LGA average (use station:CODE when multiple stations share a name)
uv run python -m fuel_signal.compare station:182 "lga:penrith"

# Two stations head-to-head
uv run python -m fuel_signal.compare "Ampol Springwood" "Shell Blaxland"

# Brand average vs Sydney average
uv run python -m fuel_signal.compare "brand:Ampol" sydney

# Treat prices within 0.2c as equal (default 0.5c)
uv run python -m fuel_signal.compare "BP Springwood" sydney --within 0.2
```

Each series can be:
- A station name (partial match against station name only; must be unique) or `station:CODE`
- `sydney` — Sydney metro E10 average
- `lga:<name>` or `council:<name>` — average for a specific LGA
- `brand:<name>` — average for a specific brand

If a name search matches multiple stations, a list of `station:CODE` alternatives is shown.

## Getting the signal

```bash
# Signal as of today (latest date in DB)
uv run python -m fuel_signal.signal

# Signal as of a specific historical date (useful for validation)
uv run python -m fuel_signal.signal --as-of 2026-02-15

# Custom DB path
uv run python -m fuel_signal.signal --db /path/to/fuel_signal.db
```

Output is the combined verdict (one line per preferred station) followed by the contributing signals:

```
[as of 2026-01-10]
BUY  | Day 27/35 of cycle | E10 @ BP Valley Heights: 159.9c
BUY  | Day 27/35 of cycle | E10 @ Shell Blaxland: 157.5c
Combined: BUY (mean signal +1.00)
  AverageCycleTimeSignal: BUY — cycle ending soon (73% through cycle; day 26 / 35.5)
  AverageGradientAfterPeakSignal: NEUTRAL — price has not flatlined
  AverageNearPreviousMinMaxSignal: BUY — price close to low in last cycle
  FavouriteServiceStationPriceGradientSignal: NEUTRAL — no preferred stations raising sharply
```

## Daily snapshots

GitHub Actions commits one snapshot CSV per day to `data/snapshots/`. To enable it, add `FUELAPI_API_KEY` and `FUELAPI_API_SECRET` as repository secrets under **Settings → Secrets and variables → Actions**.

## Generating ML training labels

Assemble a training table with one row per (station, date) that has a computable label:

```bash
# Default: 7-day horizon, 3c threshold, output to data/labels.csv
uv run python -m fuel_signal.labels

# Custom horizon and threshold
uv run python -m fuel_signal.labels --horizon 14 --threshold 5.0

# Custom output path
uv run python -m fuel_signal.labels --output /tmp/labels.csv
```

Each row contains `station_code`, `price_date`, `today_price_cents`, `future_min_cents`, and `label`. `label=1` (BUY) when **both** conditions hold: no cheaper price arrives within `--horizon` days (by more than `--threshold` cents), **and** today's price is at or below the `--percentile`th percentile of the past `--lookback` days. Rows without sufficient forward or lookback history are excluded.

### Diagnosing label distributions

```bash
# Produce two diagnostic plots in data/
uv run python -m fuel_signal.label_viz

# Custom input / output location
uv run python -m fuel_signal.label_viz --input /tmp/labels.csv --output /tmp/plots/
```

Writes `positive_rate_by_date.png` (fraction of stations labeled BUY per day — should oscillate with the ~45d price cycle) and `positive_rate_by_station.png` (histogram of per-station BUY rates — healthy distribution clusters near the marginal rate with no stations stuck at 0% or 100%).

### Inspecting individual label decisions

```bash
# Show per-day label decomposition for one station (21 days from --date)
uv run python -m fuel_signal.label_inspect --station 585 --date 2024-01-15

# Adjust window length or label parameters
uv run python -m fuel_signal.label_inspect --station 414 --date 2023-06-01 --days 30
uv run python -m fuel_signal.label_inspect --station 585 --date 2024-08-01 --horizon 14 --threshold 5.0
```

Prints a table showing `today_price`, `future_min`, the rolling `P33` threshold, and the two condition flags (`Cheap?`, `NoDrop?`) alongside the final label for each day. Useful for understanding why a specific date was or wasn't labeled BUY.

## Assembling ML feature rows

Join cycle features onto the labels table to produce a model-ready training set:

```bash
# Default: 7-day horizon, 3c threshold, output to data/features.csv
uv run python -m fuel_signal.features

# Custom horizon and threshold
uv run python -m fuel_signal.features --horizon 14 --threshold 5.0

# Custom output path
uv run python -m fuel_signal.features --output /tmp/features.csv
```

Requires `classify` to have run first — the LGA/brand mean joins fail silently to NULL otherwise.

Output includes all label columns (`station_code`, `price_date`, `today_price_cents`, `future_min_cents`, `label`) plus 15 feature columns:

- **Cycle features:** `cycle_pct_through`, `cycle_days_since_peak`, `cycle_mean_length`, `cycle_last_min_cents`, `cycle_last_max_cents`, `cycle_peak_count`
- **Station-vs-aggregate features:** `station_price_cents`, `station_minus_last_min_cents`, `station_minus_last_max_cents`, `station_minus_sydney_avg_cents`
- **LGA/brand mean features (Phase 3):** `lga_mean_cents`, `station_minus_lga_mean_cents`, `brand_mean_cents`, `station_minus_brand_mean_cents`
- **Station identity features:** `stickiness_score` — 45-day median of `station_price − LGA-Competitive-cluster median` (cents), sourced from `station_class.median_premium_decicents`. Provides a dedicated channel for the persistent-price-identity signal. Sticky stations receive the largest scores. NaN when no `station_class` row exists for that (station, date) pair.

The `stickiness_score` and the four LGA/brand mean columns can be NaN when `station_class` data is absent for that (station, date) pair. Rows are kept rather than dropped — downstream training scripts must handle the NaN (e.g. with imputation or a NaN-tolerant model like LightGBM).

Rows with insufficient history for cycle detection are excluded.

## Evaluation harness

`fuel_signal/evaluate.py` defines the canonical train/val/test split for the ML model and provides scoring utilities. The split is fixed — never adjust it after results are in.

| Split    | Start      | End        |
|----------|------------|------------|
| Train    | 2016-08-01 | 2025-03-17 |
| Val      | 2025-03-25 | 2025-06-23 |
| Test     | 2025-07-01 | 2025-12-31 |

7-day buffers between splits prevent label leakage (labels look 7 days forward). Val is sized for hyperparameter tuning; test is touched only for final evaluation.

Experiment results are appended to `experiments/results.csv` via `log_experiment()`. The baseline row (constant predictor at the train marginal rate ≈ 0.26) is the floor every model must beat:

```python
from fuel_signal.evaluate import split, baseline_prior, log_loss, brier, log_experiment
import numpy as np

train, val, test = split(df)  # df has price_date and label columns
p = baseline_prior(train)
pred = np.full(len(test), p)
print(log_loss(test["label"].values, pred))   # ≈ 0.573
print(brier(test["label"].values, pred))      # ≈ 0.192

log_experiment("my_model", features=["cycle_pct_through"], holdout_logloss=0.52, brier=0.18)
```

## Training the logistic regression baseline

The first real ML model — a vanilla logistic regression on the cycle features. Train on the canonical train split, score on val. Test is reserved for the locked final-model evaluation, so this command does **not** write to `experiments/results.csv`.

```bash
# Default: reads data/features.csv, writes data/models/logreg.joblib
# and experiments/reliability_logreg_val.png
uv run python -m fuel_signal.train_logreg

# Custom paths
uv run python -m fuel_signal.train_logreg \
    --features-csv /tmp/features.csv \
    --model-out /tmp/logreg.joblib \
    --reliability-out /tmp/reliability.png
```

Pipeline: `StandardScaler` → `LogisticRegression(max_iter=1000)`. Output prints train/val sizes and class balance, val log-loss / Brier, and the delta versus the constant-predictor baseline. The reliability plot uses 10 quantile bins with a `y=x` reference line; points below the diagonal indicate over-confidence, above indicate under-confidence.

## Walk-forward cross-validation report

Paired comparison of two joblib model artifacts across all pre-test folds. Re-trains both on each 90-day fold and reports per-fold logloss delta (model − baseline). Use this before locking a Phase upgrade to confirm that the val improvement holds across fold windows and is not an artifact of the canonical val window.

```bash
uv run python -m fuel_signal.cv_report \
  --model data/models/lgbm.joblib \
  --baseline data/models/lgbm_phase3c.joblib \
  --features data/features.csv \
  --seed 42 \
  --output experiments/cv_phase4/results.csv
```

Output: one line per fold (`val start→end`, `n_val`, `baseline=`, `model=`, `Δ=`) followed by a summary line (`folds`, `wins/n`, `median Δ`, `mean Δ`). Folds where `Δ > +0.05` are listed as named regressions. The `--output` CSV has columns: `fold_idx, train_start, train_end, val_start, val_end, n_val, baseline_logloss, model_logloss, delta`.

## SHAP analysis

`fuel_signal/shap_report.py` runs TreeExplainer on a fitted joblib model and emits three artifacts:

| Artifact | Contents |
|---|---|
| `shap_values.npy` | `(n_rows, n_features)` raw SHAP values — reused by ad-hoc notebooks and the `/features` inspect view |
| `summary.csv` | Per-feature: `mean_abs_shap`, `rank`, `sign_of_r` (sign of Pearson r between feature and SHAP), `nan_fraction` |
| `dependence/<feature>.png` | Scatter of feature value vs. SHAP value, one PNG per feature |

```bash
# Val split (default) — standard per-phase diagnostic
uv run python -m fuel_signal.shap_report \
    --model data/models/lgbm.joblib \
    --features data/features.csv \
    --output experiments/shap_phase4/

# Test split
uv run python -m fuel_signal.shap_report \
    --model data/models/lgbm.joblib \
    --features data/features.csv \
    --split test \
    --output experiments/shap_test/
```

Prints a ranked table (top 25 features) with `mean|SHAP|`, sign of correlation, and NaN fraction. NaN-bearing LGA features (stations below the 3-station floor) are handled without error; their SHAP contributions are computed from non-NaN rows.

## Training the LightGBM baseline (Phase 3a.1)

Vanilla LightGBM on the **same 10 features** as Phase 2 — no new features, no tuning, `random_state=42`. No `StandardScaler` (trees are scale-invariant). This is the apples-to-apples model-class comparison. Does **not** write to `experiments/results.csv`.

```bash
# Default: every feature in the CSV — FEATURE_COLUMNS + LGA_FEATURE_COLUMNS
# + any brand trough columns discovered in the header. Phase 4b when brand
# cols are present, Phase 4 otherwise. Reads data/features.csv, writes
# data/models/lgbm.joblib.
uv run python -m fuel_signal.train_lgbm

# Opt-out: Phase 4 (ignore brand trough columns even when present)
uv run python -m fuel_signal.train_lgbm --no-brand-features

# Opt-out: Phase 3c schema (15 features only)
uv run python -m fuel_signal.train_lgbm --no-lga-features

# Custom paths
uv run python -m fuel_signal.train_lgbm \
    --features-csv /tmp/features.csv \
    --model-out /tmp/lgbm.joblib \
    --reliability-out /tmp/reliability_lgbm.png
```

**Phase 3a.1 val result** (2026-05-14, real DB): val logloss 0.3926 (baseline 0.6428, Δ −0.2501) vs logreg val logloss 0.4112. LGBM captures non-linearities logreg cannot.

## Feature diagnostics (LightGBM)

Prints feature importance, FN/FP mean-delta analysis, and an error-group summary against the canonical val split. Use this to understand which features drive misclassifications.

```bash
# Default: reads data/models/lgbm_calibrated.joblib and data/features.csv
uv run python -m fuel_signal.feature_diagnostics

# Custom model artifact or threshold
uv run python -m fuel_signal.feature_diagnostics --model-path data/models/lgbm_calibrated.joblib
uv run python -m fuel_signal.feature_diagnostics --threshold 0.35
```

Outputs three sections: (1) gain % and split count per feature sorted by gain; (2) FN−TP and FP−TN mean delta per feature sorted by |FN−TP|, showing where the model mis-ranks BUY vs WAIT rows; (3) TP/FP/TN/FN counts and predicted-BUY rate.

## LOO ablation (feature contribution check)

Measures whether dropping one or more features hurts, helps, or has no effect on val logloss. Fits LightGBM at multiple seeds with and without the dropped column(s), then reports mean ± std and a one-line verdict.

```bash
# Ablate a single feature (5-seed protocol)
uv run python -m fuel_signal.loo_ablation \
    --features-csv data/features.csv \
    --drop station_minus_lga_mean_cents \
    --seeds 1,7,42,99,2024

# Ablate a group of features at once
uv run python -m fuel_signal.loo_ablation \
    --features-csv data/features.csv \
    --drop lga_mean_cents \
    --drop station_minus_lga_mean_cents \
    --seeds 1,7,42,99,2024
```

`--drop` is repeatable. Omitting it entirely is an error (`nothing to ablate`). Each named column must appear in `FEATURE_COLUMNS` or the command exits with a clear error.

**Verdict thresholds** (relative to baseline std across seeds):

| Condition | Verdict |
|---|---|
| \|Δ\| < baseline_std | `within noise / redundant` |
| Δ > 0, outside band | `feature contributes (starved)` |
| Δ < 0, outside band | `feature harmful (unexpected)` |

Δ = LOO mean − baseline mean; positive means removing the feature(s) raised logloss (feature was useful).

## Calibrating the model

Check calibration quality and produce a calibrated model artifact. Works with any fitted model (logreg or LightGBM).

```bash
# Default: reads data/features.csv + data/models/lgbm.joblib,
# writes data/models/lgbm_calibrated.joblib
uv run python -m fuel_signal.calibrate

# Skip writing to experiments/results.csv (e.g. during pipeline rebuild)
uv run python -m fuel_signal.calibrate --skip-results-csv

# Custom model artifact (e.g. logreg)
uv run python -m fuel_signal.calibrate \
    --model-in data/models/logreg.joblib \
    --model-out data/models/logreg_calibrated.joblib \
    --model-name logreg
```

Reports class balance (BUY rate) for all splits and prints a 10-bin reliability table on val. If miscalibrated (max |gap| > 0.05), compares sigmoid (Platt) vs isotonic calibration wrappers and saves the better one. Calibration uses `sklearn.base.clone` of the input model, so it works generically for any sklearn-compatible estimator. Appends a result row to `experiments/results.csv`.

## Cost model diagnostics

Three commands ground the TP reward and FP/FN penalties used in `score_phase2.py` in empirical data.

### TP benefit

Measures how much cheaper label=1 days are compared to the subsequent `--horizon` days at the same station:

```bash
uv run python -m fuel_signal.tp_benefit
uv run python -m fuel_signal.tp_benefit --horizon 14 --plot data/tp_benefit_14d.png
```

### FP cost

Shows the actual damage of a false-positive BUY on label=0 days. The label=0 population is bimodal: cluster A (only the percentile gate failed — small damage) vs cluster B (a cheaper price was coming — larger damage):

```bash
uv run python -m fuel_signal.fp_cost
uv run python -m fuel_signal.fp_cost --features-csv data/features.csv --plot data/fp.png --threshold 3.0
```

### FN cost

Measures the cost of a false-negative WAIT on label=1 days — the price `--delay` days after a missed BUY opportunity:

```bash
uv run python -m fuel_signal.fn_cost
uv run python -m fuel_signal.fn_cost --delay 14 --plot data/fn_cost_14d.png
```

## Phase 2 final evaluation (lock the model)

Threshold sweep on val → pick τ → **score test once** → append to `experiments/results.csv`. Run this command once to lock Phase 2. Do not re-run to tune τ after seeing test results.

Also used for Phase 3+ models via `--model-path` and `--model-name`.

```bash
# Default: loads data/models/lgbm_calibrated.joblib, runs realised-spend backtest
# against ./fuel_signal.db, writes lgbm_cycle_features to results.csv
uv run python -m fuel_signal.score_phase2

# Custom artifact or name
uv run python -m fuel_signal.score_phase2 \
    --model-path data/models/lgbm_calibrated.joblib \
    --model-name lgbm_cycle_features

# Point the backtest at a non-default DB
uv run python -m fuel_signal.score_phase2 --db /path/to/fuel_signal.db

# Skip the backtest (e.g. quick CSV-only sniff-test)
uv run python -m fuel_signal.score_phase2 --no-backtest
```

**What it does:**

1. Loads the model artifact at `--model-path` (default: `data/models/lgbm_calibrated.joblib`) and scores val directly.
2. Sweeps τ ∈ [0.05, 0.95] (step 0.05) on val — prints precision, recall, F1, BUY%, and expected-cents-saved per row.
3. Picks τ = argmax(expected cents/row on val), then applies a model-aware adjustment for val's elevated BUY rate: **+0.05 for raw artifacts, 0.00 for isotonic-calibrated artifacts**. Override with `--tau-adjustment <float>` (e.g. `--tau-adjustment 0.00` to disable).
4. Runs the realised-spend backtest at chosen τ over the test window using `--db` (default: `./fuel_signal.db`), populating `realised_spend_cpl` and `realised_savings_vs_always_buy_pct`. Skipped silently when the DB file or `--model-path` are absent (e.g. CI); use `--no-backtest` to skip explicitly.
5. Scores test at chosen τ. Appends one row to `experiments/results.csv` using `--model-name`.

**Cost model:** TP → +6.37c; FP → −5.80c; FN → −11.14c.

**Calibration warning:** If `--model-path` points at a raw (uncalibrated) artifact and `--tau-adjustment` is not passed explicitly, the CLI prints a `WARNING:` line surfacing the implicit `+0.05` default — the artifact filename (`lgbm_calibrated.joblib`) does not distinguish raw from isotonic, so this warning is the only visible signal that a raw model was loaded.

**Phase 2 result** (2026-05-09, real DB):

| Model | Test logloss | Test brier | vs baseline |
|---|---|---|---|
| Marginal-rate baseline | 0.5821 | 0.1966 | — |
| Logreg cycle features (τ=0.40) | 0.4029 | 0.1346 | −0.1792 logloss, −0.0620 brier |

At τ=0.40: precision=0.618, recall=0.581, F1=0.599, BUY rate=25.0% on test.

## Phase 2 τ re-validation on realised spend (Issue #64)

Sweeps τ ∈ [0.30, 0.70] on the test window via the backtest engine. Use `--no-patch` to dry-run; without it, patches `experiments/results.csv` with realised-spend columns for the Phase 2 and always-buy baseline rows.

```bash
# Dry-run: print sweep table only, do not patch results.csv
uv run python -m fuel_signal.backtest_phase2 \
    --model-path data/models/logreg_calibrated.joblib --no-patch

# Run and patch results.csv (default)
uv run python -m fuel_signal.backtest_phase2 \
    --model-path data/models/logreg_calibrated.joblib
```

**Phase 2 realised-spend result** (2026-05-10, preferred stations, test window 2025-07-01 → 2025-12-31):

| Strategy | CPL (c/L) | vs always-buy |
|---|---|---|
| Always-buy baseline | 191.78 | — |
| Logreg τ=0.40 (Phase 2) | 190.35 | +0.74% |
| Logreg τ=0.30 (spend-optimal) | 189.35 | +1.27% |

Spend-optimal τ=0.30 beats τ=0.40 by 1.01 c/L (≈0.5%). Phase 3 must beat 190.35 c/L.

## Phase 3a.1 LightGBM baseline (Issue #73)

Apples-to-apples LightGBM vs logreg — same 10 features, vanilla defaults, `random_state=42`.

```bash
# Full sequence: train → calibrate → score → backtest (all use Phase 4 defaults)
uv run python -m fuel_signal.train_lgbm
uv run python -m fuel_signal.calibrate --skip-results-csv
uv run python -m fuel_signal.score_phase2 --model-name lgbm_cycle_features
uv run python -m fuel_signal.backtest_phase2 \
    --model-path data/models/lgbm_calibrated.joblib --no-patch
```

**Phase 3a.1 result** (2026-05-14, real DB):

Calibration: LGBM is heavily over-confident out of the box (max |gap| = 0.38). Isotonic calibration chosen: val logloss 0.3926 (raw) → 0.3613 (isotonic), vs sigmoid 0.3904.

| Model | Val logloss | Test logloss | Test brier | vs baseline |
|---|---|---|---|---|
| Marginal-rate baseline | — | 0.5579 | 0.1855 | — |
| Logreg (Phase 2, τ=0.40) | 0.4112 | 0.4029 | 0.1346 | −0.155 logloss, −0.051 brier |
| LightGBM (Phase 3a.1, τ=0.65) | 0.3613 | 0.3444 | 0.1110 | −0.214 logloss, −0.074 brier |

LGBM val logloss beats raw-logreg val logloss (0.3926 < 0.4112) ✓

Realised-spend backtest (τ sweep, preferred stations, test window 2025-07-01 → 2025-12-31):

| τ | CPL (c/L) | vs always-buy |
|---|---|---|
| 0.30 | 185.02 | +3.53% |
| 0.35 | 186.60 | +2.70% |
| 0.40 | 186.60 | +2.70% |
| 0.45 | 187.34 | +2.31% |
| 0.50 | 188.83 | +1.54% |
| 0.55 | 189.37 | +1.26% |
| 0.60 | 189.57 | +1.15% |
| **0.65** | **189.57** | **+1.15%** ← chosen |
| 0.70 | 189.69 | +1.09% |

LGBM τ=0.65: **189.57 c/L (+1.15% vs always-buy)**, beating Phase 2 logreg τ=0.40 (190.35, +0.74%) by **0.78 c/L**. Spend-optimal τ=0.30 (185.02 c/L); gap to chosen τ is 4.55 c/L — the val-based τ selection is conservative relative to the realised-spend optimum.

## Backtesting purchasing strategies

Replay a purchasing strategy over historical prices and compare realised spend against an always-buy baseline:

```bash
# All preferred stations, rule-based signal, 2023–2024
uv run python -m fuel_signal.backtest --preferred --strategy rule_based \
    --start 2023-01-01 --end 2024-12-31

# Single station, model strategy (requires fitted model)
uv run python -m fuel_signal.backtest \
    --station 414 --strategy model \
    --model-path data/models/logreg.joblib --threshold 0.40 \
    --start 2023-01-01 --end 2024-12-31

# Compare all strategies side-by-side (threshold defaults to 0.40)
uv run python -m fuel_signal.backtest --preferred --strategy all \
    --model-path data/models/logreg.joblib \
    --start 2023-01-01 --end 2024-12-31

# Custom tank size and consumption (default: 50L tank, 50L/14d)
uv run python -m fuel_signal.backtest --preferred --strategy rule_based \
    --start 2023-01-01 --end 2024-12-31 \
    --tank-size 60 --daily-use 4.5 --eval-interval 7
```

Output is a table per station showing cents-per-litre (CPL), savings vs always-buy, fill events, and total litres for each strategy. Available strategies: `always_buy` (baseline, always included), `rule_based` (four-signal heuristic), `model` (logistic regression at `--threshold`), `all` (all three side-by-side).

## Running tests

```bash
uv run pytest
```
