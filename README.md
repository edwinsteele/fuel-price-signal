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

The signal runs from a local SQLite database (`fuel_signal.db`, gitignored). Build it once, then refresh as needed.

### 1. Download and clean historical CSVs

Downloads all bulk price history from data.nsw.gov.au (~2016–present) into `data/raw/`, then cleans into `data/cleaned/`. Both directories are gitignored. Files already present are skipped, so re-running is safe.

```bash
uv run python -m fuel_signal.history
```

Takes a few minutes on first run (100+ files).

### 2. Collect a live snapshot (populates station reference data)

The database needs station reference data (codes, addresses) from the FuelCheck API before historical data can be loaded. Run this first:

```bash
uv run --env-file .env python -m fuel_signal.live
```

This writes one snapshot CSV to `data/snapshots/YYYY/MM/YYYY-MM-DD.csv` and is also what GitHub Actions runs daily. You only need to run it manually when bootstrapping or if you need today's prices immediately.

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

## Inspecting the data

Starts a local Flask workbench and opens it in your browser:

```bash
uv run python -m fuel_signal.inspect
# Custom host/port, no auto-open:
uv run python -m fuel_signal.inspect --port 5001 --no-browser
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

Output includes all label columns (`station_code`, `price_date`, `today_price_cents`, `future_min_cents`, `label`) plus cycle features (`cycle_pct_through`, `cycle_days_since_peak`, `cycle_mean_length`, `cycle_last_min_cents`, `cycle_last_max_cents`, `cycle_peak_count`) and station-vs-aggregate features (`station_price_cents`, `station_minus_last_min_cents`, `station_minus_last_max_cents`, `station_minus_sydney_avg_cents`). Rows with insufficient history for cycle detection are excluded.

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

## Calibrating the model

Check calibration quality and produce a calibrated model artifact:

```bash
# Default: reads data/features.csv + data/models/logreg.joblib,
# writes data/models/logreg_calibrated.joblib
uv run python -m fuel_signal.calibrate

# Custom paths or skip writing to experiments/results.csv
uv run python -m fuel_signal.calibrate \
    --features-csv /tmp/features.csv \
    --model-in /tmp/logreg.joblib \
    --model-out /tmp/logreg_calibrated.joblib \
    --skip-results-csv
```

Reports class balance (BUY rate) for all splits and prints a 10-bin reliability table on val. If miscalibrated (max |gap| > 0.05), compares sigmoid (Platt) vs isotonic calibration wrappers and saves the better one. Appends a result row to `experiments/results.csv`.

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

```bash
# Default: reads data/features.csv
uv run python -m fuel_signal.score_phase2

# Custom features CSV
uv run python -m fuel_signal.score_phase2 --features-csv /tmp/features.csv
```

**What it does:**

1. Trains the logreg pipeline on the train split (reuses `train_logreg` internals).
2. Sweeps τ ∈ [0.05, 0.95] (step 0.05) on val — prints precision, recall, F1, BUY%, and expected-cents-saved per row.
3. Picks τ = argmax(expected cents/row on val) + 0.05 adjustment. The +0.05 corrects for val's elevated BUY rate (36.1% vs test's 26.9%): the cost-optimal τ on val is slightly too aggressive for the test distribution.
4. Scores test at chosen τ. Appends one row to `experiments/results.csv`.

**Cost model:** TP → +3.0c saved; FP → −1.5c penalty; FN/TN → 0.

**Phase 2 result** (2026-05-09, real DB):

| Model | Test logloss | Test brier | vs baseline |
|---|---|---|---|
| Marginal-rate baseline | 0.5821 | 0.1966 | — |
| Logreg cycle features (τ=0.40) | 0.4029 | 0.1346 | −0.1792 logloss, −0.0620 brier |

At τ=0.40: precision=0.618, recall=0.581, F1=0.599, BUY rate=25.0% on test.

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
