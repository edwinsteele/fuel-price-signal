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

### 2b. (Optional) Refresh the AIP TGP series

Pulls the AIP Sydney ULP Terminal Gate Price (wholesale floor, c/L) and refreshes `data/tgp/tgp_sydney.csv` — the upstream series for the pending `tgp_delta_7d` feature (#271).

```bash
uv run python -m fuel_signal.tgp                       # scrape + download the latest AIP xlsx
uv run python -m fuel_signal.tgp --from-xlsx FILE.xlsx # parse a local xlsx instead (offline/backfill)
```

The AIP file always carries the full 2004→present history, so each run overwrites the CSV; unchanged days produce no commit. The `Daily AIP TGP fetch` GitHub Action runs this and commits `data/tgp/`.

Step 3 below ingests this CSV into the `tgp` table (full-rewrite/self-reconciling, so re-running is safe), the upstream of the `tgp_delta_7d` feature.

### 3. Load everything into SQLite

```bash
uv run python -m fuel_signal.db
```

Loads all snapshot CSVs (from `data/snapshots/`) then all historical cleaned CSVs (from `data/cleaned/`), and finally the Sydney TGP CSV (from `data/tgp/tgp_sydney.csv`, if present) into the `tgp` table.

### 4. Forward-fill daily price gaps

```bash
# Full rebuild (use this for first-time setup):
uv run python -m fuel_signal.fill

# Incremental — only recompute recent days (cheap, cost doesn't scale with total history):
uv run python -m fuel_signal.fill --since-date 2026-08-01
```

Rebuilds the `daily_prices` table by forward-filling gaps between observations. Required after `db` — analysis commands read from `daily_prices`, not from the raw observations. `--since-date` limits the rebuild to that date onward, using each station's most recent prior observation as the forward-fill anchor so gaps straddling the boundary still resolve correctly; earlier `daily_prices` rows are left untouched. Omit it for a full rebuild.

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

## Makefile shortcuts

A `Makefile` wraps the local daily routine so you don't have to remember the step order:

```bash
make daily      # update local DB with today's data, then print the signal
make dashboard  # update local DB with today's data, then open the inspect workbench
make update     # just the DB refresh (pull + db + fill + classify + lga-leadership)
make help       # list all targets, including the individual steps
```

`update` mirrors the `Daily DB update` GitHub Actions workflow below, run locally against `fuel_signal.db`: `git pull` (to fetch today's committed snapshot CSV, if the daily snapshot workflow has already run), then `db` → `fill` → `classify` → `lga_leadership`, all with no-arg (today) defaults.

## Daily snapshots

GitHub Actions commits one snapshot CSV per day to `data/snapshots/`. To enable it, add `FUELAPI_API_KEY` and `FUELAPI_API_SECRET` as repository secrets under **Settings → Secrets and variables → Actions**.

### Snapshot retirement

`data/snapshots/` is a bridge until NSW's bulk historical CSVs cover the same period — see AGENTS.md § Data strategy for why. This is a manual, human-run check (not scheduled): run it whenever you notice a new bulk CSV has been published that might overlap committed snapshot months.

```bash
uv run python -m fuel_signal.snapshot_retire                  # report only — no files touched
uv run python -m fuel_signal.snapshot_retire --apply           # delete months that met the agreement threshold
```

For each committed snapshot month that a published bulk CSV now covers, it downloads + cleans that CSV (cached under `data/raw`/`data/cleaned`, same as `history.py`), compares snapshot prices against the historical series using as-of forward-fill (the bulk CSV is a price-change event log, not a daily census — matches `fill.py`'s gap-filling logic), and reports the agreement rate. A month is `ELIGIBLE` for retirement once its agreement rate clears `--min-agreement` (default 95%, within `--tolerance` cents, default 0.05). `--apply` deletes only the eligible months' files — review `git status` and open a PR for the deletion as usual, it isn't committed automatically. A month with low agreement is left alone; investigate the divergence before retiring it.

## CI: DB and model pipeline

Three workflows keep a `fuel_signal.db` current in CI and publish trained models, decoupled from each other since retraining doesn't need to happen on the same cadence as data ingest:

- **`Seed / reset DB`** (`workflow_dispatch` only) — full from-scratch rebuild (`history` → `db --force` → `fill` → `classify --start-date 2016-08-01` → `lga_leadership --start-date 2016-08-01`), saved as a `db-<date>-<run_id>` cache entry. Rare/manual — run once to bootstrap, or again for a clean reset.
- **`Daily DB update`** (runs after `Daily E10 snapshot` succeeds) — restores the most recent `db-*` cache entry, incrementally loads today's newly-landed snapshot (`db.py` skips already-loaded files via its `loaded_files` table), re-runs `fill`/`classify`/`lga_leadership` for today only, saves a fresh `db-<date>-<run_id>` entry, and prunes older ones. `run_id` is in the key (not just the date) because an exact-match cache save can't overwrite — without it, a same-day re-run (e.g. a manual test trigger) would collide with that day's already-saved entry.
- **`Build model`** (`workflow_dispatch` only) — restores the latest `db-*` cache entry and runs `features` → `train_lgbm --no-brand-features` → `calibrate --skip-results-csv`, then publishes `data/models/lgbm.joblib` and `lgbm_calibrated.joblib` as assets on the `model-latest` GitHub Release (`--clobber`'d each run). `--no-brand-features` reproduces the locked `RAC_full` baseline (docs/STATUS.md) — `train_lgbm`'s own default is the brand-inclusive Phase 4b schema, which this project evaluated and walked away from.

The DB uses a cache (not a release) because it's only ever consumed by other Actions workflows, which already have `GITHUB_TOKEN` for free — no need for a stable public URL. The model uses a release because its real consumer is an external, unattended box (deploy target) that isn't running inside Actions and needs a plain-HTTPS download with no stored credential.

Each `.joblib` embeds `git_sha` (the commit it was trained from) and `feature_columns` alongside the fitted pipeline. `calibrate.py` checks the loaded model's `feature_columns` against the features CSV before using them and fails with a clear error on mismatch, rather than a raw `KeyError` partway through calibration — this is the automated version of the "check `feature_columns` before trusting a model on disk" rule from the on-disk model paths note. `inspect.py`'s `/features` page shows the deployed model's `git_sha` and feature count when available, for eyeballing whether it looks current.

## ML model development

Training, evaluating, calibrating, and diagnosing the ML model behind the signal —
label/feature generation, LightGBM training, SHAP analysis, backtesting, the
AI-sourced feature pipeline — is CLI/dev reference material, not day-to-day usage.
See [docs/ML_PIPELINE.md](docs/ML_PIPELINE.md).

## Running tests

```bash
uv run pytest
```

