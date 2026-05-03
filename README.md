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

Each row contains `station_code`, `price_date`, `today_price_cents`, `future_min_cents`, and `label` (1 if the minimum price over the next `--horizon` days falls more than `--threshold` cents below today's price, else 0). Rows near the end of the data where a full horizon isn't available are excluded.

## Running tests

```bash
uv run pytest
```
