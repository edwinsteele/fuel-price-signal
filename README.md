# fuel-price-signal

A Python CLI that outputs a one-line buy/don't-buy signal for E10 fuel at preferred stations near postcode 2777 (Springwood/Blue Mountains corridor).

```
BUY  | Day 41/46 of cycle | E10 @ Caltex Springwood: 161.9c | Trough est. ~5 days
WAIT | Day 12/46 of cycle | E10 @ Caltex Springwood: 179.2c | Trough est. ~34 days
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

Takes a few minutes on first run (100+ files). Cleaning handles known data quality issues in the source:
- YYYY-DD-MM / YYYY-MM-DD date format bug (pre-2019 files)
- Postcode typos and ACT stations that slipped into NSW data
- Extra fuel-code lines where station details are omitted
- Duplicate rows for the same station + timestamp
- Missing brand fields (inferred from station name)

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

Loads all snapshot CSVs (from `data/snapshots/`) then all historical cleaned CSVs (from `data/cleaned/`), matching historical rows to stations by normalised address. Skipped rows mean the station predates the FuelCheck API reference data — expected and harmless.

## Inspecting the data

Generates `inspect.html` and opens it in your browser:

```bash
uv run python -m fuel_signal.inspect
```

Shows station count, date range, a Sydney E10 average price chart, data coverage by month, and recent prices.

## Station lookup

Find station codes by suburb or name — useful when adding entries to `PREFERRED_STATIONS` in `config.py`:

```bash
# Free-text search (matches suburb and name)
fuel-signal stations blaxland
fuel-signal stations "emu plains"

# Field-specific filters
fuel-signal stations --suburb springwood
fuel-signal stations --name ampol

# List all stations
fuel-signal stations
```

Output includes `station_code`, suburb, name, and brand. Use the `station_code` value in `PREFERRED_STATIONS`.

## Getting the signal

```bash
uv run --env-file .env python -m fuel_signal.signal
```

## Daily snapshots

GitHub Actions commits one snapshot CSV per day to `data/snapshots/`. To enable it, add `FUELAPI_API_KEY` and `FUELAPI_API_SECRET` as repository secrets under **Settings → Secrets and variables → Actions**.

## Running tests

```bash
uv run pytest
```
