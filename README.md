# fuel-price-signal

A Python CLI that outputs a one-line buy/don't-buy signal for E10 fuel at preferred stations near postcode 2777 (Springwood/Blue Mountains corridor).

```
BUY  | Day 41/46 of cycle | E10 @ Caltex Springwood: 161.9c | Trough est. ~5 days
WAIT | Day 12/46 of cycle | E10 @ Caltex Springwood: 179.2c | Trough est. ~34 days
```

## Setup

```bash
# Install dependencies
uv sync
```

## Building the database

The signal runs from a local SQLite database. Building it is a one-time setup (plus occasional refresh to pull in new history).

### 1. Download and clean historical CSVs

Downloads all bulk price history files (~2016–present) from data.nsw.gov.au into `data/raw/`, then cleans them into `data/cleaned/`. Files already present are skipped, so re-running is safe.

```bash
uv run python -m fuel_signal.history
```

This takes a few minutes on first run (100+ files). `data/raw/` and `data/cleaned/` are gitignored — local derived artifacts.

Cleaning handles known data quality issues in the source files:
- YYYY-DD-MM / YYYY-MM-DD date format bug (pre-2019 files)
- Postcode typos and ACT stations that slipped into NSW data
- Extra fuel-code lines where station details are omitted
- Duplicate rows for the same station + timestamp
- Missing brand fields (inferred from station name)

### 2. *(coming soon)* Load into SQLite

```bash
uv run python -m fuel_signal.db
```

## Daily snapshots

GitHub Actions commits one snapshot file per day to `data/snapshots/YYYY/MM/YYYY-MM-DD.csv` (E10, Sydney metro stations). These are included in the repo and loaded automatically when building the database.

## Getting the signal

```bash
uv run python -m fuel_signal.signal
```

Requires `FUELAPI_API_KEY` and `FUELAPI_API_SECRET` environment variables (FuelCheck API credentials).

## Running tests

```bash
uv run pytest
```
