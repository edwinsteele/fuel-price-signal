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

Takes a few minutes on first run (100+ files). Cleaning handles known data quality issues in the source:
- YYYY-DD-MM / YYYY-MM-DD date format bug (pre-2019 files); for files where every date has day ≤ 12 a constant-day-across-varying-months fingerprint is used to detect the true month
- Postcode typos and ACT stations that slipped into NSW data
- Extra fuel-code lines where station details are omitted
- Duplicate rows for the same station + timestamp
- Missing brand fields (inferred from station name)

> **Note:** if you have existing `data/cleaned/` files built before this fix, delete the cleaned versions of the four affected 2019 files (`6d5fd229`, `efcbe322`, `ba5a2055`, `8a29ce30`) and re-run to recover Feb 1–12, Oct 1–9, and Nov 1–8 2019.

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

Shows:
- **Cycle state** — current % through cycle, days since last peak, mean cycle length, last-cycle min/max, boundary plateau status
- **Full history chart** — Sydney E10 average (gap-filled) with scipy-detected peaks as red dashed verticals, last-cycle window shaded, data gaps marked grey
- **6-month zoomed chart** — same peak overlays plus preferred station series
- Coverage by month and recent prices tables

## Station lookup

Find station codes by suburb or name — useful when adding entries to `PREFERRED_STATIONS` in `config.py`:

```bash
# Free-text search (matches suburb and name)
uv run fuel-signal stations blaxland
uv run fuel-signal stations "emu plains"

# Field-specific filters
uv run fuel-signal stations --suburb springwood
uv run fuel-signal stations --name ampol

# List all stations
uv run fuel-signal stations
```

Output includes `station_code`, suburb, name, and brand. Use the `station_code` value in `PREFERRED_STATIONS`.

## Getting the signal

```bash
# Signal as of today (latest date in DB)
uv run python -m fuel_signal.signal

# Signal as of a specific historical date (useful for validation)
uv run python -m fuel_signal.signal --as-of 2026-02-15

# Custom DB path
uv run python -m fuel_signal.signal --db /path/to/fuel_signal.db
```

Output is the combined verdict (one line per preferred station) followed by the four contributing signals and their reasons:

```
[as of 2026-01-10]
BUY  | Day 27/35 of cycle | E10 @ BP Valley Heights: 159.9c
BUY  | Day 27/35 of cycle | E10 @ Shell Blaxland: 157.5c
Combined: BUY (mean signal +1.00)
  AverageCycleTimeSignal: BUY — cycle ending soon (73% through cycle; day 26 / 35.5)
  AverageGradientAfterPeakSignal: NEUTRAL — price has not flatlined (last 3 gradients: [-0.81, -0.52, -0.5])
  AverageNearPreviousMinMaxSignal: BUY — price close to low in last cycle (current 159.3c; last cycle min 168.3c, max 200.0c)
  FavouriteServiceStationPriceGradientSignal: NEUTRAL — no preferred stations raising sharply
```

The four signals (`AverageCycleTimeSignal`, `AverageGradientAfterPeakSignal`, `AverageNearPreviousMinMaxSignal`, `FavouriteServiceStationPriceGradientSignal`) each return BUY / WAIT / DONT_BUY / NEUTRAL. Directional values are averaged (NEUTRAL excluded); mean ≥ 0.5 → BUY, ≤ -0.5 → DON'T BUY, else WAIT.

If `--as-of` falls within the forward-fill gap (the period between the end of historical CSVs and the first daily snapshot), a warning is printed to stderr and the signal should not be trusted.

## Daily snapshots

GitHub Actions commits one snapshot CSV per day to `data/snapshots/`. To enable it, add `FUELAPI_API_KEY` and `FUELAPI_API_SECRET` as repository secrets under **Settings → Secrets and variables → Actions**.

## Running tests

```bash
uv run pytest
```
