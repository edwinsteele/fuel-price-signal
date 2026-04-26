# fuel-price-signal

A Python CLI that outputs a one-line buy/don't-buy signal for E10 fuel at preferred stations near postcode 2777 (Springwood/Blue Mountains corridor).

## Output format

```
BUY  | Day 41/46 of cycle | E10 @ Caltex Springwood: 161.9c | Trough est. ~5 days
WAIT | Day 12/46 of cycle | E10 @ Caltex Springwood: 179.2c | Trough est. ~34 days
```

## Module structure

```
fuel_signal/
├── config.py        # API key, preferred station list, postcode
├── history.py       # Download + clean bulk CSVs; dynamic resource discovery
├── db.py            # SQLite schema + read/write helpers
├── live.py          # FuelCheck API snapshot → append to DB
├── cycle.py         # Cycle detection + current phase calculation
├── signal.py        # Combine phase + live price → one-line output
└── backtest.py      # Replay historical prices through signal + purchasing strategy
```

## Project setup

- Package manager: **uv** (`uv init`, `uv add`, `uv run`)
- Standard `pyproject.toml` (not Poetry's custom format)

## Data strategy

### Sources
- Bulk historical CSVs from data.nsw.gov.au (back to 2016), resource IDs scraped dynamically
- Daily API snapshots committed to `data/snapshots/YYYY/MM/YYYY-MM-DD.csv` by GitHub Actions
- Live FuelCheck API call at signal-check time for exact current price

### Layout
```
data/
  snapshots/YYYY/MM/YYYY-MM-DD.csv   # committed; E10, Sydney metro stations only
  raw/                                # .gitignored; bulk historical CSVs, downloaded once
fuel_signal.db                        # .gitignored; SQLite, rebuilt from raw + snapshots
```

- `data/raw/` and `fuel_signal.db` are local derived artifacts — not committed
- Snapshot scope: **E10 only, Sydney metro stations** — filtered at collection time in GH Actions
- Other fuel types (diesel, U91, etc.) available in historical CSVs if ever needed
- SQLite is rebuilt by running `history.py` (downloads raw CSVs) then `db.py` (assembles DB)
- GitHub Actions runs daily, commits one snapshot file per day

### Snapshot CSV schema

```
station_code, name, address, suburb, postcode, brand, price, date
```

- `station_code`: FuelCheck API station ID (stable across rebrands)
- `name`/`brand`: current at time of snapshot — included for human readability and to keep `stations` table current
- `address`: included for self-contained matching with historical CSVs
- `price`: E10 cents
- `date`: YYYY-MM-DD

### SQLite schema

```sql
CREATE TABLE stations (
    station_code       INTEGER PRIMARY KEY,
    address_normalized TEXT NOT NULL UNIQUE,  -- join key for historical CSV matching
    suburb             TEXT NOT NULL,
    postcode           TEXT NOT NULL,
    name               TEXT NOT NULL,   -- current; updated on rebrand, prices unaffected
    brand              TEXT,
    latitude           REAL,
    longitude          REAL
);

CREATE TABLE prices (
    station_code  INTEGER NOT NULL REFERENCES stations(station_code),
    fuel_code     TEXT NOT NULL,
    price_date    DATE NOT NULL,
    price_cents   REAL NOT NULL,
    PRIMARY KEY (station_code, fuel_code, price_date)
);
```

Station names/brands change over time (e.g. Caltex → Ampol). `stations.name` and `stations.brand` reflect current state; historical prices are unaffected since they are keyed by `station_code`, not name. No name-history table needed for MVP.

### Historical CSV → DB matching

1. Normalize address from CSV row (expand abbreviations: "St"→"Street", "Rd"→"Road", strip trailing state/postcode suffixes)
2. Look up `station_code` in `stations` by `address_normalized`
3. If no match: station predates API reference data — log and skip for now (rare)

Address normalization needs care — the CSV addresses include state and postcode suffixes ("123 Main St, Springwood NSW 2777") that the API reference data may not. Check what `petrol_prices` transformer already handles before writing new normalization logic.

### FuelCheck API
- Snapshot-only — no historical retrieval endpoint
- Auth: OAuth2 client_credentials flow (API key + secret → Bearer token)
- Env vars: `FUELAPI_API_KEY`, `FUELAPI_API_SECRET`
- Reference implementation: `~/Code/ff-aws-backend/frugalfuel/nswfuel/tasks/retrieve_price_snapshot_from_fuelapi.py`

## Signal logic

- Cycle detection: `scipy.signal.find_peaks(distance=7, prominence=1.0)` on smoothed daily E10 price series
- Use peaks (not troughs) to define cycles; mean inter-peak distance = cycle length
- BUY when in last ~25% of cycle (approaching trough) — adjust from 66% used in ff-aws-backend
- Supporting signals: gradient flatline detection, price relative to last cycle min/max
- Cycle detection runs on Sydney-wide E10 average (more data = stronger signal)
- Preferred station prices used for the actual buy price display
- Data is cyclic but NOT seasonal — do not apply seasonal decomposition
- Plateau-at-boundary detection: handle the case where the current price is at a peak but scipy won't detect it yet (implemented in ff-aws-backend `PriceCycleDetector._plateau_width_at_boundary`)

## Reuse from old projects

### `~/Code/ff-aws-backend` (primary — most complete)
- `ff_aws_backend/recommendations.py` — port this; contains `PriceCycleDetector`, all signal classes, `RecommendationManager`. Do not rewrite from scratch.
- `ff_analysis/purchasing_strategy.py` — backtest engine, fully implemented
- `frugalfuel/nswfuel/tasks/retrieve_price_snapshot_from_fuelapi.py` — OAuth API auth pattern
- `ff_aws_backend/cli.py` — CLI structure with Click

### `~/Code/petrol_prices` (secondary — transformer + downloader)
- `petrol_prices/management/commands/transformer.py` — CSV cleaner (date format bug, postcode corrections, dedup, brand inference)
- `petrol_prices/management/commands/downloader.py` — bulk CSV downloader (CSV + XLSX fallback)
- `petrol_prices/management/commands/fill_daily_gaps.py` — forward-fill to daily resolution
- `postcode_council_map.py` — postcode → LGA mapping

### What NOT to carry over
- DynamoDB, S3, SQS, SNS, Serverless framework, Django ORM — all the AWS/web infra
- jsonpickle (slow; use plain JSON)
- `msrest` / AutoRest generated client (overly complex)

## Historical CSV format
Schema: `ServiceStationName, Address, Suburb, Postcode, Brand, FuelCode, PriceUpdatedDate, Price`

Known data quality issues (handled by transformer):
- YYYY-DD-MM ↔ YYYY-MM-DD date format bug (detectable when day > 12)
- Postcode errors (hardcoded correction map)
- Missing Brand field (infer from station name)
- Duplicate rows for same station + same timestamp
- Extra fuel-code lines (station details not repeated in source — blank name/address rows)
- `PriceUpdatedDate` has a time component in all files from ~2019 onwards (three formats: ISO `YYYY-MM-DDTHH:MM:SS`, space-separated `YYYY-MM-DD HH:MM:SS`, Australian `D/MM/YYYY H:MM:SS AM/PM`). Only the oldest pre-2019 files are truly date-only.
- Stations commonly update price multiple times per day (intraday resets are normal in the NSW price cycle). The transformer keeps the **latest timestamp per station/fuel/day** (end-of-day price) to avoid morning-reset spikes creating artificial day-to-day gyrations that confuse scipy peak detection.

## Station strategy
- User manually maintains preferred station list (known from two weekly routes)
- Match to FuelCheck station IDs by name/address at setup time
- Preferred stations: Blaxland, East Blaxland, Valley Heights, Faulconbridge, Emu Plains, Glenbrook, Winmalee area

## Testing
Tests are required alongside all implementation. Key areas:
- Transformer cleaning logic (date format bug, postcode corrections, dedup)
- Cycle detection correctness (synthetic price series with known cycle lengths)
- Signal threshold logic (edge cases at cycle boundaries)
- Gap-filling / forward-fill behaviour
- DB read/write roundtrips
- Backtest engine: known price series + known strategy → verify simulated spend

## Model/effort guidance
- Sonnet for implementation (downloader, transformer, DB layer, tests)
- Opus for analytically hard design: cycle detection math, backtest engine architecture, leading indicator analysis
