# fuel-price-signal — agent context

A Python CLI that outputs a one-line buy/don't-buy signal for E10 fuel at preferred stations near postcode 2777 (Springwood/Blue Mountains corridor).

## Output format

```
BUY  | Day 41/46 of cycle | E10 @ Caltex Springwood: 161.9c | Trough est. ~5 days
WAIT | Day 12/46 of cycle | E10 @ Caltex Springwood: 179.2c | Trough est. ~34 days
```

See [docs/STATUS.md](docs/STATUS.md) for current build status and pending phases.
See [docs/ML_SIGNAL.md](docs/ML_SIGNAL.md) for ML model design decisions and results.
See [docs/CONVENTIONS.md](docs/CONVENTIONS.md) for code style, test patterns, definition-of-done, and git workflow rules.

## Module structure

```
fuel_signal/
├── config.py          # API key, preferred station list, postcode
├── history.py         # Download + clean bulk CSVs; dynamic resource discovery
├── db.py              # SQLite schema + read/write helpers
├── fill.py            # Forward-fill daily price gaps → daily_prices table
├── live.py            # FuelCheck API snapshot → append to DB
├── series.py          # Series resolution (station/lga/brand/sydney) used by compare + inspect
├── cycle.py           # Cycle detection + current phase calculation
├── signal.py          # Combine phase + live price → one-line output
├── compare.py         # Compare two price series (station vs station, station vs LGA mean, etc.)
├── inspect.py         # Flask workbench: interactive chart + cycle state (dev server)
├── stations.py        # Station lookup CLI
├── labels.py          # ML label generation + training-row assembly
├── label_viz.py       # Diagnostic plots for label distributions
├── label_inspect.py   # Per-station per-day label decomposition table
├── features.py        # Join cycle features onto labels → model-ready CSV
├── evaluate.py        # Canonical train/val/test split + scoring utilities
├── train_logreg.py    # Train logistic regression baseline (val only)
├── calibrate.py       # Calibration diagnostics + calibrated model artifact
├── score_phase2.py    # Threshold sweep on val → score test → append results.csv
├── tp_benefit.py      # Diagnostic: empirical TP benefit distribution
├── fp_cost.py         # Diagnostic: empirical FP cost distribution (bimodal)
├── fn_cost.py         # Diagnostic: empirical FN cost distribution
└── backtest.py        # Replay historical prices through purchasing strategies
```

## CLI pattern

Each command is its own module with a `@click.command` named `main` and an `if __name__ == "__main__": main()` block. Invoked as:

```
uv run python -m fuel_signal.signal [--as-of DATE] [--db PATH]
uv run python -m fuel_signal.compare SERIES_A SERIES_B [--fuel E10] [--within 0.5] [--db PATH]
uv run python -m fuel_signal.stations [QUERY]
uv run python -m fuel_signal.labels [--output PATH] [--horizon DAYS] [--threshold CENTS] [--db PATH]
```

**Do not** add new commands to a shared CLI group or create new `[project.scripts]` entries — each module is its own entry point invoked via `python -m`.

## Key db.py read helpers

For analysis and new commands, these are the two series-fetching functions:

```python
# Gap-filled daily prices for one station → [(date_str, price_cents)]
get_daily_prices(conn, station_code: int, fuel_code: str = "E10")

# Gap-filled average across all Sydney metro stations (or filtered by LGA) → [(date_str, avg_price_cents)]
average_price_series(conn, fuel_code: str = "E10", councils: frozenset[str] | None = None)
```

`daily_prices` (gap-filled) is the right table for analysis. `prices` is raw observations only.

`SYDNEY_METRO_COUNCILS` in `postcode_council.py` is the frozenset of valid council names for the `councils=` parameter.

## Test patterns

Standard fixture for DB-backed tests:
```python
@pytest.fixture
def conn(tmp_path):
    c = open_db(tmp_path / "test.db")
    create_schema(c)
    yield c
    c.close()
```

Insert gap-filled test data with `upsert_daily_prices(conn, [(station_code, fuel_code, date_str, price_cents), ...])`. For standalone command tests, invoke via `CliRunner().invoke(main, [...])` where `main` is imported from the module under test.

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

### Snapshot retirement
Snapshots are a bridge until historical CSVs cover the same period — keep the committed count as small as possible.

When a new bulk CSV is released that overlaps `data/snapshots/` dates: (1) verify snapshot prices ≡ historical prices per station/date; (2) if they agree, delete the retired snapshot CSVs; (3) if they diverge, investigate before retiring — divergence reveals something about the data.

`db.py` loads snapshots before historical CSVs and uses `INSERT OR IGNORE`, so snapshot prices win silently on conflict. When the first overlap occurs, compare per-station prices to decide whether snapshot-wins is the right policy. Also check whether the GH Actions cron time (currently 10:00 UTC = 8pm AEST / 9pm AEDT) aligns with the historical CSV rollup time.

### Aggregation
`sydney_average_series` / `average_price_series` is a temporary convenience for cycle detection. Future analyses will need flexible groupings — by region, corridor, LGA cluster, etc. Don't treat it as permanent infrastructure; don't patch it when new groupings are needed, design a proper aggregation layer instead.

### Station classification (Competitive / Discount / Sticky)

LGA- and Brand-level mean features used by the ML model must reflect **current pricing that buyers can actually act on**. Stations fall into three behavioural classes; aggregation policy depends on which class they're in. See [issue #108](https://github.com/edwinsteele/fuel-price-signal/issues/108) for the design discussion and open implementation questions.

**The three classes:**

| Class | Description | Examples |
|---|---|---|
| **Competitive** | Price tracks the cycle; sits within ±10c of the LGA competitive cluster | Metro Tuggerah, Pearl Energy Wyong North, most BP/Caltex/Ampol metro stations |
| **Sticky** | Set-and-forget pricing; sits persistently above the competitive cluster | Shell Reddy Express Woy Woy, EG Ampol Umina, Ampol Foodary motorway sites, BP Berowra |
| **Discount** | Sits persistently below the competitive cluster; real, accessible cheap prices | Costco, Powerfuel, Speedway, Budget Petroleum |

**Aggregation policy:** blended Competitive + Discount means **exclude Sticky only**. Discount stations stay in because their low prices are real and accessible to a buyer; Sticky stations leave because their stale peak prices don't reflect what buyers are currently being offered.

**Why blended (not Competitive-only):** the level shift introduced by including Discount stations (LGAs with discounters look cheaper) reflects real prices buyers can access. Competitive-only would be a cleaner cycle-position signal but discards level information that matters for a purchasing decision.

**Brand aggregates use the same classification.** Brand mean is computed Sydney-wide across stations of brand B where `class != Sticky` (using the same per-LGA-derived classification). One classifier, one `station_class` table; LGA and Brand aggregates are just different slicings. The principle "ML features for pricing decisions must exclude stations that aren't informative about current pricing" applies regardless of slicing dimension — a Sticky Shell in Woy Woy is stale whether you aggregate it by LGA or by brand.

**Out of scope here:** Members-only stations (e.g. Costco) have prices that aren't accessible to a non-member. A separate accessibility filter may be warranted before they enter any "available to buyer" feature — deferred.

**The classifier (1D on premium):**

| Setting | Value |
|---|---|
| Classification axis | Median price-vs-cluster premium |
| Window | 45 days (NSW mean cycle length) |
| Band | ±10c (Sticky if median premium > +10c; Discount if < −10c; else Competitive) |
| Frequency role | Bootstrap seeding of the initial cluster only. Not in the classifier itself, and not a recency filter at aggregation time (see below). |

The classifier is deliberately 1D on premium, not 2D on (frequency, premium). Frequency was a noisy proxy for the property premium measures directly — Sticky stations update less because they're set-and-forget at high prices. A high-frequency station with persistently high premium (e.g. BP Berowra) is still Sticky.

The 45-day window is the empirical mean NSW cycle length. The classifier does **not** try to model cycle-length variation (cycles run 35–70 days) — cycle modelling is out of scope for the current ML model.

**PIT discipline:** The classification window for a training row at date D ends at D−1. Past price classifying past behaviour is not target leakage; the prediction target (future price) is not in the classification window.

**Median is computed in Python**, not SQL — SQLite has no native MEDIAN. Classification is a batch step (daily re-computation across ~800 stations), so the per-call cost doesn't matter.

**Materialisation:** classifications are pre-computed and stored in a `station_class` table:

```sql
CREATE TABLE station_class (
    station_code             INTEGER NOT NULL REFERENCES stations(station_code),
    snapshot_date            INTEGER NOT NULL,   -- YYYYMMDD; classification valid as of this date
    class                    TEXT    NOT NULL,   -- 'Competitive' | 'Sticky' | 'Discount'
    median_premium_decicents INTEGER NOT NULL,   -- median (station_price − cluster_mean) over 45d
    PRIMARY KEY (station_code, snapshot_date)
);
```

**Daily cadence.** Each day's classification uses the 45-day window ending at `snapshot_date − 1`. Daily (rather than monthly) snapshots avoid step-changes in LGA aggregates when borderline stations flip class — the rolling window smooths drift, so daily materialisation just propagates that smoothness into the feature. Storage cost is trivial (~290k rows/year × 5 years ≈ 1.5M rows). Compute cost is sub-second per day.

**No active-reporter recency filter.** An earlier design floated a 14d "last raw observation" filter at aggregation time as a guard against stale forward-filled prices. It was dropped (2026-05-19) for KISS reasons: empirical measurement showed it would exclude 5–10% of non-Sticky stations during peak plateaus where the forward-fill is *correct* (price genuinely unchanged), in exchange for limited protection against downcycle staleness that the 28d forward-fill cap and the classifier already partially handle. If model artefacts traceable to ramp-day staleness appear later, revisit — likely with a phase-aware filter rather than a static threshold.

**Aggregation floor:** if fewer than 3 non-Sticky stations are available for a given LGA/brand/date, emit NULL rather than fall back. A silently-thin aggregate is worse than a gap. The floor protects against *thin samples* (high-variance aggregates from few stations); it does **not** protect against staleness — staleness protection lives entirely in the 28d forward-fill cap and the classifier.

**Cold-start handling:** a station gets a classification entry as soon as it has at least one raw observation in the 45-day window. No minimum-observation threshold, no `is_classified` boolean, no default class. Stations with zero observations in the window have no entry and are excluded from aggregates (consistent with their absence from `daily_prices`). Because the pooled ML model has no station/brand/suburb categorical features (only 10 numeric features from cycle and price deltas — see `fuel_signal/features.py:FEATURE_COLUMNS`), there is no OOV problem at inference for a brand-new station — its numeric features can be computed from a single observation and the model produces a prediction without special handling.

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

## Signal logic

- Cycle detection: `scipy.signal.find_peaks(distance=7, prominence=1.0)` on smoothed daily E10 price series
- Use peaks (not troughs) to define cycles; mean inter-peak distance = cycle length
- BUY when in last ~25% of cycle (approaching trough) — adjust from 66% used in ff-aws-backend
- Supporting signals: gradient flatline detection, price relative to last cycle min/max
- Cycle detection runs on Sydney-wide E10 average (more data = stronger signal)
- Preferred station prices used for the actual buy price display
- Data is cyclic but NOT seasonal — do not apply seasonal decomposition
- Plateau-at-boundary detection: handle the case where the current price is at a peak but scipy won't detect it yet (implemented in ff-aws-backend `PriceCycleDetector._plateau_width_at_boundary`)
- Atypical periods (COVID demand collapse, 2026 Middle East war supply shock) distort mean cycle length, peak prominence, and last-cycle min/max. When building the backtest engine or calibrating signal thresholds, consider a mechanism to mark/exclude date ranges — but don't bake it in prematurely; add it when backtest results show anomalies traceable to a known shock.

### Backtest constraints
- Backtests must be runnable at arbitrary historical dates ("at date D, would strategy X have been cheaper?"), not just today.
- `daily_prices` is point-in-time safe (forward-fill uses no lookahead), but derived metrics built on top may not be. When adding new metrics, explicitly validate whether they can be recomputed on-the-fly by querying `WHERE price_date <= D`, or whether they need to be pre-computed and stored per day.
- Backtest performance: load the full series ONCE at startup; `detect(as_of_date)` is an in-memory numpy slice (~0.5 ms × 3650 dates ≈ 2 s total). `CycleDetector` must cache `pd.Series` in `__init__` — if conversion happens inside `detect()`, you pay it 3650× per backtest run.

## Historical CSV format
Schema: `ServiceStationName, Address, Suburb, Postcode, Brand, FuelCode, PriceUpdatedDate, Price`

Known data quality issues (handled by transformer):
- YYYY-DD-MM ↔ YYYY-MM-DD date format bug (detectable when day > 12). For files where every date has day ≤ 12, a constant day value across varying months is the YYYY-DD-MM fingerprint — the constant is the true month (e.g. Feb 2019, Oct 2019, Nov 2019 files).
- Postcode errors (hardcoded correction map)
- Missing Brand field (infer from station name)
- Duplicate rows for same station + same timestamp
- Extra fuel-code lines (station details not repeated in source — blank name/address rows)
- `PriceUpdatedDate` has a time component in all files from ~2019 onwards (three formats: ISO `YYYY-MM-DDTHH:MM:SS`, space-separated `YYYY-MM-DD HH:MM:SS`, Australian `D/MM/YYYY H:MM:SS AM/PM`). Only the oldest pre-2019 files are truly date-only.
- Stations commonly update price multiple times per day (intraday resets are normal in the NSW price cycle). The transformer keeps the **latest timestamp per station/fuel/day** (end-of-day price) to avoid morning-reset spikes creating artificial day-to-day gyrations that confuse scipy peak detection.

Known unrecoverable gaps (source data never published):
- Aug 9–31 and Sep 5–30, 2017 — those bulk CSV files only captured 8 and 4 days respectively
- Sep 18–30, Oct 10–31, Nov 9–30, 2019 — source files for Oct/Nov 2019 only captured 9 and 8 days; confirmed via price-level cross-check (not a format bug)

## Station strategy
- User manually maintains preferred station list (known from two weekly routes)
- Match to FuelCheck station IDs by name/address at setup time
- Preferred stations: Blaxland, East Blaxland, Valley Heights, Faulconbridge, Emu Plains, Glenbrook, Winmalee area

## inspect.py (Flask workbench)

`inspect.py` is a local Flask dev server — `uv run python -m fuel_signal.inspect` starts it (default port 5000). State lives in the URL query string. Series types: `sydney`, `lga:Name`, `brand:Name`, `station:CODE`. Chart types: line, scatter, gradient heatmap, coverage heatmap. See README for full usage.

Leading indicators (deferred — not yet built):
- Hypothesis: some LGAs and/or macro signals (TGP, crude) precede BM price rises
- Architecture supports this: new series → new `CycleDetector` → new signal class → register in `RecommendationManager`

## Testing
Tests are required alongside all implementation. Key areas:
- Transformer cleaning logic (date format bug, postcode corrections, dedup)
- Cycle detection correctness (synthetic price series with known cycle lengths)
- Signal threshold logic (edge cases at cycle boundaries)
- Gap-filling / forward-fill behaviour
- DB read/write roundtrips
- Backtest engine: known price series + known strategy → verify simulated spend

## Automation workflow

See [docs/automation.md](docs/automation.md) for the full state machine and operational details.

### Issue label taxonomy

| Label | Meaning | Who works it |
|-------|---------|--------------|
| `chore` | Formatting, dead code, doc fixes, dependency bumps, trivial cleanup | Automated worker |
| `polish` | Small contained features, test additions, minor refactors | Automated worker |
| `design` | Cycle detection, signal logic, ML work, architecture decisions | Owner only — never automated |
| `claude-authored` | PR was opened by the automated worker | Identifies worker-opened PRs |
| `auto-merge-ok` | Safe to auto-merge once CI passes | Applied by worker to `chore` PRs |

**Classification examples:**
- `chore`: add a missing type hint, bump a dev dependency, fix a typo in a docstring, delete unused import
- `polish`: add a missing test for an existing function, extract a helper that duplicates two callers, add a `--verbose` flag to an existing CLI command
- `design`: change cycle detection algorithm, add a new signal class, modify the DB schema, anything that touches `cycle.py`, `signal.py`, or ML work

**Escape hatch — polish → design upgrade:**
If while implementing a `polish` issue you discover it actually requires design work:
1. Relabel the issue: `gh issue edit N --add-label design --remove-label polish`
2. Post a comment explaining why you stopped and what the design question is.
3. Do not write any code.

### Branch and PR conventions

- Branch naming: `worker/issue-<N>-<short-slug>` (e.g. `worker/issue-7-add-type-hints`)
- PR title: `fix: <issue title> (closes #N)` for chore; `feat: <issue title> (closes #N)` for polish
- PR body: 3–5 bullet plan (what changed, what didn't, what test was added)
- Target branch: always `main` (`--base main`)
- Run `uv run ruff check . && uv run pytest -q` before pushing; fix any failures
