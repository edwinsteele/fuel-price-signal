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
See [docs/ML_PIPELINE.md](docs/ML_PIPELINE.md) for the ML model training/evaluation CLI reference (README.md covers setup and day-to-day signal usage only).
See [docs/feature-pipeline.md](docs/feature-pipeline.md) for the AI-sourced candidate-feature pipeline's machinery.

## Module structure

```
fuel_signal/
├── config.py          # API key, preferred station list, postcode
├── history.py         # Download + clean bulk CSVs; dynamic resource discovery
├── snapshot_retire.py # Report/delete committed snapshots now covered by bulk CSVs
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
├── backtest.py        # Replay historical prices through purchasing strategies
├── backtest_phase2.py # Phase 2 τ re-validation on realised spend
├── train_lgbm.py      # Train LightGBM baseline; --no-brand-features for the locked 54-feat model
├── classify.py        # Competitive/Discount/Sticky classifier → station_class table
├── cv_report.py       # Paired walk-forward CV for feature add/drop/swap decisions
├── lga_leadership.py  # Phase 4 LGA event-based leadership features
├── brand_leadership.py# Brand trough features (computed; not in locked model)
├── feature_redundancy.py  # SHAP-redundancy cluster analysis
├── feature_diagnostics.py # Feature-level diagnostic utilities
├── shap_report.py     # SHAP importance + per-prediction explanation
├── loo_ablation.py    # Leave-one-out feature ablation
└── postcode_council.py    # Postcode → LGA mapping; SYDNEY_METRO_COUNCILS
```

## CLI pattern

Each command is its own module with a `@click.command` named `main` and an `if __name__ == "__main__": main()` block. Invoked as:

```
uv run python -m fuel_signal.signal [--as-of DATE] [--db PATH]
uv run python -m fuel_signal.compare SERIES_A SERIES_B [--fuel E10] [--within 0.5] [--db PATH]
uv run python -m fuel_signal.stations [QUERY]
uv run python -m fuel_signal.labels [--output PATH] [--horizon DAYS] [--threshold CENTS] [--db PATH]
uv run python -m fuel_signal.shap_report --model MODEL --features CSV --split val --output DIR
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

## Backfill (`--start-date`) paths: load once, slice in memory

Any `*_range` backfill that walks snapshots over a trailing window **must not** re-query per snapshot. Successive windows overlap ~98%, so the re-query is the from-scratch rebuild's dominant cost — measured at 59% of `classify.py` and **99%** of `lga_leadership.py`. Keep the single-snapshot path (what `daily-db-update.yml` runs daily) querying directly; only the range path preloads.

Which shape depends on how far SQL has already aggregated — memory is the binding constraint on Viking:

- **Aggregated output (≲100k rows for the decade)** → load the full range once, bucket by `date_int`, `searchsorted` per snapshot. `_load_lga_sums` is 30 MB for all history.
- **Raw per-station rows (~2.2M for the decade)** → a naive full load is 540 MB resident / 881 MB peak, which will OOM Viking. Use one `ORDER BY price_date` scan (free — `daily_prices_fuel_date` already delivers that order, no sort step in the plan) feeding a `deque` of per-day buckets: append day `D-1`, evict day `D-window`. ~7 MB resident. Yield the buckets, not a flattened row list — flattening rebuilds every window's rows and eats most of the win.

Backfills also commit per snapshot; batch commits in range mode only, where the fsync cost is disproportionate on Viking's storage.

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
- Snapshot files themselves are **unfiltered** — all NSW stations, all fuel types, captured by `fuel_signal/live.py`. Filtering to **E10, Sydney metro** happens at DB-load time (`db.py`'s `load_snapshot_csv`/`load_all_snapshots` `postcodes`/`fuel_codes` params, defaulted from `SYDNEY_METRO_POSTCODES` + `{"E10"}`), not at collection time.
- Other fuel types (diesel, U91, etc.) available in historical CSVs if ever needed
- SQLite is rebuilt by running `history.py` (downloads raw CSVs) then `db.py` (assembles DB)
- GitHub Actions runs daily, commits one snapshot file per day

### Snapshot retirement
Snapshots are a bridge until historical CSVs cover the same period — keep the committed count as small as possible.

When a new bulk CSV is released that overlaps `data/snapshots/` dates, run `uv run python -m fuel_signal.snapshot_retire` (report only) and review the agreement numbers; re-run with `--apply` to delete eligible months, then commit the deletion via a PR. If a month diverges below the agreement threshold, investigate before retiring — divergence reveals something about the data. See [README.md § Snapshot retirement](README.md#snapshot-retirement) for usage.

**Validated 2026-08-17 (first overlap, gh#4 / fps-1785999730823-12-2fd8326a):** the bulk historical CSV is an **event log**, not a daily census — a station only gets a row on a day its price changed, not every day. This is exactly what `fill.py`'s forward-fill already exists to reconstruct (same mechanism used for all pre-2026 history with no snapshots at all). Comparing April–July 2026 snapshots against the newly-published bulk CSVs for the same months, using as-of forward-fill: **99.3% of snapshot rows exactly match the historical-derived price** (prices are recorded to 0.1c precision; the `snapshot_retire.py` tolerance default of 0.05c only absorbs floating-point rounding, not real divergence). Remaining divergence is small (median ~2c) and one-directional in a way consistent with the historical file recording each day's *last* price update while the snapshot is taken once ~9pm — i.e. explained by intraday timing, not a data-quality problem.

**Conclusion: `db.py` loads snapshots before historical CSVs and uses `INSERT OR IGNORE`, so snapshot prices win silently on conflict — confirmed to be a reasonable default**, since the two sources agree closely and no systematic bias was found. April–July 2026 snapshots were retired (deleted) on this basis; August 2026 stays committed until a bulk CSV covering it is published.

### Known source data limitations

**2022-03-12 to 2022-03-21: NSW source data collapses ~85-98%, unrecoverable.** `prices` row counts (raw event log, all fuel grades, all NSW stations) drop from a ~3,500-1,100/day baseline to double- and triple-digit counts (e.g. 2022-03-20: 49 rows), with the surrounding week each side (2022-03-10 to 2022-04-01) also thinner than normal. P95/P98 grades hit **zero events statewide** 2022-03-11 to 2022-03-29 while E10/U91 continued at reduced volume, so this is a genuine source-side reporting collapse, not a real 19-day market freeze — the network E10 median kept moving through the window (183.9 → 176.9 → 185.8 → 198.9 c/L). The 2022-03-30/31 spike (515/549 stations repricing E10 in two days) is the federal fuel excise cut (~22 c/L, effective 2022-03-30), not stations reopening after a genuine pause.

Checked 2026-08-25, no alternative source exists: data.nsw.gov.au's one March-2022 resource (`d707be7a-dbb9-47a7-ae57-599428731fac`) has never been revised since 2022-04-04; rows-per-MB of the published XLSX is consistent with neighbouring months (Feb 19.6k/MB, Mar 19.0k/MB, Apr 19.8k/MB), so this project's extraction is complete — the *source file itself* is short, not our parsing of it. data.gov.au only mirrors the same resource. The NSW real-time Fuel API is snapshot-only and cannot be queried retrospectively. `data/snapshots/` starts 2026-08. **The gap is permanent; plan around it, don't try to fill it.**

Within a single station's `fuel_signal.db` history this mostly self-heals: `fill.py`'s `MAX_GAP_FILL_DAYS = 28` forward-fills across the ~10-19 day hole without triggering exclusion. It becomes a much bigger problem in combination with `labels.py`'s exclusion window for the minority of stations that also have a gap on the *other* side of March 2022 — tracked separately as `fps-ghr` (turns a 2-3 week source hole into a 125+ day exclusion for 81 stations). This entry is the underlying data fact; `fps-ghr` is the code defect it interacts with.

**Evaluation treatment (decided 2026-08-27, `fps-tpy`): flag, don't exclude.** The canonical train/val/test split (`docs/STATUS.md` § Canonical split) never scores this window directly — Val and Test both start in 2025. It only touches the pre-test `walk_forward_folds()` used for feature-change CV (`docs/CONVENTIONS.md` § Changing the production feature set): with the project's standard `train_min_days=1825, val_days=90, step_days=90`, exactly one fold's **val window** is `2022-02-03 to 2022-05-03` (its train window for any later fold that reaches this far also absorbs the hole via forward-filled rows, same as any other station gap). Only ~10-19 of that fold's 90 val days are degraded, `fill.py`'s cap keeps `val_df` non-empty (so `cv_report.py`'s `if val_df.empty: continue` skip does not fire), and the same window already coincides with the Ukraine-invasion price shock (2022-02-24) that regime-segmented CV runs treat as elevated-variance. Given the effect is small, partial, and confined to one fold of many, excluding it would be special-casing evaluation logic for a bounded, already-measured effect — proportionate response is to treat a March-2022-spanning fold the same way `docs/CONVENTIONS.md`'s existing override clause treats any other "known price-shock period": eligible for the fold-regression override, not a reason to rebuild the CV harness. Don't re-litigate this; if a specific CV run shows a March-2022 fold behaving anomalously, cite this note rather than re-deriving the cause.

**A structurally different, non-alarming gap also exists in 2017.** Several months in 2017 (03-27–03-31, 05-19–05-31, 06-18–06-30, 07-19–07-31, 09-05–09-30, 10-14–10-31 — 5 to 26 days each) show **exactly zero** `prices` rows for the tail of the month, resuming cleanly on the 1st of the next month. Unlike March 2022 this is not mid-month and not partial — it lines up precisely with month boundaries in both directions, which is the signature of an early-history bulk-CSV resource that simply didn't cover the full calendar month, not a reporting collapse. The longest (Sept 2017, 26 days) sits just under `fill.py`'s 28-day cap, so it forward-fills without tripping the `fps-ghr` exclusion chain. No other window in 2016-2026 shows either signature — the remaining single- and double-day dips found by the same scan (holidays, weekends) sit at normal-baseline magnitude and don't warrant documentation here.

Composition-drift measurement (the panel-size question this event also raised, originally filed as `fps-ghr`): **answered and closed**, see `fps-tpy`'s bd description or `bd show fps-tpy` — chain-linked index test over 2021-11-05..2025-04-17 (1,260 dates × 714 stations) found total drift +0.461 c/L over 3.4 years, daily |drift change| median exactly 0, concentrated in ~8 days, largest single-day move 0.567 c/L on 2022-06-29 (a different event, not this one). Small and bounded; feeds `station_minus_sydney_avg_cents` and the LGA/brand-mean derivatives on those specific days but is not a first-order threat. Do not re-investigate.

### Aggregation
`sydney_average_series` / `average_price_series` is a temporary convenience for cycle detection. Future analyses will need flexible groupings — by region, corridor, LGA cluster, etc. Don't treat it as permanent infrastructure; don't patch it when new groupings are needed, design a proper aggregation layer instead.

### Station classification (Competitive / Discount / Sticky)

LGA- and Brand-level mean features used by the ML model must reflect **current pricing that buyers can actually act on**. Stations fall into three behavioural classes; aggregation policy depends on which class they're in. The classifier is built (`classify.py` → `station_class` table); [issue #108](https://github.com/edwinsteele/fuel-price-signal/issues/108) (closed) holds the original design discussion.

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

**Cold-start handling:** a station gets a classification entry as soon as it has at least one raw observation in the 45-day window. No minimum-observation threshold, no `is_classified` boolean, no default class. Stations with zero observations in the window have no entry and are excluded from aggregates (consistent with their absence from `daily_prices`). Because the pooled ML model uses numeric features only — no station/brand/suburb categorical (the locked baseline is 54 features; see § Canonical feature set) — there is no OOV problem at inference for a brand-new station: its numeric features can be computed from a single observation and the model produces a prediction without special handling. The LGA/brand mean features may be NULL (NaN) if fewer than 3 non-Sticky stations are available; the model handles NaN natively.

### Snapshot CSV schema

```
station_code, name, address, suburb, postcode, brand, fuel_code, price, date
```

- `station_code`: FuelCheck API station ID (stable across rebrands)
- `name`/`brand`: current at time of snapshot — included for human readability and to keep `stations` table current
- `address`: included for self-contained matching with historical CSVs
- `fuel_code`: all fuel types are captured (E10, U91, P95, P98, PDL, DL, LPG, etc.) — filtering to E10 happens at DB-load time, not here (see § Data strategy)
- `price`: cents, for the fuel type in `fuel_code`
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

## Canonical feature set (54-feat baseline, locked issue #216)

The production model (`data/models/lgbm.joblib`, `lgbm_calibrated.joblib`) is trained on:

| Group | Count | Source constant |
|-------|-------|-----------------|
| Core cycle + station features | 15 | `FEATURE_COLUMNS` |
| LGA trough features | 35 | `LGA_FEATURE_COLUMNS` (one per `SYDNEY_METRO_COUNCILS` LGA) |
| RAC_full network features | 4 | `NETWORK_FEATURE_COLUMNS` |
| **Total** | **54** | **`LOCKED_FEATURE_COLUMNS`** |

`fuel_signal.features.LOCKED_FEATURE_COLUMNS` is the one symbol for this contract — import it (or `experiments.lib.constants.BASELINE_COLUMNS` from an experiment script) rather than retyping the group composition. Its order is production order and must not be sorted; `data/models/lgbm_calibrated.joblib`'s `feature_columns` is ground truth, and `tests/test_feature_contract.py` asserts ordered equality against it whenever the (gitignored) artifact is present. `baseline_fingerprint()` gives the contract a `'54:<sha12>'` identity that every experiment `meta.json`, batch `freeze.json` and run `results.json` records. Columns computed into `features.csv` but held out of the lock are registered in `NON_MODEL_COLUMNS` with a reason — **graduating one means deleting its entry in the same change that adds it here**, or `resolve_baseline_columns()` raises. See [docs/CONVENTIONS.md § The baseline feature set is declared, never discovered](docs/CONVENTIONS.md).

**RAC_full group** (`network_px_std`, `network_px_std_delta_3d`, `lga_phase_std`, `lga_phase_std_delta_3d`): graduated via within-family ablation in #212; adds −0.045 Δh25 over the 50-feat LGA-only baseline.

To reproduce the locked 54-feat model: `uv run python -m fuel_signal.features` (regenerates `data/features.csv`), then `uv run python -m fuel_signal.train_lgbm --no-brand-features`. Brand trough columns are excluded from the locked baseline until a separate ablation graduates them.

## Multi-seed test-logloss policy

At **lock time** (phase boundaries, results you will compare future changes against), run `score_phase2.py` with `--seeds 1,7,42,99,2024`. This banks a per-seed raw (uncalibrated) LightGBM test-logloss vector in `experiments/results.csv` columns `seed_test_logloss_vector`, `seed_test_logloss_mean`, `seed_test_logloss_std`.

For **development sniff-tests**, omit `--seeds`. Single-seed is sufficient for checking direction; multi-seeding every experiment defeats the 3×std comparison gate.

Comparison gate: a new model's delta vs the baseline must exceed `3 × seed_test_logloss_std` of the baseline to be considered real (not seed noise).

Metric is always **raw (uncalibrated)** LightGBM test logloss so that the calibration choice doesn't confound the comparison. The `holdout_logloss` column in the same row records the final (possibly calibrated) model score and is a separate quantity.

## Testing
Tests are required alongside all implementation. Key areas:
- Transformer cleaning logic (date format bug, postcode corrections, dedup)
- Cycle detection correctness (synthetic price series with known cycle lengths)
- Signal threshold logic (edge cases at cycle boundaries)
- Gap-filling / forward-fill behaviour
- DB read/write roundtrips
- Backtest engine: known price series + known strategy → verify simulated spend

## Beads

Work items (what was previously GitHub Issues) live in [Beads](https://github.com/gastownhall/beads) (`bd`), a git-native, dependency-aware issue tracker. GitHub Issues were retired for this project 2026-08-06; PRs, CI, and reviews still live on GitHub as before — only the backlog moved.

- `.beads/` holds bd's config (git-tracked) and its Dolt database (`.beads/embeddeddolt/`, gitignored — it does not travel via ordinary `git push`). Cross-checkout sync is `bd dolt pull` / `bd dolt push` against the `origin` Dolt remote, not git.
- Finding work: `bd ready` (open, unblocked), `bd show <id>` (full detail + deps), `bd search <query>`.
- Working an issue: `bd update <id> --claim` (marks in_progress), `bd close <id>` when done — there is no GitHub auto-close-on-merge equivalent, closing is always an explicit step. Run `bd dolt push` after any write you want visible elsewhere.
- Filing work: `bd create --title "..." --description "..." --labels chore|polish|design` — see [§ Issue label taxonomy](#issue-label-taxonomy) below for which label. The `spawn_task` redirect in [CLAUDE.md](CLAUDE.md) uses this.
- The 12 open design issues carried over from GitHub keep their original number as `external_ref` (e.g. `https://github.com/edwinsteele/fuel-price-signal/issues/271`) for traceability into old PR/commit history. New issues created directly in bd have no GitHub counterpart.
- **Decision pointer convention:** when a closed `design` issue represents a settled decision (an approach tried and accepted/rejected), file a thin `bd create --type=decision` — title + one-line takeaway + `--deps discovered-from:<resolved-issue-id>` + a reference to the doc section with the actual argument (e.g. "see ML_SIGNAL.md § TGP leading indicator"). The bd record is a queryable pointer so `bd search`/`bd find-duplicates` can catch re-litigation of settled ground; it is **not** a second copy of the argument. Two things make that pointer actually work: **close it on filing** (an open record means work outstanding and pollutes `bd ready` forever — closed records stay findable via `bd search <word> --status all`), and **put the searchable words in the title**, because `bd search` matches a contiguous substring of the *title* only and does not read descriptions (`--desc-contains` does). First use: `fps-bsb`. [docs/STATUS.md](docs/STATUS.md)/[docs/ML_SIGNAL.md](docs/ML_SIGNAL.md)/`PLAN_ml_signal.md` remain the only place the reasoning itself lives — see [docs/CONVENTIONS.md § One source of truth](docs/CONVENTIONS.md#one-source-of-truth-for-current-model-state). Backfill lazily as decisions come up in conversation, not as a batch project.
- **Ignore bd's own generic priming advice where it conflicts with this project's conventions.** `bd prime` (bd's built-in AI-context command) tells agents to stop using TodoWrite/TaskCreate and to stop keeping MEMORY.md files, in favour of `bd remember`. That's bd's generic pitch for projects adopting it as the *sole* state layer; this project made a deliberate narrower choice instead — bd holds work items plus a handful of atomic technical gotchas (`bd memories`), while process rules stay in CLAUDE.md/CONVENTIONS.md, decision narratives stay in docs, and Claude's own per-user memory (preferences, teaching style) stays in its private memory system outside this repo. Follow this file's conventions over `bd prime`'s generic ones when they conflict.

## Automation workflow

See [docs/automation.md](docs/automation.md) for the full state machine and operational details.

### Issue label taxonomy

| Label | Meaning | Who works it |
|-------|---------|--------------|
| `chore` | Formatting, dead code, doc fixes, dependency bumps, trivial cleanup | Automated worker |
| `polish` | Small contained features, test additions, minor refactors | Automated worker |
| `design` | Cycle detection, signal logic, ML work, architecture decisions | Owner only — never automated |
| `claude-authored` | PR was opened by the automated worker | Identifies worker-opened PRs |
| `experiment` | One candidate feature, run unattended by the local nightly runner | Local runner (`bd ready --label experiment`) — never the remote worker |
| `auto-merge-ok` | Safe to auto-merge once CI passes | Applied by worker to `chore` PRs |

The labels above are the **routing** axis: they decide *who* picks an issue up. They say nothing about what it is about, which is why a backlog of ~35 was unreadable by 2026-09-02 (every item `design` or `chore`, 20 of 38 at P2, and `bd ready` opening with eight never-scoped wishlist items).

**Topic labels — the second, orthogonal axis.** Every issue carries exactly one, alongside its routing label:

| Label | Scope |
|-------|-------|
| `batch1` | The batch1 close-out chain, and only that. Empty it and retire the label; do not repurpose it for batch2 — file `batch2` |
| `pipeline` | Experiment-pipeline machinery: `experiments/pipeline/**`, `experiments/lib/**`, dossier/retrospective/noise-floor defects and hoists |
| `research` | Feature and analysis tracks — candidate features, the phase axis, arbiter design, anything whose deliverable is a finding rather than code |
| `data` | Ingest and frame quality: FuelCheck/TGP sources, `fill.py`, panel membership |
| `product` | The end-user signal itself — delivery, CLI, what the owner actually acts on |
| `infra` | CI, scheduled tasks, the worker Routine, `.github/**` |

`bd list --label research` is now the way to read one thread; `bd ready` is the way to read across them.

**Priority is a queue position, not a severity.** P1 is reserved for the current focus's critical path and should hold under ~6 issues — if everything is P1, nothing is. P2 is real work with a near-term claim, P3 is real work that is not now, P4 is a parking lot that should be periodically emptied by closing rather than by demoting further.

**Classification examples:**
- `chore`: add a missing type hint, bump a dev dependency, fix a typo in a docstring, delete unused import
- `polish`: add a missing test for an existing function, extract a helper that duplicates two callers, add a `--verbose` flag to an existing CLI command
- `design`: change cycle detection algorithm, add a new signal class, modify the DB schema, anything that touches `cycle.py`, `signal.py`, or ML work

**Escape hatch — polish → design upgrade:**
If while implementing a `polish` issue you discover it actually requires design work:
1. Relabel the issue: `bd update <id> --add-label design --remove-label polish`
2. `bd note <id> "<why you stopped and what the design question is>"`, then `bd dolt push`.
3. Do not write any code.

### Branch and PR conventions

- Branch naming: `worker/<bd-id>-<short-slug>` (e.g. `worker/fps-1785999730120-5-048485f0-add-type-hints`)
- PR title: `fix: <issue title> (bd-<id>)` for chore; `feat: <issue title> (bd-<id>)` for polish — plus a `Resolves: <id>` line in the PR body (no GitHub auto-close magic on a bd issue; see [CLAUDE.md](CLAUDE.md#automated-worker-vs-interactive-session))
- PR body: 3–5 bullet plan (what changed, what didn't, what test was added)
- Target branch: always `main` (`--base main`)
- Run `uv run ruff check . && uv run pytest -q` before pushing; fix any failures
### Reviewing PRs

- PR branches authored in this repo are usually checked out under `.claude/worktrees/<slug>`. Before fetching anything, run `git worktree list`, match the PR's head branch name, and verify `git -C <path> rev-parse HEAD` equals the PR head sha. If it matches, review files directly from the worktree — far cheaper than paging `gh pr diff` or `git show FETCH_HEAD:<path>` per file. Fetch only if no worktree matches or it's stale.
- When reading PR metadata, skip comments unless they're actually needed (e.g. `gh pr view --json title,body,files` rather than a full view) — review bots (CodeRabbit et al.) attach large noise blobs.
