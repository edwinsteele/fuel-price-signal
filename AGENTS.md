# fuel-price-signal — agent context

A Python CLI that outputs a one-line buy/don't-buy signal for E10 fuel at preferred stations near postcode 2777 (Springwood/Blue Mountains corridor).

## Output format

```
BUY  | Day 41/46 of cycle | E10 @ Caltex Springwood: 161.9c | Trough est. ~5 days
WAIT | Day 12/46 of cycle | E10 @ Caltex Springwood: 179.2c | Trough est. ~34 days
```

See [docs/STATUS.md](docs/STATUS.md) for current build status and pending phases.
See [docs/ML_SIGNAL.md](docs/ML_SIGNAL.md) for ML model design decisions and results.
See [docs/CONVENTIONS.md](docs/CONVENTIONS.md) for code style, test patterns, definition-of-done, git workflow, issue tracking and the label taxonomy.
See [docs/ML_PIPELINE.md](docs/ML_PIPELINE.md) for the ML model training/evaluation CLI reference (README.md covers setup and day-to-day signal usage only).
See [docs/feature-pipeline.md](docs/feature-pipeline.md) for the AI-sourced candidate-feature pipeline's machinery.
See [docs/data-semantics.md](docs/data-semantics.md) for station classification, price-data shape, and the traps in filtering on class.

**This file has a byte budget.** Codex loads `AGENTS.md` eagerly up to 32 KiB and silently
drops the remainder, so it stays architecture, module layout and review rules only — everything
else goes in the docs above and is reached by link. `tests/test_agents_md_budget.py` enforces
the limit; when it fails, move content out rather than deleting it.

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

The NSW source data has holes and quirks that will otherwise look like bugs in our code — most
consequentially the unrecoverable 2022-03-12 → 2022-03-21 gap. Full list, with what each one
does to a fold or a series: [docs/data-semantics.md § Known source data limitations](docs/data-semantics.md#known-source-data-limitations).

### Aggregation
`sydney_average_series` / `average_price_series` is a temporary convenience for cycle detection. Future analyses will need flexible groupings — by region, corridor, LGA cluster, etc. Don't treat it as permanent infrastructure; don't patch it when new groupings are needed, design a proper aggregation layer instead.

### Station classification (Competitive / Discount / Sticky)

Every station carries one of three classes in the `station_class` table, assigned by `classify.py`. Definitions, thresholds, the sticky-exclusion rule and the traps in filtering on class live in [docs/data-semantics.md § Station classification](docs/data-semantics.md#station-classification-competitive--discount--sticky). Read it before touching `classify.py` or filtering on class.

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

See [docs/CONVENTIONS.md § Multi-seed test-logloss policy](docs/CONVENTIONS.md#multi-seed-test-logloss-policy) — when to pass `--seeds`, which metric is banked, and the 3×std comparison gate.

## Testing

Tests are required alongside all implementation. Required coverage areas and the DB fixture / `CliRunner` patterns: [docs/CONVENTIONS.md § Tests](docs/CONVENTIONS.md#tests).

## Technical memories

[docs/memory/](docs/memory/INDEX.md) holds this repo's atomic technical gotchas — one fact
per file, `name`/`description`/`type` frontmatter, wiki-style `[[name]]` links between related facts, and
an [INDEX.md](docs/memory/INDEX.md) whose hooks are written to tell you whether you need a
file without substituting for it. They are short, load-bearing, and cheap to read; several
are rules you will otherwise break before noticing. Read the index at orientation and `grep
-l <term> docs/memory/*.md` when you hit something surprising.

They live in the repo rather than in an agent's private memory store **because every agent
working here must see them** — Claude and Codex both. Scope: traps in specific code paths,
specific data, and specific tools. Rules for *how we work* belong in
[docs/CONVENTIONS.md](docs/CONVENTIONS.md); architecture belongs in this file; current model
state belongs in [docs/STATUS.md](docs/STATUS.md).

### Writing one

**Every agent working here writes new technical memories to `docs/memory/`, not to its own
private memory store.** A fact filed privately is invisible to the other agent, and to the
next session of a different tool — which is the entire reason these 53 were moved out of
Beads. This applies to Claude and Codex identically.

1. **Classify first.** *Would this be true for a different person working on this repo?*
   Yes → `docs/memory/`. No — it's about the owner (their background, how they like to be
   taught, their working hours) → an agent's own private memory, not the repo. If it's a
   rule for *how we work* rather than a trap in a specific thing, it belongs in
   [docs/CONVENTIONS.md](docs/CONVENTIONS.md) instead.
2. **Check for an existing file** — `grep -l <term> docs/memory/*.md`. Update it rather than
   adding a near-duplicate; the corpus is only cheap to read while it stays small.
3. **One fact per file**, named in kebab-case after the fact, with `name` / `description` /
   `type` frontmatter (`project` for code/data gotchas, `reference` for tool and environment
   traps, `feedback` for working-practice guidance). `description` is what a reader scans to
   decide whether to open the file, so make it the claim, not the topic.
4. **Add a one-line hook to [INDEX.md](docs/memory/INDEX.md)** under the right heading. A file
   with no index line will not be found.
5. **Link related facts** with `[[name]]`. Nothing resolves these automatically — they are
   breadcrumbs for the next reader, so a link to a file that doesn't exist yet is fine.
6. **Delete a memory that turns out to be wrong.** A stale memory is worse than a missing one:
   `github-issue-state-meaningless-post-migration` inverted rather than decayed, and would have
   broken the worker routine if it had been migrated unread.

**No PR needed** — `docs/memory/**` is exempt, like the experiments lab book. Commit **and
push**; a memory sitting on a local `main` is not filed.

These were `bd remember` entries until 2026-09-08 —
[docs/memory/MIGRATION.md](docs/memory/MIGRATION.md) accounts for all 53.

## Issue tracking

Work items live in **GitHub Issues**, driven from the `gh` CLI. The live backlog opened at **#365–#388** on 2026-09-07, when the Beads (`bd`) experiment was cut back over; PRs, CI and reviews never moved. Two rules carry most of the weight: **the assignee IS the claim** (no separate in-progress state), and **`Closes #<N>` in the PR body** is what closes an issue on merge.

Command reference, the decision-pointer convention, the `--label` vs `--search` consistency trap, and `fps-*` archive lookup: [docs/CONVENTIONS.md § Issue tracking](docs/CONVENTIONS.md#issue-tracking). Filing discipline: [§ Filing and finding issues](docs/CONVENTIONS.md#filing-and-finding-issues).

## Automation workflow

See [docs/automation.md](docs/automation.md) for the full state machine and operational details, and [docs/CONVENTIONS.md § Issue label taxonomy](docs/CONVENTIONS.md#issue-label-taxonomy) for the routing/topic/priority labels that decide who picks an issue up.

## Code Review Rules

For `@codex review`. General correctness review needs no instruction here — these are the
repository-specific rules.

**Do not report:**

- **Missing or incomplete docstrings.** This repo uses WHY-comments over formal docstrings; a
  docstring written to clear a coverage threshold fights the convention. Comment when an
  invariant is non-obvious, not otherwise.
- **Style, formatting, or import order.** `ruff` owns these and gates every PR in CI.
- **Anything in `experiments/**` that predates `experiments/lib/`.** Those dirs are frozen
  lab-book records of what was run, not the template. Review them only when the diff changes them.

**Verify before reporting.** Every reviewer this repo has had — CodeRabbit, Sourcery, and an
independent Claude session — has filed a confident false positive, and each was caught by one
of these two checks:

- **Read the whole function a hunk lives in, not just the changed lines.** A review of PR #301
  flagged a test fixture's `price_date` dtype as mismatched against `runner.py`; the
  `pd.to_datetime()` that made it correct was in the unchanged lines just above the hunk.
- **Never cite prose as authority for a code constant.** `snapshot_retire.py`'s
  `DEFAULT_TOLERANCE = 0.05` was flagged as "should be 5.0 cents" on the strength of a nearby
  doc sentence reading "agrees within 5c". The prose was the imprecise one. Check what the code
  does, not how a doc rounds it off in words.

**Report these — each has shipped a real bug here:**

- **`std > 0` guards on float arrays.** `np.std` of identical floats is `1.78e-18`, not `0.0`,
  so the guard reads a degenerate band as a confident result. Safe path: pair it with a
  `np.ptp(...) == 0` exactness check. Shipped twice (#377, #406).
- **A `pandas` equality assertion in a leakage or point-in-time test without
  `check_exact=True`.** The default `rtol=1e-5` silently passes a real leak at YYYYMMDD
  magnitudes.
- **A column appended to `LOCKED_FEATURE_COLUMNS` without being deleted from
  `NON_MODEL_COLUMNS`.** Graduating a feature is both edits in one change, or
  `resolve_baseline_columns()` raises. The order of the locked list is part of the contract —
  never sort it.
- **A new feature family wired into `features.py` but not into `backtest.py`'s `decide()`.**
  `decide()` recomputes features independently of `features.py`; wiring only one aborts the
  realised backtest.
- **`Closes #<N>` in a PR body naming an issue not meant to close.** GitHub matches the keyword
  as a substring anywhere in the body, including inside a sentence disclaiming it.

**How to report:**

- **P0 and P1 only, at most five findings, most severe first.** This repo already runs Sourcery
  and CodeRabbit; a third source of nits is worse than none.
- **If you are not confident it is a real defect, do not file it.** A false positive costs more
  here than a missed nit.
- **State the safe path**, not only the objection.

This list samples a larger corpus rather than replacing it: the false-positive case studies are
in [docs/CONVENTIONS.md § Code review caution](docs/CONVENTIONS.md#code-review-caution), and the
numerical, pipeline and tooling traps are one hook each in
[docs/memory/INDEX.md](docs/memory/INDEX.md). Read those when a diff touches ground this section
does not cover.
