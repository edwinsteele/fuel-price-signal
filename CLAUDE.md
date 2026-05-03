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
├── compare.py       # Compare two price series (station vs station, station vs LGA mean, etc.)
└── backtest.py      # Replay historical prices through signal + purchasing strategy
```

## CLI pattern

Each command is its own module with a `@click.command` named `main` and an `if __name__ == "__main__": main()` block. Invoked as:

```
uv run python -m fuel_signal.signal [--as-of DATE] [--db PATH]
uv run python -m fuel_signal.compare SERIES_A SERIES_B [--fuel E10] [--within 0.5] [--db PATH]
uv run python -m fuel_signal.stations [QUERY]
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
- Atypical periods (COVID demand collapse, 2026 Middle East war supply shock) distort mean cycle length, peak prominence, and last-cycle min/max. When building the backtest engine or calibrating signal thresholds, consider a mechanism to mark/exclude date ranges — but don't bake it in prematurely; add it when backtest results show anomalies traceable to a known shock.

### Backtest constraints
- Backtests must be runnable at arbitrary historical dates ("at date D, would strategy X have been cheaper?"), not just today.
- `daily_prices` is point-in-time safe (forward-fill uses no lookahead), but derived metrics built on top may not be. When adding new metrics, explicitly validate whether they can be recomputed on-the-fly by querying `WHERE price_date <= D`, or whether they need to be pre-computed and stored per day.
- Backtest performance: load the full series ONCE at startup; `detect(as_of_date)` is an in-memory numpy slice (~0.5 ms × 3650 dates ≈ 2 s total). `CycleDetector` must cache `pd.Series` in `__init__` — if conversion happens inside `detect()`, you pay it 3650× per backtest run.

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

## inspect.py rewrite (Flask)

`inspect.py` is currently a static HTML generator. It will be rewritten as a local Flask dev server — `uv run python -m fuel_signal.inspect` starts the server rather than writing a file.

**Reason:** the desired features require round-trip queries (station search/add, named groups with aggregated series) that are impractical to do client-side with all data pre-loaded.

**Target features:**
- Add/remove individual stations from the chart by name search
- Define named groups of stations (e.g. "Shell servos in Sydney", "all stations in region X") with an aggregated price series displayed alongside individual ones
- Hide/show series (Chart.js legend click already does this — keep it)

**Planned architecture:**
- Flask app replaces `generate_html()` — same entry point
- Chart.js stays for rendering; data fed via `fetch()` to Flask endpoints rather than baked-in JSON
- Endpoints needed roughly: station search, get series for station or group, list current chart contents
- Keep the summary stats, coverage table, and recent prices sections from the current page

**Current static generator state (as of 2026-04-27):**
- Single `generate_html()` bakes Sydney avg + preferred station series as JSON into a `<script>` block
- Two charts: full history (scipy peaks as red dashed verticals, last-cycle shaded, data-gap grey shading) + 6-month zoom with preferred stations
- Cycle state box at top; coverage table and recent prices table included

**Leading indicators (deferred until after Flask rewrite):**
- Hypothesis: some LGAs and/or macro signals (TGP, crude) precede BM price rises
- One preferred Penrith station observed to lead on falls, lag on rises — worth examining
- Architecture supports this: new series → new `CycleDetector` → new signal class → register in `RecommendationManager`

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

## Automation workflow

### Label taxonomy

| Label | Meaning | Who acts |
|-------|---------|----------|
| `chore` | Formatting, dead code, doc fixes, dependency bumps, trivial cleanup | Worker picks up; auto-merges on green CI |
| `polish` | Small contained features, test additions, minor refactors | Worker picks up; owner review required before merge |
| `design` | Cycle detection, signal logic, ML work, architecture decisions | **Worker never picks these** — owner works manually |
| `claude-authored` | PR was opened by the worker routine | Applied automatically by the worker |
| `auto-merge-ok` | Safe to auto-merge once CI passes | Applied by worker to `chore` PRs after CI is green |

**Classification examples:**
- `chore`: add a missing type hint, bump a dev dependency, fix a typo in a docstring, delete unused import
- `polish`: add a missing test for an existing function, extract a helper that duplicates two callers, add a `--verbose` flag to an existing CLI command
- `design`: change cycle detection algorithm, add a new signal class, modify the DB schema, anything that touches `cycle.py`, `signal.py`, or ML work

### If you are the scheduled worker routine

You are a Sonnet worker. You run hourly. Your job is to pick up `chore` and `polish` issues and open PRs.

**Pickup rules:**
1. Check for open `claude-authored` PRs that need maintenance: merge conflicts (`gh pr view N --json mergeable` returns `CONFLICTING`) or unresolved review threads with no `[worker]` reply. If any exist, perform maintenance (see **PR maintenance** below), then exit.
2. Check for open `claude-authored` PRs (any). If any exist, **exit immediately** — one at a time.
3. Query `gh issue list --label "chore,polish" --state open --no-assignee --json number,title,labels,createdAt` ordered by label (`chore` before `polish`), then by age (oldest first). Take 1.
4. Create a branch `worker/issue-<N>-<slug>` for the issue.

**For each PR:**
1. Implement the minimal change — do not scope-creep.
2. Run `uv run ruff check . && uv run pytest -q` locally before pushing. Fix any failures.
3. Open PR titled `fix: <issue title> (closes #N)` targeting `main` (`--base main`) with labels `claude-authored` + the issue's original label. PR body must include a 3–5 bullet plan (what changed, what didn't, what test was added).

**PR maintenance:**
When pickup rule 1 triggers, for each qualifying PR:

*Merge conflicts:*
1. Check out the branch locally.
2. `git fetch origin && git rebase origin/main`. Resolve any conflicts — prefer the incoming (`main`) change unless the branch change is clearly intentional, in which case keep both.
3. Run `uv run ruff check . && uv run pytest -q`. Fix any failures.
4. `git push --force-with-lease`.

*Unresolved review threads:*
1. Run `gh pr view N --json reviewThreads` and filter to threads where `isResolved` is false and no comment body starts with `[worker]`.
2. Read all such threads together to understand the full set of requested changes.
3. For any thread that is ambiguous or requires a design decision: reply `[worker] Needs owner input — <question>` and skip it. Do not make changes for that thread.
4. Make the minimal changes to address the remaining threads.
5. Run `uv run ruff check . && uv run pytest -q`. Fix any failures.
6. Push.
7. Reply to each addressed thread: `[worker] Done — <one sentence describing what changed>`.

Handle conflicts first, then review threads, in a single pass per PR.

**Escape hatch — polish → design upgrade:**
If while implementing a `polish` issue you discover it actually requires design work:
1. Relabel the issue to `design` using `gh issue edit N --add-label design --remove-label polish`.
2. Post a comment explaining why you stopped and what the design question is.
3. Do not write any code. Move on to the next issue in the batch.

**Branch naming:** `worker/issue-<N>-<short-slug>` (e.g. `worker/issue-7-add-type-hints`)

**PR title format:** `fix: <issue title> (closes #N)` for chore; `feat: <issue title> (closes #N)` for polish.

### If you are an interactive session

- **Do not pick up `chore` or `polish` issues yourself.** Those belong to the worker routine. If you notice something that belongs in one of those categories, file a `gh issue create` instead.
- **`design` issues are fair game** for interactive work — work them with the owner as you would any other task.
- Do not open PRs with `claude-authored` label — that label is exclusively for the worker.

### spawn_task → gh issue create redirect

When `mcp__ccd_session__spawn_task` would normally be the right call (you noticed an out-of-scope issue while working), **do not spawn a session**. Instead:

```bash
gh issue create \
  --title "Short imperative title" \
  --label "chore"  # or polish or design \
  --body "$(cat <<'EOF'
## What
<what needs doing>

## Why I noticed this
<file paths + context — e.g. "saw this while implementing X in fuel_signal/cycle.py">

## Files likely affected
- fuel_signal/foo.py

## Acceptance criteria
- [ ] ...
EOF
)"
```

Use `chore` for cleanup, `polish` for small features/tests, `design` for anything architectural. Include file paths and enough context for the worker (or future-you) to act cold.
