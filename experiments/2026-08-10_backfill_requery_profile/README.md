# Backfill re-query profile (seed-db.yml's two slowest steps)

- **Date:** 2026-08-10
- **Branch:** main
- **SHA:** 6319895
- **Status:** done (design landed; implementation tracked in fps-53j + fps-2us)

## Hypothesis

`classify_range` and `score_leadership_range` are the two heaviest steps of a from-scratch
DB rebuild (~25 of 46 min on a GitHub runner, disproportionately worse on Viking's APU3).
Both re-query SQLite once per snapshot over a trailing window that overlaps the previous
iteration's by ~98%. Expected: the redundant re-query dominates, and eliminating it is a
data-access change, not an algorithm change.

## How to invoke these scripts

```bash
PYTHONPATH=. uv run python experiments/2026-08-10_backfill_requery_profile/profile_backfill.py fuel_signal.db 2>&1 | tee experiments/2026-08-10_backfill_requery_profile/run.log
```

```bash
PYTHONPATH=. uv run python experiments/2026-08-10_backfill_requery_profile/proto_backfill.py fuel_signal.db
```

`profile_writes.py` **mutates** the DB it is given — pass a throwaway copy, never
`fuel_signal.db`:

```bash
cp fuel_signal.db /tmp/scratch-copy.db && PYTHONPATH=. uv run python experiments/2026-08-10_backfill_requery_profile/profile_writes.py /tmp/scratch-copy.db
```

## Setup

Real DB, 492 MB: 2.23M E10 `daily_prices` rows, 3597 distinct dates (2016-08-01 →
2026-06-06), 744 councilled stations, 32 LGAs. Backfill sizes: 3597 daily snapshots for
classify, 523 weekly for leadership. Timings are Mac (arm64); the GitHub-runner
comparison points are from PR #279, `gh run 31300192738`.

- `profile_backfill.py` — read-only. Per-snapshot SQL vs python compute for both steps;
  memory cost of the naive full-range load; `EXPLAIN QUERY PLAN` for the ordered scan.
- `profile_writes.py` — write-path breakdown for classify (baseline / commits suppressed /
  writes+commits suppressed, 30 snapshots each).
- `proto_backfill.py` — reference implementations of both fixes plus parity checks against
  the current queries. **This is the file to lift from.**

`run.log` is gitignored (`*.log`, repo-wide) — it exists only in the dir where the run
happened. Every number it held is transcribed into the Results tables below, so re-run the
scripts rather than looking for it.

## Results

Hypothesis confirmed, and more lopsided than expected — SQL is 79% of classify's
read+compute and **99%** of leadership's. scipy/trough detection is 1% of leadership.

| step | snapshots | per-snap SQL | per-snap compute | write | commit | projected (Mac) | measured (GHA) |
|---|---|---|---|---|---|---|---|
| `classify.py` | 3597 daily | 39 ms | 11 ms | 5 ms | 8–18 ms | 276 s | 471 s |
| `lga_leadership.py` | 523 weekly | 1042 ms | 12 ms | — | — | 583 s | 1019 s |

Mac→GHA ratio is 1.66× / 1.75× — consistent, so the projections carry to both runners.
Commit cost is the one noisy number (8–18 ms/snap across runs; fsync under WAL +
`synchronous=FULL`).

**Memory is what forces two different fixes.** The obvious "load the full range once" is
fine for one step and disqualifying for the other:

| full-range load | rows | time | resident | peak |
|---|---|---|---|---|
| `_load_lga_sums` (aggregated to `(date, LGA)`) | 106k | 5.8 s | **30 MB** | 30 MB |
| `daily_prices_in_window` (raw per-station) | 2.23M | 8.4 s | **540 MB** | **881 MB** |

540 MB resident / 881 MB peak will OOM Viking, so classify needs a sliding window rather
than a preload. `ORDER BY dp.price_date` costs nothing — `daily_prices_fuel_date` already
delivers that order, no sort step in the plan, and the ordered full scan clocks the same
2.35 s as the unordered one.

Prototype results, all windows byte-identical to the current queries:

| | parity | before | after | speedup |
|---|---|---|---|---|
| classify read+compute | 45/45 windows, identical class counts | 198 s | ~16 s | 12× |
| leadership read | 4/4 sampled windows | 497 s | 5.4 s | 92× |

Leadership parity is structural rather than merely sampled: `_load_lga_sums` has
`HAVING COUNT(*) >= MIN_STATION_FLOOR` grouping on `(dp.price_date, s.council)`, every
group lives inside a single date, and the window filter only selects which dates are
included — so no window can change a group's membership.

Expected step times on a GitHub runner: classify 471 s → ~90 s, leadership 1019 s → ~40 s.
The two steps go 25 min → ~2 min and `seed-db.yml` 46 min → ~23 min, leaving `history.py`
(16 min, network-bound) as the bottleneck.

## Conclusion

Two fixes, same root cause, different shapes — dictated by how far SQL has already
aggregated. Convention recorded in AGENTS.md § "Backfill (`--start-date`) paths: load
once, slice in memory". Single-snapshot paths (`daily-db-update.yml` runs both without
`--start-date`) stay as they are; only the range paths change.

## Followups

- **fps-53j** — `lga_leadership`: preload sums once, slice per snapshot. Do first (bigger
  win, simpler, provable parity).
- **fps-2us** — `classify`: single ordered scan + sliding day-bucket deque, plus batched
  commits in range mode.
- Before/after wall-clock on a real backfill is carried on those two issues — this dir has
  measured *before* numbers and prototype projections only.
