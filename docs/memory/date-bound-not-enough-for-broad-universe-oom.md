---
name: date-bound-not-enough-for-broad-universe-oom
description: Bounding a per-station daily_prices query by date range does not by itself avoid the Viking-OOM shape when the station count is large (a broad universe) — concatenating all stations' rows into one frame still does.
metadata:
  type: project
---

`experiments/lib/universe.py::_window_label_rows` (#387, PR #413) originally called
`fuel_signal.labels.assemble_training_rows(conn, station_codes=codes)` with NO date
bound — a raw per-station-row load over EVERY station's FULL history, exactly the
"~2.2M raw rows for the decade" shape [docs/AGENTS.md § Backfill paths](../../AGENTS.md)
warns can OOM Viking. The first fix (bounding the query to the smallest date envelope
that still gives every window's calendar-gap mask its full lookback/horizon context)
was necessary but NOT sufficient: for a 410-station universe it still concatenated
every station's label rows into ONE DataFrame, resident alongside whatever the caller
(e.g. `homogeneity.py`, already holding an ~800k-row feature frame) had loaded —
Codex flagged this as a SEPARATE P1 finding on the SAME PR, after the date-bound fix
had already landed.

The actual fix: call the per-station function ONE STATION AT A TIME and discard each
station's frame immediately after counting, rather than passing the whole `codes` list
to one call. Peak resident label data becomes one station's rows (at most a few
thousand), never all of `codes`'s concatenated together. Cost is N small queries
instead of one big one, worth it only because `codes` in this path is O(100s) and
`windows` is O(10s) (outer folds) — a caller with the reverse shape (few stations,
thousands of windows) would want the streaming-bucket rewrite AGENTS.md's memory
section actually describes instead.

The general lesson: when a per-entity metric needs a bounded date range AND the
entity COUNT is what makes a naive load OOM-shaped, bounding the date range alone
only shrinks the query — check separately whether the RESULT is still concatenated
across all entities into one resident structure. Two different axes, two different
fixes, and a reviewer (or this file) checking one does not mean the other is checked.
