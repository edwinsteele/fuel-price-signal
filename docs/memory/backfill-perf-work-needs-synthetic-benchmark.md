---
name: backfill-perf-work-needs-synthetic-benchmark
description: The 492MB profiling DB is gitignored and usually absent; write a synthetic parity+direction benchmark instead.
metadata:
  type: project
---

Backfill-perf issues (the fps-53j/fps-2us family: preload-once-slice-per-snapshot fixes) usually carry an acceptance criterion like 'before/after wall-clock captured for a multi-snapshot backfill', modeled on the real 492MB/2.2M-row fuel_signal.db used in the profiling experiment (experiments/2026-08-10_backfill_requery_profile/). That DB is gitignored and was NOT present in any worktree as of 2026-08-10 (checked whats-next-ecf413 and the primary checkout, both empty) — don't assume it's there; check with 'find . -iname "*.db"' first since a future session's machine state may differ.

If absent: write a throwaway synthetic-DB benchmark (open_db(':memory:'), seed a few dozen stations across several LGAs with distinct addresses — INSERT OR IGNORE on stations dedupes on address_normalized, so identical placeholder addresses across stations silently drop all but the first — and a few hundred days of history), time the old per-snapshot-loop shape against the new batched/streamed function on identical data, and assert row-for-row parity. The absolute speedup will be much smaller than the production-scale profiled number (small row counts don't stress the re-query cost the fix targets) — report it as a parity+directional-speedup check, and cite the experiment README's production-scale projection as the actual evidence for the acceptance criterion, not the synthetic number alone.
