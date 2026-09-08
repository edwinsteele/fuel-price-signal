---
name: handover-before-results-csv-write
description: Stop and hand over to the user before any step that writes experiments/results.csv — running it is the violation.
metadata:
  type: project
---

When a rebuild reaches a step that writes to experiments/results.csv (score_phase2 without a skip flag, calibrate without --skip-results-csv, or any lock-row writer), stop and hand over to the user BEFORE running it. The branch-side half of the rule (results.csv is only ever modified on main) lives in docs/CONVENTIONS.md § Git workflow. Don't run-then-revert — running the write is itself the violation, not just what's left afterward.
