---
name: batch-freeze-stale-features
description: batch_freeze now hard-gates both DB and features staleness — the old manual pre-freeze workaround is obsolete.
metadata:
  type: project
---

batch_freeze.py's refresh_db() now hard-gates BOTH DB staleness and features.csv/.parquet staleness — after 'make update' it also runs 'make features' (uv run python -m fuel_signal.features), aborting with DbRefreshError if either step fails. Fixed fps-3vo (PR #307, merged 2026-08-18); the old gap (freeze could silently pin months-stale features right after a fresh DB pull) no longer exists. No manual 'run fuel_signal.features before freezing' workaround needed anymore.
