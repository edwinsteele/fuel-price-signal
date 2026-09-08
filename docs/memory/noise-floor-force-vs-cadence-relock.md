---
name: noise-floor-force-vs-cadence-relock
description: --force recovers a floor after a column-lock graduation but is refused after a cadence re-lock; freeze a new batch.
metadata:
  type: project
---

noise_floor.py --force recovers a batch's floor after a COLUMN-LOCK graduation (fingerprint drift) but is refused by check_freeze_cadence after a CADENCE re-lock (e.g. fps-oqz 7d->1d) — the guard deliberately won't let freeze.json and noise_floor.json disagree on cadence. After a cadence re-lock, freeze a NEW batch instead of trying to recompute the old one in place (see docs/CONVENTIONS.md 'A re-lock invalidates the batch's noise floor'). This is what actually happened to batch0 on 2026-08-22 (fps-cds's original plan assumed --force would work and had to be superseded by fps-aay).
