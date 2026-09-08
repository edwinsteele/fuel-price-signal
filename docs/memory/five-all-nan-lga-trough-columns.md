---
name: five-all-nan-lga-trough-columns
description: Five locked LGA trough columns are permanently 100% NaN, so a complete-case mask over the locked set selects zero rows.
metadata:
  type: project
---

Five LGA trough columns are 100% NaN in the real features frame: days_since_trough_entry_{bayside,botany_bay,hunters_hill,lane_cove,waverley}. They ARE in LOCKED_FEATURE_COLUMNS (part of the 54) — their LGA never produced a confirmed trough event, so this is permanent structure, not a data bug to fix.

Consequence for anything that fits/regresses ACROSS the locked set at once: a complete-case mask (predictors.notna().all(axis=1)) selects ZERO rows and every result returns NaN, silently. This bit experiments/pipeline/redundancy.py's block-R2 check (fps-3jj.15): the screen printed a clean report and measured nothing. Fixed there by usable_predictors(), which drops all-NaN and constant predictors and discloses the drop in the report.

placebo.py meets the same five from the other direction (experiments/INDEX.md 2026-08-22: '5 all-NaN skipped, 49 usable' of 54).

Synthetic test frames will NOT reproduce this — smoke-run against data/features.parquet before opening the PR.
