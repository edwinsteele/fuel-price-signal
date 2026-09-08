---
name: worktree-missing-gitignored-batch-data
description: Gitignored batch/candidate artifacts don't exist in a fresh worktree; copy them in before running the pipeline.
metadata:
  type: reference
---

Batch/candidate pipeline artifacts (data/features.parquet, fuel_signal.db, and per-batch/per-candidate features.parquet, fuel_signal.db, r0_cache.joblib, fills.parquet, rowpreds.parquet under experiments/batches/<batch>/ and experiments/candidates/<batch>/<candidate>/) are gitignored and do NOT exist in a fresh worktree checkout. Before running experiments.pipeline.shock_folds, dossier_tables, or any script that reads them, copy them from wherever they exist (e.g. the primary checkout) into the worktree first, or the run fails with FileNotFoundError. Confirmed live during the fps-hnp backfill (PR #351).
