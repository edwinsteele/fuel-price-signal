---
name: graduating-a-column-is-two-edits
description: Graduating a column means appending to LOCKED_FEATURE_COLUMNS and deleting from NON_MODEL_COLUMNS, in one change.
metadata:
  type: project
---

Graduating a held-out feature column into the locked model contract is TWO edits, in the same change:
  1. add it to fuel_signal.features.LOCKED_FEATURE_COLUMNS — APPEND, never insert or sort
  2. DELETE its entry from fuel_signal.features.NON_MODEL_COLUMNS

Do only step 1 and resolve_baseline_columns() raises NonModelColumnLeak; batch_freeze and the candidate runner abort. This is a hard stop, not a warning, on purpose: graduation is a declaration, not something the code should infer.

Why it works this way (fps-zci, PR #312): non_model_columns() decides a column's category from what the column IS — named in NON_MODEL_COLUMNS, or trough-prefixed and not one of the LGA troughs — and deliberately does NOT check whether the lock already claims it. A detector that skipped whatever the lock contains cannot detect a column wrongly IN the lock, which is the entire fps-sa1 failure ([[baseline-fingerprint-before-comparing-runs]] is the detector for the general case) (10 rejected brand-trough columns sitting in a 64-column R0 for two months).

Two traps at re-lock time:
  - ORDER is part of the contract. LightGBM breaks equal-gain split ties by feature index, so a permutation fits a different model (measured: 0.038 c/L on the pooled realised delta). Append leaves existing indices untouched; inserting or sorting does not.
  - tests/test_feature_contract.py asserts LOCKED_FEATURE_COLUMNS == the artifact's feature_columns (ORDERED ==), but SKIPS when data/models/ is absent — and it always is in CI, since the dir is gitignored and ci.yml runs a bare `pytest -q`. Verify the re-lock LOCALLY; green CI does not mean the contract was checked.

Live case: tgp_delta_7d at the #271 chip-4 re-lock — bd fps-1785999729707-1-0301bf82, component 5 carries the full revised sequence.
