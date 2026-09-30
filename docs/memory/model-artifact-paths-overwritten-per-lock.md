---
name: model-artifact-paths-overwritten-per-lock
description: data/models/lgbm.joblib and lgbm_calibrated.joblib are fixed paths overwritten by every lock — identify what's on disk by its feature_columns, never by filename or a log's mention.
metadata:
  type: project
---

`train_lgbm.py` and `calibrate.py` write to fixed paths (`data/models/lgbm.joblib`,
`data/models/lgbm_calibrated.joblib`). Each phase lock silently replaces the previous artifact;
the phase lives in `experiments/results.csv` (`name` column) and commit history, never in the
filename. The binaries are gitignored.

**Identify an artifact by its metadata:**

```python
import joblib
m = joblib.load("data/models/lgbm_calibrated.joblib")
print(len(m["feature_columns"]), m["feature_columns"][-1])
```

Compare against `LOCKED_FEATURE_COLUMNS` ([docs/STATUS.md](../STATUS.md) names the current
lock). Historical signatures: 54 ending `lga_phase_std_delta_3d` (post-#216, RAC_full);
60 → Phase 4b; 50 ending `days_since_trough_entry_woollahra` → Phase 4; 15 ending
`stickiness_score` → 3c; 14 ending `station_minus_brand_mean_cents` → 3b; 10 → 3a.

A filename in a run log that isn't in `data/models/` (e.g. `lgbm_stickiness_calibrated.joblib`
in `experiments/phase3c_score.log`) was a worker-worktree alias, never persisted.

**Reconstructing an old lock:** `git checkout <commit>` and re-run train + calibrate; the binary
itself is not recoverable. **Evaluating a candidate that retrains the canonical path:** first
train the baseline to a side path (`train_lgbm --no-brand-features --model-out
data/models/lgbm_baseline.joblib`); if the candidate doesn't lock, `cp` it back and
`python -m fuel_signal.calibrate --skip-results-csv` (the flag prevents a duplicate
calibration row).

Related: [[graduating-a-column-is-two-edits]].
