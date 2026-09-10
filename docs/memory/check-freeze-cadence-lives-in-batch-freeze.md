---
name: check-freeze-cadence-lives-in-batch-freeze
description: check_freeze_cadence moved out of noise_floor.py into batch_freeze.py (#382) so runner.py could call it without a circular import.
metadata:
  type: project
---

`check_freeze_cadence` (the guard refusing to grade/write at a cadence a batch's
`freeze.json` doesn't declare, fps-oqz) now lives in `experiments/pipeline/batch_freeze.py`,
not `noise_floor.py` where it originated — moved in #382 so `runner.py`'s
`run_candidate()` could call it too without creating the module-level cycle
[[experiments-pipeline-import-cycle]] describes (`noise_floor.py` imports from
`runner.py` at load time, so `runner.py -> noise_floor.py` would cycle back).
`noise_floor.py` still re-exports it under the same name via a top-level import
from `batch_freeze`, so existing `from experiments.pipeline.noise_floor import
check_freeze_cadence` call sites (including `tests/test_noise_floor.py`) keep
working unchanged.
