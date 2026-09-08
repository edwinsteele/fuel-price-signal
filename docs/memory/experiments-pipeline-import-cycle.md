---
name: experiments-pipeline-import-cycle
description: dossier_tables -> runner -> batch_freeze is load-bearing; noise_floor must be imported inside function bodies.
metadata:
  type: project
---

experiments/pipeline module import cycle: dossier_tables -> runner -> batch_freeze is a load-bearing chain (runner.py imports from batch_freeze; dossier_tables imports from runner). noise_floor.py imports from BOTH batch_freeze and dossier_tables, so it must never be imported at module level by batch_freeze.py (would cycle: batch_freeze -> noise_floor -> batch_freeze) or by dossier_tables.py transitively. Same applies to dossier_tables symbols (e.g. NOISE_FLOOR_FILENAME) imported into batch_freeze.py — top-level import cycles back through runner. Fix used in fps-cf8: import inside the function body (deferred to call time), not at module top — by call time every module in the chain is already fully loaded, so the cycle never actually executes.
