---
name: batch-dir-vs-candidates-dir
description: experiments/batches/<b>/ and experiments/candidates/<b>/ are different dirs with similar names; batch.json lives in the latter.
metadata:
  type: project
---

experiments/batches/<batch>/ (runner's frozen-artifact dir: freeze.json, noise_floor.json, meta['batch_dir']) and experiments/candidates/<batch>/ (candidate modules' dir: batch.json, redundancy.write_batch_record's output) are TWO DIFFERENT directories with confusingly similar names/roles. A dossier_tables.py call site that wants batch.json must use run_dir.parent (== candidates dir), never meta['batch_dir'] — passing the latter always misses the file and produces a plausible-but-false 'record was never written' reason. Bit fps-qbv's first PR (#341); caught by review before merge. See docs/routines/dossier.md and experiments/pipeline/redundancy.py's batch_dir_for().
