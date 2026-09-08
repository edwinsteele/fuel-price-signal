---
name: status-rejected-means-graded-not-failed
description: STATUS_REJECTED means 'finished grading', not 'rejected' — read effect_delta_cpl_held for the verdict.
metadata:
  type: project
---

runner.py's STATUS_REJECTED constant (value 'rejected') is written to EVERY candidate that completes a full graded run in results.json/facts.json — win, loss, or in-between. It is NOT a verdict, just 'the pipeline finished grading, here are real numbers' vs disqualified/aborted. Don't read status=='rejected' in a dossier artifact as 'the pipeline rejected this candidate' — check effect_delta_cpl_held / noise_band_z / clears_family_wise_threshold for the actual outcome. Tracked for a rename to STATUS_GRADED in fps-0z5.
