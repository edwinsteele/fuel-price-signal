---
name: baseline-fingerprint-before-comparing-runs
description: Compare two runs' baseline fingerprints before their numbers; different fingerprint means not commensurable.
metadata:
  type: project
---

Before comparing the numbers from two experiment runs, compare their BASELINE FINGERPRINTS. Different fingerprint = not commensurable, whatever the deltas say.

The fingerprint is '<n>:<sha12>', a sha256 over the ORDERED baseline column list (fuel_signal.features.baseline_fingerprint). The current lock is 54:1a6ec2d84a69. It is recorded in:
  - experiment meta.json      -> meta["baseline"]  (stamped automatically by experiments.lib.io.write_meta)
  - batch freeze.json         -> "baseline_fingerprint"
  - run results.json / facts.json -> meta["baseline_fingerprint"] + "n_baseline_columns"
  - experiments/results.csv   -> "baseline_fingerprint" column (derived from that row's own `features` list)

A null/missing fingerprint means the run predates fps-zci (PR #312, 2026-08-19). Do NOT substitute today's constants for it — that asserts the run used today's baseline, which is exactly the claim that was false for every batch0 run. The absence IS the signal.

Why this exists: two contract defects (fps-sa1, wrong column SET: a 64-column R0 including the rejected Phase 4b brand troughs; fps-zci, wrong column ORDER: the right 54 sorted alphabetically, which fits a different LightGBM model because ties break on feature index) were compared head-to-head against a correct June run for two months. Neither left any trace in the artifacts that recorded the runs. The fingerprint is the detector; docs/CONVENTIONS.md § "The baseline feature set is declared, never discovered" is the full rule.
