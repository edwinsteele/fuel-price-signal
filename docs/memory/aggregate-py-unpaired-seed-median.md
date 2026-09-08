---
name: aggregate-py-unpaired-seed-median
description: aggregate.py takes a difference of medians over paired seeds; deliberately not fixed — compute the paired version yourself if you read per-fold signs.
metadata:
  type: project
---

experiments/lib/aggregate.py:46 computes delta_<cohort>_median as median_over_seeds(candidate_ll) - median_over_seeds(R0_ll) — a DIFFERENCE OF MEDIANS. The seeds are paired (same data, same folds; only random_state differs), so the MEDIAN OF PAIRED DIFFERENCES is the better estimator. Measured on batch1/tgp_cycle_displacement: the two disagree by up to 0.0127 nats per fold and FLIP SIGN on fold 2 (paired +0.0036, published -0.0090), enough to change a per-fold sign-agreement count. Deliberately NOT fixed (fps-2kc closed 2026-08-31): fps-e6i showed the per-fold screen delta barely predicts the arbiter at all ([[screen-and-arbiter-share-no-population]]) (r = -0.096 over all stations), so correcting how a near-uninformative number is aggregated buys little, and changing it would move published per-fold columns for every candidate already graded. If you are about to READ per-fold screen delta signs, or if fps-nas makes the screen matter again, compute the paired version yourself — see experiments/2026-08-31_tau_distance_of_logloss_gain/README.md 'Incidental finding' and run.py's per_fold_delta_ll(paired=...) for both conventions side by side.
