---
name: tau-is-calibrated-not-raw
description: tau=0.25 is calibrated; rowpreds proba is raw. The equivalent raw threshold is ~0.11-0.16.
metadata:
  type: project
---

tau=0.25 is a CALIBRATED threshold; rowpreds.parquet 'proba' is RAW LightGBM output (experiments.lib.fit.fit_score never calibrates). (See [[decision-flip-substrate-is-fills-not-rowpreds]] for the related trap of reading decisions off rowpreds at all.) They are not the same scale — the equivalent RAW threshold measured across batch1's 14 folds is 0.1136-0.1590, roughly half of 0.25. Comparing |p_raw - 0.25| is a category error worth ~0.10 of probability, and it lands in the most gain-dense bin of the distance profile, so it manufactures false confirmations. To convert: experiments/batches/<batch>/r0_cache.joblib holds the arbiter's per-fold fitted isotonic calibrator (per_fold[f]['cal_pipe'].calibrator) and own_tau. Isotonic is monotone, so p_cal >= tau <=> p_raw >= tau_raw and the whole analysis can stay in raw space once tau_raw is located by inverting the map. The screen's R0/seed-42 raw probabilities are byte-for-byte the arbiter's cached baseline scores (verified 2.98e-08 = float32 eps, fps-e6i), so this conversion is exact, not an approximation. Worked example: experiments/2026-08-31_tau_distance_of_logloss_gain/run.py (tau_raw, iso_levels, one_step_band).
