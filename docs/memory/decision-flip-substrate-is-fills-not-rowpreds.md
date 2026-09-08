---
name: decision-flip-substrate-is-fills-not-rowpreds
description: rowpreds proba comes from a different model than the realised arbiter; diff fills.parquet for decision flips.
metadata:
  type: project
---

rowpreds.parquet's proba is the wrong substrate for 'did the realised backtest's decision differ' — it comes from the WFCV screen's own multi-seed, uncalibrated fit_score (experiments/pipeline/runner.py's _run_wfcv_screen), a DIFFERENT model from the realised backtest's single-seed, OOF-calibrated, per-fold-tau-selected one (experiments/lib/realised.py's _train_calibrate_select_tau). Thresholding rowpreds' raw proba at a realised arm's own_tau compares a decision from one model against a tau chosen for another. fills.parquet is the tank simulator's own executed record and is the correct ground truth for 'did the two arms buy on the same days' — diff it directly (experiments/lib/flips.py, fps-gez) rather than re-deriving decisions from rowpreds.
