---
name: tau-selector-is-cadence-blind
description: score_phase2's tau selector never sees the tank, so an unmoved tau is not evidence a cadence re-lock worked.
metadata:
  type: project
---

score_phase2's tau selector is CADENCE-BLIND — it prints "Chosen tau = 0.25" at any cadence, so a tau that did not move after a cadence re-lock is NOT evidence the cadence change worked.

threshold_sweep() (fuel_signal/score_phase2.py:117) maximises (tp*6.37 - fp*5.80 - fn*11.14)/n over OOF PREDICTION ROWS. It takes y_true/y_pred and three cost constants — no TankParams, no tank state, no fill schedule — and tau is picked at step 3, BEFORE the tank is constructed for the backtest (score_phase2.py:495). Changing evaluation_interval_days cannot move it.

This is a known, MEASURED gap, not an oversight. fps-929 swept tau against realised CPL under the tank: best-in-hindsight is 0.35 at 1d, worth 0.062 c/L off 422 changed decisions whose effects cancel (at 7d the same comparison moves 8 decisions). Many flips + tiny delta = a real measurement of no effect, so tau stays 0.25 by decision.

Why they diverge more at 1d: at 7d the tank FORCES 67.6% of fills so the model's opinion is overridden two-thirds of the time and tau barely matters; at 1d only 21.4% are forced.

Harmless BY MEASUREMENT, not by construction. If the tank changes again (size, consumption, floor), nothing re-checks this automatically — the selector still will not be looking. Documented in docs/STATUS.md's cadence note.
