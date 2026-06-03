# Experiments lab book

Reverse-chronological index of experiment dirs under `experiments/`. One row per dir. Keep entries terse — full write-up lives in `<dir>/README.md`.

**Status legend:** `open` (running / undecided) · `done` (concluded, no graduation) · `graduated` (code landed in `fuel_signal/`) · `abandoned` (dead end).

**Adding an entry:** copy `experiments/TEMPLATE.md` into `experiments/<YYYY-MM-DD>_<slug>/README.md`, then add a row here.

| Date | Name | Hypothesis | Result | Status |
|---|---|---|---|---|
| 2026-06-03 | 2026-06-03_drop_redundant_pair | Drop `station_price_cents` vs `station_minus_last_max_cents` — pair is SHAP-redundant, so removing either should be neutral | Step 1 (5 seeds): drop_price out (sign flips); drop_minus_max real (paired Δ −0.011 ± 0.004). Step 2 (14-fold CV): **abandoned** — 7/14 wins, mean Δ +0.010, two folds regress by >+0.05 (fold 4 +0.066, fold 9 +0.103). Regime-sensitive, not safe to drop. | done |
| 2026-06-03 | redundancy_phase4b_tight | Tighter SHAP redundancy threshold drops fewer real features | — | open |
| 2026-06-03 | redundancy_phase4b_p2 | Second pass on SHAP-based redundancy at relaxed threshold | — | open |
| 2026-06-03 | redundancy_phase4b | Initial SHAP-based feature redundancy sweep | — | open |

<!-- Older dirs (pre-lab-book): brand_leadlag_regime, cv_compare_15feat, cv_compare_phase4, cv_compare_phase4b, l1_per_row_lga, lga_lead_lag, seed_bank_phase4, shap, shap_15feat, shap_phase4, trough_weakness. Backfill rows when revisited. -->
