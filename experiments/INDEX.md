# Experiments lab book

Reverse-chronological index of experiment dirs under `experiments/`. One row per dir. Keep entries terse — full write-up lives in `<dir>/README.md`.

**Status legend:** `open` (running / undecided) · `done` (concluded, no graduation) · `graduated` (code landed in `fuel_signal/`) · `abandoned` (dead end).

**Adding an entry:** copy `experiments/TEMPLATE.md` into `experiments/<YYYY-MM-DD>_<slug>/README.md`, then add a row here.

| Date | Name | Hypothesis | Result | Status |
|---|---|---|---|---|
| 2026-06-05 | 2026-06-05_phase_lookup_nonparametric | Replace step-4's linear-interp `expected = last_min + pct × (last_max − last_min)` with a non-parametric per-fold `E[norm_price \| phase]` lookup; if fold 1 still regresses the diagonal-projection diagnosis is confirmed, if it flips the shape misspecification was the real issue | **PARKED mid-design 2026-06-05.** Major mid-session finding: `cycle_pct_through` is peak-anchored (`days_since_last_peak / mean_cycle_length`), so step-4's formula was shape-inverted, not merely crude. See README for full state, decisions made (tail extends to 2.0, shock-fold taxonomy pre-committed), and pending sub-decisions (bin method/count, normalisation choice). | open |
| 2026-06-04 | 2026-06-04_cycle_pct_through_interaction | `cycle_pct_through` × cents features show ~10% partner-score plateau and clean SHAP saddles; engineering `station_price − interp(trough,peak,pct_through)` should absorb the implicit interaction and let us drop the siblings | Step 1: partner-score conflates substitution + true interaction (`station_minus_sydney_avg_cents` is the redundancy candidate, `station_minus_last_min_cents` the genuine saddle). Steps 2–3 single val window: additive neutral, ablationA looked good (Δ −0.010, mean\|SHAP\| jumped 15× → 1.56). Step 4 (14-fold paired CV): **abandoned** — ablationA regresses 10/14 folds, mean Δ +0.018, worst +0.101 in shocked-regime windows (fold 1 late-2021, fold 9 Israel-Gaza). Diagonal projection drops absolute-anchor info siblings preserve. Followup test queued: `next_session_prompt.md` (non-parametric phase model). | done |
| 2026-06-03 | 2026-06-03_drop_redundant_pair | Drop `station_price_cents` vs `station_minus_last_max_cents` — pair is SHAP-redundant, so removing either should be neutral | Step 1 (5 seeds): drop_price out (sign flips); drop_minus_max real (paired Δ −0.011 ± 0.004). Step 2 (14-fold CV): **abandoned** — 7/14 wins, mean Δ +0.010, two folds regress by >+0.05 (fold 4 +0.066, fold 9 +0.103). Regime-sensitive, not safe to drop. | done |
| 2026-06-03 | redundancy_phase4b_tight | Tighter SHAP redundancy threshold drops fewer real features | — | open |
| 2026-06-03 | redundancy_phase4b_p2 | Second pass on SHAP-based redundancy at relaxed threshold | — | open |
| 2026-06-03 | redundancy_phase4b | Initial SHAP-based feature redundancy sweep | — | open |

<!-- Older dirs (pre-lab-book): brand_leadlag_regime, cv_compare_15feat, cv_compare_phase4, cv_compare_phase4b, l1_per_row_lga, lga_lead_lag, seed_bank_phase4, shap, shap_15feat, shap_phase4, trough_weakness. Backfill rows when revisited. -->
