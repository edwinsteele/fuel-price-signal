# cycle_pct_through interaction analysis + phase-residual engineering

- **Date:** 2026-06-04
- **Branch:** main
- **SHA:** 3da7f38
- **Status:** done

## Hypothesis

User observation: in the Phase 4 partner-score table, `cycle_pct_through` showed ~10% interaction strength against several cents-denominated features (`station_price_cents`, `station_minus_last_min/max_cents`, `station_minus_sydney_avg_cents`). These features are structurally related — all anchored to the Sydney-average cycle series — but SHAP main-effect redundancy didn't fire. Was the model phase-gating its reading of those cents features? And if so, could an engineered "phase residual" feature absorb the implicit interaction and improve val logloss?

## Setup

Phase 4 LightGBM (50 features = FEATURE_COLUMNS + LGA_FEATURE_COLUMNS), default hyperparameters, seed=42 unless multi-seed noted.

- **Step 1** (`run.py`): re-rank `cycle_pct_through` partners from existing `experiments/shap_phase4/partner_scores.csv`, then compute `TreeExplainer.shap_interaction_values` on 10k val rows × 50 features. Plot signed interaction surfaces for 7 targets including non-cycle controls (stickiness_score, cycle_peak_count). Saddle-score test (4-quadrant signed mean) per pair.
- **Step 2** (`step2_engineered_feature.py`): engineer `station_minus_expected_phase_price = station_price − (last_min + pct_through × (last_max − last_min))`. Additive: 3 seeds × {baseline 50 / engineered 51}, single val window.
- **Step 3** (`step3_ablation_A.py`): replacement (ablation A): 3 seeds × {baseline / additive / engineered − siblings (49 feat)}, single val window. Compares whether engineered absorbs `station_minus_last_min_cents` and `station_minus_last_max_cents`.
- **Step 4** (`step4_paired_wfcv.py`): paired walk-forward CV per `feedback_regime_segmented_evaluation` + CONVENTIONS.md gate. 14 folds × 3 configs × seed=42.

## Results

### Step 1 — partner ranking and true SHAP interaction

- Partner-score plateau at ~10.2–10.3% across `station_minus_sydney_avg_cents`, `station_price_cents`, `station_minus_last_max_cents`, `station_minus_last_min_cents` — the Sydney-anchored cents family. LGA/brand-anchored at ~9%, raw means ~5%, stickiness ~4%, LGA trough features ~1%, slow-changing cycle scalars ~0.9%.
- True `shap_interaction_values` mean|iv| **re-orders** the list:

  | partner | partner score | mean\|iv\| | saddle |
  |---|---|---|---|
  | station_minus_last_min_cents | 10.24% | 0.0185 | 0.0448 (clean saddle) |
  | station_minus_last_max_cents | 10.24% | 0.0166 | 0.0018 (monotone amp, no flip) |
  | station_price_cents | 10.28% | 0.0143 | 0.0361 (clean saddle) |
  | stickiness_score | 4.25% | 0.0091 | 0.0322 (opposite-orientation saddle) |
  | station_minus_lga_mean_cents | 8.90% | 0.0079 | 0.0230 |
  | **station_minus_sydney_avg_cents** | **10.30%** | **0.0069** | 0.0154 |
  | cycle_peak_count | 0.89% | 0.0027 | 0.0003 |

- **Key finding:** the partner-score heuristic conflates substitution and true interaction. `station_minus_sydney_avg_cents` ranks #1 by heuristic but dead-last by true interaction → pure substitute, redundancy candidate. `station_minus_last_min_cents` shows the cleanest saddle → phase-gated reading, decomposition candidate. Stickiness × pct_through has its own saddle (opposite orientation) — unexpected, worth its own thread. Captured in memory as `project-partner-score-substitution`.

### Step 2 — additive engineered feature, single val window

3-seed deltas vs baseline: [+0.0052, −0.0166, −0.0083]. Mean Δ = −0.0066, seed std = 0.0110, |m|/std = 0.60 — well below the 3× threshold per `feedback-seed-discipline`. SHAP shows the engineered feature lands at #11 (mean|SHAP| = 0.101); siblings shrink modestly (−0.10 and −0.07) but remain #1 and #2 features. Inconclusive.

### Step 3 — replacement (ablation A), single val window

3-seed deltas vs baseline: [+0.0044, −0.0180, −0.0154]. Mean Δ = **−0.0097**, |m|/std = 0.79. AblationA beat additive on every seed. Engineered feature jumped to **mean|SHAP| = 1.559** (15× from additive) — inheriting the siblings' lead role. Looked like genuine "concentrate signal into one feature" win. **(This finding turned out to be a regime artefact — see step 4.)**

### Step 4 — paired walk-forward CV (14 folds, seed=42)

| comparison | mean Δ | **median Δ** | std | helps | hurts | worst-fold |
|---|---|---|---|---|---|---|
| additive − baseline | −0.0028 | −0.0003 | 0.012 | 7/14 | 7/14 | +0.014 |
| **ablationA − baseline** | **+0.0175** | **+0.0098** | 0.045 | **4/14** | **10/14** | **+0.101** |
| ablationA − additive | +0.0204 | +0.0078 | 0.048 | 3/14 | 11/14 | +0.114 |

Concentrated regressions for ablationA in shocked-regime folds:

| fold | window | Δ |
|---|---|---|
| 1 | 2021-11 → 2022-02 (late-2021 surge) | +0.101 |
| 9 | 2023-10 → 2024-01 (Israel-Gaza onset) | +0.097 |
| 13 | 2024-10 → 2025-01 | +0.044 |
| 4 | 2022-08 → 2022-10 (Ukraine continuation) | +0.043 |

Diagnosis: the diagonal projection (interp between trough and peak by phase) drops information that the siblings preserve via per-anchor distances. During normal-cycle windows (including step 2/3's val window Mar–Jun 2025) the diagonal is sufficient and the engineered feature looks competitive; during shock windows the projection assumption breaks. Captured in memory as `project-phase-residual-fragile`.

## Conclusion

**abandoned** — neither additive nor ablationA configurations graduate. The engineered feature is internally consistent and the saddle hypothesis is real, but the diagonal-projection encoding is fragile across regimes that the siblings handle gracefully. Trees were already learning the interaction from the siblings; engineering it explicitly adds nothing and replacing the siblings actively regresses.

Methodology wins to keep:

- The "partner-score conflates substitution and complement" finding (memory: `project-partner-score-substitution`).
- Confirmation that single-window deltas under the 3× seed-std bar can flip sign in paired CV — same pattern as 2026-06-03_drop_redundant_pair fold 9.
- Calibration data for fit / SHAP / CV wall times on this machine (memory: `feedback-instrument-walltime`).

## Followups

- **`station_minus_sydney_avg_cents` redundancy drop test** — the step-1 finding (highest partner score, lowest true SHAP interaction) makes this a cleaner candidate than the engineered feature ever was. Same protocol as step 4. ~2 min compute.
- **Non-parametric phase model retest** — `next_session_prompt.md` in this dir. Tests whether the linear-interpolation crudeness was the issue (likely no, per step-4 diagnosis) using a learn-from-train `E[price | phase]` lookup. Sized as a teaching exercise.
- **Stickiness × cycle_pct_through opposite-orientation saddle** — surfaced unexpectedly in step 1. Not pursued. Worth a separate thread.
