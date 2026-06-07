# Step 2 analysis — late-descent triplet

**Date:** 2026-06-06. Sign convention in this doc:

- Attribution tables (`standalone_improvement`, `marginal_improvement`): **positive = log-loss reduction (family helps)**, matching `step2_meta.json`'s convention.
- Raw `delta_*` columns in the per-fold CSV: **negative = improvement** (these are `Rx − R0`).

## ⚠ Correction note — mean vs median aggregation

The experiment script aggregated across 5 seeds with **mean**. One cell out of 560 (fold 2, R1_ABC, seed 44) produced a LightGBM fit failure (`ll_all = 4.73` vs ~0.24 for the other 4 seeds on that cell — almost certainly an unlucky bagging/feature-subsample RNG draw; feature values on fold 2's val window were checked and clean). That single cell dragged the mean enough to flip the headline.

**This document reports median-across-seeds aggregation as the headline.** Mean numbers are shown alongside where the gap matters. The verdict is based on the median picture. See [[feedback-check-seed-variance-before-trusting-mean]] for the methodological lesson.

## Headline

The triplet (R1_ABC) delivers a modest but real lift, concentrated on the hard cohorts.

| Cohort | mean-agg delta | median-agg delta |
|---|---|---|
| `all`     | +0.0586 (worse, driven by s44) | **−0.0060** (neutral / slight help) |
| `hard25`  | −0.0168 (helps) | **−0.0427** (helps, ~2.5× larger) |
| `hard10`  | −0.0891 (helps) | **−0.0975** (helps) |
| `lated`   | +0.0719 (worse, driven by s44) | **−0.0161** (helps) |

- On the median-aggregated picture R1 helps on every reported cohort.
- Per-fold deltas under median agg: 12 of 14 folds help on hard25; fold 9 (shock) regresses by +0.12 and fold 7 by +0.05. The "no fold blows up more than the avg lift gives back" rule fires on fold 9, but the regression is shock-fold (per `project_shock_gate_scoping`, normal-fold median is the primary read, not shock-fold worst-case).

## Per-family verdicts (median-aggregated)

| Family | Standalone (hard25) | Marginal (hard25) | Standalone (hard10) | Marginal (hard10) | Standalone (lated) | Marginal (lated) |
|---|---|---|---|---|---|---|
| **A** px_std (+Δ)     | +0.030 | **+0.049** | +0.068 | +0.069 | +0.017 | **+0.033** |
| **B** disc_gap (+Δ)   | -0.005 | -0.002 | +0.008 | -0.012 | -0.010 | -0.014 |
| **C** lga_phase_std (+Δ) | +0.014 | +0.010 | +0.036 | +0.019 | +0.007 | +0.018 |

### A (cross-station price dispersion) — KEEP, run within-family ablation (#212)

- Standalone hard25 **+0.030** (the clean number).
- Standalone hard10 **+0.068** — A's lift roughly doubles on the hardest decile. That's the right shape for a feature genuinely picking off hard rows.
- Standalone lated **+0.017** — modest help on the targeted late-descent cohort. Marginal-given-C on lated is **+0.018** — additive again, no synergy.
- Shock-fold cut (folds 1/4/9/13, R5_A_only hard25): -0.009 / -0.057 / +0.022 / +0.004 (mean -0.010). Normal-fold mean ~-0.034. A is a normal-regime signal; fold-9 shock regression is small.
- **Note on the +0.049 marginal:** R1's A-marginal was inflated by B's drag (see "B's drag inflated A's apparent marginal" section below). The +0.030 standalone is the trustworthy headline for what A buys.
- **Elongation read:** A's per-fold delta_hard25 correlates r = −0.25 with elongation exposure — slight tendency to help more, not less, on elongated folds. A is not hiding a positive-on-normal + negative-on-elongation pattern at the per-fold-elongation-score level. (See "Elongation investigation" section.) #212 carries this through as an informational diagnostic, not a graduation gate.
- **Verdict:** keep. Within-family ablation (level / Δ / both) in #212 to decide which subset graduates.

### B (network discount-cohort gap) — DROP for parsimony

- Standalone hard25 flat (-0.005); marginal hard25 effectively neutral (-0.002) on the **fold-averaged** read.
- The 3-fold stratification on {7, 9, 13} initially looked like B was actively harmful on the elongation regime (B standalone hard25 −0.024 on those folds vs +0.005 elsewhere; fold 9 worst single cell after s44). But the continuous per-fold elongation analysis (see "Elongation investigation" section below) contradicts this — across 14 folds B's deltas correlate **negatively** with elongation (r = −0.37 hard25, −0.39 hard10), i.e. B trends to help more on elongated folds, opposite of the 3-fold story. The {7,9,13} grouping doesn't survive continuous scoring.
- Highest mean |SHAP| in R1 (0.10–0.61); the model splits heavily on `network_disc_gap` but the splits don't generalise — neither the helpful direction nor the harmful direction is stable.
- **Mechanism implication:** B's design hypothesis was "discount-cohort gap widens when the floor slides in extended descent → flag elongation." Data is mixed; no clean signal either way.
- **Verdict:** drop for parsimony, not for harm. No positive marginal contribution under any aggregation.

### B's drag inflated A's apparent marginal — re-attribution under B-free baseline

The Step 2 attribution table compared R1 (A+B+C) vs R2_drop_A (B+C). Both include B. With B's elongation-regime drag, the marginal numbers are contaminated. The cleaner read uses R3 (A+C, B dropped):

| Comparison | hard25 lift vs R0 (median) |
|---|---|
| R5 (A alone) | +0.030 |
| R7 (C alone) | +0.014 |
| R3 (A + C, B dropped) | +0.045 |

From this:
- **A's marginal-given-C** (clean): +0.030 — equal to A's standalone.
- **C's marginal-given-A** (clean): +0.015 — equal to C's standalone.
- **A↔C interaction**: 0.045 − (0.030 + 0.014) = **+0.001** — additive, no real synergy.

The "+0.019 A interaction synergy" originally attributed to A+C was actually **A partially compensating for B's drag**, not genuine A↔C complementarity. A's true marginal contribution is +0.030, not the inflated +0.049 reported in R1's marginal column.

This means the expected lift from any subset of {A, C} added to the 50-feat baseline is approximately *additive*: A buys ~+0.030 hard25, C buys ~+0.014 hard25, A+C buys ~+0.045 hard25. No combinatorial bonus to chase.

### C (LGA phase-std divergence) — INCLUDE IN #212, don't drop pre-emptively

- Standalone hard25 +0.014, marginal-given-A (clean) **+0.015** — modestly positive, additive with A.
- Standalone hard10 **+0.036** but marginal collapses to +0.019 — on the hardest decile A absorbs much of C's signal.
- Standalone lated +0.007, marginal-given-A **+0.018** — on the targeted slice C still adds value on top of A.
- SHAP-corr with `network_px_std_delta_3d` ≥ 0.5 in 6/14 folds — real SHAP-level redundancy with A's Δ, but predictive contribution remains positive across cohorts.
- **Reversed from initial verdict.** Initial said "drop, harmful in combo" — that was s44-poisoned. C is modestly useful and additive with A (no synergy, but no anti-synergy either).
- **Elongation read:** C's per-fold delta_hard25 correlates r = +0.16 with elongation exposure — slight tendency to help LESS on elongated folds. Weakest of the three families on the targeted regime, but the absolute lift on average is still positive.
- **Verdict:** include in the #212 ablation. The clean question is "does the cost of one extra column buy +0.014 hard25 / +0.018 lated marginal-given-A" — and whether that holds up under the elongation cut.

## SHAP read (unchanged)

Mean |SHAP| ranking across folds (R1_ABC seed 42):

1. `network_disc_gap` — 0.10–0.61 (model splits heavily, doesn't generalise → drop family B regardless)
2. `network_px_std_delta_3d` — 0.09–0.18
3. `network_px_std` — 0.06–0.18
4. `lga_phase_std` — 0.03–0.10
5. `network_disc_gap_delta_3d` — 0.03–0.10
6. `lga_phase_std_delta_3d` — 0.02–0.07

SHAP-corr ≥ 0.5 candidates (flag per `project_shap_redundancy_regime_caveat`, don't auto-drop):

- `network_px_std_delta_3d` ↔ `lga_phase_std`: 6 folds (0.54, 0.65, 0.60, 0.65, 0.55, 0.52). Most material.
- `network_px_std_delta_3d` ↔ `lga_phase_std_delta_3d`: 3 folds.
- Within-A `network_px_std` ↔ `network_px_std_delta_3d`: 2 folds.

These motivate the within-family ablation in #212.

## Regime split (median-aggregated)

- **Shock folds (1, 4, 9, 13)** R1 hard25 delta: -0.154 / -0.026 / +0.123 / +0.027 → mean -0.007. Same as mean-agg picture (the s44 outlier was on a normal fold).
- **Normal folds** R1 hard25 delta (median agg): now consistently negative (helping) on every fold except f2 (small, -0.060) and f7 (+0.062). Mean ~-0.058. Under mean agg this was +0.000 because of s44.

The triplet helps materially on normal late-descent rows under correct aggregation. Shock behaviour is unchanged.

## Elongation investigation (negative result)

User raised a side hypothesis: R1 regresses on folds 7, 9, 13 because they share an **elongated descent cycle** character. The targeted question is whether this generalises beyond the three flagged folds, or whether they're isolated anomalies.

Three operationalisations were tried; only the third is methodologically clean:

1. `days_since_peak > 1.3 × cycle_mean_length` — fails because `cycle_mean_length` is adaptive and recalibrates during sustained elongation.
2. Per-fold descent gradient — wrong concept. Gradient (slope) is independent of elongation (calendar duration) by construction.
3. Per-fold elongation exposure using cycle-level identification (≥7d real cycles), frozen 730-day trailing baseline, weighted by per-day cycle ratio.

Method (3) result — Pearson r between per-fold elongation exposure and per-fold delta (negative r = more elongation → more help):

| Run | hard25 r | hard10 r | lated r |
|---|---|---|---|
| R1_ABC | −0.21 | −0.20 | −0.16 |
| A only | −0.25 | −0.19 | −0.29 |
| B only | **−0.37** | **−0.39** | −0.26 |
| C only | +0.16 | +0.05 | +0.02 |

**Hypothesis did not generalise.** R1's correlation is negative — triplet trends to help *more* on elongated folds, opposite of the 3-fold story. B has the strongest negative r, contradicting the {7,9,13} stratification that said B was actively harmful on elongation.

Key fold-level surprises:

- Fold 3 (elongation 1.35; user-flagged as "elongated and a little shallow") is where R1 helps most: delta_hard25 = −0.24. A single huge counterexample driving much of the negative correlation.
- Fold 9 — the strongest regressor (delta_hard25 = +0.118) — has only mid-low elongation (1.11). Not explained by this metric.
- Folds 3 and 6 are MORE elongated than 7 and 13 but R1 helps on them. The per-fold relationship isn't monotone in elongation.

**Remaining open hypothesis (untested):** elongation × descent-gradient interaction. Gradient is independent of elongation by construction. Plausibly: extreme elongation + shallow gradient = bad (fold 7, user described as "very shallow"); extreme elongation + normal gradient = fine (fold 3, "a little shallow"). A 2D surface (elongation, gradient) → R1 delta might separate them; single-axis elongation doesn't.

**Operational consequence:** #212 carries the per-fold elongation diagnostic as an informational read, not a graduation gate. The user's {7, 9, 13} folds may not be a single regime — they could be regressing for different reasons.

Plot: `elongation_real_vs_per_fold_delta.png`.

## Recommendation for Step 3

1. **Graduate A** (subject to within-family ablation in #212: level only, Δ only, both).
2. **Include C in the #212 ablation** — test A-only / A+C / C-only against baseline. SHAP redundancy is real but predictive redundancy is not established under correct aggregation.
3. **Drop B.** No positive marginal contribution under any aggregation; not actively harmful, just not pulling weight.
4. **Late-descent track is NOT exhausted.** The corrected verdict reopens A (and possibly C) as viable intra-series signals. External-data move per `project_late_descent_investigation` is contingent on #212's outcome, not a foregone conclusion.

## Compute

Total wall: ~1838s (~30.6 min) for 8 runs × 14 folds × 5 seeds ≈ 560 LightGBM fits. The 5-seed budget was correct per `feedback_seed_discipline`; the methodological correction is to use median-across-seeds for the headline aggregation (or scan seed_std before trusting the mean) per [[feedback-check-seed-variance-before-trusting-mean]].
