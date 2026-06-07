# Step 4 — fold-level (elongation × gradient) 2D diagnostic

**Date:** 2026-06-07. Follow-up to the negative-result elongation investigation summarised in the "Elongation investigation" section of `step2_analysis.md`.

**Question (user, 2026-06-07):** does an elongation × descent-gradient axis separate the folds where R5_A_only regresses ({7, 9, 13}) from those where it helps? Goal: either (a) a constraint that suppresses A/C on extended-descent rows, or (b) a broad-population signal where {7, 9, 13} are just the loudest exemplars of a phenomenon spread across folds.

## Method

Fold-level only (no re-fit). Per fold val window:

- **Elongation score:** median over val rows of `cycle_days_since_peak / frozen_baseline`. Frozen baseline = per-station median `cycle_mean_length` over the 730d window ending at `val_start − 1`. Computed in this script, NOT a re-use of the prior session's exact method (which wasn't preserved on disk). My r-values disagree with the handover-quoted values (R1 −0.37 vs −0.21; R6 −0.11 vs −0.37; R7 −0.38 vs +0.16), so this is a *similar but not identical* elongation proxy — directional reads are robust, exact correlations are not.
- **Gradient score:** median 14d backward slope of `station_price_cents`, restricted to descending rows. More negative = steeper. All folds cluster between −0.71 and −1.39 cents/day — a narrow band.
- **Per-fold delta:** median across 5 seeds (per `feedback_check_seed_variance_before_trusting_mean`), R_x − R0 on `ll_hard25`.

## Per-fold table (sorted by R5 delta_hard25 descending = most-regressed first)

| fold | elong | grad   | R1     | R5 (A) | R6 (B) | R7 (C) |
|------|-------|--------|--------|--------|--------|--------|
| 7    | 0.51  | −0.86  | +0.084 | **+0.090** | −0.007 | −0.011 |
| 9    | 0.53  | −1.07  | +0.117 | +0.014 | +0.073 | +0.031 |
| 13   | 0.48  | −0.80  | +0.046 | +0.013 | +0.024 | +0.003 |
| 2    | 0.61  | −0.89  | −0.060 | +0.013 | +0.007 | −0.062 |
| 14   | 0.57  | −1.11  | −0.025 | +0.011 | −0.029 | −0.003 |
| 12   | 0.36  | −1.39  | +0.011 | +0.009 | −0.005 | −0.047 |
| 6    | 0.59  | −0.79  | −0.008 | −0.008 | −0.005 | −0.016 |
| 10   | 0.34  | −0.93  | −0.008 | −0.008 | +0.019 | −0.009 |
| 1    | 0.69  | **−0.71** | −0.184 | −0.028 | −0.058 | −0.119 |
| 11   | 0.58  | −1.07  | −0.054 | −0.032 | +0.005 | +0.024 |
| 4    | 0.51  | −1.07  | −0.017 | −0.057 | +0.102 | +0.093 |
| 8    | 0.57  | −0.93  | −0.143 | −0.080 | +0.059 | −0.082 |
| 5    | 0.73  | −1.07  | −0.121 | −0.142 | −0.008 | −0.048 |
| 3    | 0.47  | −1.07  | −0.237 | **−0.213** | −0.113 | +0.044 |

## Regime-mix filter (user insight 2026-06-07)

The original 2D table averages per-fold over all val rows including ascending-arm rows, but the regression hypothesis is specifically about descent regimes. User clarified that **fold 1 is in a generally price-increasing regime**, so its "A helps" verdict is averaged over mostly-ascending rows and the fold-level read on fold 1 isn't a clean counterexample to "extended descent + shallow descent → A regresses".

Added two diagnostics:

- **`descent_frac`** = fraction of val rows with `cycle_pct_through < 0.5` (peak-anchored; pct < 0.5 = descending toward trough per `project_cycle_pct_through_semantics`).
- **`cycle_descent_slope`** = median over descending val rows of `(station_price_cents − cycle_last_max_cents) / cycle_days_since_peak`. Cycle-anchored, replaces the 14d backward proxy as the y-axis.

| fold | descent_frac | cycle_descent_slope | R5 (A) | regime |
|------|-------------|---------------------|--------|--------|
| 1    | 0.42        | −0.74               | −0.028 | ascent-dominated |
| 2    | 0.44        | −1.53               | +0.013 | ascent-dominated |
| 3    | 0.54        | −1.10               | −0.213 | descent-dominated |
| 4    | 0.50        | −4.09               | −0.057 | descent (right at threshold) |
| 5    | 0.36        | −1.77               | −0.142 | ascent-dominated |
| 6    | 0.46        | −0.82               | −0.008 | ascent-dominated |
| 7    | 0.50        | −1.20               | **+0.090** | descent (right at threshold) |
| 8    | 0.45        | −0.58               | −0.080 | ascent-dominated |
| 9    | 0.49        | −1.74               | +0.014 | **borderline** (user says descent) |
| 10   | 0.62        | −1.13               | −0.008 | descent-dominated |
| 11   | 0.44        | −1.76               | −0.032 | ascent-dominated |
| 12   | 0.68        | −2.60               | +0.009 | descent-dominated |
| 13   | 0.52        | −1.53               | +0.013 | descent-dominated |
| 14   | 0.47        | −1.50               | +0.011 | ascent-dominated |

Confirms user's domain read for fold 1: dfrac=0.42, most ascent-dominated of any fold flagged earlier as a regressor-candidate. (User did NOT extend the regime claim to folds 7/9/13 — only fold 1 was specified. The descent-frac diagnostic is my own classification for those folds, not the user's.)

Notable for context but not a verified user claim: fold 9's dfrac=0.49 sits right at the threshold; fold 9's structure has "two back-to-back elongated cycles" per `project_late_descent_elongation_regime`, which may break the peak-anchored `cycle_pct_through` measure.

## What the descent-dominated subset shows (n=6)

Pearson r on descent-dominated subset only:

| Pair | r | n |
|---|---|---|
| (elongation, R5 delta_hard25) | −0.06 | 6 |
| (cycle_descent_slope, R5 delta_hard25) | +0.00 | 6 |
| (elongation, R6 delta_hard25) | +0.11 | 6 |
| (cycle_descent_slope, R6 delta_hard25) | −0.70 | 6 |
| (elongation, R7 delta_hard25) | +0.61 | 6 |
| (cycle_descent_slope, R7 delta_hard25) | −0.50 | 6 |

R5 correlations collapse to zero — the elongation+gradient story doesn't survive even on the regime-filtered subset. f3 (descent + mid elong + normal grad) is where A helps most (−0.213) and f7 (descent + mid elong + shallow grad) is where A regresses most (+0.090), at near-identical positions in the plane. Fold-level resolution can't separate them.

R6 and R7 show stronger correlations on the descent subset (R6 r=−0.70 with gradient; R7 r=+0.61 with elongation) but n=6 — not robust enough to act on.

## Resolution ceiling

Fold-level averaging is the wrong resolution for the user's question. Both the "constraint" and "broad-population" framings need per-row data: within each fold, *which rows* does A regress on, and do those rows share an extended-descent character that the existing 50-feat baseline can't capture?

`step2_paired_wfcv.py` only persisted (fold, run, seed) aggregates. Per-row predicted probabilities for R0 / R5 are needed to drop to row resolution. A reduced re-run (R0 + R5 only, 140 fits, ~7 min wall) would generate them.

## Verdict

The fold-level 2D diagnostic, even after the regime-mix filter, is **inconclusive**:

- The user's fold-1 instinct lands — fold 1 is genuinely ascent-dominated; removing it from the comparison is correct.
- The remaining descent-dominated subset (n=6) is too small for meaningful fold-level statistics.
- Fold 3 (descent + mid elong + normal grad) helps most; fold 7 (descent + mid elong + shallow grad) regresses most. Near-identical fold-level coordinates, opposite outcomes — fold-level aggregation cannot separate them.

This is not strong enough on its own to design a constraint feature. Next step is row-level resolution.

## Open questions and options

1. **Row-level diagnostic (recommended next step)**: re-run the WFCV harness saving per-row predicted probabilities for R0 and R5 (140 fits, ~7 min wall). Then per-row: `delta_logloss = ll_R5 − ll_R0`; stratify by per-row `is_extended_descent` (and by descent-arm membership). If A's positive delta concentrates on extended-descent rows across many folds, the "broad-population signal" hypothesis lands and constraint design is justified. If only fold 7/13's extended-descent rows light up, it's a localised pattern and the constraint is fold-specific (low value).
2. **Constraint feature design (downstream of 1)**: once row-level confirms (or denies) the regime, build `is_extended_descent` as a feature and test A × constraint vs A alone.
3. **Close the thread.** Documented fallback per `project_late_descent_investigation` — move to #211 (external data).

## Artefacts

- `step4_elongation_gradient_2d.py` — script (4.4s wall).
- `step4_fold_scores.csv` — per-fold scores + deltas.
- `step4_2d_panels.png` — 2x2 (one per run).
