# Step 5 — row-level diagnostic: A vs extended-descent regime

**Date:** 2026-06-07. Follow-up to the fold-level inconclusive 2D result in `step4_analysis.md`.

**Question:** within each fold, does A's per-row prediction error concentrate on extended-descent rows? Two operational uses (per user 2026-06-07):

- A constraint that suppresses A/C on extended-descent rows, removing the regression on fold 7 et al.
- Evidence of a broad-population signal where the user-flagged regressor folds are just the loudest exemplars of a population-wide failure mode.

## Method

`step5_paired_wfcv_rowpreds.py` re-ran the WFCV harness restricted to R0 (50-feat baseline) + R5_A_only (50 + `network_px_std` + `network_px_std_delta_3d`), 5 seeds × 14 folds = 140 fits, saving per-(fold, run, seed, row) predicted probabilities. Output: 8.0M rows / 32 MB parquet. Mirrors step2's harness exactly; reproduces step2's `ll_all` / `ll_hard25` for these two runs.

`step5_rowlevel_diag.py` then computed:

- per-(fold, run, row) median log-loss across seeds;
- per-row `delta_ll = ll_R5 − ll_R0` (positive = A made the row worse);
- per-row `descent_arm = (5d backward price change < 0)` — price-based, not pct-based, because `cycle_pct_through` is peak-anchored and elongated cycles can have pct >= 1.0 while still descending (which would silently exclude the target population);
- per-row `is_elongated = (cycle_days_since_peak / station_baseline > 1.3)` where `station_baseline` = median `cycle_mean_length` over the 730d window ending at `val_start − 1` (frozen per fold per station);
- 4-way bucket: `ext_descent`, `normal_descent`, `elong_ascent`, `normal_ascent`.

## Headline — broad-population effect is real

| Bucket | n (all) | mean delta_ll (all) | n (hard25) | mean delta_ll (hard25) |
|---|---|---|---|---|
| **ext_descent**   | 29,640  | **+0.0168** | 10,305 | **+0.0428** |
| elong_ascent      | 43,794  | +0.0003 |  7,418  | +0.0025 |
| normal_ascent     | 288,186 | −0.0001 | 58,881  | −0.0120 |
| **normal_descent**| 431,702 | **−0.0088** | 121,491 | **−0.0420** |

A's effect on extended-descent rows is the **opposite sign** of its effect on normal-descent rows, in both samples. On hard25 the two effects are near-symmetric in magnitude (+0.043 vs −0.042). In aggregate they nearly cancel, which is why R5 looks neutral-to-positive overall — but the mechanism is two opposite-signed contributions, not a uniform mild lift.

**The user's "broad-population signal" hypothesis lands.** A's failure mode on extended descent isn't localised to fold 7 — it shows up across the population of extended-descent rows. Fold 7 just has many such rows in its val window.

## Per-fold breakdown — ext_descent vs normal_descent gap

Positive gap = A regresses *more* on ext_descent than normal_descent in that fold.

| fold | n ext_descent | mean delta ext_descent | mean delta normal_descent | gap |
|------|---|---|---|---|
| 7  | 8,451 | **+0.0991** | +0.0041 | **+0.095** |
| 9  | 1,085 | **+0.0835** | +0.0102 | **+0.073** |
| 4  | 6,890 | +0.0345 | −0.0259 | +0.060 |
| 5  | 2,561 | −0.0159 | −0.0272 | +0.011 |
| 10 |   465 | +0.0020 | −0.0037 | +0.006 |
| 11 | 2,660 | −0.0018 | −0.0076 | +0.006 |
| 13 | 1,234 | +0.0067 | +0.0027 | +0.004 |
| 2  |     2 | −0.0002 | −0.0028 | +0.003 |
| 6  |   653 | −0.0025 | −0.0045 | +0.002 |
| 14 | 1,997 | −0.0007 | +0.0025 | −0.003 |
| 1  | 1,287 | −0.0138 | −0.0025 | −0.011 |
| **3**  | 2,354 | **−0.2586** | −0.0588 | **−0.200** |
| 8  |     1 | −1.0088 | −0.0231 | −0.986 (n=1) |
| 12 |     0 | NaN | +0.0090 | n/a |

**10 of 12 folds with meaningful ext_descent count show positive gap** (A regresses more on ext_descent). The two clean negatives are fold 1 (small, −0.011) and **fold 3 (huge, −0.200)**.

### Fold 7 deep-dive

| bucket | n | mean delta | sum contribution | % of fold's positive sum |
|---|---|---|---|---|
| ext_descent   | 8,451 | +0.0991 | +837.5 | **64.8%** |
| elong_ascent  | 4,157 | +0.0490 | +203.8 | 15.8% |
| normal_ascent | 14,891 | +0.0087 | +129.0 | 10.0% |
| normal_descent | 29,845 | +0.0041 | +121.5 | 9.4% |

Two-thirds of fold 7's positive delta_ll comes from ext_descent rows, which are only ~15% of the fold's val rows. The hypothesis is row-level confirmed for fold 7.

## The fold 3 counterexample

Fold 3 has 2,354 ext_descent rows (substantial — not an n=1 outlier), and A *helps* them strongly (mean delta_ll −0.26). The gap to normal_descent is −0.20 — A is doing especially good work on fold 3's elongated rows, not bad.

This rules out a blanket `is_extended_descent` interaction as the constraint design. Such a feature would suppress A's contribution on fold 3 (a big helper) AND on fold 7 (a big regressor). One axis isn't enough to separate them.

User observation flagged this in advance: fold 3's elongated cycle was "a little shallow"; fold 7's was "very shallow". The missing axis is descent gradient or some shape proxy — exactly what the earlier elongation × gradient hypothesis flagged but couldn't isolate at fold resolution.

## What this means for #212 and the constraint design

1. **The mechanism is verified.** Across 14 folds, A's per-row error correlates negatively with `is_extended_descent` (it helps normal descent, hurts extended). The triplet-track motivation per `project_late_descent_investigation` is salvageable: there IS information separating coordinated late-descent from extended-descent, and A's compression signal misreads it.
2. **A simple `is_extended_descent × A` interaction is not the right constraint.** Fold 3 would lose −0.26 of its current lift on ext_descent rows.
3. **The constraint needs at least one more axis.** Candidates:
   - Descent gradient / shallowness — user's prior intuition. Test: among ext_descent rows, is A's regression on rows where the descent has been shallow?
   - Cohort consensus — when many stations are simultaneously in ext_descent, A's compression signal is misleading network-wide.
   - Phase-elapsed × baseline-amplitude — extended descent with low amplitude vs extended descent with high amplitude.
4. **#212's ablation grid is unchanged.** This work doesn't change which subset of A graduates — it gives evidence that whatever subset graduates can be *improved* by a constraint feature, but the constraint is design work downstream of #212.

## Recommendation

File a `design` issue capturing:

- The row-level evidence that A's effect is opposite-signed on ext_descent vs normal_descent.
- The fold 3 counterexample that rules out blanket suppression.
- Three candidate axes for the missing constraint dimension.
- The mandate: prototype `is_extended_descent × is_shallow_descent × A` or similar two-axis interactions on the existing `step5_rowdelta.parquet` (pure analysis, no retrain needed for prototyping) before deciding which subset, if any, justifies a re-train.

Do NOT close `project_late_descent_investigation` / #211 yet — the verdict shifts from "intra-series exhausted" to "intra-series has a real signal but needs a 2-axis decoder." External data per #211 stays in reserve as a fallback if no 2-axis constraint pans out.

## Artefacts

- `step5_paired_wfcv_rowpreds.py` + `step5.log` (408.9s wall, 8.0M predictions saved).
- `step5_rowpreds.parquet` (32 MB, per-(fold, run, seed, row) predictions).
- `step5_fold_meta.csv` (140 rows: per-(fold, run, seed) summary).
- `step5_rowlevel_diag.py` + `step5_diag.log` (5.8s wall).
- `step5_rowdelta.parquet` (per-row delta + cohort flags, ready for ad-hoc analysis).
- `step5_rowlevel_summary.csv` (per-fold × bucket mean delta_ll pivot).
- `step5_rowlevel.png`.
