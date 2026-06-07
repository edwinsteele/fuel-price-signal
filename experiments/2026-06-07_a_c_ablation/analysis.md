# Analysis — within-family A ablation (+ C as complement)

Issue #212. Median-across-seeds is the headline; mean shown alongside.

## Verdict

**Graduate the full RAC_full set (4 columns):**

- `network_px_std`
- `network_px_std_delta_3d`
- `lga_phase_std`
- `lga_phase_std_delta_3d`

Retrain the production model on 50 → 54 features.

## Headline numbers

Δh25_median = mean across 14 folds of the per-fold (median across 5 seeds)
delta_hard25. Negative = improvement.

| Run        | Δh25 med  | Δh25 mean | Δh10 med  | Δlated med | helps_h25 | Δh25 normal | Δh25 shock |
|------------|-----------|-----------|-----------|------------|-----------|-------------|------------|
| RA_level   | −0.0317   | −0.0382   | −0.0774   | −0.0159    | 9 / 14    | −0.0318     | −0.0316    |
| RA_delta   | −0.0059   | −0.0090   | −0.0195   | −0.0072    | 6 / 14    | −0.0089     | +0.0016    |
| RA_both    | −0.0299   | −0.0268   | −0.0681   | −0.0169    | 8 / 14    | −0.0360     | −0.0145    |
| **RAC_full** | **−0.0448** | **−0.0465** | **−0.0992** | **−0.0296** | **10 / 14** | **−0.0616** | **−0.0026** |

R0 reference: median ll_h25 across folds = 1.0913.

## Applying the decision rule

1. **Best A subset (parsimony):** RA_level beats RA_both outright (−0.0317 vs −0.0299),
   so the smallest A subset matching RA_both is `network_px_std` alone. RA_delta
   fails the threshold (−0.0059 lift, far short of RA_level).

2. **Does C graduate?** RAC_full Δh25 = −0.0448 vs RA_level −0.0317 → +0.0131
   improvement. ≥ +0.010 gate cleared. **C graduates.**

3. **Why keep `network_px_std_delta_3d` despite RA_both ≈ RA_level?** See
   "Parsimony vs regime-sensitivity" below — the Δ feature is net-flat on
   average but materially regime-sensitive across folds. Dropping it would throw
   away pre-#214 help on fold 7 (the failure mode #214 is built to address).

## Seed-variance drill-in

Hard25 (the headline cohort): **0 flagged cells**. The decision rule is built on
clean data.

8 cells flagged in `all` (4) and `lated` (4):

| Cohort | Fold | Run       | seed_std | ratio | per-seed pattern |
|--------|------|-----------|----------|-------|------------------|
| all    | 2    | RA_both   | 0.039    | 5.2×  | s44 = 0.333 (others ~0.245); median 0.246 robust |
| all    | 3    | RAC_full  | 0.070    | 9.3×  | s45 = 0.487 (others ~0.34); median 0.345 robust |
| all    | 5    | RA_delta  | 0.053    | 7.1×  | s43 = 0.386 (others ~0.27); median 0.274 robust |
| all    | 11   | RAC_full  | 0.039    | 5.2×  | s45 = 0.416 (others ~0.33); median 0.346 robust |
| lated  | 3    | R0        | 0.117    | 7.4×  | wide spread across all seeds; cohort-fold inherent noise |
| lated  | 3    | RAC_full  | 0.104    | 6.6×  | same cohort-fold noise carries through all runs |
| lated  | 3    | RA_both   | 0.103    | 6.5×  | same |
| lated  | 3    | RA_delta  | 0.092    | 5.8×  | same |

Pattern in `all`: every flagged cell has a single outlier seed; the other 4 are
tightly clustered; median is robust. Same shape as the s44 outlier from the
2026-06-06 step2 investigation (see `project_a_dual_effect_mechanism`).

Pattern in `lated`: fold 3's lated subset (n=8321) is intrinsically noisy across
seeds — `R0` itself is flagged. The noise is a property of the cohort/fold, not
the runs, and the delta is meaningful because it's a difference of medians where
both terms are subject to the same source of noise. RAC_full's median lated lift
on fold 3 = −0.245 (strong help) is real signal.

## Parsimony vs regime-sensitivity (why 4 cols not 3)

The grid never directly tested `network_px_std` + `lga_phase_std` +
`lga_phase_std_delta_3d` (3 columns, dropping `network_px_std_delta_3d`). On
aggregate the Δ feature looks net-flat (RA_both ≈ RA_level). But per-fold:

| Fold | Elong. | RA_level Δh25 | RA_both Δh25 | Δ contributes |
|------|-------:|--------------:|-------------:|--------------:|
| 1    | 0.69 | −0.050 | −0.028 | +0.022 (hurts) |
| 4    | 0.51 | −0.098 | −0.057 | +0.041 (hurts) |
| 5    | 0.73 | −0.095 | −0.142 | −0.047 (helps) |
| **7**  | 0.51 | **+0.127** | **+0.090** | **−0.037 (helps)** |
| 11   | 0.58 | −0.057 | −0.032 | +0.025 (hurts) |
| 14   | 0.57 | −0.026 | +0.011 | +0.037 (hurts) |

The averages cancel, but `network_px_std_delta_3d` materially helps on **fold 7**
— the worst regressor, and the exact failure mode #214 is being built to
address. Dropping it ahead of #214 would discard pre-#214 relief on the right
fold.

This is the same shape as the 2026-06-03 SHAP-redundancy lesson
(`project_shap_redundancy_regime_caveat`): a feature that looks aggregate-flat
can be regime-sensitive, and the per-fold variance is the thing to trust.

Cost of carrying the column: +1 in the 54-wide feature matrix. Trivial.

## Elongation diagnostic (informational)

Pearson r between per-fold elongation exposure and per-fold delta_hard25_median:

| Run       | r       |
|-----------|---------|
| RA_level  | −0.27   |
| RA_delta  | −0.26   |
| RA_both   | −0.21   |
| RAC_full  | −0.32   |

Negative r means **more elongation correlates with more improvement** — the
opposite of the original `project_late_descent_elongation_regime` hypothesis.
Folds where RA_level regresses (7, 10, 12, 13) are clustered at the *low* end
of fold elongation (0.34–0.51); folds 1 (0.69) and 5 (0.73) — high elongation —
are among the biggest helpers.

Reconciliation with step5: fold 7 regresses because ~15% of its rows
(ext_descent shallow) carry 65% of its positive delta_ll. A per-fold elongation
average drowns that subset out. The row-level (elongation × shallowness)
framing from `project_a_dual_effect_mechanism` is the surviving story; the
fold-level elongation hypothesis is retired.

## Contingency for #215

#212 has graduated a useful subset (RAC_full). #214 is now the gating
experiment for the planning decision in #215. Per #215's decision matrix:

- If #214 closes the fold-7 regression on ext_descent_shallow rows: intra-series
  late-descent track has delivered two useful pieces. External-data move stays
  in reserve.
- If #214 fails: external-data case strengthens; scope one cheap external probe.

#215 stays open, blocked on #214.

## Acceptance criteria

- [x] Lab-book entry committed (README + script + result CSVs + meta JSON + log
      + this analysis).
- [x] Median-aggregated attribution tables per cohort {all, hard25, hard10, lated}
      for the 4 non-baseline runs (above).
- [x] Seed-variance diagnostic reported. All 8 flagged cells drilled into.
- [x] Per-fold delta vs elongation-exposure diagnostic reported.
- [x] Decision recorded with reasoning: RAC_full graduates (4 columns).
- [x] Follow-up issue filed to land the columns in `fuel_signal/features.py`
      and retrain the 50 → 54-feat baseline. (See issue link.)

## Outputs

- `runs.csv` — 350 rows, one per (fold, run, seed).
- `fold_run.csv` — 70 rows, mean + median + seed_std + deltas per (fold, run).
- `elongation_per_fold.csv` — frozen-baseline elongation per fold.
- `meta.json` — config + summary + seed-variance flags + elongation r.
- `run.log` — captured stdout, total wall 1227.5s.
