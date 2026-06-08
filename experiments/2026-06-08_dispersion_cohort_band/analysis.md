# Dispersion-cohort band ablation — analysis

**Verdict: SWITCH_TO_10C** (canonical Competitive cohort wins).
**Wall:** 585s (9.75 min).

## Headline

Median seed-aggregation, mean across 14 folds. Δ negative-is-helpful.

| Run        | Δall    | Δh25       | Δh10    | Δlated  | helps_h25 |
|------------|---------|------------|---------|---------|-----------|
| R0         | 0.0000  | 0.0000     | 0.0000  | 0.0000  | —         |
| R1 (±5c)   | -0.0052 | **-0.0299**| -0.0681 | -0.0169 | 8/14      |
| R2 (±10c)  | -0.0073 | **-0.0386**| -0.0864 | -0.0221 | 10/14     |

**R2 − R1 margin on Δh25: -0.0088.** Canonical beats tight by ~30% on lift, well outside the +0.005 parsimony tolerance. Decision: SWITCH_TO_10C.

Mean-agg agrees with median (R2 -0.0372 vs R1 -0.0268). No outlier-suspect divergence.

## What changed vs the framing

The issue body assumed ±5c was "roughly half" the canonical cohort. Reality:

```
±5c  = 1,604,459 rows
±10c = 1,876,421 rows  (ratio 1.17x)
```

The bulk of stations cluster near zero stickiness; the 5–10c band is sparse tail. So the experiment isn't really "doubling the cohort," it's "adding 17% more rows from the lower-density outer ring." That ~17% materially improves the std signal, which is surprising — naively the outer ring should be noisier and dilute the cross-station dispersion read.

Hypothesis: tight ±5c selects only the most centrally-priced stations, which by construction move tightly together. The std collapses are real but the signal is narrower (fewer contributors to capture the dispersion). Widening to canonical includes mid-band stations that diverge more in extended descent (where Family A struggles), giving the model more information about cohort-wide phase coherence. Untested — record as a hypothesis, not a finding.

## Regime split

| Regime  | R1 Δh25 | R2 Δh25 | R2 − R1 |
|---------|---------|---------|---------|
| normal  | -0.0360 | -0.0506 | -0.0146 |
| shock   | -0.0145 | -0.0086 | +0.0059 |

R2 wins big on normal folds, loses a little on shock folds. Net win. Not a #214-style hidden-failure pattern — the loss on shock is small (~0.006), the gain on normal is large (~0.015), and the per-fold helps_h25 count favours R2 (10/14 vs 8/14).

But worth flagging: future work targeting shock-regime improvements should not assume the canonical cohort is strictly Pareto. There may be a shock-conditional refinement (e.g. narrower cohort during regime transitions) — out of scope here.

## Per-fold pattern: R2 amplifies, doesn't smooth

| fold | regime | R1 Δh25 | R2 Δh25 | R2 − R1 | dir       |
|------|--------|---------|---------|---------|-----------|
| 1    | shock  | -0.0278 | -0.0114 | +0.016  | both help |
| 2    | normal | +0.0133 | -0.0313 | -0.045  | **R2 rescues** |
| 3    | normal | -0.2131 | -0.2684 | -0.055  | both help |
| 4    | shock  | -0.0572 | -0.0906 | -0.033  | both help |
| 5    | normal | -0.1424 | -0.1538 | -0.011  | both help |
| 6    | normal | -0.0085 | -0.0112 | -0.003  | both help |
| 7    | normal | +0.0900 | +0.1242 | +0.034  | **R2 worse** |
| 8    | normal | -0.0796 | -0.1248 | -0.045  | both help |
| 9    | shock  | +0.0141 | +0.0478 | +0.034  | **R2 worse** |
| 10   | normal | -0.0083 | -0.0056 | +0.003  | both help |
| 11   | normal | -0.0317 | -0.0699 | -0.038  | both help |
| 12   | normal | +0.0095 | +0.0408 | +0.031  | **R2 worse** |
| 13   | shock  | +0.0130 | +0.0198 | +0.007  | **R2 worse** |
| 14   | normal | +0.0108 | -0.0064 | -0.017  | **R2 rescues** |

Bad-fold counts:
- R1 hurts on **6 folds**: 2, 7, 9, 12, 13, 14.
- R2 hurts on **4 folds**: 7, 9, 12, 13.
- R2 rescues folds 2 and 14 (both small R1 regressions, around +0.011–+0.013).
- R2 makes 7, 9, 12, 13 **worse**, not better.

The canonical cohort isn't smoothing the signal — it's amplifying it. Helps get bigger AND hurts get bigger. R2 wins on aggregate because the helps grow faster than the hurts (the rescues are small, the deepenings of large helps are large).

**Fold 7 takeaway.** Fold 7 was the headline `ext_descent_shallow` failure mode from `project_a_dual_effect_mechanism`. R1 had it at +0.090; R2 takes it to **+0.124** — the worst single-cell regression in the whole experiment. The canonical cohort doesn't fix the mechanism. Giving A a wider base of contributors makes the compression signal *more confident* in extended descent — exactly when A is wrong.

**Implication for #214.** The shallow-elongated constraint motivation is **stronger** under the canonical cohort, not weaker. Once #221 lands, fold 7's residual error budget grows from +0.090 to +0.124. #214's potential lift gets bigger — but so does the cost of not doing it. Update #214 to re-baseline its decision rule against the canonical-cohort A.

**Fold 12 is new.** Fold 9 (back-to-back-elongation, the unrescuable side-quest in #214) was already on the radar. Fold 12 (normal regime, R1 +0.010 → R2 +0.041) wasn't. It's small under R1 but grows under R2. Not blocking; record as a follow-up if it persists after the 54-feat retrain.

## Seed-variance gate

`hard25`, `hard10`, `all` cohorts: **zero flagged cells**.

`lated` cohort: 3 flagged cells, all at ratio 5.1–6.4× (modest):

| cohort | fold | run    | seed_std | ratio |
|--------|------|--------|----------|-------|
| lated  | 3    | R0     | 0.117    | 6.4×  |
| lated  | 3    | R1_5c  | 0.103    | 5.6×  |
| lated  | 8    | R2_10c | 0.093    | 5.1×  |

The flags are **fold-level noise**, not feature pathology:
- Two of three live on fold 3 — including R0 (no new features at all). The fold itself is noisy on the lated cohort.
- The third is on fold 8 R2_10c, isolated. R0 and R1 on fold 8 are not flagged.
- Lated is the smallest cohort per fold (tens of rows), so naturally higher seed_std.

The headline metric (Δh25) has zero flags. Verdict stands without per-seed drill-in.

## Why the verdict is robust

1. Margin (-0.0088) exceeds the parsimony tolerance (0.005) by ~2×.
2. R2 wins on **all four** cohorts (`all`, `hard25`, `hard10`, `lated`), not just the headline.
3. Per-fold helps count favours R2 (10/14 vs 8/14).
4. Mean-agg agrees with median-agg — no single-seed contamination.
5. Seed-variance gate clean on the headline cohort.
6. Cohort filter is mechanically simpler: reuse the existing `sc.class = 'Competitive'` label instead of a feature-specific magic constant.

## Implications

1. **`COMP_BAND_CENTS = 5.0` is a regression**, not just a stylistic drift. The constant was inherited from `step2_paired_wfcv.py` and never ablated against the canonical band. The #212 ablation that graduated RAC_full used ±5c throughout — the **measured lift of RAC_full was understated** by ~0.009 on Δh25 because the suboptimal cohort was held fixed across all runs. The #212 verdict (graduate RAC_full) still stands directionally; the absolute Δ should be larger under the canonical cohort.

2. **The 54-feat retrain (pending per `project_late_descent_triplet_outcome`) should use the canonical cohort.** Implementing this *before* the retrain saves a re-retrain cycle.

3. The `cycle_descent_slope` × `elongation_ratio` constraint (#214) was designed against the ±5c version of A. The mechanism (A's compression signal misreading in extended descent) is unlikely to be cohort-dependent, but the absolute fold-7 regression numbers may shift. Re-baseline #214's decision rule against the canonical-cohort A once landed.

## Follow-ups

- **#221 (filed):** implement the switch in `fuel_signal/features.py` — replace the `ABS(median_premium_decicents) <= ±5c` filter in `_network_px_std_per_date` with `sc.class = 'Competitive'`, drop `COMP_BAND_CENTS`, update docstring. Tests updated. Worker-eligible.
- **User-run:** after #221 merges, retrain to 54-feat baseline with the canonical cohort. The retrain blocks #214 (per the planning chain in #215).
- **Closed:** #219.

## Files

- `runs.csv` — per (fold, run, seed): ll per cohort + fit seconds.
- `fold_run.csv` — per (fold, run): mean + median per cohort + Δ vs R0.
- `meta.json` — config, summary, seed-variance flags, decision.
- `run.log` — full stdout.
