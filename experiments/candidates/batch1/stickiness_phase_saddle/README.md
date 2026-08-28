# batch1 / stickiness_phase_saddle

- **Date:** 2026-08-15 (snapshot) / dossiered 2026-08-27
- **Branch:** main
- **SHA:** 9362d46d74f63ccc2a430e96379dfd8658f2036d
- **Status:** done — **inconclusive**
- **Bead:** fps-6yi
- **Cadence:** `50/3.571/1d/10%` (batch1's canonical daily cadence)

## Hypothesis

`cycle_pct_through` is a NETWORK clock (one value per date), but a station's
willingness to follow that clock is a per-station property. A chronically
cheap, competitive station tracks the cycle closely: at phase 0.9 it really
is near its own trough. A chronically dear or erratic station is only
loosely coupled to it — the same phase reading carries much less
information about where that station's own price is going next. The lock
hands the model `stickiness_score` and `cycle_pct_through` separately and
nothing else; a gradient-boosted tree can only approximate their product
through nested axis-aligned splits. Columns: `sticky_x_phase`,
`abs_sticky_x_phase`.

**Predicted signature:** decision-timing — for stations with near-zero
stickiness the decisions should be essentially unchanged (the product is
~0 there), and the flips should concentrate on the tails of the
stickiness distribution, buying EARLIER on high-`|stickiness|` stations at
late phase. The distinctive signature is that the SIGNED and ABSOLUTE
columns are both used and DISAGREE in orientation somewhere in the SHAP
surface — if only `abs_sticky_x_phase` is used, the real content is
"outlier stations are decoupled," a weaker and different claim. Expected
to spread across folds rather than concentrate, since station composition
is cross-sectional with no particular fold affinity.

## How to invoke this script

```bash
PYTHONPATH=. uv run python -m experiments.pipeline.runner \
  --batch-dir experiments/batches/batch1 \
  --candidate experiments/candidates/batch1/stickiness_phase_saddle.py
```

## Facts

Transcribed from `facts.json` — no new arithmetic.

**Provenance:** batch `batch1`, snapshot 2026-08-15, seeds [42, 43, 44, 45,
46], wall 1010.6s, status `graded`, 54-column baseline (`54:1a6ec2d84a69`).

**Headline (realised CPL, held τ — the arbiter):**

| | R0 | candidate | delta |
|---|---|---|---|
| CPL held | 187.8211 | 187.7476 | **-0.0735** |

`effect_resolved`: **true**. Note the seeds above are the WFCV screen
only — the realised arbiter ran on a single seed
(`results.json` `meta.realised_seed` = 42). That is provenance, not a
missing error bar: the realised delta is a PAIRED difference in which
both arms share the seed, the data, the folds and the 54 baseline
columns, differing only by this candidate's 2 added columns, so the
fit's seed idiosyncrasy is common-mode and cancels. Its error bar is the
noise floor reported below, built the same paired way at the same fixed
seed (`PLACEBO_SEED_DEFAULT = SEEDS[0]` = 42) — that matching is what
makes the two comparable.

**WFCV log-loss** (descriptive colour, NOT the arbiter): Δll_all mean
+0.00355, Δll_hard25 mean -0.00651. Mixed — nominally slightly worse on
`all`, slightly better on `hard25`.

**Zone:** `resolved`: **null** — `TARGET` declared neither `folds` nor
`axis`/`expect_concentration_in`, so there is no declared-zone test to
report for this candidate (unlike `station_descent_dynamics` or
`lga_trough_propagation` in the same batch).

**Per-fold:**

| fold | regime | n_fills | delta_cpl_own |
|---|---|---|---|
| 1 | shock | 190 | 0.0000 |
| 2 | normal | 200 | +0.7729 |
| 3 | normal | 186 | -0.5079 |
| 4 | shock | 239 | +0.6069 |
| 5 | normal | 185 | -0.0107 |
| 6 | normal | 125 | -0.2567 |
| 7 | normal | 235 | +0.1544 |
| 8 | normal | 84 | +0.0190 |
| 9 | shock | 112 | +0.0718 |
| 10 | normal | 96 | -0.5536 |
| 11 | normal | 161 | -0.2953 |
| 12 | normal | 184 | +0.0528 |
| 13 | shock | 142 | **-0.8646** |
| 14 | normal | 169 | -0.0302 |

No cell is suppressed (`min_row_cell_n` = 30, smallest cell here is 84).

**Per-regime:** shock -0.0614 c/L (n=687), normal -0.0768 c/L (n=1645) —
both favour the candidate, unlike several batch1 siblings where the two
regimes disagree in sign.

`per_axis` is `null` — this candidate declared no `add_axis` beyond the
(unresolved) zone check above.

**Seed stability:** 3 cells exceed 5× the cohort-median seed_std, all
`all`-cohort:

| fold | run | seed_std | ratio vs cohort median |
|---|---|---|---|
| 2 | R0 | 0.04577 | 5.56× |
| 2 | candidate | 0.07379 | 8.96× |
| 3 | R0 | 0.04510 | 5.47× |

Fold 2's flag is on the **candidate arm**, so per dossier convention here
is the fold × arm comparison rather than prose: R0's seed_std jumps from
0.04577 to 0.07379 on the candidate arm in the same fold — a ~1.6×
increase, consistent with the added columns mildly destabilising an
otherwise-ordinary fit in that fold. Raw seed_std against the quantity
that actually matters (`delta_ll_*`, not the CPL delta, which is not
seed-measured at all): candidate seed_std 0.07379 vs fold 2's
`delta_ll_all_median` +0.02152 (~3.4×) and vs `delta_ll_hard25_median`
-0.05522 (~1.3×) — a real but not extreme gap, nothing like the 1700×
case seen elsewhere in this project. Fold 3's flag is R0-only (a property
of that window, not the candidate) — one line is enough for it.

**Validation:** PIT passed (reached WFCV/realised stages, so the
differential PIT truncation test found no leak). INPUTS check passed.
Candidate column NaN rates: `sticky_x_phase` 0.0%, `abs_sticky_x_phase`
0.0%.

**Noise floor** (`experiments/batches/batch1/noise_floor.json`, 10 draws
at arity 3 — this candidate is only arity 2, so `floor_arity_exceeds_run`:
**true**, meaning this run was graded against a ruler built from MORE
columns than it has; allowed, since a wider ruler only raises the bar,
never lowers it, but it means a candidate that failed here has not been
shown to fail against its own arity's band): band mean +0.0251 c/L, band
std 0.0999 c/L. The candidate's -0.0735 c/L held-τ sits at the **80th
percentile** of the 10 observed noise draws
(`candidate_percentile_better_than_noise` = 80.0), and the parametric
single-candidate call is `z = -0.987` against `t = 1.923`
(`single_candidate_z_threshold`, the authoritative threshold — not the
nominal one) — inside the band on both sides
(`abs(z) < t`), so this is a two-sided pass into "inconclusive," not a
clean win.

Bar decomposition, in c/L: `single_candidate_bar_cpl_held` -0.1670 (the
threshold in project units); of that, `bar_draw_count_shift_cpl_held`
-0.0278 is the thinness penalty (10 draws vs the large-sample limit) and
`bar_reuse_shift_cpl_held` is **0.0** — this bank's draws carry no source-column-reuse discount at this candidate's arity. `floor_arity_excess`
is **1** (floor built at arity 3, run at arity 2).

**Decision flips** (`facts["decision_flips"]`, diffs R0's and the
candidate's `fills.parquet` directly on `(fold, station_code, date)`):
**303 total flips**, present in every fold except fold 1. Per fold:

| fold | regime | n_baseline_only | n_candidate_only | flip_cpl_baseline | flip_cpl_candidate | flip_cpl_delta |
|---|---|---|---|---|---|---|
| 1 | shock | 0 | 0 | — | — | — |
| 2 | normal | 12 | 38 | 181.63 | 183.37 | +1.74 |
| 3 | normal | 21 | 16 | 200.36 | 193.98 | -6.38 |
| 4 | shock | 15 | 39 | 184.34 | 185.22 | +0.88 |
| 5 | normal | 6 | 22 | 198.14 | 190.77 | -7.37 |
| 6 | normal | 18 | 8 | 185.47 | 185.32 | -0.15 |
| 7 | normal | 6 | 10 | 186.22 | 176.95 | -9.27 |
| 8 | normal | 20 | 16 | 205.37 | 207.96 | +2.58 |
| 9 | shock | 0 | 3 | — (no baseline-only fills) | 179.65 | — |
| 10 | normal | 10 | 20 | 208.66 | 198.32 | **-10.34** |
| 11 | normal | 2 | 5 | 223.90 | 204.22 | **-19.68** |
| 12 | normal | 3 | 2 | 191.33 | 181.80 | -9.52 |
| 13 | shock | 5 | 1 | 205.63 | 172.90 | **-32.73** |
| 14 | normal | 2 | 3 | 179.90 | 177.67 | -2.23 |

Favourable (`flip_cpl_delta` < 0) flips dominate — 9 of the 12 folds with
a computable delta — with the largest concentration in fold 13 (shock)
and fold 11 (normal); folds 2, 4, and 8 move the other way. `n_flips`
(303) spread across 13 of the 14 folds is consistent with the predicted
"spread across folds, not concentrated" shape, though `decision_flips`
carries no station-level stickiness data, so it cannot confirm the more
specific "flips concentrate on the stickiness-distribution tails" claim.

![](per_fold_delta_bars.png)
![](seed_mean_vs_median.png)
![](realised_cpl_by_fold.png)
![](tau_sweep.png)
![](candidate_over_time.png)
![](cycle_phase_breakdown.png)

## Judgement

**Signature grading: inconclusive.** Realised delta (-0.0735 c/L,
favourable direction) and 303 decision flips spread across 13 of 14
folds match the predicted diffuse, not fold-concentrated, pattern. But
the effect sits inside this batch's single-candidate noise band
(z=-0.99 vs t=1.92, 80th percentile of the noise draws — graded against a
ruler wider than this candidate's own arity), and `facts.json` carries no
SHAP importance/orientation data at all, so the run's own distinguishing
signature — signed vs absolute column disagreement in the SHAP surface —
was never tested. Filed as `fps-1l1`: this is a real gap in
`dossier_tables.py`, not something to compute in this session. Neither
the effect size nor the candidate's own proposed mechanism test is
resolved by this run.

**not_tested:**

- Whether the SHAP signature the candidate predicted (signed and absolute
  columns both used, disagreeing in orientation somewhere in the surface)
  exists at all — `dossier_tables.py` captures no per-column SHAP
  importance/orientation, so this run's own distinguishing test was never
  run. Tracked as `fps-1l1`.
- Whether a floor matching this candidate's own arity (2 columns, not the
  3 the batch's floor was built at — `floor_arity_exceeds_run`) would move
  the call. The candidate cleared the 80th percentile of the wider
  ruler's noise draws; a same-arity floor could resolve it either way.
- Whether the more specific claim — flips concentrated on the tails of
  the stickiness distribution, near-zero elsewhere — holds at the row
  level. `decision_flips` has no station-level stickiness data, so only
  the coarser "spread across folds" half of the prediction is checkable
  here, and it matches.

**Recommendation:** not a rejection. The effect direction favours the
candidate in both regimes and the flip pattern matches the predicted
diffuse shape, but the run cannot distinguish the candidate from noise at
its own arity, and its own mechanism test was never captured. Worth a
second look once `fps-1l1`'s SHAP gap is closed, or once a floor matching
this candidate's exact 2-column arity exists — either could move this out
of the noise band without a new run.

## Addendum — batch-wide coverage caveat

This run's `results.json` `meta` records `extra_feature_provider_hits`
5693 / `extra_feature_provider_misses` 607 — the same split every batch1
candidate carries, since all of them replay the same 5-preferred-station,
14-fold grid. Full analysis (per-fold decomposition, which stations/folds
are affected, and why it's reduced-not-corrupted sample):
`experiments/candidates/batch1/lga_trough_propagation/README.md` §
Review addendum. Not specific to this candidate and does not change this
dossier's verdict.

## Followups

- `fps-1l1` — `dossier_tables.py` carries no SHAP importance/orientation
  data, so this run's own predicted signature (signed/absolute
  disagreement) could not be graded.
