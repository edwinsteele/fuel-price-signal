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

**Signature grading: inconclusive.** Two separate things carry that label
and they should not be confused. The ledger `outcome` is mechanical
(`docs/routines/dossier.md` step 3: `-t < z < t` -> `inconclusive`), and
z=-0.99 against t=1.92 puts this run inside the band — that field is
correct and stays. Separately, the candidate's own `PREDICTED_SIGNATURE`
was written in terms of SHAP importance/orientation, which
`docs/routines/dossier.md` (decided `fps-1l1`) grades `inconclusive`
**always**, because no SHAP field will ever exist in `facts.json`. So the
signature grade here is a property of how the claim was phrased, not a
measurement of the claim.

What the run itself showed: realised delta -0.0735 c/L in the favourable
direction, 303 decision flips across 13 of 14 folds, both regimes
favouring the candidate. Inside the noise band on the ruler used.

**This dossier's original recommendation was wrong and has been replaced**
— see the post-dossier review addendum below. Both routes it named for a
"second look" were already closed at the time of writing, and the one
mechanism question it declared uncheckable was checkable from
`facts.json`'s own `decision_flips.rows`.

**not_tested:**

- ~~Whether the predicted SHAP signature exists.~~ **Permanently
  untestable, not pending.** `fps-1l1` closed 2026-08-27 down its
  "don't capture" branch: no SHAP field will exist in `facts.json`, and
  `docs/routines/dossier.md` now grades any SHAP-shaped signature
  `inconclusive` on sight. This is a defect in how the candidate's
  signature was *phrased*, not a gap in the pipeline — see `fps-3jj.10`
  lead 6 and `fps-i2o` below.
- ~~Whether a floor matching this candidate's own 2-column arity would
  move the call.~~ **Answered — it would not.** Already resolvable from
  banks committed in the batch directory (see addendum): z=-0.81 against
  the arity-1 bank, -0.88 against the ICC bank, -0.99 against the arity-3
  bank the run was graded on. No comparable ruler brings |z| near t=1.92.
- Whether flips concentrate on the tails of the **stickiness**
  distribution specifically. Still open — `decision_flips` carries no
  station-level stickiness data, so the mechanism was localised by market
  condition (cycle turns) rather than by station type. The stickiness-tail
  formulation of the claim remains unmeasured.
- **New, and the live one:** whether a turn-GATED construction pays.
  The mechanism is real but mispriced (addendum below); nothing here tests
  a version that is inert outside turn conditions. Filed as `fps-3jj.10`
  lead 6. The open sub-question is what a PIT-safe turn gate is built
  from — the addendum's turn definition is forward-looking and is a
  diagnostic, not a constructible feature.

**Recommendation:** reject **this construction**; graduate the
**mechanism** to a redesign lead. The effect size question is settled —
inside the band on three independently-built rulers, with no remaining
route to move it without a new run. The mechanism question is answered
well enough to act on: the interaction genuinely helps at cycle turns and
mildly hurts everywhere else, and turns are too small a share of volume
for that trade to pay. Do not re-run this candidate as written. The live
successor is `fps-3jj.10` lead 6 (turn-gated stickiness x phase).

## Addendum — post-dossier review (2026-08-29)

New arithmetic, deliberately kept out of the Facts section above (which
stays a transcription of `facts.json`). Sources: `facts.json`'s
`decision_flips.rows`, this run's `fills.parquet`, and
`experiments/batches/batch1/fuel_signal.db` daily prices. All of it was
available when the dossier was written.

**1. The arity question was already answerable.** `_noise_band()` reads
`noise_floor.json` by fixed filename. Two further banks sit in the same
directory with matching `baseline_fingerprint` (`54:1a6ec2d84a69`), tank
params (`50/3.571/1d/10%`) and 14 windows:

| bank | draws | arity | null_method | band mean / std | candidate z |
|---|---|---|---|---|---|
| `noise_floor.json` (graded on) | 10 | 3 | `placebo_column` | +0.0251 / 0.0999 | **-0.99** |
| `noise_floor_k1.json` | 20 | 1 | `placebo_column` | -0.0089 / 0.0800 | **-0.81** |
| `noise_floor_icc.json` | 32 (eff. 14.7) | 1 | `placebo_column_pinned_source` | +0.0278 / 0.1152 | **-0.88** |

An arity-2 band sits between the arity-1 and arity-3 banks by
construction. None approaches t=1.92. Filed as `fps-30p`: make the dossier
consult every comparable bank rather than one.

**2. The mechanism is real, and located at cycle turns.** Taking each
flipped fill and looking forward at that station's own price (a
post-hoc diagnostic — forward data is the point, and would be a leak in a
feature):

- **Turn catches** (bought within 3 days of a >=15 c/L rise): candidate
  11, baseline 4 — spread across folds 2, 3, 8, 10 and 13, not one
  window. Not significant on its own (Fisher p=0.28-0.42).
- **Litres-weighted 7d regret** (price paid minus the cheapest price at
  that station in the next 7 days; 0 = bought the optimum):

  | flips | baseline | candidate |
  |---|---|---|
  | turn-adjacent (n=8 / 11) | 21.62 c/L | **0.33 c/L** |
  | ordinary (n=112 / 172) | 5.64 c/L | 6.52 c/L |
  | **all flips** | **6.22 c/L** | **6.21 c/L** |

**3. Why it does not pay.** Turn-adjacent fills are 150 of 2982 flipped
litres (~5%). The candidate buys near-optimally there and is 0.88 c/L
worse across the other 95%, so litres-weighted regret over all flips is a
dead heat — which is exactly why the realised delta is ~0. Fold 13's
much-quoted -32.73 c/L flip delta contributes **-0.0644 c/L** to the run;
folds 2 and 4 hand back **+0.0995** between them (litres-weighted
shift-share off `fills.parquet`, reconciling to the headline -0.0735 via
a -0.0129 composition residual).

**4. Withdrawn during this review.** An apparent "the candidate never
buys before a sharp fall" result (0/183 vs the baseline's 4/120, Fisher
p=0.024) did **not** survive a threshold sweep — p=1.000 at 8-10 c/L,
0.202 at 12, 0.156 at 20, 0.396 at 25. An isolated island of significance
across 14 tests. Not evidence; recorded here so it is not rediscovered.

**5. The per-fold flip table cannot be read as written.** Against each
cell's own standard error (per-fill flip price dispersion 13.7 c/L,
litres-weighted effective n of 1.0-19.5 per arm), **0 of 12 folds print a
`flip_cpl_delta` larger than twice its own SE** — fold 13's -32.73 sits
inside a 2*SE of 36.77. The Facts section's "favourable flips dominate —
9 of the 12 folds" is a vote count over cells that cannot resolve their
own numbers, and should not be read as corroboration. A redesign of the
flip table (counts + cascade-collapsed decisions + litres + a
headline-reconciling contribution column + each delta beside its own
2*SE) is in hand but not yet filed.

**Power caveat on all of the above:** the turn cells are n=11 and n=8, the
two arms' flips fall on different days (two decision sets, not a paired
test), and 303 flips are ~87 independent decisions once cascades are
collapsed. The effect-size conclusion is solid; the mechanism
localisation is suggestive and under-powered.

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

- `fps-3jj.10` lead 6 — **the live successor.** Turn-gated stickiness x
  phase: gate the interaction so it is active near a cycle turn and inert
  otherwise. Carries this review's numbers and the open PIT question
  (what a same-day turn proxy is built from).
- `fps-i2o` — `TEMPLATE.py` has no rule requiring a `PREDICTED_SIGNATURE`
  to be gradeable from persisted facts, so untestable signatures like this
  one's keep being written and now auto-grade `inconclusive` forever.
- `fps-30p` — `_noise_band()` reads one floor by fixed filename; two
  comparable banks in the same directory already answered this run's
  arity question.
- `fps-1p1` — `not_tested` lines must state feasibility, not just absence.
  All three of this dossier's original lines were wrong.
- `fps-1l1` — **CLOSED 2026-08-27**, down the "don't capture" branch: no
  SHAP field will exist in `facts.json`. Listed here because the original
  version of this dossier recommended waiting on it.
