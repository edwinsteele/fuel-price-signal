# batch1 / tgp_cycle_displacement

- **Date:** 2026-08-15 (snapshot) / dossiered 2026-08-28 / zone corrected 2026-08-30
  (`fps-hnp`) / flip table re-rendered 2026-08-31 (`fps-j6w`) / realised figures
  refreshed to the post-#359 regen 2026-09-03 (`fps-3ug`)
- **Branch:** main
- **SHA:** 12dc7629f1f440bbf71fa7f50304e6adb781d048 (the #359 regen run; the original
  2026-08-27 run was 4a2016dc8b5cad4dab661801bc789336e10b5e65 — the model fits are
  byte-identical between the two, only the realised backtest was re-run)
- **Status:** done — **contradicted signature, favourable headline**
- **Bead:** fps-2sf
- **Cadence:** `50/3.571/1d/10%` (batch1's canonical daily cadence)

## Hypothesis

The retail trough is bounded below by the wholesale floor, so how far the floor
has MOVED since this cycle's retail peak sets how deep this particular trough
can go. Neither prior TGP expression asks this: `station_minus_tgp_cents`
reads today's floor LEVEL (rejected — accurate when the floor is stable,
misleading when moving); `tgp_delta_7d` reads the floor's velocity over a
fixed 7-day window unrelated to where the retail cycle sits (inconclusive).
Anchoring the displacement to the cycle's own peak (`tgp_cycle_displacement_cents`,
`tgp_cycle_displacement_frac_amp`) makes it a depth-remaining statistic
instead of a level or a fixed-window momentum term.

**Predicted signature:** on rows where the floor has fallen a long way since
the peak, the model should WAIT past the point the lock would buy (the trough
ahead is deeper than the network clock implies); where the floor has been
flat or risen, it should buy earlier. **Expect concentration on NON-SHOCK
(normal) folds** — the mechanism is about floor-set trough depth in an
ordinary cycle, and shock folds are where the prior TGP gap candidate
measurably HURT. **A shock-concentrated win would be evidence AGAINST the
stated mechanism, and should be read as such.** Magnitude expectation: small
(0.03–0.26 c/L, per `results.csv` history), and both prior TGP expressions
landed inside the noise band.

## How to invoke this script

```bash
PYTHONPATH=. uv run python -m experiments.pipeline.runner \
  --batch-dir experiments/batches/batch1 \
  --candidate experiments/candidates/batch1/tgp_cycle_displacement.py
```

## Facts

Transcribed from `facts.json` — no new arithmetic.

**Provenance:** batch `batch1`, snapshot 2026-08-15, seeds [42, 43, 44, 45, 46]
(WFCV screen only — the realised arbiter ran on the single seed `42`, so
every `delta_cpl_own`/`delta_cpl_held` figure below is one draw with no error
bar), wall 1053.2s, status `graded`, 54-column baseline (`54:1a6ec2d84a69`).

**Headline (realised CPL, held τ — the arbiter):**

| | R0 | candidate | delta |
|---|---|---|---|
| CPL held | 188.0376 | 187.8299 | **-0.2077** |

`effect_resolved`: **true**.

**WFCV log-loss** (descriptive colour, NOT the arbiter): Δll_all mean
-0.0091, Δll_hard25 mean -0.0411. Both favour the candidate.

**Zone (regime axis, the candidate's declared target — safe to cut; a fold
is an independent simulation, not a row-label slice). Graded against the
corrected empirical shock-fold set `{1, 3, 4, 14}`
(`experiments/batches/batch1/shock_folds.json`, `fps-3tu`/`fps-hnp`), not
the fixed `{1, 4, 9, 13}` this dossier originally used — see the Correction
note below the Judgement section:**

| | target (normal) | other (shock) |
|---|---|---|
| delta_cpl_own | -0.1613 | -0.3079 |
| n_fills | 1389 | 674 |

`resolved`: **false** — the "other" (shock) zone is MORE negative than the
declared target (normal) zone, the opposite of the predicted concentration.
The margin is narrower than under the old fixed shock set
(-0.1392/-0.3405 → -0.1603/-0.2884 as measured on 2026-08-30, i.e. pre-#359;
the current post-regen corrected-set pair is -0.1613/-0.3079, a slightly wider
margin than the pre-regen one), but the verdict is unchanged.

**Per-fold** (regime column reflects the corrected empirical shock-fold set
`{1, 3, 4, 14}` — folds 3 and 9 swap regime labels, as do 13 and 14, vs.
the old fixed set):

| fold | regime | n_fills | delta_cpl_own |
|---|---|---|---|
| 1 | shock | 190 | +0.9884 |
| 2 | normal | 186 | +0.6307 |
| 3 | shock | 106 | -1.0892 |
| 4 | shock | 207 | -1.2103 |
| 5 | normal | 134 | -0.4760 |
| 6 | normal | 112 | +0.1780 |
| 7 | normal | 171 | -0.1340 |
| 8 | normal | 88 | +0.9725 |
| 9 | normal | 112 | +0.3879 |
| 10 | normal | 96 | -0.6291 |
| 11 | normal | 160 | -0.5000 |
| 12 | normal | 175 | -0.5325 |
| 13 | normal | 140 | **-1.6021** |
| 14 | shock | 169 | +0.0771 |

No cell is suppressed (`min_row_cell_n` = 30, smallest cell here is 88).
`per_axis` is `null` — this candidate declared no `add_axis` beyond the
regime split above, so there is no path-coupled per-row zone to caveat.

Within the 4 shock folds the pattern is mixed, not uniform: folds 1 and 14
hurt the candidate (+0.99, +0.08), while folds 3 and 4 help it (-1.09,
-1.21). **The #359 regen materially weakened the single-fold reading here,
and this paragraph has been rewritten rather than renumbered.** Pre-regen,
excluding fold 3 (litre-weighted against the then-current `fills.parquet`)
took the shock-zone delta from -0.2884 c/L to **+0.0050 c/L**, essentially a
sign flip — that figure is gitignored-artifact-derived, cannot be restated,
and stands as the pre-#359 measurement. Post-regen fold 4's own cell moved
from -0.9512 to **-1.2103**, so fold 3 no longer carries the zone alone: the
same leave-one-out recomputed fill-count-weighted from `facts.json` on both
sides reads **-0.0097 pre-regen against -0.0878 post-regen**, against a
fill-weighted shock zone of -0.2458. So fold 3 still accounts for **most** of
the shock number — 64% of its magnitude — but no longer for nearly all of it
(95% pre-regen), and removing it no longer flips the sign. What survives
unchanged is the
weaker, count-based version of the same caution: **2 of the 4 shock folds
hurt the candidate**, so the shock aggregate is not a uniform shock story.

**Decision flips** (`facts["decision_flips"]`, diffs R0's and the
candidate's `fills.parquet` directly on `(fold, station_code, date)` — not a
`rowpreds` proba-threshold crossing, which is a different, uncalibrated fit
from the one that drove the realised decision): run-level **359 flips / 83
decisions**. `flips` overstates independent evidence — one real flip
cascades into a run of later differing fills at the same station as the
tank's state diverges, which `decns` collapses over this run's own
`cascade_window_days` = 7. **Read `decns`, not `flips`, as this table's
sample size**, and note it is not comparable across runs at different
cadences (a coarser cadence reports a systematically LOWER `decns` for the
identical underlying divergence).

Timing is rendered as **regret** (`decision_flips["regret"]`,
`horizon_days` = 13, `cadence_days` = **1**): `price paid − the cheapest
price that same station reached inside the tank's feasible wait`, so `0` is
the attainable optimum and lower is better. Unlike the flip-CPL columns it
is defined identically on both arms and cycle-normalised, so it pools. The
horizon is a fixed ceiling no real fill attains, making regret levels
conservative by construction; regret is comparable only WITHIN one cadence.
`flip_cpl_delta` is retained in `facts.json` but **no longer rendered**
(`fps-2js`) — `flip_cpl_baseline` and `flip_cpl_candidate` pool disjoint
fills on different days at different points in the price cycle, so their
difference has no stable denominator.

| fold | regime | flips | decns | litres R0 | litres cand | regret R0 | regret cand | regret Δ | Δ interval (2·SE) | run_contribution_cpl |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | shock | 8 | 4 | 64.3 | 182.1 | 14.33 | 14.80 | +0.47 | [-12.98, +13.92] | +0.0693 |
| 2 | normal | 66 | 11 | 564.3 | 385.7 | 7.74 | 14.66 | +6.92 | [-5.04, +18.88] | +0.0474 |
| 3 | shock | 113 | 10 | 767.9 | 300.0 | 11.72 | 7.94 | -3.78 | [-17.09, +9.53] | -0.0851 |
| 4 | shock | 33 | 12 | 317.9 | 89.3 | 14.89 | 12.32 | -2.57 | [-12.67, +7.54] | -0.0801 |
| 5 | normal | 20 | 7 | 192.9 | 107.1 | 12.80 | 15.56 | +2.76 | [-2.56, +8.08] | -0.0297 |
| 6 | normal | 7 | 3 | 42.9 | 46.4 | 8.02 | 3.71 | -4.31 | [-15.88, +7.26] | +0.0107 |
| 7 | normal | 6 | 3 | 82.1 | 0.0 | 15.37 | — | — (`one arm only`) | — | -0.0081 |
| 8 | normal | 21 | 3 | 164.3 | 185.7 | 8.35 | 13.25 | +4.90 | [-15.76, +25.56] | +0.0748 |
| 9 | normal | 8 | 3 | 64.3 | 117.9 | 11.00 | 8.05 | -2.95 | [-15.98, +10.09] | +0.0302 |
| 10 | normal | 36 | 7 | 282.1 | 339.3 | 12.72 | 8.18 | -4.54 | [-15.77, +6.70] | -0.0450 |
| 11 | normal | 5 | 3 | 78.6 | 25.0 | 24.36 | 19.43 | -4.94 | [-26.20, +16.33] | -0.0350 |
| 12 | normal | 12 | 7 | 217.9 | 3.6 | 21.48 | 20.00 | -1.48 | [-6.85, +3.89] | -0.0400 |
| 13 | normal | 18 | 6 | 171.4 | 175.0 | 12.73 | 5.39 | -7.34 | [-26.27, +11.59] | -0.1251 |
| 14 | shock | 6 | 4 | 67.9 | 50.0 | 4.37 | 12.57 | +8.20 | [-1.66, +18.07] | +0.0060 |
| **all** | — | **359** | **83** | **3078.6** | **2007.1** | **12.29** | **10.96** | **-1.34** | **[-5.83, +3.16]** | **-0.2098** |

**Every resolvable regret delta in this table sits inside its own 2·SE
interval — 13 of 13, plus the pooled `all` row
(`regret_cpl_delta_inside_own_se`: `true` throughout).** Per
`docs/routines/dossier.md`, a cell marked `inside_own_se` must never be cited
as evidence in the Judgement section, so nothing in this table corroborates or
contradicts the zone verdict above. One cell is not resolvable at all and its
reason is stated verbatim rather than blanked: **fold 7 is `one arm only`** —
6 baseline-only fills against zero candidate-only ones, i.e. "the candidate
stopped buying there" rather than "the two arms swapped timing". That is new
in the #359 regen; pre-regen fold 7 had fills on both arms and a computable
-7.03 delta. It is not the same as a resolved zero.
The pooled reading is the only run-level statement this table can make, and it
is that **both arms buy about equally well relative to what they could have
reached** (12.29 vs 10.96 c/L of regret against a ±4.50 interval), consistent
with a -0.2077 c/L headline.

Two things the table surfaces that are not deltas. **Litres:** the flip-only
fills are 3078.6 L on R0 against 2007.1 L on the candidate (248 vs 111
flip-fills) — a 35% gap. `pooled_cpl` is spend-weighted, so a 3.57 L top-up
and a 42.86 L fill count identically as flips but not in the CPL; the two
arms are not diverging on equal volume. **Dark fills:** `dark_fill_days` =
**0** of 359 scored. Pre-#359 this run recorded **64** dark fills in folds
**4–7** (the Blue Mountains preferred-station outage, then also behind this
run's 9.63% provider miss rate — now 4.66%), scored at the forward-filled
price the tank simulator had bought at. `fps-2i4`/#356 stopped the realised
backtest transacting at a station's stale frozen price through a closure, so
those station-days are no longer bought at all. Folds 4–7 are exactly the
folds whose cells moved in the regen, and only those.

**`run_contribution_cpl`** is what each fold's divergence is actually worth
at the scale of the whole run — a litres-weighted shift-share of that fold's
own `delta_cpl_own` (the whole fold, matching and flipped fills alike), not
of the flip-only figures. It reconciles exactly:
`run_contribution_total_cpl` **-0.2098** + `composition_residual_cpl`
**+0.0021** = **-0.2077**, the headline `delta_cpl_held`. The residual is
real, not rounding — a litres-weighted average of per-fold ratios is not the
ratio of pooled totals when the arms' litres mix differs fold to fold. This
run's `tau_diverges_any` is `None` (**unavailable, never read as "checked,
agrees"**), so the residual must be stated as composition drift **and
possibly tau drift**, not composition drift alone.

Summed by regime, `run_contribution_cpl` splits **normal -0.1199** (10
folds, 53 decns) against **shock -0.0899** (4 folds, 30 decns) — i.e. the
normal cell supplies the larger share of the headline, which is the emphasis
the candidate's signature predicted and the opposite of the `delta_cpl_own`
zone read that grades it. **This does not overturn the zone verdict**, which
is graded on `delta_cpl_own` per the routine, and it is not independent
corroboration either: fold 13 alone contributes **-0.1251** of that normal
**-0.1199**, so the other nine normal folds net **+0.0052** between them and
the cell is one fold plus cancellation. The shock cell has the same shape,
though less starkly than pre-regen — folds 3 and 4 supply -0.1652 (fold 4
having grown from -0.0687 to -0.0801), folds 1 and 14 hand back +0.0753. The
normal cell is a single-fold artefact and the shock cell is a two-fold one,
which is the reason for caution on the zone read, not a second arbiter to
weigh against it.

**Seed stability:** 4 cells exceed 5× the cohort-median seed_std. Three are
on R0 (folds 2 and 3, `all` cohort; fold 2, `hard25` cohort) — a property of
the window, not the candidate. **One is on the CANDIDATE arm** (fold 3,
`hard25` cohort), so per the fold × arm rule, here is that comparison rather
than prose:

| fold | cohort | R0 seed_std | candidate seed_std |
|---|---|---|---|
| 2 | all | 0.0458 (flagged, 5.61×) | not flagged |
| 3 | all | 0.0451 (flagged, 5.53×) | not flagged |
| 2 | hard25 | 0.1326 (flagged, 5.17×) | not flagged |
| 3 | hard25 | not flagged | **0.1564 (flagged, 6.10×)** |

Fold 3's hard25 cohort is flagged on the candidate arm only — R0 was not
flagged there. That is the within-fold R0-vs-candidate jump the routine
treats as a redundancy tell: the added columns may be destabilising an
otherwise-ordinary fit in exactly the fold with this candidate's most
favourable `delta_cpl_own` reading (-1.09 c/L). Fold 3 was `normal` under
the old fixed shock-fold set; under the corrected empirical set it is
`shock` and is now the single largest driver of the (unresolved) shock-zone
verdict — see Judgement below — so this candidate-arm instability sits
directly inside the zone the verdict rests on, not off to the side of it.
Every `seed_std` above is WFCV **log-loss**, not CPL — never read this as
instability in `delta_cpl_own`, which is not seed-measured at all.

**Validation:** PIT passed (differential PIT truncation test found no leak).
INPUTS check passed. Candidate column NaN rate 0.0% for both columns.
`extra_feature_provider_miss_rate`: **4.66%** (278 misses against 5693 hits)
— this is the REPLAY's own coverage, distinct from the clean 0.0% offline NaN
rate above. Pre-#359 it was 9.63% (607/6300), the 6300 being the full
5 stations × 14 folds × 90 days grid; the regen leaves the hits unchanged and
removes 329 misses, because `fps-2i4`/#356 stopped the arms visiting a closed
station's days at a stale frozen price at all. It is the same batch-wide gap
documented in
`experiments/candidates/batch1/lga_trough_propagation/README.md` § Review
addendum (Blue Mountains preferred-station outages, folds 1–8 only, not
specific to this candidate).

**Noise floor** (`experiments/batches/batch1/noise_floor.json`, 10 draws at
arity 3 — **wider than this candidate's own 2-column arity**,
`floor_arity_exceeds_run`: **true**, excess 1 column. This is a stricter
ruler than the candidate's own arity would require, which only raises the
bar, never lowers it — a candidate that clears it has not been shown to
fail against an easier one, only a harder one): band mean +0.0251 c/L, band
std 0.0999 c/L. The candidate's -0.2077 c/L sits at the **100th
percentile** of the 10 observed draws
(`candidate_percentile_better_than_noise` = 100.0), and the parametric call
is `z = -2.3295` against threshold `t = 1.9226` — **past the bar, in the
favourable direction** (`abs(z)` > `t`, sign favourable).

Bar in c/L, all four required components: **`single_candidate_bar_cpl_held`
= -0.1670 c/L** (the detection threshold in the units the rest of the
project quotes, comparable against `results.csv`'s 0.03–0.26 c/L history).
**`bar_draw_count_shift_cpl_held` = -0.0278 c/L** (how much of that bar is
draw-count thinness, the `df = n_draws - 1` t-penalty). **`bar_reuse_shift_cpl_held`
= 0.0** (no reuse-hardening component at this arity/bank). **`floor_arity_excess`
= 1** (the ruler is 1 column wider than the run).

**Redundancy** (`facts["redundancy"]`, `available`: true): `block_r2` =
0.574, `max_column_r2` = **0.583** (`tgp_cycle_displacement_cents`) against
49 lock predictors, with `tgp_cycle_displacement_frac_amp` close behind at
0.565. Both columns are meaningfully reconstructible from the existing lock
— `predictors_dropped` (columns too collinear to include as predictors in
the redundancy fit) is entirely the `days_since_trough_entry_*` LGA family
(Bayside, Botany Bay, Hunters Hill, Lane Cove, Waverley), which is the same
trough-timing family this candidate's mechanism (floor displacement since
the retail peak) is adjacent to.

![](per_fold_delta_bars.png)
![](seed_mean_vs_median.png)
![](realised_cpl_by_fold.png)
![](tau_sweep.png)
![](candidate_over_time.png)
![](cycle_phase_breakdown.png)
![](external_series_overlay.png)

## Judgement

**Signature grading: contradicted — unchanged by the 2026-08-30 shock-fold
correction (`fps-hnp`), though the specific fold responsible has moved.**
See the Correction note below for what changed. The realised headline is
favourable and clears this batch's single-candidate noise band on the
good-direction side (z = -2.3295 past t = 1.9226, against a ruler one
column wider than the candidate's own arity — this z is computed from the
headline realised delta, which is unaffected by the shock-fold set) — a
stronger result than either prior TGP expression, both of which sat inside
their noise bands. But the *where* of the effect is the opposite of what
was predicted: the candidate's own signature named concentration on
**normal** folds as the expected pattern and explicitly pre-registered a
**shock**-concentrated win as evidence against its wholesale-lead
mechanism. What was measured is exactly that, still: shock delta_cpl_own
(-0.31) is larger in magnitude than normal (-0.16), and the zone's own
`resolved` flag reads `false` for the same reason. The shock aggregate is
not a clean shock story on its own terms — 2 of 4 shock folds (now 1 and 14)
hurt the candidate — though the strength of the **fold 3** single-driver
reading has dropped: fold 4's own cell grew to -1.21 in the regen, so fold 3
now accounts for 64% of the shock number rather than 95%, and excluding it no
longer flips the sign (see Facts above for the leave-one-out recomputed on
both sides). Either way, as measured, the
pattern the candidate itself said would falsify its mechanism is the pattern
that showed up — and the regen made that pattern slightly *more* pronounced,
not less.

Per `docs/routines/dossier.md`'s guardrail, none of the per-fold or
decision-flip colour above overrides this — it explains why the aggregate
looks the way it does, not a second arbiter.

**not_tested:**

- Whether the shock-fold gain is fold-3-led (-1.09 c/L, 64% of the shock-zone
  magnitude, with fold 4's -1.21 c/L supplying most of the rest) rather than a
  genuine shock mechanism — 2 of the 4 shock folds (1, 14) actually hurt the
  candidate, so the regime aggregate is not a uniform shock story. (Two
  earlier revisions of this line have named a single driver: fold 13 under the
  old fixed shock-fold set, then fold 3 under the corrected set with pre-#359
  figures, where excluding it flipped the sign. The #359 regen moved fold 4 to
  -1.21, so fold 3's share fell from 95% to 64% and the sign no longer flips —
  which is itself the point: how much "one fold carries it" has not been
  stable across two re-measurements.)
- Whether testing the actual claim — decision timing keyed to the
  *magnitude* of floor displacement since the peak, not the coarse
  shock/normal regime label — would recover the predicted normal-fold
  concentration. No per-axis breakdown by displacement magnitude was built
  for this run; `per_axis` is `null`.
- Whether the ~58% max per-column R² against the lock (concentrated
  entirely in the `days_since_trough_entry_*` family) means this is
  substantially a proxy for existing trough-timing columns rather than new
  information — the redundancy check flags the column, not the mechanism,
  and the one candidate-arm seed-instability flag (fold 3, hard25) now
  lands in the same shock-fold cell that alone drives the (unresolved)
  zone verdict, which sharpens rather than merely coexists with this
  question.
- Per `fps-1l1`: whether the model's own SHAP orientation would corroborate
  or refute the wholesale-lead story — `facts.json` has no field for this,
  so the mechanism itself (as opposed to the regime-location prediction) is
  untestable with this pipeline as built.

**Recommendation:** not a graduation call on this run's own terms — the
predicted mechanism's most specific, falsifiable claim (normal-fold
concentration) is contradicted by its own criterion, and this holds under
the corrected shock-fold set — but the headline is real, favourable, and
clears noise more convincingly than either prior TGP expression. Worth a
second look if a displacement-magnitude axis (rather than the coarse
regime split) can be built, and worth checking against the
`days_since_trough_entry_*` family directly before treating this as new
information rather than a proxy for it.

## Followups

- A per-axis breakdown keyed on displacement *magnitude* (not just the
  regime label) would let a future run test the actual predicted mechanism
  directly rather than through the coarse shock/normal proxy.
- `fps-1l1` (SHAP importance/orientation in `facts.json`) remains open and
  would let a future run test the wholesale-lead mechanism itself.

## Correction note — 2026-08-30 (`fps-hnp`, backfill of PR #350 / `fps-3tu`)

PR #350 replaced the batch's fixed shock-fold index (`SHOCK_FOLDS =
{1, 4, 9, 13}`) with a per-batch empirical set derived from the baseline's
own val-fold log-loss. batch1's corrected set is **`{1, 3, 4, 14}`**
(`experiments/batches/batch1/shock_folds.json`) — folds 3 and 9 swap
regime labels, as do 13 and 14, relative to the old fixed set. This
dossier's zone verdict (`resolved: false`, "contradicted") is unchanged by
the correction — the margin narrows (-0.1392/-0.3405 →
-0.1603/-0.2884 c/L, both pairs pre-#359; the current post-regen
corrected-set pair is -0.1613/-0.3079) but the sign relationship between the
two zones does not flip. What does change: **fold 3, not fold 13, is now the
single fold driving the shock-zone reading** (fold 13 moved to `normal`) —
though the #359 regen has since spread that driver across folds 3 and 4, see
the Facts section — and the one
candidate-arm seed-instability flag in this run (fold 3, hard25 cohort)
now sits inside that same driving fold rather than off to the side of it
— see the Facts and Judgement sections above, rewritten in place. The
realised CPL headline (-0.2015 c/L as measured then, -0.2077 after the #359
regen; past the noise band either way) and its z-score are unaffected — only the regime-axis zone grade and its supporting
per-fold/flip/seed commentary depended on the shock-fold set. This
backfill reused the run's already-persisted `fills.parquet` and
`rowpreds.parquet`; no re-run of the realised backtest was needed.

## Re-render note — 2026-08-31 (`fps-j6w`)

**The figures in this section are as they stood on 2026-08-31 and are superseded by the
Facts section above** — kept as the record of what that pass corrected. All of them were
re-measured in the #359 regen; see the Refresh note below for the current values. The
substantive correction it made (the flip-CPL vote count was inadmissible) is unaffected.

The **Decision flips** section above was rendering the pre-`fps-2js` table shape
(`n_baseline_only` / `n_candidate_only` / `flip_cpl_baseline` / `flip_cpl_candidate` /
`flip_cpl_delta`). Every field the current `docs/routines/dossier.md` spec asks for was
already present in this run's `facts.json` — it was emitted and never rendered. No new
computation: the section was re-rendered from the committed artifact.

What the re-render changes for a reader:

- **`decns` alongside `flips`.** The run is 423 flips but **92 decisions** once cascades
  are collapsed at this run's `cascade_window_days` = 7. Folds carry 3–13 decisions each.
  The old table's raw flip counts materially overstated per-fold evidence.
- **Regret replaces `flip_cpl_delta`** as the rendered timing column, with each delta
  beside its own 2·SE interval. **All 14 folds and the pooled `all` row are
  `inside_own_se`.** The superseded prose cited a "+6.04 c/L shock-flip average, now
  UNFAVOURABLE" and a "-2.94 c/L normal, now FAVOURABLE" and built a caution on their
  divergence — a vote count over cells that cannot resolve their own numbers, which the
  routine forbids citing. Those sentences are gone.
- **`run_contribution_cpl` is now shown**, reconciling exactly to the headline
  (-0.1937 + -0.0078 residual = -0.2015). Summed by regime it reads normal **-0.1158** /
  shock **-0.0780** — the opposite emphasis to the `delta_cpl_own` zone read that grades
  this candidate. It does not overturn that grade (the routine grades the zone on
  `delta_cpl_own`) and is not corroboration either: fold 13 alone supplies -0.1194 of the
  normal -0.1158, so both regime cells are single-fold artefacts.
- **Litres and dark fills surfaced.** Flip-only volume is 3517.9 L (R0) vs 2375.0 L
  (candidate), a 32% gap on a spend-weighted metric; `dark_fill_days` = 64 of 423 in folds
  4–7, the same Blue Mountains outage behind this run's 9.63% provider miss rate.

**The Judgement section is unchanged and did not depend on any of the removed figures** —
it cites `delta_cpl_own` and the noise band, both untouched by this re-render. The
`not_tested` list and the 2026-08-30 correction note stand as written.

## Refresh note — 2026-09-03 (`fps-3ug`)

Every realised-backtest figure in the Facts and Judgement sections was refreshed from
this run's committed `facts.json` as regenerated by PR #359 (`fps-32h`), which re-ran
the realised backtest under the `fps-2i4`/#356 price-accessor fix and the #358
oracle-DP fix. No re-grading: the **contradicted** verdict and the favourable-headline
reading both survive, and the regen strengthens rather than weakens the contradiction.

**Every CPL figure in this note is at this run's cadence, `50/3.571/1d/10%`**
(`fps-15c`) — the same cadence on both sides of every → below, since the regen
re-ran the identical tank config.

What moved:

- **Headline** -0.2015 → **-0.2077** c/L; R0 187.8211 → 188.0376, candidate 187.6196 →
  187.8299. `z` -2.2676 → **-2.3295** against the same 1.9226 — still past the
  single-candidate bar in the favourable direction, and by slightly more. This remains
  batch1's only candidate clearing that bar, and the only one whose signature is
  contradicted while doing so.
- **Zone** target -0.1603 → **-0.1613**, other -0.2884 → **-0.3079**, n_fills 1493/684
  → **1389/674**. `resolved: false` and the shock-more-negative-than-normal ordering —
  the verdict's load-bearing facts — unchanged, on a slightly *wider* margin.
- **Per-fold: only folds 4–7 moved**, which is where the 64 pre-regen dark fills sat.
  Fold 3's -1.0892 and fold 13's -1.6021 are byte-identical. Fold 4 moved most,
  -0.9512 → **-1.2103**.
- **`dark_fill_days` 64 → 0**, and the provider miss rate 9.63% (607) → **4.66% (278)**.
- **Flips 423 → 359 / decns 92 → 83.** One shape change: **fold 7 is now `one arm
  only`** (6 baseline-only fills, zero candidate-only), where pre-regen it had a
  computable -7.03 regret delta. The table's conclusion is unchanged — every resolvable
  delta is still `inside_own_se`, so still nothing citable — but it is now 13 of 13
  rather than 14 of 14.
- **Reconciliation** -0.1937 + -0.0078 → **-0.2098 + +0.0021**; the residual changed
  sign at a negligible size. By regime, normal -0.1158/shock -0.0780 → **normal
  -0.1199/shock -0.0899**; fold 13 supplies -0.1194 → **-0.1251** of the normal cell.
- **Pooled regret** 10.80/9.26 (Δ -1.54) → **12.29/10.96 (Δ -1.34)**, still
  `inside_own_se`. Flip-only litres 3517.9/2375.0 (32%) → **3078.6/2007.1 (35%)**.
- **The one claim that materially weakened:** "excluding fold 3 takes the shock zone to
  +0.005, essentially a sign flip". That was litre-weighted from the pre-regen
  `fills.parquet` and cannot be restated. Recomputed fill-count-weighted from
  `facts.json` on both sides it reads **-0.0097 pre-regen against -0.0878 post-regen**
  (fill-weighted shock zone -0.2458), because fold 4's cell grew to -1.2103. Fold 3's
  share of the shock zone falls from 95% to 64% — still most of it, no longer nearly all,
  and removing it no longer flips the sign. The Facts, Judgement and `not_tested`
  sections are restated with those figures. **This weakens a
  caveat against the verdict, so the verdict is if anything better supported than
  before** — the surviving caution is the count-based one, that 2 of 4 shock folds hurt
  the candidate.
- **Unchanged by the regen, and verified so:** every WFCV / log-loss / seed field —
  including the fold-3 hard25 candidate-arm 6.10× flag the Judgement cites — and the
  redundancy block (`block_r2` 0.574, `max_column_r2` 0.583).
