# batch1 / station_descent_dynamics

- **Date:** 2026-08-15 (snapshot) / dossiered 2026-08-25 / regime labels corrected
  2026-08-30 (`fps-hnp`) / flip table re-rendered 2026-08-31 (`fps-j6w`) / realised
  figures refreshed to the post-#359 regen 2026-09-03 (`fps-3ug`)
- **Branch:** main
- **SHA:** 12dc7629f1f440bbf71fa7f50304e6adb781d048 (the #359 regen run; the original
  2026-08-24 run was f536e98b9c183e754620d46ee0a6ef13681e7c35 — the model fits are
  byte-identical between the two, only the realised backtest was re-run)
- **Status:** done — **inconclusive**
- **Bead:** fps-6to
- **Cadence:** `50/3.571/1d/10%` (batch1's canonical daily cadence)

## Hypothesis

The retail descent is a sequence of discrete cuts, and the SHAPE of a
station's own cut sequence says how much of the descent is left. A station
still cutting 1-2 c/day is mid-descent; a station whose recent pace has
collapsed toward zero while its 14-day pace was steep has arrived at its
floor and is now exposed to the restoration. The lock cannot express this:
every dynamic column in it is network-level (one value per date), so the
model sees the Sydney-average clock and each station's contemporaneous
LEVEL, but never the station's own rate of change.
`station_minus_last_min_cents` is the closest substitute and is a distance
to a NETWORK trough, not a station velocity. Columns:
`station_px_change_3d`, `station_px_change_14d`, `station_descent_decel`.

**Predicted signature:** decision-timing, not accuracy — the model should
stop waiting earlier on rows where `station_descent_decel` has gone positive
(recent pace slower than the trailing pace) at moderate `cycle_pct_through`,
and keep waiting longer where it's negative (cuts still accelerating). A
MODEST number of buy/wait flips, concentrated in elongated cycles where the
network clock and the station's own pace diverge most — i.e. the
shallow-vs-steep-within-elongated distinction #214's oracle diagnostic found
invisible in `network_px_std` until phase > 1.4. The per-fold contribution
should be largest in folds 3 and 7 (the elongation-axis folds) and
near-zero elsewhere; a uniform improvement across all 14 folds would instead
mean the columns are working as generic momentum, not a descent-shape read.

## How to invoke this script

```bash
PYTHONPATH=. uv run python -m experiments.pipeline.runner \
  --batch-dir experiments/batches/batch1 \
  --candidate experiments/candidates/batch1/station_descent_dynamics.py
```

## Facts

Transcribed from `facts.json` — no new arithmetic.

**Provenance:** batch `batch1`, snapshot 2026-08-15, seeds [42, 43, 44, 45,
46], wall 1062.8s, status `graded`, 54-column baseline (`54:1a6ec2d84a69`).

**Headline (realised CPL, held τ — the arbiter):**

| | R0 | candidate | delta |
|---|---|---|---|
| CPL held | 188.0376 | 188.0138 | **-0.0237** |

`effect_resolved`: **true**.

**WFCV log-loss** (descriptive colour, NOT the arbiter): Δll_all mean
-0.00057, Δll_hard25 mean -0.02066. Both nominally favour the candidate.

**Zone (target = folds 3/7, the candidate's own declared elongation axis):**

| | target (folds 3, 7) | other |
|---|---|---|
| delta_cpl_own | +0.0148 | -0.0298 |
| n_fills | 380 | 1847 |

`resolved`: **false** — the target zone shows a small WORSENING, not the
predicted improvement; the improvement instead sits in the "other" fills.

**Per-fold** (regime column reflects the corrected empirical shock-fold set
`{1, 3, 4, 14}`, not the fixed `{1, 4, 9, 13}` this dossier originally used
— folds 3 and 9 swap regime labels, as do 13 and 14 — see the Correction
note below the Judgement section; this candidate's TARGET grades on folds
3/7, not regime, so `resolved` above is unaffected, but the descriptive
per-regime split below is not):

| fold | regime | n_fills | delta_cpl_own |
|---|---|---|---|
| 1 | shock | 169 | -0.2459 |
| 2 | normal | 200 | +0.3586 |
| 3 | shock | 191 | -0.1517 |
| 4 | shock | 226 | +0.1577 |
| 5 | normal | 144 | +0.5266 |
| 6 | normal | 114 | -0.1270 |
| 7 | normal | 177 | +0.2290 |
| 8 | normal | 88 | -0.1318 |
| 9 | normal | 112 | +0.1273 |
| 10 | normal | 95 | +0.2791 |
| 11 | normal | 161 | -0.5651 |
| 12 | normal | 185 | -0.2809 |
| 13 | normal | 146 | **-1.2787** |
| 14 | shock | 169 | +0.3727 |

No cell is suppressed (`min_row_cell_n` = 30, smallest cell here is 88).

**Per-regime:** shock **+0.0504** c/L (n=770), normal **-0.0694** c/L
(n=1424) — **the sign of both cells flips relative to the old fixed shock
set** (which read shock -0.2837, normal +0.0706 — pre-#359 figures, kept as
the record of what the correction moved), because fold 13 (this run's single
largest favourable cell, -1.28 c/L) moves from shock to normal under the
correction. See the Correction note below.

`per_axis` is `null` — this candidate declared no `add_axis` beyond the
folds-3/7 zone check above, so there is no path-coupled per-row zone to
caveat here.

**Seed stability:** 2 cells exceed 5× the cohort-median seed_std, both on
run R0, `all` cohort — folds 2 (5.99×) and 3 (5.90×). Both are R0 (baseline)
flags, not candidate-arm flags.

**Validation:** PIT passed (differential PIT truncation test found no
leak). INPUTS check passed. Candidate column NaN rates:
`station_px_change_3d` 0.18%, `station_px_change_14d` 0.85%,
`station_descent_decel` 0.85%.

**Noise floor** (`experiments/batches/batch1/noise_floor.json`, 10 draws at
arity 3 — matches this candidate's 3-column arity, `floor_arity_exceeds_run`:
false): band mean +0.0251 c/L, band std 0.0999 c/L. The candidate's -0.0237
c/L sits at the **60th percentile** of the 10 observed noise draws
(`candidate_percentile_better_than_noise` = 60.0), and the parametric
single-candidate call is `z = -0.489` against a threshold of `t = 1.923`:
well inside the band. The band's own computed single-candidate bar is
**-0.16703 c/L** (`single_candidate_bar_cpl_held`).

**Decision flips** (`facts["decision_flips"]`, diffs R0's and the
candidate's `fills.parquet` directly on `(fold, station_code, date)` — not a
`rowpreds` proba-threshold crossing, which is a different, uncalibrated fit
from the one that drove the realised decision): run-level **337 flips / 99
decisions**. `flips` overstates independent evidence — one real flip
cascades into a run of later differing fills at the same station as the
tank's state diverges, which `decns` collapses over this run's own
`cascade_window_days` = 7. **Read `decns`, not `flips`, as this table's
sample size**, and note it is not comparable across runs at different
cadences.

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
difference has no stable denominator. **The `litres R0` / `litres cand`
columns below are FLIP-ONLY litres, not fold totals** — the fold-total
identity that the deferral analysis further down rests on is a different
quantity.

| fold | regime | flips | decns | litres R0 | litres cand | regret R0 | regret cand | regret Δ | Δ interval (2·SE) | run_contribution_cpl |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | shock | 21 | 3 | 160.7 | 0.0 | 2.49 | — | — (`one arm only`) | — | -0.0172 |
| 2 | normal | 85 | 15 | 378.6 | 403.6 | 6.76 | 14.38 | +7.61 | [-1.72, +16.95] | +0.0264 |
| 3 | shock | 37 | 9 | 310.7 | 485.7 | 8.14 | 7.65 | -0.49 | [-7.35, +6.37] | -0.0118 |
| 4 | shock | 27 | 10 | 160.7 | 196.4 | 10.37 | 16.75 | +6.38 | [-5.89, +18.65] | +0.0104 |
| 5 | normal | 21 | 7 | 135.7 | 253.6 | 14.65 | 16.41 | +1.76 | [-3.56, +7.08] | +0.0328 |
| 6 | normal | 11 | 5 | 89.3 | 164.3 | 12.80 | 11.96 | -0.84 | [-10.94, +9.26] | -0.0078 |
| 7 | normal | 7 | 5 | 7.1 | 125.0 | 17.50 | 13.94 | -3.56 | [-8.77, +1.64] | +0.0139 |
| 8 | normal | 15 | 7 | 67.9 | 121.4 | 1.53 | 1.45 | -0.08 | [-2.10, +1.94] | -0.0102 |
| 9 | normal | 12 | 3 | 110.7 | 146.4 | 13.35 | 13.51 | +0.15 | [-11.90, +12.20] | +0.0099 |
| 10 | normal | 39 | 7 | 417.9 | 453.6 | 12.03 | 10.87 | -1.16 | [-11.76, +9.43] | +0.0199 |
| 11 | normal | 9 | 4 | 82.1 | 64.3 | 24.13 | 11.00 | -13.13 | [-31.00, +4.74] | -0.0395 |
| 12 | normal | 14 | 9 | 139.3 | 200.0 | 23.18 | 14.56 | **-8.62** | **[-16.98, -0.26]** | -0.0215 |
| 13 | normal | 26 | 8 | 153.6 | 310.7 | 12.42 | 12.43 | +0.01 | [-15.21, +15.23] | -0.0997 |
| 14 | shock | 13 | 7 | 128.6 | 203.6 | 9.00 | 12.23 | +3.23 | [-3.72, +10.18] | +0.0289 |
| **all** | — | **337** | **99** | **2342.9** | **3128.6** | **10.80** | **12.06** | **+1.26** | **[-2.15, +4.68]** | **-0.0654** |

**Twelve of the thirteen resolvable regret deltas sit inside their own 2·SE
interval, as does the pooled `all` row.** Per `docs/routines/dossier.md`, an
`inside_own_se` cell must never be cited as evidence in the Judgement
section. **Fold 12 is the single exception** — Δ -8.62 c/L against
`[-16.98, -0.26]`, an interval that excludes zero — and it is the only cell
in this dossier that may be cited as resolved. Fold 1 is not resolvable at
all: `one arm only`, which is the regret-side expression of the same
zero-candidate-fills shape analysed below.

**The pooled row is the most informative line in this table, and it points
the other way from the headline.** Regret is a WHEN measure, same station by
construction. R0's pooled regret is **10.80 c/L** and the candidate's is
**12.06 c/L** — Δ **+1.26**, i.e. the candidate times its purchases
*worse* — while the realised headline is **-0.0237 c/L in the candidate's
favour**. That is not a contradiction and it is the most useful thing this
pair says: whatever this candidate gains, it does **not** gain by buying at
better moments at the same station. It gains on **what** was bought —
volume and composition. Flip-only litres corroborate: 2342.9 L on R0
against 3128.6 L on the candidate, a **34% gap**, on a spend-weighted
metric. (The pooled regret delta is itself `inside_own_se`, ±3.41, so this
is a direction, not a resolved magnitude.)

**Dark fills:** `dark_fill_days` = **0** of 337 scored — and that zero is the
single largest thing the #359 regen changed in this dossier. The pre-regen run
recorded **102** dark fills in folds **4–7** (the Blue Mountains
preferred-station outage, then batch1's largest dark-fill count), scored at the
forward-filled price the tank simulator had bought at. `fps-2i4`/#356 stopped
the realised backtest transacting at a station's stale frozen price through a
closure, so those station-days are no longer bought and no longer appear as
fills at all. Folds 4–7 are exactly the folds whose per-fold cells moved in the
regen, and only those; their evidence weight is still not what a row count
suggests, but the mechanism is now exclusion rather than stale-price purchase.

**`run_contribution_cpl`** is what each fold's divergence is worth at the
scale of the whole run — a litres-weighted shift-share of that fold's own
`delta_cpl_own` (the whole fold, matching and flipped fills alike), not of
the flip-only figures. It reconciles: `run_contribution_total_cpl`
**-0.0654** + `composition_residual_cpl` **+0.0417** = **-0.0237**, the
headline `delta_cpl_held`. **This run has the largest composition residual in
batch1, and post-#359 it is 176% of the headline's own size — larger than the
effect it sits beside** — which is consistent
with the regret reading above: a litres-weighted average of per-fold ratios
diverges from the ratio of pooled totals precisely when the two arms' litres
mix differs fold to fold, and here it does. This run's `tau_diverges_any` is
`None` (**unavailable, never read as "checked, agrees"**), so that residual
is composition drift **and possibly tau drift**.

Summed by regime: **normal -0.0756** (10 folds, 70 decns) against **shock
+0.0102** (4 folds, 29 decns). Fold 13 alone supplies **-0.0997** of the
normal total, so the other nine normal folds net **+0.0241** between them —
the normal cell is one fold plus cancellation.

**Fold 1, analysed** (`one arm only` above). *The fill-level counts in this
paragraph were computed from the pre-#359 `fills.parquet`, which is not a
committed artifact and so cannot be re-derived here. They are carried forward
unchanged, and the regen gives no reason to doubt them: every fold-1 cell that
IS in `facts.json` is untouched by it — 169 fills, delta_cpl_own -0.2459, 21
baseline-only / 0 candidate-only, regret 2.49 c/L. Only folds 4–7 moved.*
R0 has 21 baseline-only fills
there and the candidate has zero candidate-only fills to compare against, so
neither the flip nor the regret delta can be formed. The shape is **deferral, not
reduced buying**: total litres are identical (1539.286 L in both arms —
consumption is fixed, so the arms always buy the same fuel), and the
candidate's fill *dates* are a strict subset of R0's. Reconstructing tank
level day by day, the candidate is never AHEAD of R0 on cumulative litres
in this fold (behind on 14 days, level on 76) — it runs a leaner tank, buys
later, and because its later buys are large it never needs R0's string of
3.571 L top-ups (exactly one day's consumption at this cadence — the
"wait" outcome). Fill-level: 153 byte-identical, 16 same-date-different-
volume, 21 R0-only dates, 0 candidate-only. Example, station 18517:
R0 takes 46.4 L @165.9 on 2021-11-17; the candidate takes 21.4 L that day
and 28.6 L on 11-18 @**155.9** — same fuel, one day later, 10 c/L cheaper.

Note fold 13 is the mirror image (candidate ahead on 41 days, level on 44,
behind on 5 — pre-#359 fill-level counts, same provenance caveat as above;
fold 13's `facts.json` cells are likewise untouched by the regen): the same
three columns produce opposite timing bias in the two folds, which is
consistent with them reading descent SHAPE rather than carrying a fixed
buy-earlier or buy-later tilt.

**Instrument caveat:** `decision_flips` keys on `(fold, station_code,
date)`, so a fill that occurs on the same day in both arms at a DIFFERENT
volume is counted as common, not as a flip. Measured on the **pre-#359** run,
that hid 150 volume-only changes behind its 439 counted date-flips — fold 12
was the sharpest case (13 volume-only changes against only 14 date-flips), and
fold 1's null is entirely an artifact of this blind spot. Those two counts come
from `fills.parquet` and were not re-derived for the regenerated fills, so they
are stated against the run they were measured on rather than against the 337
flips in the table above; the blind spot itself is structural and unchanged.
Tracked as `fps-gzz`.

![](per_fold_delta_bars.png)
![](seed_mean_vs_median.png)
![](realised_cpl_by_fold.png)
![](tau_sweep.png)
![](candidate_over_time.png)
![](external_series_overlay.png)

## Post-hoc analysis (2026-08-25, NOT pipeline-generated)

Added while reviewing this dossier, to separate two readings the graded run
cannot tell apart: are the columns inert, or is the edge real but too small
and too fat-tailed for 14 folds to resolve? Script and output:
`posthoc_decel_conditional.py` / `.log`. No new data, no refit — a
conditional-distribution read of this run's own `rowpreds.parquet`, across
all **714 stations / 770,799 rows**, far wider than the 5 preferred
stations the backtest fills cover. **Unaffected by the #359 regen:** that
change re-ran the realised backtest only — every WFCV / log-loss / seed field
in `results.json` is byte-identical across it — so the row predictions this
section reads are the same ones it was written against. Forward 3-day change is recovered from
the backward change observed three days later.

Base rate: mean forward-3d **-0.06 c/L**, P(cliff > +25 c/L) **5.37%**.

Within steep descents (`station_px_change_14d <= -15`), by
`station_descent_decel`:

| decel | n | mean fwd 3d | edge vs base | P(cliff >+25c) | label |
|---|---|---|---|---|---|
| [-99, 0) cuts still accelerating | 79,082 | -2.41 | -2.35 | **2.22%** | 0.203 |
| [0, +2) | 41,747 | -1.10 | -1.03 | 4.25% | 0.292 |
| [+2, +4) | 51,164 | +0.45 | +0.52 | 6.82% | 0.352 |
| **[+4, +6)** (both decisive fold-13 buys) | 32,010 | +0.61 | **+0.67** | 6.62% | 0.411 |
| [+6, +8) | 10,052 | +0.29 | +0.35 | 5.41% | 0.442 |
| [+8, +99) | 2,157 | -0.13 | -0.07 | 3.71% | 0.503 |

**The columns carry real information.** A station whose cuts are still
accelerating is **3x safer to wait on** (2.22% cliff risk) than one whose
descent has stalled (6.6-6.8%) — the direction the hypothesis predicted,
on a sample far too large to be noise.

**But the edge per opportunity is ~0.67 c/L**, and the payoff is
fat-tailed: in the [+4,+6) cell you gain roughly nothing ~93% of the time
and dodge a 25-45 c/L step ~7% of the time.

**Fold 13 sampled that tail twice** — 18517 on 2024-10-20 (decel +4.93,
174.9 -> 207.9 overnight, pinned 17 days) and 585 on 2024-11-20 (decel
+5.36, 172.9 -> 207.9, pinned 9 days). A second multiplier then stacked on
top: R0's tank ran dry inside both plateaus, so it did not merely pay more
later, it was FORCED to buy — 964 L (56% of the fold's litres) at 195.57
c/L against 176.14 voluntary, a 19.4 c/L forced-fill premium. Fold 13's
-1.28 c/L is a real edge's lucky realisation, not a repeatable magnitude.

Caveats: forward return is not realised saving — it ignores the tank
mechanics, so +0.67 c/L is a LOWER bound on what acting on the signal is
worth, and is silent on whether the MODEL can capture it. The buckets are
also not clean everywhere: unconditioned on `px_change_14d`, high decel
mixes "deep descent stalled" with "just restored" (a +34/+16 post-jump row
scores decel +30.6), which is why the table conditions on steep descents
first.

## Judgement

**Signature grading: contradicted.** The candidate made a specific,
falsifiable prediction — concentration in folds 3 and 7, near-zero
elsewhere, with a MODEST flip count — and the pipeline's own instruments
say the opposite on every count. The zone test built from the candidate's
own declared target (folds 3, 7) reads `resolved: false`: those folds show
a small worsening (+0.0148 c/L) while the rest of the run carries the
improvement (-0.0298 c/L). Folds 3 and 7 individually are unremarkable
(-0.15 and +0.23 c/L) next to fold 13's -1.28 c/L or fold 11's -0.57 c/L —
neither of which the candidate predicted. The flip count (337, spread across
all 14 folds) is large and diffuse, not the "modest, concentrated"
shape predicted. **Corrected 2026-08-30 (`fps-hnp`):** what aggregate
improvement exists now traces mostly to **normal**-regime folds
(per-regime: shock +0.0504 vs normal -0.0694 c/L, against the corrected
empirical shock-fold set `{1, 3, 4, 14}`) — the reverse of the original
2026-08-25 reading below, which (against the old fixed set `{1, 4, 9, 13}`)
attributed the improvement to shock-regime folds instead. Fold 13, the
single largest favourable cell, is the reason: it was `shock` under the
old set and is `normal` under the corrected one. Read literally, this is
now closer to — not further from — the candidate's own framing (elongated
*normal*-regime descents), though the candidate's declared axis was
elongation (folds 3/7) specifically, not the regime label, and the zone
test above still reads `resolved: false` against that declared axis
regardless of this regime reattribution.

None of this reaches the arbiter, though: the headline -0.0237 c/L sits
well inside this batch's own noise band (z=-0.49 vs threshold 1.92, 60th
percentile of the noise draws) — this run cannot distinguish the candidate
from noise at all, regardless of the mechanism question above.

**not_tested:**

- **Whether an evaluation that can SEE a fat-tailed payoff would resolve
  these columns.** Superseded the old "is there a shock-regime mechanism"
  item: the post-hoc analysis above shows there is one, and names it —
  stalled-descent rows carry 3x the cliff risk of still-cutting rows. What
  remains untested is whether pooled realised CPL over 14 folds can ever
  resolve an edge of ~0.67 c/L whose payoff arrives in ~7% of firings.
  Candidate designs: many more folds; scoring on tail-avoidance (frequency
  of buying within N days of a >25 c/L step) rather than pooled CPL; or
  block-bootstrap CIs on the per-fold deltas instead of a single pooled
  headline. Note the graded run's own instruments were not wrong — the
  noise-band verdict is correct — the question is whether the arbiter has
  the resolution for this shape of effect at all.
- **Whether the effect is real but underpowered**, not absent. The
  headline sits inside the noise band, which per the ledger's own header is
  not the same as rejection — more data, or a different arm reading the
  same station-level cut sequence, is legitimate open ground.
- **Whether fold 1's deferral shape is mechanism or incident.** The
  candidate's fill dates are a strict subset of R0's there and it is never
  ahead on cumulative litres — is that informative about what
  `station_descent_decel` reads on a shock fold's early cuts, or a
  coincidence of that fold's price path? Not tested.
- **Per-column attribution** (which of the three columns is doing
  anything, vs. inert or reinforcing) needs SHAP interaction values on the
  fitted candidate model, which the pipeline doesn't currently persist —
  same gap noted on `network_move_breadth`.

**Recommendation:** not a graduation candidate on this run, and the
specific mechanism predicted (elongated-fold concentration) is
contradicted by the pipeline's own zone test — but the headline itself is
inside the noise band, so this is inconclusive rather than a rejection of
the underlying columns.

The post-hoc analysis sharpens what to do next. The columns are **not
inert** — `station_descent_decel` separates cliff risk 2.22% vs 6.6-6.8%
within steep descents, in the predicted direction, on 200k+ rows. The
problem is resolution, not existence: an edge of ~0.67 c/L per firing,
realised in ~7% of firings, is not something a 14-fold pooled-CPL arbiter
can distinguish from noise. So if this series is revisited, the change
should be to the EVALUATION before the hypothesis — give the run an
instrument that can see a fat-tailed payoff — rather than re-targeting the
columns at a shock-restoration story and scoring it the same way.

## Followups

- None filed — the per-column-attribution gap (SHAP, needs the candidate
  model persisted) is a repeat of the same open item noted on
  `network_move_breadth`; not worth a second bead for the same ask.

## Correction note — 2026-08-30 (`fps-hnp`, backfill of PR #350 / `fps-3tu`)

PR #350 replaced the batch's fixed shock-fold index (`SHOCK_FOLDS =
{1, 4, 9, 13}`) with a per-batch empirical set derived from the baseline's
own val-fold log-loss. batch1's corrected set is **`{1, 3, 4, 14}`**
(`experiments/batches/batch1/shock_folds.json`) — folds 3 and 9 swap
regime labels, as do 13 and 14, relative to the old fixed set. **This
candidate's TARGET grades on folds 3/7 directly, not on the regime axis,
so the zone `resolved: false` verdict is entirely unaffected.** What does
change is the descriptive per-regime breakdown used as colour in the
Facts and Judgement sections: fold 13 (this run's single largest
favourable cell) moves from shock to normal, which **flips the sign of
both per-regime cells** (shock -0.2837/normal +0.0706 →
shock +0.0703/normal -0.0720; both pairs as computed on 2026-08-30, i.e.
pre-#359 — the current post-regen values are shock +0.0504/normal -0.0694,
same signs) and reverses which regime the Judgement
section credits with the aggregate improvement — both sections above have
been rewritten in place. This backfill reused the run's already-persisted
`fills.parquet` and `rowpreds.parquet`; no re-run of the realised backtest
was needed.

## Addendum — 2026-08-26: batch-wide coverage caveat

Found while reviewing `lga_trough_propagation`; it applies identically here, because
every batch1 candidate replays the same station-day grid. **Not specific to this
candidate, and it does not change this dossier's verdict.**

This run's `results.json` `meta` records `extra_feature_provider_hits` 5693 /
`extra_feature_provider_misses` **278** — the same split as all four sibling
candidates. **Both the total and the per-fold decomposition below moved in the #359
regen** and the paragraph has been restated accordingly: pre-regen the split was
5693 / **607**, summing to exactly 6300 = 5 preferred stations × 14 folds × 90 days.
Post-regen the hits are unchanged and 329 of the misses are gone, so the arms now
visit 5971 station-days rather than all 6300. That is the same `fps-2i4`/#356 fix as
the dark-fill zero above, seen from the other side: station-days behind a closure are
no longer visited at a stale frozen price, so they are no longer counted as a provider
miss either. The residual 278 are genuine no-feature-row days.

The per-fold decomposition below is the **pre-#359** one (it was derived from the
feature frame against the arms' visited station-days, not from a committed artifact,
and was not re-derived). It is kept because it is what established the shape of the
gap; read it as "where the 607 sat", not as a breakdown of the surviving 278:

| fold | window | missing (pre-#359) | which |
|---|---|---|---|
| 1 | 2021-11-05..2022-02-02 (shock) | **180/450** | 414 and 18517, all 90 days |
| 2 | 2022-02-03..2022-05-03 | 8 | 414, 18517 |
| 4-8 | 2022-08-02..2023-10-25 | 67/90/90/90/82 | 414 |

Verified against `data/cleaned/` raw prices: the 2021 gap is **not** a closure — both
stations kept reporting through it (414: 44 raw rows inside, 18517: 75), at roughly a
third of their prior update rate, which trips `fill.py`'s 28-day fill cap and then
`labels.py`'s 7-before/90-after strip. 414's 2022-23 gap **is** a real year-long
rebuild, but it reopened ~2023-07 while the frame stays dark to 2023-10-17, so fold 8's
82 missing days are label-strip tail on a live, trading station.

So folds 1-8 ran on a smaller — and council-skewed — universe than folds 9-14. Fold 1
drops from 5 stations to 3, flipping the mix from 3 Blue Mountains / 2 Penrith to 1/2,
which matters wherever a fold-level read is being compared across the run. This is
reduced sample, **not corrupted sample**: a missing feature row means no fill was
available and both arms skip it identically. The direction survives the regen — fold
1's cells are untouched by it, and the folds that did move (4-7) are inside the
1-8 band this addendum names.

Full analysis, including the per-fold decomposition and the `fps-ghr` implications:
`experiments/candidates/batch1/lga_trough_propagation/README.md` § Review addendum.

## Re-render note — 2026-08-31 (`fps-j6w`)

**The figures in this section are as they stood on 2026-08-31 and are superseded
by the Facts section above** — they are kept as the record of what that pass
added, in the same way the Correction note above keeps its pre-correction pair.
Every one of them was re-measured in the #359 regen; see the Refresh note below
for the current values.

The **Decision flips** section was rendering the pre-`fps-2js` table shape. Every field the
current spec asks for was already in this run's `facts.json`. No new computation;
re-rendered from the committed artifact and verified row-by-row. The fold-1 deferral
analysis, the fold-13 mirror-image note and the `fps-gzz` instrument caveat are preserved
unchanged.

- **`decns` alongside `flips`.** 439 flips are **106 decisions**; folds carry 3-15 each.
- **Regret replaces `flip_cpl_delta`**, each delta beside its own 2·SE interval. Twelve of
  thirteen resolvable folds and the pooled `all` row are `inside_own_se`; **fold 12 alone
  resolves** (-8.62, `[-16.98, -0.26]`).
- **The pooled regret row is the most informative line added, and it points the other way
  from the headline.** R0 8.51 vs candidate **10.26** c/L — the candidate times its
  purchases *worse* — against a **-0.0272 c/L favourable** headline. Regret is a WHEN
  measure at the same station, so this says the candidate's gain came from **what** was
  bought, not when: flip-only litres are 3014.3 vs 3678.6 L, a 22% gap. This is consistent
  with, and helps explain, this run having batch1's largest `composition_residual_cpl`
  (+0.0173, 64% of the headline's own size).
- **`run_contribution_cpl` added**, reconciling to the headline (-0.0445 + +0.0173 =
  -0.0272). By regime: **normal -0.0717 / shock +0.0272**, with fold 13 supplying -0.0951
  of the normal total.
- **Dark fills surfaced:** `dark_fill_days` = **102** in folds 4-7, the largest in batch1.

The Judgement is unchanged — it did not cite the flip table.

## Refresh note — 2026-09-03 (`fps-3ug`)

Every realised-backtest figure above was refreshed from this run's committed
`facts.json` as regenerated by PR #359 (`fps-32h`), which re-ran the realised
backtest under the `fps-2i4`/#356 price-accessor fix and the #358 oracle-DP fix.
No re-grading: the Judgement's conclusion (**contradicted**) and every claim it
rests on survive the refresh with the same sign and the same reading.

**Every CPL figure in this note is at this run's cadence, `50/3.571/1d/10%`**
(`fps-15c`) — the same cadence on both sides of every → below, since the regen
re-ran the identical tank config.

What moved:

- **Headline** -0.0272 → **-0.0237** c/L; R0 187.8211 → 188.0376, candidate
  187.7940 → 188.0138. Noise-band `z` -0.523 → **-0.489** against the same
  threshold 1.923 and the same 60th percentile — still well inside the band.
- **Zone** target +0.0138 → **+0.0148**, other -0.0345 → **-0.0298**, n_fills
  431/1906 → **380/1847**. `resolved: false` on the declared folds-3/7 axis,
  unchanged — the verdict's load-bearing fact.
- **Per-fold: only folds 4–7 moved at all**, which is exactly where the 102
  pre-regen dark fills sat. Every other fold's `n_fills` and `delta_cpl_own` is
  byte-identical, fold 13's -1.2787 and fold 11's -0.5651 included.
- **`dark_fill_days` 102 → 0**, and `extra_feature_provider_misses` 607 → 278.
  Two faces of the same fix; the largest qualitative change in this dossier.
- **Flips 439 → 337 / decns 106 → 99.** Still large and diffuse across all 14
  folds, which is what the Judgement cites it for. Fold 12 remains the single
  regret delta that resolves outside its own 2·SE.
- **Reconciliation** -0.0445 + 0.0173 → **-0.0654 + 0.0417**. The composition
  residual goes from 64% of the headline's size to **176%** of it — still
  batch1's largest, now larger than the effect it accompanies.
- **Pooled regret** 8.51/10.26 (Δ +1.75) → **10.80/12.06 (Δ +1.26)**, still
  `inside_own_se`, still pointing the opposite way from the headline. Flip-only
  litres 3014.3/3678.6 → **2342.9/3128.6**, a 22% gap becoming **34%**.
- **Not re-derived, and labelled in place:** the fold-1 deferral counts, the
  fold-13 mirror-image counts and the 150 volume-only changes. All three come
  from `fills.parquet`, which is gitignored; the on-disk copy predates the regen
  and cannot be used to restate them. Folds 1 and 13 are among the folds the
  regen left untouched in `facts.json`, so nothing contradicts them.
- **Unchanged by the regen, and verified so:** every WFCV / log-loss / seed
  field, hence the whole post-hoc conditional analysis, whose only input is
  `rowpreds.parquet`.
