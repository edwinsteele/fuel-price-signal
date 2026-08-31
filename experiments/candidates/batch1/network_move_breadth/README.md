# batch1 / network_move_breadth

- **Date:** 2026-08-15 (snapshot) / dossiered 2026-08-24 / zone corrected 2026-08-30
  (`fps-hnp`) / flip table re-rendered + Judgement corrected 2026-08-31 (`fps-j6w`)
- **Branch:** main
- **SHA:** 19b65549181cd94bca63fffde682f180faabd606
- **Status:** done — **inconclusive**
- **Bead:** fps-2l9
- **Cadence:** `50/3.571/1d/10%` (batch1's canonical daily cadence)

## Hypothesis

A Sydney cycle ends with a coordinated restoration: a subset of stations jumps
20–30 c/L within a day or two, and the rest follow over the next few days.
`network_px_std` measures the spread of the cross-section and is invariant to
direction, so it can't tell "a few stations jumped up" from "a few stations
dropped down." Counting the share of stations moving up vs. down over a fixed
window (`network_rise_breadth_3d`, `network_fall_breadth_3d`,
`network_rise_breadth_delta_2d`) recovers that sign, as a participation count
robust to the size of any one station's jump.

**Predicted signature:** rows where `network_rise_breadth_delta_2d` turns
sharply positive should flip WAIT→BUY one to three days earlier than the lock
manages, with a large per-flip saving (a restoration is worth 20–30 c/L vs.
~1–2 c/L for a day of descent) — i.e. a **few large flips**, concentrated on
**shock folds**, with `network_fall_breadth_3d` mostly pushing the opposite
direction (WAIT longer).

## How to invoke this script

```bash
PYTHONPATH=. uv run python -m experiments.pipeline.runner \
  --batch-dir experiments/batches/batch1 \
  --candidate experiments/candidates/batch1/network_move_breadth.py
```

## Facts

Transcribed from `facts.json` — no new arithmetic.

**Provenance:** batch `batch1`, snapshot 2026-08-15, seeds [42, 43, 44, 45, 46],
wall 1782.6s, status `graded`, 54-column baseline (`54:1a6ec2d84a69`).

**Headline (realised CPL, held τ — the arbiter):**

| | R0 | candidate | delta |
|---|---|---|---|
| CPL held | 187.8211 | 187.6659 | **-0.1552** |

`effect_resolved`: **true**.

**WFCV log-loss** (descriptive colour, NOT the arbiter): Δll_all mean
-0.0114, Δll_hard25 mean -0.0570. Both favour the candidate.

**Zone (regime axis — safe to cut; a fold is an independent simulation, not a
row-label slice). Graded against the corrected empirical shock-fold set
`{1, 3, 4, 14}` (`experiments/batches/batch1/shock_folds.json`,
`fps-3tu`/`fps-hnp`), not the fixed `{1, 4, 9, 13}` this dossier originally
used — see the Correction note below the Judgement section:**

| | target (shock) | other (normal) |
|---|---|---|
| delta_cpl_own | -0.2111 | -0.1310 |
| n_fills | 785 | 1539 |

`resolved`: **true** — shock is more negative than normal, in the direction
the candidate predicted. The margin is narrower than under the old fixed
shock set (-0.3065 / -0.0925 → -0.2111 / -0.1310), but the verdict is
unchanged.

**Per-fold** (regime column reflects the corrected empirical shock-fold set
`{1, 3, 4, 14}` — folds 3 and 9 swap regime labels, as do 13 and 14, vs. the
old fixed set):

| fold | regime | n_fills | delta_cpl_own |
|---|---|---|---|
| 1 | shock | 188 | +0.0573 |
| 2 | normal | 197 | -0.4233 |
| 3 | shock | 178 | -0.6392 |
| 4 | shock | 239 | -0.1729 |
| 5 | normal | 185 | +0.3301 |
| 6 | normal | 127 | +0.1767 |
| 7 | normal | 232 | -0.0622 |
| 8 | normal | 88 | -0.0602 |
| 9 | normal | 112 | +0.2894 |
| 10 | normal | 96 | +0.8968 |
| 11 | normal | 161 | -0.3535 |
| 12 | normal | 177 | -0.7462 |
| 13 | normal | 146 | **-1.3250** |
| 14 | shock | 168 | -0.0281 |

No cell is suppressed (`min_row_cell_n` = 30, smallest cell here is 88).

`per_axis` is `null` — this candidate declared no `add_axis` beyond the
regime split above, so there is no path-coupled per-row zone to caveat here.

**Seed stability:** 5 cells exceed 5× the cohort-median seed_std, all on run
R0 — folds 2 and 3 (`all` cohort) and folds 2, 3, 5 (`hard25` cohort), each
around 5.4–6.0× the median. Under the old fixed shock set none of these
were shock folds; under the corrected empirical set **fold 3 is now
shock**, so one of the four shock folds (both its `all` and `hard25` R0
cells) is seed-unstable. Fold 3 also turns out to be this dossier's
single largest driver of the shock-zone effect (see Judgement below), so
this instability is directly relevant to the zone verdict in a way it
wasn't under the old labelling.

**Validation:** PIT passed (differential PIT truncation test found no leak).
INPUTS check passed. Candidate column NaN rate 0.0% for all three columns.

**Noise floor** (`experiments/batches/batch1/noise_floor.json`, 10 draws at
arity 3 — matches this candidate's 3-column arity, `floor_arity_exceeds_run`:
false): band mean +0.0251 c/L, band std 0.0999 c/L. The candidate's -0.1552
c/L sits at the **100th percentile** of the 10 observed noise draws
(`candidate_percentile_better_than_noise` = 100.0 — better than every one of
them), but the parametric single-candidate call is `z = -1.80` against a
threshold of `t = 1.92`: inside the band, not past it.

**Decision flips** (`facts["decision_flips"]`, added 2026-08-24 once `fps-gez`
merged — diffs R0's and the candidate's `fills.parquet` directly on
`(fold, station_code, date)`, not a `rowpreds` proba-threshold crossing):
run-level **336 flips / 106 decisions**. `flips` overstates independent
evidence — one real flip cascades into a run of later differing fills at the
same station as the tank's state diverges, which `decns` collapses over this
run's own `cascade_window_days` = 7. **Read `decns`, not `flips`, as this
table's sample size**, and note it is not comparable across runs at different
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
difference has no stable denominator. **The 2026-08-24 revision of this
dossier built its central claim on those unrendered numbers; see the
Judgement, corrected in place.**

| fold | regime | flips | decns | litres R0 | litres cand | regret R0 | regret cand | regret Δ | Δ interval (2·SE) | run_contribution_cpl |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | shock | 6 | 3 | 35.7 | 42.9 | 4.41 | 4.00 | -0.41 | [-7.05, +6.23] | +0.0038 |
| 2 | normal | 59 | 13 | 382.1 | 414.3 | 8.03 | 9.03 | +0.99 | [-9.41, +11.40] | -0.0304 |
| 3 | shock | 37 | 11 | 457.1 | 246.4 | 6.09 | 4.45 | -1.64 | [-8.54, +5.26] | -0.0476 |
| 4 | shock | 64 | 15 | 342.9 | 525.0 | 12.44 | 11.87 | -0.57 | [-10.53, +9.39] | -0.0125 |
| 5 | normal | 44 | 11 | 196.4 | 346.4 | 6.88 | 8.52 | +1.63 | [-5.62, +8.89] | +0.0240 |
| 6 | normal | 22 | 6 | 189.3 | 175.0 | 0.34 | 4.20 | +3.86 | [-3.96, +11.69] | +0.0125 |
| 7 | normal | 13 | 7 | 189.3 | 132.1 | 8.72 | 3.41 | -5.31 | [-17.36, +6.74] | -0.0042 |
| 8 | normal | 13 | 5 | 92.9 | 121.4 | 1.78 | 3.55 | +1.77 | [-0.71, +4.25] | -0.0044 |
| 9 | normal | 8 | 4 | 0.0 | 153.6 | — | 5.56 | — (`one arm only`) | — | +0.0215 |
| 10 | normal | 17 | 4 | 125.0 | 246.4 | 9.67 | 11.36 | +1.69 | [-13.74, +17.12] | +0.0612 |
| 11 | normal | 8 | 5 | 78.6 | 100.0 | 24.36 | 7.50 | -16.86 | [-37.23, +3.50] | -0.0236 |
| 12 | normal | 14 | 10 | 217.9 | 75.0 | 21.38 | 7.75 | **-13.62** | **[-24.13, -3.12]** | -0.0541 |
| 13 | normal | 20 | 7 | 164.3 | 282.1 | 12.65 | 10.80 | -1.85 | [-18.79, +15.08] | -0.0987 |
| 14 | shock | 11 | 5 | 100.0 | 92.9 | 7.66 | 3.88 | -3.78 | [-10.72, +3.16] | -0.0021 |
| **all** | — | **336** | **106** | **2571.4** | **2953.6** | **9.38** | **8.19** | **-1.19** | **[-4.82, +2.43]** | **-0.1545** |

**Twelve of the thirteen resolvable regret deltas sit inside their own 2·SE
interval, as does the pooled `all` row.** Per `docs/routines/dossier.md`, an
`inside_own_se` cell must never be cited as evidence in the Judgement
section. **Fold 12 is the single exception in this run** — Δ -13.62 c/L
against `[-24.13, -3.12]`, an interval that excludes zero — and it is the
only cell in this dossier that may be cited as resolved. Fold 9 is not
resolvable at all: `one arm only`, 8 candidate-only fills against zero
baseline-only ones, i.e. "candidate bought more often" rather than "the two
arms swapped timing" — a different shape from every other row, and not the
same as a resolved zero.

The pooled row is the only run-level statement this table can make: **both
arms buy about equally well relative to what they could have reached**
(9.38 vs 8.19 c/L of regret, Δ -1.19 against a ±3.63 interval), consistent
with a -0.1552 c/L headline.

Two things the table surfaces that are not deltas. **Litres:** flip-only
volume is 2571.4 L on R0 against 2953.6 L on the candidate, a 15% gap on a
spend-weighted metric — the arms are not diverging on equal volume. **Dark
fills:** `dark_fill_days` = **77** of 336 scored, in folds **4–7** — the
Blue Mountains preferred-station outage. Those fills ARE scored, at the
forward-filled price the tank simulator itself bought at, so the outage does
not bias the estimate, but it thins and tilts exactly those folds.

**`run_contribution_cpl`** is what each fold's divergence is worth at the
scale of the whole run — a litres-weighted shift-share of that fold's own
`delta_cpl_own` (the whole fold, matching and flipped fills alike), not of
the flip-only figures. It reconciles: `run_contribution_total_cpl`
**-0.1545** + `composition_residual_cpl` **-0.0007** = **-0.1552**, the
headline `delta_cpl_held`. This run's `tau_diverges_any` is `None`
(**unavailable, never read as "checked, agrees"**), so the residual is
composition drift **and possibly tau drift**.

Summed by regime: **normal -0.0962** (10 folds, 72 decns) against **shock
-0.0583** (4 folds, 34 decns). On the folds-11-13 cluster the Judgement
discusses, run contribution is **-0.1764** between them — more than the
whole headline — but fold 10, immediately adjacent, hands back **+0.0612**,
so the cluster is not cleanly bounded even on this measure.

![](per_fold_delta_bars.png)
![](seed_mean_vs_median.png)
![](realised_cpl_by_fold.png)
![](tau_sweep.png)
![](candidate_over_time.png)
![](external_series_overlay.png)

## Judgement

**Signature grading: inconclusive.** Zone `resolved` is still **true** under
the corrected empirical shock-fold set (`{1, 3, 4, 14}`, `fps-hnp`) — see
the Correction note below — but the account of *why* changes materially
from the 2026-08-24 revision below, which was written against the old
fixed set `{1, 4, 9, 13}` and is superseded on every point involving fold 3
or fold 13's regime label.

**The folds-11-13 cluster survives re-measurement, but weaker and
differently shaped than the 2026-08-24 revision claimed.** That revision
read three CONSECUTIVE folds — 11, 12, 13 — as showing "a flip-only CPL
delta of similar, large magnitude," and concluded the effect was "real and
sustained, not a single-fold artifact." Those three numbers were
`flip_cpl_delta` cells, and on re-render (`fps-j6w`, 2026-08-31) **two of
the three cannot resolve their own value**: on regret, fold 11 is -16.86
against `[-37.23, +3.50]` and fold 13 is **-1.85** — essentially flat
timing — against `[-18.79, +15.08]`. Only **fold 12** resolves (-13.62,
`[-24.13, -3.12]`, excluding zero), and it is the one cell in this run that
may be cited. "Three folds of similar large magnitude" was a vote count
over cells that could not resolve their own numbers.

What does survive is a different, better-grounded statement: on
`run_contribution_cpl` — which decomposes the headline additively rather
than pooling disjoint flip baskets — folds 11, 12 and 13 contribute
**-0.1764 c/L between them against a -0.1552 c/L headline**, i.e. more than
all of it. So the window still carries the run. But it is not cleanly
bounded: fold 10, immediately adjacent, hands back **+0.0612**, and fold 13
supplies -0.0987 of the cluster's -0.1764 on flat timing, meaning its
contribution came from WHAT was bought rather than WHEN. "A genuine
multi-week period where these columns paid off" remains the most plausible
reading; "sustained across three folds of similar magnitude" does not.

**Under the corrected shock-fold set, none of those three folds is
shock — all of 11, 12, and 13 are `normal`.** Under the old fixed set, fold
13 alone carried a shock label and the other two were normal, which read as
"straddling the shock/normal boundary." The correction removes that
straddle entirely: the whole folds-11-13 favourable cluster now sits
inside one regime, sharpening rather than resolving the open question —
this looks even more like a calendar-adjacent event than a regime-adjacent
one. **The zone still resolves true, but for a different reason than
before**: fold 3, which moved INTO shock under the correction, now drives
most of the shock-zone effect on its own — excluding it (litre-weighted,
against the actually-persisted `fills.parquet`) takes the shock-zone delta
from -0.2111 c/L to -0.0546 c/L, roughly a 74% reduction, though the sign
does not flip the way it does for `lga_trough_propagation`'s zone. Fold 3
is also one of this run's seed-instability flags (both `all` and `hard25`
cohorts, R0 run — see Facts above), so the zone verdict now rests
partly on an unstable-fitting window in a way it didn't under the old
labelling.

Elsewhere the picture is flat: outside fold 12, every resolvable regret
delta in the run sits inside its own 2·SE interval, so no other fold —
favourable or unfavourable — can be cited as evidence either way. Fold 9 is
a different shape again and is not resolvable at all (`one arm only`): 8
candidate-only fills with zero matched baseline-only fills, i.e. more
frequent buying rather than swapped timing. The headline realised delta (-0.1552 c/L, unaffected by the shock-
fold correction) still sits inside this batch's single-candidate noise
band, so nothing here overrides that — per `docs/routines/dossier.md`'s
guardrail, this flip-level detail is colour explaining *why* the aggregate
looks the way it does, not a second arbiter.

**not_tested:**

- **Why folds 11–13 specifically.** Under the correction these three
  calendar-adjacent folds carrying the whole favourable flip pattern are
  now unambiguously all-`normal`, sharpening rather than answering the
  original question: is there a real event in that window (visible in
  `candidate_over_time.png` or the raw snapshot) that these columns caught,
  independent of any regime labelling, or is 3-folds-in-a-row itself within
  the range pure noise would produce at this batch's fold count? The
  batch's own noise floor is fold-pooled (10 draws, no per-fold breakdown),
  so it can't answer the second half directly.
- Whether the shock-zone read survives excluding fold 3 the way it doesn't
  survive for `lga_trough_propagation`'s normal zone excluding fold 13 —
  here it shrinks (-0.2111 → -0.0546 c/L) but stays the predicted sign, a
  materially different fragility than a sign flip, worth stating precisely
  rather than lumping the two candidates' single-fold dependencies together.
- **The rise/fall asymmetry.** Nothing here checks whether
  `network_fall_breadth_3d` is actually pushing WAIT longer as predicted, or
  whether it's inert (or even reinforcing) — that needs per-column
  attribution (e.g. SHAP interaction values), which needs the candidate's
  fitted model persisted (it currently isn't — only R0's baseline is
  cached). Discussed but deliberately not built as part of `fps-gez`.
- Fold 9's "more frequent buying" shape is untouched — is that itself a
  restoration-adjacent pattern (buying every small dip because participation
  breadth stayed elevated) or an unrelated effect of these columns.
- The three columns here are one arm of a two-candidate cross-sectional-
  consensus filing in this batch (see `lga_trough_propagation`, same batch) —
  whether a combined rise/fall breadth + LGA-trough feature does better than
  either alone is untested.

**Recommendation:** not a graduation candidate on this run — the headline
sits inside the batch's own noise floor — but this is now better-evidenced
open ground, not a coin-flip. The favourable pattern is carried by
folds 11–13 (-0.1764 c/L of run contribution between them), which rules out
"one lucky fold" even though only fold 12 resolves its own timing, and
under the corrected regime split it is now unambiguous that this cluster
sits entirely outside
shock — so if this series gets revisited, the next hypothesis should be
built around "what happened in this calendar window" rather than
"restoration timing during shocks," with the same recommendation now
stronger than it was under the old, boundary-straddling read.

## Followups

- `fps-gez` (decision-flip attribution tool) is now built and merged —
  this dossier is its first real application, and the folds-11–13 finding
  above is the direct payoff. No further tooling work needed to act on this
  candidate's own not_tested items.
- The rise/fall per-column attribution gap (SHAP, needs the candidate model
  persisted) remains open — worth its own bead if a future candidate's
  grading depends on it, rather than building it speculatively now.

## Correction note — 2026-08-30 (`fps-hnp`, backfill of PR #350 / `fps-3tu`)

PR #350 replaced the batch's fixed shock-fold index (`SHOCK_FOLDS =
{1, 4, 9, 13}`) with a per-batch empirical set derived from the baseline's
own val-fold log-loss. batch1's corrected set is **`{1, 3, 4, 14}`**
(`experiments/batches/batch1/shock_folds.json`) — folds 3 and 9 swap
regime labels, as do 13 and 14, relative to the old fixed set. This
dossier's zone verdict (`resolved: true`) is unchanged by the correction,
but the margin narrows (-0.3065/-0.0925 → -0.2111/-0.1310 c/L) and, more
consequentially, the folds-11-13 favourable cluster discussed at length in
the Judgement section moves from "straddling the shock/normal boundary" to
"entirely inside normal" — the Judgement section above has been rewritten
in place to reflect this; the 2026-08-24 revision it describes is kept
inline as historical context but its shock/normal claims about folds 3, 9,
13, 14 no longer hold. The realised CPL headline (-0.1552 c/L, still
inside the noise band) is unaffected — only the regime-axis zone grade and
its supporting per-fold/flip commentary depended on the shock-fold set.
This backfill reused the run's already-persisted `fills.parquet` and
`rowpreds.parquet`; no re-run of the realised backtest was needed.

## Addendum — 2026-08-26: batch-wide coverage caveat

Found while reviewing `lga_trough_propagation`; it applies identically here, because
every batch1 candidate replays the same station-day grid. **Not specific to this
candidate, and it does not change this dossier's verdict.**

This run's `results.json` `meta` records `extra_feature_provider_hits` 5693 /
`extra_feature_provider_misses` 607 — the same split as both sibling candidates. The
6300 total is 5 preferred stations × 14 folds × 90 days. All 607 misses are
station-days where a Blue Mountains station has no feature row, and they are confined
to **folds 1-8**; folds 9-14 are completely clean:

| fold | window | missing | which |
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
available and both arms skip it identically.

Full analysis, including the per-fold decomposition and the `fps-ghr` implications:
`experiments/candidates/batch1/lga_trough_propagation/README.md` § Review addendum.

## Re-render note — 2026-08-31 (`fps-j6w`)

The **Decision flips** section was rendering the pre-`fps-2js` table shape, and unlike the
other batch1 dossiers **this one's Judgement rested on it**, so both were corrected. No new
computation; re-rendered from the committed `facts.json` and verified row-by-row.

- **`decns` alongside `flips`.** 336 flips are **106 decisions**; folds carry 3-15 each.
- **Regret replaces `flip_cpl_delta`**, each delta beside its own 2·SE interval. Twelve of
  thirteen resolvable folds and the pooled `all` row are `inside_own_se`. **Fold 12 is the
  single cell in this run that resolves** (-13.62, `[-24.13, -3.12]`).
- **The 2026-08-24 "three consecutive folds" claim is corrected in the Judgement.** It read
  folds 11, 12 and 13 as showing flip deltas "of similar, large magnitude" and concluded
  the effect was "real and sustained, not a single-fold artifact." On regret, fold 11 and
  fold 13 cannot resolve their own values, and fold 13's timing is **flat** (-1.85). What
  survives is a different statement, now made on `run_contribution_cpl`: folds 11-13
  contribute **-0.1764 c/L against a -0.1552 headline**, so the window does carry the run —
  but fold 10 hands back +0.0612, so the cluster is not cleanly bounded.
- **`run_contribution_cpl` added**, reconciling to the headline (-0.1545 + -0.0007 =
  -0.1552). By regime: **normal -0.0962 / shock -0.0583**.
- **Litres and dark fills surfaced** (2571.4 vs 2953.6 L flip-only, a 15% gap;
  `dark_fill_days` = 77 in folds 4-7).

The zone grade, the headline and the noise-band call are unchanged — none depended on the
flip table.
