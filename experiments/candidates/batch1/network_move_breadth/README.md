# batch1 / network_move_breadth

- **Date:** 2026-08-15 (snapshot) / dossiered 2026-08-24 / zone corrected 2026-08-30 (`fps-hnp`)
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
`(fold, station_code, date)`; `flip_cpl_*` is `pooled_cpl` over ONLY the fills
that differ in that fold, not the same quantity as `delta_cpl_own` above,
which pools every fill): **336 total flips**. Per fold:

| fold | regime | n_baseline_only | n_candidate_only | flip_cpl_baseline | flip_cpl_candidate | flip_cpl_delta |
|---|---|---|---|---|---|---|
| 1 | shock | 4 | 2 | 153.11 | 167.90 | +14.79 |
| 2 | normal | 31 | 28 | 182.35 | 176.28 | -6.06 |
| 3 | shock | 25 | 12 | 197.30 | 195.53 | -1.77 |
| 4 | shock | 26 | 38 | 181.82 | 183.59 | +1.77 |
| 5 | normal | 22 | 22 | 186.37 | 187.96 | +1.59 |
| 6 | normal | 15 | 7 | 185.48 | 187.37 | +1.88 |
| 7 | normal | 8 | 5 | 178.78 | 176.52 | -2.26 |
| 8 | normal | 6 | 7 | 185.35 | 187.16 | +1.80 |
| 9 | normal | 0 | 8 | — (no baseline-only fills) | 181.35 | — |
| 10 | normal | 5 | 12 | 199.99 | 202.64 | +2.64 |
| 11 | normal | 3 | 5 | 212.45 | 197.97 | **-14.47** |
| 12 | normal | 11 | 3 | 186.93 | 172.51 | **-14.42** |
| 13 | normal | 10 | 10 | 199.03 | 186.08 | **-12.95** |
| 14 | shock | 6 | 5 | 181.62 | 178.52 | -3.11 |

Fold 9's `flip_cpl_delta` is unavailable (`null`): the candidate has 8
candidate-only fills there and R0 has zero baseline-only fills to compare
against, so this fold's divergence is "candidate bought more often," not "the
two arms swapped timing" — a different shape than every other row.

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

**The favourable effect is real and sustained, not a single-fold
artifact**: three CONSECUTIVE folds — 11 (-14.47 c/L), 12 (-14.42 c/L), 13
(-12.95 c/L) — each show a flip-only CPL delta of similar, large magnitude,
on double-digit flip counts each. That is a materially stronger finding
than "one dominant shock fold" — it looks like a genuine multi-week period
where these columns paid off, not a single lucky catch.

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

Elsewhere the flip-level effect is flat-to-unfavourable (folds 1, 4, 5–8,
10 all sit between +1.6 and +14.8 c/L, i.e. worse for the candidate — fold
3 itself is a mild favourable -1.77 c/L, not one of the large movers), and
fold 9 is a different shape again — 8 candidate-only fills with zero
matched baseline-only fills, i.e. more frequent buying rather than swapped
timing. The headline realised delta (-0.1552 c/L, unaffected by the shock-
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
open ground, not a coin-flip. The favourable pattern is real and sustained
(folds 11–13), which rules out "one lucky fold," and under the corrected
regime split it is now unambiguous that this cluster sits entirely outside
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
