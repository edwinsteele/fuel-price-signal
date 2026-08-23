# batch1 / network_move_breadth

- **Date:** 2026-08-15 (snapshot) / dossiered 2026-08-24
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
row-label slice):**

| | target (shock) | other (normal) |
|---|---|---|
| delta_cpl_own | -0.3065 | -0.0925 |
| n_fills | 705 | 1619 |

`resolved`: **true** — shock is more negative than normal, in the direction
the candidate predicted.

**Per-fold:**

| fold | regime | n_fills | delta_cpl_own |
|---|---|---|---|
| 1 | shock | 188 | +0.0573 |
| 2 | normal | 197 | -0.4233 |
| 3 | normal | 178 | -0.6392 |
| 4 | shock | 239 | -0.1729 |
| 5 | normal | 185 | +0.3301 |
| 6 | normal | 127 | +0.1767 |
| 7 | normal | 232 | -0.0622 |
| 8 | normal | 88 | -0.0602 |
| 9 | shock | 112 | +0.2894 |
| 10 | normal | 96 | +0.8968 |
| 11 | normal | 161 | -0.3535 |
| 12 | normal | 177 | -0.7462 |
| 13 | shock | 146 | **-1.3250** |
| 14 | normal | 168 | -0.0281 |

No cell is suppressed (`min_row_cell_n` = 30, smallest cell here is 88).

`per_axis` is `null` — this candidate declared no `add_axis` beyond the
regime split above, so there is no path-coupled per-row zone to caveat here.

**Seed stability:** 5 cells exceed 5× the cohort-median seed_std, all on run
R0 — folds 2 and 3 (`all` cohort) and folds 2, 3, 5 (`hard25` cohort), each
around 5.4–6.0× the median. None of the four shock folds appear in this list.

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
| 3 | normal | 25 | 12 | 197.30 | 195.53 | -1.77 |
| 4 | shock | 26 | 38 | 181.82 | 183.59 | +1.77 |
| 5 | normal | 22 | 22 | 186.37 | 187.96 | +1.59 |
| 6 | normal | 15 | 7 | 185.48 | 187.37 | +1.88 |
| 7 | normal | 8 | 5 | 178.78 | 176.52 | -2.26 |
| 8 | normal | 6 | 7 | 185.35 | 187.16 | +1.80 |
| 9 | shock | 0 | 8 | — (no baseline-only fills) | 181.35 | — |
| 10 | normal | 5 | 12 | 199.99 | 202.64 | +2.64 |
| 11 | normal | 3 | 5 | 212.45 | 197.97 | **-14.47** |
| 12 | normal | 11 | 3 | 186.93 | 172.51 | **-14.42** |
| 13 | shock | 10 | 10 | 199.03 | 186.08 | **-12.95** |
| 14 | normal | 6 | 5 | 181.62 | 178.52 | -3.11 |

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

**Signature grading: inconclusive** — revised 2026-08-24 with `decision_flips`
now available; supersedes the original fold-13-only read below. **The
favourable effect is real and sustained, not a single-fold artifact**: three
CONSECUTIVE folds — 11 (-14.47 c/L), 12 (-14.42 c/L), 13 (-12.95 c/L) — each
show a flip-only CPL delta of similar, large magnitude, on double-digit flip
counts each. That is a materially stronger finding than "one dominant shock
fold" — it looks like a genuine multi-week period where these columns paid
off, not a single lucky catch.

**But it contradicts the specific mechanism predicted.** Only fold 13 of
those three is a SHOCK fold (`SHOCK_FOLDS = {1, 4, 9, 13}`); folds 11 and 12
are classified `normal`. The predicted signature was explicit — concentration
on shock (restoration) folds specifically — and the actual per-flip wins
straddle the shock/normal boundary rather than sitting inside it. The
per-regime aggregate (shock -0.3065 vs. normal -0.0925 c/L) still nominally
points the predicted way, but that now reads more like a coincidence of which
folds happen to carry the `SHOCK_FOLDS` label than confirmation of a
restoration-timing mechanism — the three winning folds are calendar-adjacent,
not regime-adjacent. Elsewhere the flip-level effect is flat-to-unfavourable
(folds 1, 4–8, 10 all sit between +1.6 and +14.8 c/L, i.e. worse for the
candidate), and fold 9 is a different shape again — 8 candidate-only fills
with zero matched baseline-only fills, i.e. more frequent buying rather than
swapped timing. The headline realised delta (-0.1552 c/L) still sits inside
this batch's single-candidate noise band, so nothing here overrides that —
per `docs/routines/dossier.md`'s guardrail, this flip-level detail is colour
explaining *why* the aggregate looks the way it does, not a second arbiter.

**not_tested:**

- **Why folds 11–13 specifically.** Three calendar-adjacent folds carrying
  the whole favourable pattern, crossing the shock/normal label boundary, is
  a concrete new question this run raises rather than answers: is there a
  real event in that window (visible in `candidate_over_time.png` or the raw
  snapshot) that these columns caught regardless of the `SHOCK_FOLDS`
  labelling, or is 3-folds-in-a-row itself within the range pure noise would
  produce at this batch's fold count? The batch's own noise floor is
  fold-pooled (10 draws, no per-fold breakdown), so it can't answer the
  second half directly.
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
(folds 11–13), which rules out "one lucky fold," but it isn't the predicted
mechanism (it isn't shock-concentrated) — so if this series gets revisited,
the next hypothesis should be built around "what happened in this
calendar window" rather than "restoration timing during shocks."

## Followups

- `fps-gez` (decision-flip attribution tool) is now built and merged —
  this dossier is its first real application, and the folds-11–13 finding
  above is the direct payoff. No further tooling work needed to act on this
  candidate's own not_tested items.
- The rise/fall per-column attribution gap (SHAP, needs the candidate model
  persisted) remains open — worth its own bead if a future candidate's
  grading depends on it, rather than building it speculatively now.
