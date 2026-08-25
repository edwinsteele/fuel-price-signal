# batch1 / station_descent_dynamics

- **Date:** 2026-08-15 (snapshot) / dossiered 2026-08-25
- **Branch:** main
- **SHA:** f536e98b9c183e754620d46ee0a6ef13681e7c35
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
46], wall 1079.2s, status `graded`, 54-column baseline (`54:1a6ec2d84a69`).

**Headline (realised CPL, held τ — the arbiter):**

| | R0 | candidate | delta |
|---|---|---|---|
| CPL held | 187.8211 | 187.7940 | **-0.0272** |

`effect_resolved`: **true**.

**WFCV log-loss** (descriptive colour, NOT the arbiter): Δll_all mean
-0.00057, Δll_hard25 mean -0.02066. Both nominally favour the candidate.

**Zone (target = folds 3/7, the candidate's own declared elongation axis):**

| | target (folds 3, 7) | other |
|---|---|---|
| delta_cpl_own | +0.0138 | -0.0345 |
| n_fills | 431 | 1906 |

`resolved`: **false** — the target zone shows a small WORSENING, not the
predicted improvement; the improvement instead sits in the "other" fills.

**Per-fold:**

| fold | regime | n_fills | delta_cpl_own |
|---|---|---|---|
| 1 | shock | 169 | -0.2459 |
| 2 | normal | 200 | +0.3586 |
| 3 | normal | 191 | -0.1517 |
| 4 | shock | 239 | +0.3787 |
| 5 | normal | 179 | +0.4037 |
| 6 | normal | 130 | -0.0735 |
| 7 | normal | 231 | +0.1968 |
| 8 | normal | 88 | -0.1318 |
| 9 | shock | 112 | +0.1273 |
| 10 | normal | 95 | +0.2791 |
| 11 | normal | 161 | -0.5651 |
| 12 | normal | 185 | -0.2809 |
| 13 | shock | 146 | **-1.2787** |
| 14 | normal | 169 | +0.3727 |

No cell is suppressed (`min_row_cell_n` = 30, smallest cell here is 88).

**Per-regime:** shock -0.2837 c/L (n=683), normal +0.0706 c/L (n=1645).

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
false): band mean +0.0251 c/L, band std 0.0999 c/L. The candidate's -0.0272
c/L sits at the **60th percentile** of the 10 observed noise draws
(`candidate_percentile_better_than_noise` = 60.0), and the parametric
single-candidate call is `z = -0.523` against a threshold of `t = 1.923`:
well inside the band.

**Decision flips** (`facts["decision_flips"]`, diffs R0's and the
candidate's `fills.parquet` directly on `(fold, station_code, date)`;
`flip_cpl_*` is `pooled_cpl` over ONLY the fills that differ in that fold,
not the same quantity as `delta_cpl_own` above, which pools every fill):
**439 total flips**. Per fold:

| fold | regime | n_baseline_only | n_candidate_only | flip_cpl_baseline | flip_cpl_candidate | flip_cpl_delta |
|---|---|---|---|---|---|---|
| 1 | shock | 21 | 0 | 166.83 | — (no candidate-only fills) | — |
| 2 | normal | 39 | 46 | 181.28 | 182.37 | +1.09 |
| 3 | normal | 14 | 23 | 205.56 | 200.98 | -4.57 |
| 4 | shock | 23 | 30 | 183.29 | 183.18 | -0.12 |
| 5 | normal | 32 | 26 | 188.61 | 189.24 | +0.63 |
| 6 | normal | 17 | 12 | 187.36 | 186.34 | -1.02 |
| 7 | normal | 16 | 12 | 178.31 | 177.15 | -1.16 |
| 8 | normal | 6 | 9 | 183.85 | 182.31 | -1.54 |
| 9 | shock | 6 | 6 | 198.55 | 195.06 | -3.48 |
| 10 | normal | 20 | 19 | 205.93 | 204.78 | -1.15 |
| 11 | normal | 4 | 5 | 212.03 | 197.57 | **-14.46** |
| 12 | normal | 5 | 9 | 188.75 | 181.79 | **-6.96** |
| 13 | shock | 8 | 18 | 199.81 | 186.16 | **-13.65** |
| 14 | normal | 6 | 7 | 180.73 | 184.55 | +3.82 |

Fold 1's `flip_cpl_delta` is unavailable (`null`): R0 has 21 baseline-only
fills there and the candidate has zero candidate-only fills to compare
against, so `flip_cpl_candidate` is 0/0. The shape is **deferral, not
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
behind on 5): the same three columns produce opposite timing bias in the
two folds, which is consistent with them reading descent SHAPE rather than
carrying a fixed buy-earlier or buy-later tilt.

**Instrument caveat:** `decision_flips` keys on `(fold, station_code,
date)`, so a fill that occurs on the same day in both arms at a DIFFERENT
volume is counted as common, not as a flip. Across this run that hides 150
volume-only changes behind the 439 counted date-flips — fold 12 is the
sharpest case (13 volume-only changes against only 14 date-flips), and
fold 1's null is entirely an artifact of this blind spot. Tracked as
`fps-gzz`.

![](per_fold_delta_bars.png)
![](seed_mean_vs_median.png)
![](realised_cpl_by_fold.png)
![](tau_sweep.png)
![](candidate_over_time.png)
![](external_series_overlay.png)

## Judgement

**Signature grading: contradicted.** The candidate made a specific,
falsifiable prediction — concentration in folds 3 and 7, near-zero
elsewhere, with a MODEST flip count — and the pipeline's own instruments
say the opposite on every count. The zone test built from the candidate's
own declared target (folds 3, 7) reads `resolved: false`: those folds show
a small worsening (+0.0138 c/L) while the rest of the run carries the
improvement (-0.0345 c/L). Folds 3 and 7 individually are unremarkable
(-0.15 and +0.20 c/L) next to fold 13's -1.28 c/L or fold 11's -0.57 c/L —
neither of which the candidate predicted. The flip count (439, tens per
fold across all 14) is large and diffuse, not the "modest, concentrated"
shape predicted. What aggregate improvement exists traces mostly to
shock-regime folds (per-regime: shock -0.2837 vs normal +0.0706 c/L) — a
mechanism the candidate never proposed; its whole framing was about
elongated *normal*-regime descents.

None of this reaches the arbiter, though: the headline -0.0272 c/L sits
well inside this batch's own noise band (z=-0.52 vs threshold 1.92, 60th
percentile of the noise draws) — this run cannot distinguish the candidate
from noise at all, regardless of the mechanism question above.

**not_tested:**

- **Whether these columns carry a real shock-regime mechanism.** The
  run's actual (noise-indistinguishable) improvement concentrated in shock
  folds, not the elongated-normal folds the candidate was built around. A
  follow-up could re-target `station_px_change_3d`/`_14d`/
  `station_descent_decel` at shock-restoration dynamics — "has this
  station's own pace already stalled going into a shock" — rather than
  the late-descent-shape story tested here.
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
the underlying columns. If this series is revisited, the next hypothesis
should be built around the shock-regime pattern this run actually showed,
not the elongated-descent story tested here.

## Followups

- None filed — the per-column-attribution gap (SHAP, needs the candidate
  model persisted) is a repeat of the same open item noted on
  `network_move_breadth`; not worth a second bead for the same ask.
