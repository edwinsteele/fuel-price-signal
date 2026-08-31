# batch1 / lga_trough_propagation

- **Date:** 2026-08-15 (snapshot) / dossiered 2026-08-26 / zone corrected 2026-08-30
  (`fps-hnp`) / flip table re-rendered 2026-08-31 (`fps-j6w`)
- **Branch:** main
- **SHA:** b8c01f8aa5e73d9ceaa57f7d41b9ec9393b5dd61
- **Status:** done — **inconclusive** (headline CPL); zone mechanism **confirmed** as of the 2026-08-30 correction — see § Correction note
- **Bead:** fps-cgf
- **Cadence:** `50/3.571/1d/10%` (batch1's canonical daily cadence)

## Hypothesis

The Sydney trough is not a moment, it is a wave: individual LGAs bottom on
different days and the wave runs from a known set of early LGAs outward. The
35 `days_since_trough_entry_<lga>` clocks are all in the lock, but the model
reduces them through 35 axis-aligned splits, and the one cross-sectional
summary it is handed (`lga_phase_std`) is a dispersion statistic — sign-blind
and order-blind. Std cannot distinguish "a third of the network has just
bottomed and the rest is about to follow" from "a third bottomed six weeks
ago and the rest is mid-descent," yet those are opposite buy/wait situations.
Breadth (`lga_trough_breadth_7d`), its velocity
(`lga_trough_breadth_delta_3d`), and the leaders' head start
(`lga_leader_lead_days`) recover exactly what the std discards.

**Predicted signature:** the model should keep waiting while
`lga_trough_breadth_delta_3d` is positive and `lga_leader_lead_days` is large
(the wave is still arriving), and should buy once breadth velocity rolls
over toward zero with leader lead already collapsed. Flips should be
WAIT→BUY moves a few days earlier than the lock manages, **concentrated on
non-shock folds** — the wave is a property of the ordinary cycle, and a
shock's wholesale move overrides propagation order entirely. Falsification
signature: gains driven by `lga_trough_breadth_7d`'s LEVEL alone with the
velocity and leader-lead columns inert.

## How to invoke this script

```bash
PYTHONPATH=. uv run python -m experiments.pipeline.runner \
  --batch-dir experiments/batches/batch1 \
  --candidate experiments/candidates/batch1/lga_trough_propagation.py
```

## Facts

Transcribed from `facts.json` — no new arithmetic.

**Provenance:** batch `batch1`, snapshot 2026-08-15, seeds [42, 43, 44, 45,
46], wall 1270.9s, status `graded`, 54-column baseline (`54:1a6ec2d84a69`).
Those five seeds are the **WFCV screen only**. The realised backtest — the arbiter —
ran on a **single seed** (`results.json` `meta.realised_seed` = 42); `facts.json`'s
provenance block does not carry this field. That is provenance, not a missing error
bar: the realised delta is a PAIRED difference in which both arms share the seed, the
data, the folds and the 54 baseline columns, so the fit's seed idiosyncrasy is
common-mode and cancels. Its error bar is the noise floor, built the same paired way
at the same fixed seed (`PLACEBO_SEED_DEFAULT = SEEDS[0]` = 42).

**Headline (realised CPL, held τ — the arbiter):**

| | R0 | candidate | delta |
|---|---|---|---|
| CPL held | 187.8211 | 187.7537 | **-0.0675** |

`effect_resolved`: **true**.

**WFCV log-loss** (descriptive colour, NOT the arbiter): Δll_all mean
-0.0003, Δll_hard25 mean -0.0053. Both favour the candidate, marginally.

**Zone (regime axis — safe to cut; a fold is an independent simulation, not
a row-label slice). Target = normal (the concentration the candidate
predicted), other = shock. Graded against the corrected empirical
shock-fold set `{1, 3, 4, 14}` (`experiments/batches/batch1/shock_folds.json`,
`fps-3tu`/`fps-hnp`) — see the Correction note below the Judgement section
for what changed and why:**

| | target (normal) | other (shock) |
|---|---|---|
| delta_cpl_own | -0.1073 | +0.0026 |
| n_fills | 1541 | 785 |

`resolved`: **true** — the target zone (normal) is favourable for the
candidate, and the zone the candidate predicted would be inert (shock) is
in fact indistinguishable from zero.

**Per-fold** (regime column reflects the corrected empirical shock-fold set
`{1, 3, 4, 14}`, not the fixed `{1, 4, 9, 13}` this dossier originally used
— folds 3 and 9 swap regime labels, as do 13 and 14 vs. the set below):

| fold | regime | n_fills | delta_cpl_own |
|---|---|---|---|
| 1 | shock | 190 | 0.0000 |
| 2 | normal | 186 | +0.6506 |
| 3 | shock | 188 | +0.2313 |
| 4 | shock | 239 | -0.0732 |
| 5 | normal | 185 | -0.1116 |
| 6 | normal | 135 | +0.0323 |
| 7 | normal | 235 | +0.0530 |
| 8 | normal | 87 | +0.7971 |
| 9 | normal | 112 | +0.2326 |
| 10 | normal | 96 | -0.5082 |
| 11 | normal | 161 | -0.5000 |
| 12 | normal | 184 | -0.0078 |
| 13 | normal | 143 | **-1.5646** |
| 14 | shock | 165 | -0.0864 |

No cell is suppressed (`min_row_cell_n` = 30, smallest cell here is 87).

`per_axis` is `null` — this candidate declared no `add_axis` beyond the
regime split above, so there is no path-coupled per-row zone to caveat here.

**Seed stability:** 7 cells exceed 5× the cohort-median seed_std. Six sit in
the ordinary 5.4–6.6× range (fold 2 `all`/R0, fold 3 `all`/R0, fold 2
`hard25`/R0, fold 3 `hard25`/R0, fold 3 `hard25`/candidate, fold 5
`hard25`/R0). The seventh is an outlier among the outliers: **fold 3, `all`
cohort, candidate run, 75.4× the cohort median** — an order of magnitude
past every other flagged cell. Fold 3 is a `shock` (other-zone) fold under
the corrected empirical split — it was `normal` (target-zone) under the
old fixed-index set, so this instability no longer sits inside the zone
the resolved verdict below rests on. It is also the single worst per-fold
flip-delta in the table below.

Every `seed_std` here is **WFCV log-loss**, not realised CPL —
`experiments/lib/gates.py`'s `seed_variance_gate` runs on `ll_all`/`ll_hard25`.
Fold 3's raw candidate `seed_std` is 0.5359 against a `Δll_all` of -0.0003: the
fit's seed noise is ~1700× the log-loss effect being measured. The blow-up is in
the EASY rows — the `hard25` cell for the same fold/arm is only 6.6×.
**No seed-variance measurement on CPL exists in this run.**

**Validation:** PIT passed (differential PIT truncation test found no
leak). INPUTS check passed. Candidate column NaN rate 0.0% for all three
columns.

**Noise floor** (`experiments/batches/batch1/noise_floor.json`, 10 draws at
arity 3 — matches this candidate's 3-column arity, `floor_arity_exceeds_run`:
false): band mean +0.0251 c/L, band std 0.0999 c/L. The candidate's -0.0675
c/L sits at the 80th percentile of the 10 observed noise draws
(`candidate_percentile_better_than_noise` = 80.0), and the parametric
single-candidate call is `z = -0.93` against a threshold of `t = 1.92`:
inside the band, not past it either direction.

**Decision flips** (`facts["decision_flips"]`, diffs R0's and the
candidate's `fills.parquet` directly on `(fold, station_code, date)` — not a
`rowpreds` proba-threshold crossing, which is a different, uncalibrated fit
from the one that drove the realised decision): run-level **272 flips / 84
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
difference has no stable denominator.

| fold | regime | flips | decns | litres R0 | litres cand | regret R0 | regret cand | regret Δ | Δ interval (2·SE) | run_contribution_cpl |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | shock | 0 | 0 | 0.0 | 0.0 | — | — | — (`no flips`) | — | +0.0000 |
| 2 | normal | 62 | 9 | 439.3 | 410.7 | 9.07 | 12.86 | +3.79 | [-8.90, +16.49] | +0.0469 |
| 3 | shock | 15 | 6 | 157.1 | 114.3 | 8.45 | 12.34 | +3.89 | [-3.80, +11.58] | +0.0172 |
| 4 | shock | 49 | 13 | 332.1 | 489.3 | 9.53 | 9.22 | -0.31 | [-15.43, +14.82] | -0.0053 |
| 5 | normal | 33 | 11 | 264.3 | 296.4 | 5.66 | 7.88 | +2.22 | [-5.55, +9.99] | -0.0082 |
| 6 | normal | 24 | 7 | 128.6 | 171.4 | 0.95 | 1.53 | +0.58 | [-10.87, +12.03] | +0.0023 |
| 7 | normal | 10 | 6 | 100.0 | 107.1 | 9.20 | 9.52 | +0.32 | [-12.11, +12.75] | +0.0036 |
| 8 | normal | 19 | 4 | 207.1 | 232.1 | 7.37 | 10.24 | +2.87 | [-15.99, +21.74] | +0.0587 |
| 9 | normal | 6 | 3 | 0.0 | 117.9 | — | 5.80 | — (`one arm only`) | — | +0.0172 |
| 10 | normal | 16 | 6 | 132.1 | 175.0 | 6.50 | 1.53 | -4.96 | [-10.14, +0.22] | -0.0346 |
| 11 | normal | 6 | 3 | 39.3 | 28.6 | 30.82 | 18.62 | -12.19 | [-34.62, +10.24] | -0.0333 |
| 12 | normal | 9 | 7 | 164.3 | 75.0 | 19.52 | 13.40 | -6.12 | [-21.78, +9.54] | -0.0006 |
| 13 | normal | 15 | 5 | 157.1 | 175.0 | 12.50 | 5.39 | -7.11 | [-28.76, +14.53] | -0.1163 |
| 14 | shock | 8 | 4 | 107.1 | 46.4 | 9.23 | 1.60 | -7.63 | [-16.50, +1.24] | -0.0064 |
| **all** | — | **272** | **84** | **2228.6** | **2439.3** | **9.32** | **8.49** | **-0.83** | **[-5.33, +3.67]** | **-0.0586** |

**Every resolvable regret delta sits inside its own 2·SE interval — 12 of
12, plus the pooled `all` row** (`regret_cpl_delta_inside_own_se`: `true`
throughout). Per `docs/routines/dossier.md`, an `inside_own_se` cell must
never be cited as evidence in the Judgement section, so nothing in this
table corroborates or contradicts the zone verdict. Two cells are not
resolvable at all and their reasons are stated verbatim rather than
blanked: fold 1 is `no flips` (both arms have zero differing fills that
fold), and fold 9 is `one arm only` — zero baseline-only fills against 6
candidate-only ones, the same shape seen in this batch's
`network_move_breadth` dossier, i.e. "candidate bought more often" rather
than "the two arms swapped timing." Neither is the same as a resolved zero.

The pooled row is the only run-level statement this table can make, and it
is that **both arms buy about equally well relative to what they could have
reached** (9.32 vs 8.49 c/L of regret, Δ -0.83 against a ±4.50 interval),
consistent with a -0.0675 c/L headline.

Two things the table surfaces that are not deltas. **Litres:** flip-only
volume is 2228.6 L on R0 against 2439.3 L on the candidate. `pooled_cpl` is
spend-weighted, so a 3.57 L top-up and a 42.86 L fill count identically as
flips but not in the CPL. **Dark fills:** `dark_fill_days` = **60** of 272
scored, in folds **4–7** — the Blue Mountains preferred-station outage also
behind this batch's 607-miss provider gap documented in the Review addendum
below. Those fills ARE scored, at the forward-filled price the tank
simulator itself bought at, so the outage does not bias the estimate, but it
thins and tilts exactly those folds.

**`run_contribution_cpl`** is what each fold's divergence is worth at the
scale of the whole run — a litres-weighted shift-share of that fold's own
`delta_cpl_own` (the whole fold, matching and flipped fills alike), not of
the flip-only figures. It reconciles: `run_contribution_total_cpl`
**-0.0586** + `composition_residual_cpl` **-0.0088** = **-0.0675**, the
headline `delta_cpl_held`. The residual is real, not rounding — a
litres-weighted average of per-fold ratios is not the ratio of pooled totals
when the arms' litres mix differs fold to fold. This run's
`tau_diverges_any` is `None` (**unavailable, never read as "checked,
agrees"**), so the residual is composition drift **and possibly tau drift**.

Summed by regime it reads **normal -0.0642** (10 folds, 61 decns) against
**shock +0.0055** (4 folds, 23 decns) — the same "favourable in normal,
inert in shock" shape the zone test grades `resolved=true`, now measured on
the metric that decomposes the headline rather than on pooled fold CPL. It
carries the same single-fold caveat the Judgement states below: fold 13
alone contributes **-0.1163**, so the other nine normal folds net **+0.0521**
between them. This is corroboration of the zone's shape from a second
decomposition, not independent evidence — both readings rest on fold 13.

![](per_fold_delta_bars.png)
![](seed_mean_vs_median.png)
![](realised_cpl_by_fold.png)
![](tau_sweep.png)
![](candidate_over_time.png)

## Judgement

**Signature grading: confirmed — corrected 2026-08-30 (`fps-hnp`), reversing
the original `contradicted` read. See the Correction note below for what
changed and why; the account below supersedes the original Judgement and
the 2026-08-26 Review addendum wherever they discussed the shock/normal
split.**

The candidate specifically predicted concentration in non-shock (normal)
folds, with shock folds inert because a shock's wholesale move overrides
propagation order. Graded against the corrected empirical shock-fold set
(`{1, 3, 4, 14}`, not the old fixed `{1, 4, 9, 13}`), the pipeline's zone
test now confirms this: `resolved=true`, the target (normal) zone is
favourable for the candidate (-0.1073 c/L) while the zone predicted to be
inert (shock) is statistically indistinguishable from zero (+0.0026 c/L) —
exactly the "concentrated in normal, inert in shock" shape the hypothesis
named. The aggregate is still not itself past the noise band: -0.0675 c/L
held-τ sits inside batch1's single-candidate band (z=-0.93 vs t=1.92), so
this is a zone-mechanism confirmation, not a graduation.

**But the confirmed zone rests entirely on one fold, and it is worth saying
so as plainly as the old `contradicted` read said the mirror-image thing.**
Excluding fold 13 (delta -1.5646 c/L, the single largest cell in the whole
run, litre-weighted using the actually-persisted `fills.parquet`) the
normal-zone delta flips from -0.1073 c/L to **+0.0588 c/L** — the opposite
sign. Every other normal fold nets out close to flat. This is the same
single-fold dependency the original `contradicted` verdict had (fold 13 was
then filed under shock and carried the whole -0.3794 c/L shock number); the
correction did not remove that fragility, it just re-filed which zone it
sits in and, in doing so, moved it into the zone the candidate actually
predicted. Unlike the old story, though, fold 13 itself is not one of this
run's seed-instability flags — that flag now sits on fold 3 (see below),
which under the correction moved OUT of the target zone and into shock
(where the zone reads as noise anyway), so the instability no longer
tangles with the confirmed reading the way it used to.

The flip-level detail is colour, not a second arbiter (per
`docs/routines/dossier.md`'s guardrail), but two things stand out. First,
fold 3 — now a shock (other-zone) fold — carries both the single worst
flip-delta (+10.78 c/L, the candidate's worst fold) and a 75.4×
cohort-median seed_std for the candidate run, an order of magnitude past
every other flagged cell in this run; it no longer sits inside the zone
the resolved verdict depends on. Second, the two largest favourable
flip-deltas (fold 11, -19.72 c/L, normal; fold 13, -19.05 c/L, normal) both
now sit inside the same (normal) regime — under the old labelling fold 13
was shock and this pair straddled the boundary the way `network_move_breadth`'s
folds 11–13 cluster did; under the correction all three of 11, 12, 13 are
normal for both candidates. The open question from the original dossier —
whether there's a real calendar-clustered event in that window both
candidates' columns catch, independent of any regime label — is if
anything sharpened, not resolved, by the correction: it's no longer
plausible to describe as "straddling a boundary" when the boundary moved
away from all three folds.

**not_tested:**

- Whether the propagation-wave mechanism holds with fold 13 excluded — as
  shown above, the confirmed zone flips sign without it. A wider seed
  sweep or a genuine re-run (this backfill reused the existing
  `fills.parquet`, so seed variance on CPL itself still doesn't exist for
  this candidate) would separate "the mechanism works, fold 13 is just
  where it shows up most" from "fold 13 is one unusual window carrying the
  whole read."
- The candidate's own proposed falsification test — whether gains trace to
  `lga_trough_breadth_7d`'s level alone with velocity/leader-lead columns
  inert — is untested here; it needs per-column SHAP/attribution against the
  candidate's fitted model, which this pipeline doesn't currently persist
  (same gap `network_move_breadth`'s dossier already named). Partial
  corroboration exists below (§ Correction note point 3, carried over from
  the original addendum): `lga_leader_lead_days`, not the breadth level, is
  this candidate's most lock-reconstructible column.
- Whether combining this with `network_move_breadth` (same batch, same
  cross-sectional-consensus family, same folds 11–13 pattern, now entirely
  inside `normal` for both candidates) does better than either alone.

**Recommendation:** the zone mechanism is now confirmed rather than
contradicted, but it is carried by a single fold (13) whose exclusion flips
the sign — treat this as reopened, not resolved. It should be checked
jointly with `network_move_breadth` (same folds 11-13 cluster, same
now-unambiguous regime placement) before either is pursued further, and a
fold-13-excluded re-read is the most direct next step for this candidate
specifically.

## Followups

- Per-column SHAP/attribution for this candidate's fitted model remains
  unbuilt (same gap noted in `network_move_breadth`) — worth its own bead if
  a future candidate's grading comes to depend on it, rather than building
  it speculatively now.
- If the batch retrospective revisits the folds-11-13(-4) cluster shared
  between this candidate and `network_move_breadth`, treat it as one
  question, not two.
- A genuine fold-13-excluded re-run (not just a leave-one-out read on the
  existing fills, which already flips the sign — see the Judgement section
  above) would settle whether the confirmed zone mechanism survives without
  its single largest fold.

## Correction note — 2026-08-30 (`fps-hnp`, backfill of PR #350 / `fps-3tu`)

PR #350 replaced the batch's fixed shock-fold index (`SHOCK_FOLDS =
{1, 4, 9, 13}`) with a per-batch empirical set derived from the baseline's
own val-fold log-loss. batch1's corrected set is **`{1, 3, 4, 14}`**
(`experiments/batches/batch1/shock_folds.json`) — folds 3 and 9 swap
regime labels, as do 13 and 14, relative to the old fixed set. This
dossier was originally written against the old set and its headline zone
verdict has flipped as a result: **`resolved` was `false` ("mechanism
contradicted"), it is now `true` ("mechanism confirmed")** — see the
Judgement section above, rewritten in place. The realised CPL headline
(-0.0675 c/L, still inside the noise band) is unaffected; only the
regime-axis zone grade depended on the shock-fold set.

This backfill reused the run's already-persisted `fills.parquet` and
`rowpreds.parquet` — no re-run of the realised backtest was needed or
performed. The Facts tables, the Judgement section, and this note are the
only parts of the dossier rewritten; the Review addendum below is kept
verbatim as a historical record of the 2026-08-26 review, with an
introductory flag on the one item (§1) that no longer applies to the
current regime split.

## Review addendum — 2026-08-26

**Superseded note (2026-08-30): §1 below decomposes the OLD shock zone
(`{1, 4, 9, 13}`) and is kept only as a historical record of that review —
its numbers no longer describe this dossier's live zone grade. §§2-4 do not
depend on the shock/normal split and remain current; §4's "retention gap"
is also now moot, since this backfill's own leave-fold-13-out re-read (see
Judgement above) used the `fills.parquet` that gap said was missing.**

A later interactive review of this dossier. Everything below is new analysis on the
same run, kept out of Facts because it is not transcription from `facts.json`; each
item names its own source. **The `contradicted` verdict is unchanged** — it rests on
the target (normal) zone being wrong-signed, which none of this touches.

**1. The shock-zone effect is one fold, not a zone.** Decomposing
`per_regime`'s shock `delta_cpl_own` of -0.3794 over its four folds (fill-count
weighted, so approximate — the reported figure is litre-weighted `pooled_cpl`, and
`fills.parquet` was not retained):

| fold | delta_cpl_own | n_fills | share of shock net |
|---|---|---|---|
| 1 | 0.0000 | 190 | 0% (zero flips either way) |
| 4 | -0.0732 | 239 | 8% |
| 9 | **+0.2326** | 112 | -12% (against) |
| 13 | -1.5646 | 143 | **104%** |

Excluding fold 13, the shock zone reads **+0.016** — sign flipped. The band's 0.0999
std is on the *pooled* 14-fold delta, so a single fold's null is roughly √14× wider
(~0.37); on that ruler fold 4 is 0.2σ, fold 9 is 0.6σ, and only fold 13 (4.2σ) is
distinguishable. The Judgement's original "folds 4 and 13" is corrected above.

**2. Folds 1-8 ran on a reduced station universe; folds 9-14 did not.** All 607 of
this run's `extra_feature_provider_misses` (`results.json` `meta`; 6300 requested =
5 preferred stations × 14 folds × 90 days) are station-days where two Blue Mountains
stations have no feature row:

| fold | window | missing | which |
|---|---|---|---|
| 1 | 2021-11-05..2022-02-02 (shock) | **180/450** | 414 and 18517, all 90 days |
| 2 | 2022-02-03..2022-05-03 | 8 | 414, 18517 |
| 4-8 | 2022-08-02..2023-10-25 | 67/90/90/90/82 | 414 |
| 9-14 | — | 0 | clean |

Verified against `data/cleaned/` raw prices: **the 2021 gap is not a closure.** Both
stations kept reporting through it (414: 44 raw rows inside; 18517: 75), at ~1/3 their
prior update rate — sparse enough to trip `fill.py`'s 28-day fill cap, after which
`labels.py` strips 7 days before and 90 after. 414's 2022-23 gap **is** a real
year-long rebuild (zero raw rows 2022-09..2023-04), but it reopened ~2023-07 while the
feature frame stays dark to 2023-10-17, so **fold 8's 82 missing days are label-strip
tail on a live, trading station.**

Consequences: fold 1's `delta_cpl_own` = 0.0000 with zero decision flips comes
from a 3-station fold. Coverage is not the whole explanation — this batch's other
two candidates do flip decisions in fold 1 (`network_move_breadth` 4/2,
`station_descent_dynamics` 21/0) — but a fold missing both of its dark stations
from the SAME council is a thin substrate for an LGA-propagation feature
specifically, so read the zero as thinned rather than as a clean null — and the council mix flips from 3 Blue Mountains /
2 Penrith to 1/2, which matters for a candidate whose mechanism is LGA propagation
(Blue Mountains lags the network trough by -1.45 days, Penrith by -1.14). Folds 4-7
are honest 4-station markets. Fold 1 and fold 8 are measurement gaps. Note this is
reduced sample, **not corrupted sample** — a missing feature row means no fill was
available and both arms skip it identically.

**3. The candidate's own falsification test is partly answerable already.** It asks
whether this is `lga_phase_std` in different clothes. `experiments/candidates/batch1/batch.json`
retains the generation-time R² of each column against the 54-column lock:

```
block_r2  0.685   (BLOCK_R2_FLAG = 0.9 — advisory, did not fire)
  lga_trough_breadth_7d        0.705
  lga_trough_breadth_delta_3d  0.497
  lga_leader_lead_days         0.855
```

The most-reconstructible member is **`lga_leader_lead_days`**, not the breadth level
the falsification test named. Note the block figure is the *mean* of the members, which
dilutes a single redundant column by design (`docs/routines/generator.md` § Redundancy),
and the hard |rho| >= 0.85 gate is **cross-candidate only** — it never compares a
candidate to the lock. Neither number reaches `facts.json`, so the original grading
could not see them. Correlation is not the model's *use* of a column, so this is
corroboration, not proof — but it is independent of any attribution machinery, and it
points the same way as fold 3's seed instability.

**4. Retention gap.** Neither `fills.parquet` nor the fitted model was kept, so the
fold-3-excluded re-read proposed in `not_tested` cannot actually be run against this
run's artifacts — it needs a full re-run. Worth more than the SHAP stage the Followups
contemplate.

## Re-render note — 2026-08-31 (`fps-j6w`)

The **Decision flips** section was rendering the pre-`fps-2js` table shape. Every field
the current `docs/routines/dossier.md` spec asks for was already in this run's
`facts.json` — emitted and never rendered. No new computation; re-rendered from the
committed artifact and verified row-by-row against it.

- **`decns` alongside `flips`.** 272 flips are **84 decisions** once cascades collapse at
  `cascade_window_days` = 7; folds carry 0-13 each.
- **Regret replaces `flip_cpl_delta`**, each delta beside its own 2·SE interval. **All 12
  resolvable folds and the pooled `all` row are `inside_own_se`**, so no cell in the table
  may be cited as evidence. The two unresolvable cells now state their reasons verbatim
  (`no flips` for fold 1, `one arm only` for fold 9) rather than showing a bare dash.
- **`run_contribution_cpl` added**, reconciling to the headline (-0.0586 + -0.0088 =
  -0.0675). By regime it reads **normal -0.0642 / shock +0.0055**, independently
  reproducing the "favourable in normal, inert in shock" shape the zone test grades
  `resolved=true` — with the same single-fold caveat the Judgement already states, since
  fold 13 alone supplies -0.1163 and the other nine normal folds net +0.0521.
- **Litres and dark fills surfaced** (2228.6 vs 2439.3 L flip-only; `dark_fill_days` = 60
  in folds 4-7).

**The Judgement is unchanged** — it rests on `delta_cpl_own`, the zone test and the noise
band, none of which this re-render touches, and it already named the fold-13 dependency.
