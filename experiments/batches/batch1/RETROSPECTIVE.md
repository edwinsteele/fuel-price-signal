# batch1 — retrospective

- **Date:** 2026-09-03
- **Batch:** `experiments/batches/batch1`
- **SHA:** `61ffe5f` (the `git_sha` stamped in `retrospective_facts.json`)
- **Candidates filed:** 5

## What this is

The payoff artifact for aim (b) — gaining experience with an AI-sourced candidate
pipeline — not a verdict on any individual candidate. Those live in each candidate's own
`README.md`. Everything in **Facts** traces to `retrospective_facts.json`, written by
`experiments/pipeline/retrospective.py`; **Judgement** is prose on top of it.

batch1 is the first batch where this artifact can do its job. batch0 had one candidate, so
its leaderboard could not differentiate and its calibration read was a single point.

> **Note on the SHA.** `retrospective_facts.json` was computed at `61ffe5f`, immediately
> after PR #359 merged and *before* the `fps-rlh` grading backfill (`cd3de8f`) landed. That
> ordering cannot affect any number here: `retrospective.py` reads no `grading` field
> anywhere — the outcome tally keys on `provenance.status`, calibration on
> `candidate.confidence_*` with `headline.realised.effect_resolved`/`headline.zone.resolved`,
> and the leaderboard on `noise_band_z` / `delta_cpl_held` / `tank_params`. The signature
> grades quoted under Judgement are read from the candidates' `facts.json`, not from this
> payload.

> **Status of the dossiers this rests on (`fps-3ug` — RESOLVED 2026-09-03).** Every figure
> below traces to `retrospective_facts.json` or the post-#359 `facts.json`; all were verified
> against those payloads before this was committed. The five candidate `README.md` files
> beside them quoted PRE-#359 figures when this was written; they have since been refreshed,
> and **every figure in this document was re-verified against the refreshed dossiers.**
>
> **No candidate's Judgement conclusion changed**, so the signature-grade table and the
> decoupling finding stand as written. Two supporting claims inside dossiers did move, and
> neither touches anything asserted here: `tgp_cycle_displacement`'s fold-3 exclusion is no
> longer a near sign flip (restated in `not_tested` below), and
> `stickiness_phase_saddle`'s descriptive per-regime shock cell changed sign at a negligible
> magnitude — that candidate declared no zone TARGET, so nothing is graded on it. The one
> qualitative change the regen made across all five dossiers is that `dark_fill_days` went to
> zero and `extra_feature_provider_misses` fell 607 → 278: the closed-station days are no
> longer visited at a stale frozen price. No number in this document reads either field.

---

# Facts

## Leaderboard

Ranked by `noise_band_z` (more negative is better — `delta_cpl_held` is a COST).

| candidate | mechanism family | status | cadence | `delta_cpl_held` | z | percentile (colour) | clears batch bar? |
|---|---|---|---|---|---|---|---|
| tgp_cycle_displacement | wholesale-lead | graded | 50/3.571/1d/10% | −0.2077 c/L | −2.3295 | 100.0 | **no** |
| network_move_breadth | directional-network-breadth | graded | 50/3.571/1d/10% | −0.1627 c/L | −1.8794 | 100.0 | no |
| stickiness_phase_saddle | station-heterogeneity-interaction | graded | 50/3.571/1d/10% | −0.0761 c/L | −1.0128 | 80.0 | no |
| lga_trough_propagation | lead-lag-propagation | graded | 50/3.571/1d/10% | −0.0672 c/L | −0.9233 | 80.0 | no |
| station_descent_dynamics | station-own-price-dynamics | graded | 50/3.571/1d/10% | −0.0237 c/L | −0.4886 | 60.0 | no |

**Cadence.** Every `delta_cpl_held` above is in the units of a **1-day** evaluation
interval (`tank_params` = `50/3.571/1d/10%`), and the noise band was measured at the same
cadence (`noise_floor.tank_params`). All five rows agree — `build_leaderboard` refuses to
rank a mixed-cadence batch rather than produce a table across them, and it did not fire.
The stamp is not decoration: `fps-fii` measured realised CPL at 189.67 / 187.85 / 187.82
c/L across 7 / 2 / 1-day intervals on an otherwise identical run, a spread an order of
magnitude larger than the deltas ranked here.

## Noise band and the multiple-comparisons bar

| quantity | value |
|---|---|
| draws (`n_draws`) | 10 |
| placebo columns per draw | 3 |
| band mean `delta_cpl_held` | +0.0251 c/L @ 1d |
| band std | 0.0999 c/L @ 1d |
| single-candidate z bar | 1.9226 |
| **family-wise (batch) z bar** (`family_wise_z_threshold`, N=5) | **2.9591** |
| family-wise percentile threshold (colour only) | 99.0 |

**No candidate clears the batch bar.** The best, `tgp_cycle_displacement` at z = −2.3295,
falls short of 2.9591. One candidate clears the *single-candidate* bar (|z| = 2.3295 >
1.9226) — that is the bar its own dossier was graded against, and it is a different,
easier question than being the best of five against the same band.

The two 100.0 percentiles are descriptive colour only, and are a good illustration of why:
with 10 draws the empirical rank has 11 distinct values, so "beat every draw" and "beat
every draw by a wide margin" print the same number. Two candidates share it while their z
values differ by 0.45.

**Draw-collision check (`fps-3jj.24`, decided 2026-08-27 — not re-opened here).** batch1's
committed floor carries a seed collision that makes its bar slightly too easy; the
correction is monotone-hardening, so it can only ever demote a row, never promote one. Each
row that clears a bar must therefore be checked against the flip window it would fall
through:

| gate | flip window on \|z\| | rows inside |
|---|---|---|
| single candidate | [1.9226, 1.9797) | none |
| batch (N=5) | [2.9591, 3.1030) | none |

`tgp_cycle_displacement`, the only row clearing anything, sits at |z| = 2.3295 — above the
single-candidate window, so it clears under the corrected ruler too. **The collision did
not change any verdict.** (This is the check `docs/routines/retrospective.md` deferred:
`tgp_cycle_displacement` was the one candidate still unrun when that note was written.
Note also that every `facts.json` emits `effective_n_draws: 10.0`; that field is precisely
what `fps-3jj.24` says is wrong, so it is not evidence the band is clean.)

## Outcome tally

| status | count |
|---|---|
| filed (universe) | 5 |
| `graded` | 5 |
| `disqualified` | 0 |
| `aborted_candidate` (ordinary bug) | 0 |
| `aborted_candidate:leak_by_declaration` | 0 |
| `never_run` | 0 |
| `retryable_incomplete` | 0 |
| `pending_dossier` | 0 |

**The tally is clean** — no missing data, so the conclusions below are not written around a
hole. **Leak rate: 0 / 5 = 0%.** No candidate named a label column
(`future_min_cents`/`label`) in its `INPUTS`/`COLUMNS`; the generator produced no oracle
features in this batch.

## Signature grades

From each candidate's own `facts.json` `grading` block (backfilled from the dossier prose
under `fps-rlh`, `cd3de8f`):

| candidate | signature grade |
|---|---|
| lga_trough_propagation | matched |
| network_move_breadth | inconclusive |
| stickiness_phase_saddle | inconclusive |
| station_descent_dynamics | contradicted |
| tgp_cycle_displacement | contradicted |

## Confidence calibration

**`insufficient_data: true`** — 6 usable pairs pooled across every batch, against
`min_calibration_n` = 10. Scope is cumulative across all batches, not batch1 alone. No
calibration verdict is claimed below, and no mean is reported: the module returns `None`
for both.

| candidate | batch | `confidence_effect` | `effect_resolved` | `confidence_zone` | `zone_resolved` |
|---|---|---|---|---|---|
| tgp_delta_7d | batch0 | 0.50 | false | 0.50 | null |
| lga_trough_propagation | batch1 | 0.45 | true | 0.45 | true |
| network_move_breadth | batch1 | 0.60 | true | 0.35 | true |
| station_descent_dynamics | batch1 | 0.45 | true | 0.30 | false |
| stickiness_phase_saddle | batch1 | 0.35 | true | 0.15 | null |
| tgp_cycle_displacement | batch1 | 0.30 | true | 0.55 | false |

---

# Judgement

## Did the headline candidate hold up against noise?

No — not at the bar that matters for a batch. `tgp_cycle_displacement` is the clear
economic leader (−0.2077 c/L @ 1d) and the only candidate to clear its own
single-candidate bar, but it does not reach the family-wise bar of 2.9591, and that is the
right bar for this document: five candidates were graded against one band, and picking the
best of five is exactly where a noise delta gets promoted to a finding. **batch1 produced
no graduation.**

That is not the same as producing nothing. The honest summary is that batch1 bought
information, not a feature.

## The batch's most useful finding is a decoupling

Economic rank and mechanism-prediction accuracy came apart, and they came apart at both
ends:

- The **best** economic performer, `tgp_cycle_displacement`, has a **contradicted**
  signature. Its own `PREDICTED_SIGNATURE` pre-registered a shock-concentrated win as
  evidence *against* the wholesale-lead mechanism it proposed — and that is precisely what
  showed up (shock −0.3079 vs normal −0.1613 c/L @ 1d, `zone.resolved = false`). It made
  money in the place it said would falsify it.
- The **only matched** signature, `lga_trough_propagation`, ranks 4th of 5 and sits well
  inside the band (z = −0.9233). Its zone test resolves true in the predicted shape, but on
  an aggregate that is indistinguishable from noise, and the confirmed zone rests almost
  entirely on a single fold.

So the pipeline separated "did the model make money" from "was the proposed reason right",
and in this batch those two answers pointed at different candidates. A screening process
that only ranked on `delta_cpl_held` would have promoted a candidate whose own falsification
criterion had fired, and a process that only graded signatures would have promoted one whose
effect is noise. Aim (b)'s value here is that keeping both instruments was what made the
distinction visible at all.

## Did the confidence priors predict anything?

**Not answerable yet, and the payload says so** (`insufficient_data: true`, 6 of 10 needed).
The pairs are reported above without a read. Two structural observations about the
instrument, which are findings about the pipeline rather than calibration claims:

1. **`effect_resolved` has no within-batch variance in batch1** — all five are `true`, and
   the only `false` in the pooled record is batch0's `tgp_delta_7d`. The calibration axis
   compares mean confidence when resolved-true against resolved-false, so batch1 adds five
   points to one side of that comparison and none to the other. More batches of this shape
   will grow `n_usable` without making that particular axis any more informative. Worth
   deciding, before batch2, whether `effect_resolved` is the right resolution variable or
   whether it saturates.
2. **`zone_resolved` looks more discriminating and is the axis `fps-2sf` deliberately rated
   above effect.** Among the four candidates with a non-null zone result, the *highest*
   `confidence_zone` in the batch (0.55, `tgp_cycle_displacement`) resolved **false**, while
   0.45 and 0.35 resolved **true** and 0.30 resolved false. With n=4 this is an anecdote,
   not an anti-correlation, and it is recorded here only so the next retrospective can
   append to it rather than rediscover it.

## Mechanism-family concentration — a finding about the generator

Five candidates declared five distinct `mechanism_family` labels, which reads as full
diversity on its face. It is not quite that. Two of the five —
`directional-network-breadth` (`network_move_breadth`) and `lead-lag-propagation`
(`lga_trough_propagation`) — are both **cross-sectional consensus statistics over the
station network**, differing mainly in whether the consensus is read as a contemporaneous
breadth count or as a propagation ordering. Both modules disclosed the adjacency to the
already-graduated families in their own `PRIOR_ART` rather than relabelling around it,
which is the behaviour the generator brief asks for and is worth recording as a positive.

Whether that is *two ideas* or *one idea wearing two hats* is not settled by this batch, and
the available instruments cannot settle it: the redundancy check in each `facts.json`
measures a candidate's columns against the **locked feature set**, not against the other
candidates in its own batch. Notably both landed `zone_resolved: true` — the only two in the
batch to do so — which is weak evidence they are reading the same underlying structure.

For aim (b) the actionable version is: **the generator's own family labels are not a
sufficient diversity control**, because two labels can name one mechanism viewed from two
angles. A batch-internal redundancy check (candidate columns against each other, not just
against the lock) would settle it, and does not exist today.

## One-at-a-time screening did not validate any combination

Every number here is from an additive, one-at-a-time screen: each candidate was run against
the same baseline alone. Nothing in this batch tests two of them together, and the two
best-performing candidates are the two that may be reading the same structure. If any
combination is pursued, `tgp_cycle_displacement` + `network_move_breadth` is the pair worth a
combined run — but that is a **new measurement**, not an inference available from this table.

## `not_tested`

- Whether any batch1 candidate clears the batch bar in a **combined** run. Not tested; the
  screen is one-at-a-time by construction.
- Whether `network_move_breadth` and `lga_trough_propagation` are redundant with each other.
  Not tested and **not testable with what is built** — redundancy is measured against the
  lock, never candidate-to-candidate within a batch.
- Whether the confidence priors are calibrated. Not tested — pooled n is 6 against a
  required 10; genuinely cross-batch-pending, not a null result.
- Whether `tgp_cycle_displacement`'s shock-concentrated effect is a real regime mechanism or
  fold-3-led. Its own dossier flags that 2 of the 4 shock folds actually hurt the candidate,
  so the regime aggregate is not a uniform shock story, and that fold 3 carries most of the
  rest. **How much it carries moved in the #359 regen** and the dossier was restated for it
  under `fps-3ug`: the pre-regen leave-one-out took the shock zone to roughly zero — a near
  sign flip — while post-regen, recomputed fill-count-weighted on both sides, excluding fold
  3 leaves −0.0878 c/L @ 1d against a −0.2458 fill-weighted shock zone. Fold 3's share falls
  from 95% to 64%, because fold 4's own cell grew to −1.2103. Still most of the effect, no
  longer nearly all of it, and no longer a sign flip. Not settled here — and note this is a
  caveat *against* the candidate weakening, so the contradicted signature grade above is if
  anything better supported than when this batch was first graded.
- Whether a turn-gated `stickiness_phase_saddle` pays. The mechanism was found real but
  mispriced; the gated construction is unbuilt (`fps-3jj.10` lead 6).
- Whether the batch would look different at another cadence. Everything here is 1d;
  no candidate was re-run at another interval.

---

# Recommendation

**Graduate nothing from batch1.** No candidate clears the family-wise bar, and the one that
clears its single-candidate bar has a contradicted signature — the weakest possible
combination of "made money" and "for the reason claimed".

For the next generator session, in priority order:

1. **Carry `tgp_cycle_displacement` forward as a measurement, not a graduate.** It is the
   strongest economic result the pipeline has produced and the only one to clear any bar,
   but its mechanism story is falsified by its own pre-registered criterion. `fps-cde`
   already plans to run both TGP expressions as batch2 candidates against a floor that
   applies to them — that is the right next step, and it should be read as re-testing the
   effect, not as promoting it.
2. **Add a batch-internal redundancy check before batch2 grades anything.** The
   family-label diversity control is weaker than it looks, and batch2 is planned at 10–15
   candidates, where the chance of two labels naming one mechanism is higher, not lower.
3. **Decide whether `effect_resolved` is the right calibration axis** before adding more
   batches to a pooled record where it may not discriminate.
4. **Do not re-run `stickiness_phase_saddle` as written** — its dossier settled the effect
   size on three independently-built rulers. The live successor is the turn-gated
   redesign.

The batch's headline for aim (b): the pipeline worked. It ran five AI-sourced candidates
end to end, leaked nothing, graded all five, and — most usefully — told two different
stories about its best candidate that a single-instrument screen would have collapsed into
one wrong answer.
