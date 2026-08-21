# batch0 / tgp_delta_7d — pipeline calibration run

- **Date:** 2026-08-19 (supersedes the 2026-08-18 run; see § History)
- **Branch:** main
- **SHA:** 0a69c971eb4d0bf6fc0318d9565d1039cb847e00
- **Status:** done — **calibration PASSED**
- **Beads:** fps-32p (candidate), fps-nor (investigation), fps-sa1 + fps-zci (the two defects it found)
- **Cadence:** `50/3.571/7d/10%` — **every CPL, delta and noise-band figure below is a
  7-day-cadence number.** This run predates the 2026-08-22 re-lock to daily
  (`fps-oqz`), and the numbers have not been restated: this is a lab-book record of
  what was measured, not a live claim. In particular the **~0.05 c/L single-flip
  quantum quoted below is a 7-day quantity, not a constant** — at 7d the tank forces
  67.6% of fills and only 244 are chosen, so each flip carries a large share of the
  total; at 1d there are 1832 chosen fills and the per-flip value is correspondingly
  smaller. Do not carry that 0.05 across the boundary as a resolution bar. Comparing
  anything here against a 1-day run requires a NEW batch frozen at 1d (`fps-aay`) —
  batch0 cannot be re-frozen (`batch_freeze` allows one freeze per batch) and its floor
  must not be recomputed at 1d in place, which `compute_noise_floor` now refuses
  (`fps-oqz`). `dossier_tables._noise_band()` separately refuses the cross-cadence
  run-vs-floor comparison (`fps-v8o`) rather than letting it look valid.

## What this run is for

batch0 exists to calibrate the AI-sourced pipeline, not to decide anything about
`tgp_delta_7d`. The question is narrow: **does the pipeline produce the same
number as an independently written script given the same inputs?**

The reference is `diagnostic_baseline_composition.py` in this directory — same
harness (`run_paired_realised_backtest`), same frozen snapshot, same 54-column
production baseline, same fold geometry, seed 42. `runner.py`'s realised stage is
single-seed at `SEEDS[0]=42`, so this is an exact-reproduction check.

## How to invoke this script

```bash
PYTHONPATH=. uv run python -m experiments.pipeline.runner \
  --batch-dir experiments/batches/batch0 \
  --candidate experiments/candidates/batch0/tgp_delta_7d.py
```

## Calibration result — PASSED

| | pipeline | independent reference | diff |
|---|---|---|---|
| R0 `cpl_held` | 189.6717090069284 | 189.671709 | 0 |
| candidate `cpl_held` | 189.6775604142693 | 189.677560 | 0 |
| **`delta_cpl_held`** | **+0.005851** | **+0.0059** | **0** |

The reference was computed *before* the fix that made this run possible, by a
different script, using a different feature-injection path (`.asof`-by-date
provider vs the pipeline's `add_columns` + exact-`(station_code, date)` lookup,
822 hits / 88 misses). It still matches to every printed digit — so the 88
exact-key misses again never touch an economically live decision.

**The pipeline is exact.** What was wrong twice before was the *contract handed
to it*, not the code.

## Facts

Transcribed from `facts.json` — no new arithmetic.

**Provenance:** batch `batch0`, snapshot 2026-08-10, seeds [42, 43, 44, 45, 46]
(realised stage uses 42 alone), wall 1543.3s, status `rejected`.

**Headline (realised CPL, held τ — the arbiter):**

| | R0 | candidate | delta |
|---|---|---|---|
| CPL held | 189.6717 | 189.6776 | **+0.0059** |

`effect_resolved`: **false**. `CONFIDENCE_EFFECT` is 0.5 by design for this run —
sign carries no information at this magnitude — so it scored conditionally.

**WFCV log-loss** (descriptive colour, NOT the arbiter): Δll_all mean −0.0137,
Δll_hard25 mean −0.0646. Both favour the candidate, and disagree with the
realised headline — the same divergence seen in June.

**Per-regime** (min cell n=30, nothing suppressed):

| regime | n_fills | delta_cpl_own |
|---|---|---|
| shock | 218 | +0.0320 |
| normal | 534 | −0.0174 |

**Shock folds** (the axis June's hypothesis targeted):

| fold | n_fills | delta_cpl_own |
|---|---|---|
| 1 | 51 | 0.0000 |
| 4 | 54 | +0.1094 |
| 9 | 57 | −0.2989 |
| 13 | 56 | +0.1271 |

Fold 13 — which drove essentially all of the 2026-08-18 run's apparent shock harm
at +1.09 — is +0.127 here. Its dominance was an artifact of the wrong baseline.
The largest contributor now is fold 2 (+1.147), a *normal* fold.

**Seed stability:** 8 cells exceed 5× the cohort-median seed_std; none of the four
shock folds. **Validation:** PIT passed, INPUTS passed, `tgp_delta_7d` NaN rate
0.0%.

**Noise floor** (`fps-3jj.9`, 5 R0-vs-R0 paired-seed draws on this batch): mean
−0.0135 c/L, std 0.142 c/L. The candidate's +0.00585 c/L sits at the **40th
percentile** of that distribution (`candidate_z_vs_band` = +0.14) — squarely
inside pure fit noise, not near either tail.

![](per_fold_delta_bars.png)
![](realised_cpl_by_fold.png)
![](seed_mean_vs_median.png)
![](tau_sweep.png)
![](candidate_over_time.png)
![](external_series_overlay.png)

## History — two contract defects, found by this candidate

The calibration took three attempts. Each failure was a real defect in what the
pipeline was told the baseline *was*, and neither was in the pipeline itself.

| run | R0 | delta | defect |
|---|---|---|---|
| 2026-08-18 | 64 cols, sorted | +0.0941 | wrong column **set** — carried the 10 rejected Phase 4b brand-trough columns |
| 2026-08-19 (first) | 54 cols, sorted | +0.0439 | wrong column **order** — alphabetical, not the locked artifact's |
| 2026-08-19 (final) | 54 cols, production order | **+0.0059** | — |

**Defect 1 — set (bd `fps-sa1`, PR #309).**
`batch_freeze.resolve_baseline_columns()` appended
`discover_brand_feature_columns(df)`, on the rationale that it mirrored
`train_lgbm`'s default resolution. It did — but that default is not the lock; the
lock is `train_lgbm --no-brand-features`. The 10 brand-trough columns are Phase 4b,
**evaluated and rejected 2026-06-02**. Every batch0 candidate was being graded
against production-plus-a-rejected-feature-group.

**Defect 2 — order (bd `fps-zci`, PR #310).** The same function returned the
corrected 54 columns *sorted*. The locked artifact is in group order
(`FEATURE_COLUMNS + LGA_FEATURE_COLUMNS + NETWORK_FEATURE_COLUMNS`), and the two
differ in **52 of 54 positions**. LightGBM breaks equal-gain split ties by feature
index, so a permutation fits a different model: 732/47,823 val rows change
probability (max |Δp| 0.090). Confirmed by rerunning the reference *in sorted
order* — it reproduced the pipeline's sorted run to 7 decimal places, so order was
the entire remaining discrepancy.

Both defects were the same mistake at different levels: **the baseline was derived
rather than declared.** Now `docs/CONVENTIONS.md` § "The baseline feature set is
declared, never discovered".

## Judgement

**On the pipeline: calibrated.** It reproduces an independent implementation
exactly. Its outputs can be trusted to the extent the contract it is handed is
correct — which is now pinned to the locked artifact, order included.

**On `tgp_delta_7d`: nothing, and now that has a ruler behind it.** +0.0059 c/L is
inert. It is neither June's −0.0394 graduation nor the batch0 rejection, and
none of the three was ever a measurement:

- per-fold pooled contributions span **−0.209 to +0.654 c/L** for a +0.006 total —
  the largest single fold is ~110× the pooled result, and they nearly cancel
- the two arms differ on **11 of 752 fills** (1.5%)
- one buy/wait decision flip is worth **~0.05 c/L pooled** — *larger than the
  −0.039 that graduated this feature in June*
- **against the noise floor (`fps-3jj.9`, computed 2026-08-19): the 40th
  percentile of pure fit noise** (band mean −0.0135, std 0.142 c/L; z = +0.14).
  This is the confirmation, not just informal reasoning: the pipeline's own
  seed-to-seed wobble is ~24× the size of the candidate's effect.

Three separate things each moved this headline by roughly one flip: the wrong
column set (+0.088), two months of data vintage (+0.045), and column order
(+0.038). None is large. The arbiter's headline is a near-cancellation of terms an
order of magnitude bigger, so anything that nudges a few decisions reorders the
answer.

**not_tested:**

- **Why the 54-column baseline improved 0.1206 c/L between June's vintage and the
  2026-08-10 snapshot.** `always_cpl` is identical to the decimal (193.414835), so
  the 50 preferred stations' eval-date prices did not change, and feature code
  barely moved in the window. Plausible mechanism: gap-fill adding historical rows
  for *other* stations, feeding the LGA/network/stickiness aggregates. Unproven.
- Fold 2's +1.147 — the current largest contributor. Not investigated, and on this
  evidence not worth investigating: the dominant fold's identity changed with every
  contract fix, which is a property of the metric, not of any fold.
- **The retrospective acceptance criterion** (`fps-3jj.8`, not yet built) that
  would formally rank a candidate against this band rather than eyeballing a
  percentile — this dossier reports the number but doesn't gate on it
  programmatically.

## Recommendation

**`#271` (TGP re-lock): hold — now on a resolved ruler, not just informal
reasoning.** The graduating number was below the instrument's resolution, does
not reproduce, and now measurably sits inside pipeline noise (40th percentile).
Not because the feature is harmful — it isn't — but because there has never been
realised evidence that it helps, and now there's a number confirming why the
existing evidence couldn't have shown it either way.

**Do not close the track.** The WFCV log-loss screen has favoured this feature
consistently and independently of both contract defects (here: Δll_all −0.0137,
Δll_hard25 −0.0646). A feature that reliably helps the screen while sitting inside
the arbiter's noise floor is a case the project has no tooling for yet — see
`fps-3jj.8` (retrospective, not yet built) for where that tooling would land.

## Followups

- `fps-sa1` — wrong column set. Closed, PR #309.
- `fps-zci` (P1) — single-source the feature-scope contract (set **and** order) and
  fingerprint it into every result. The fingerprint is what would have caught both
  defects in a day rather than two months; `results.json`'s `meta` still records no
  baseline identity at all.
- `fps-3jj.9` — noise band. Closed, PR #311; run against batch0 2026-08-19 (see
  above). Candidate sits at the 40th percentile of the band.
- PR #306 (`fps-3i7`) wired brand-trough columns into `ModelStrategy.decide()` so
  the live replay could produce columns the baseline should never have contained.
  Fine to keep as parity, but it measures how far a wrong contract propagated
  before anyone questioned it.
