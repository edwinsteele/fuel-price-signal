# batch0 / tgp_delta_7d — pipeline re-lock check

- **Date:** 2026-08-18
- **Branch:** main
- **SHA:** 94cdfef78f461c4687e026627f3f7673505a35b8
- **Status:** done
- **Bead:** fps-32p

## Hypothesis

7-day AIP Sydney TGP (wholesale floor) momentum predicts near-term retail descent,
concentrating in shock regimes where today's wholesale-floor position alone
misleads (the raw gap `station_minus_tgp_cents` helps calm but hurts shock;
`tgp_delta_7d` was supposed to rescue exactly that shock regression). Already
graduated the realised arbiter once, 2026-06-22
(`experiments/2026-06-20_leading_indicators/`, pooled −0.039 c/L, 14 WF folds).
This run is the AI-sourced pipeline's held-out re-lock check on that same claim,
not a new proposal — batch0 was purpose-built as the pipeline's known-graduate
sanity check.

## How to invoke this script

```bash
PYTHONPATH=. uv run python -m experiments.pipeline.launch --candidate experiments/candidates/batch0/tgp_delta_7d.py
```

## Facts

All numbers below are transcribed from `facts.json` in this directory — no new
arithmetic.

**Provenance:** batch `batch0`, snapshot 2026-08-10, seeds [42, 43, 44, 45, 46],
wall time 2022.1s, bead `fps-32p`, pipeline status `rejected`.

**Headline (realised CPL, held τ — the arbiter):**

| | R0 (baseline) | candidate | delta |
|---|---|---|---|
| CPL held | 189.66 | 189.7541 | **+0.0941** |

`effect_resolved`: **false**. Zone: not resolved — `CONFIDENCE_EFFECT` (0.9) did
not resolve true, so this run scored conditionally rather than against a
sign+magnitude gate.

**WFCV log-loss** (descriptive colour only, NOT the arbiter — see
`docs/CONVENTIONS.md` #250/#254, flat WFCV log-loss has shown opposite realised
outcomes before): Δll_all mean across folds −0.0106, Δll_hard25 mean −0.0498 —
both favour the candidate. Disagrees with the realised-CPL headline.

**Per-regime breakdown** (min cell n=30, nothing suppressed):

| regime | n_fills | delta_cpl_own |
|---|---|---|
| shock | 218 | **+0.2720** |
| normal | 535 | +0.0093 |

**Per-fold, shock folds only** (the axis the hypothesis targets):

| fold | n_fills | delta_cpl_own | seed-flagged? |
|---|---|---|---|
| 1 | 51 | 0.0000 | no |
| 4 | 54 | −0.0469 | no |
| 9 | 57 | 0.0000 | no |
| 13 | 56 | **+1.0923** | no |

(Full 14-fold table, all regimes, is in `facts.json`.)

**Seed stability:** `seed_flags` lists 8 cells exceeding 5× the cohort-median
seed_std. The largest is fold 3 (normal regime), R0 seed_std=1.595 — **~280×**
the cohort median — and its candidate-side counterpart at ~90×. Folds 2 and 4
also flag at 8–13×. None of the four **shock** folds are flagged, so fold 13's
+1.09 is not a seed-noise artifact.

**Validation:** PIT test passed (reached WFCV/realised stages — no leak).
INPUTS check passed. `tgp_delta_7d` NaN rate: 0.0%.

**Noise floor:** not available yet (`fps-3jj.9`, P3, not built) — no way yet to
check whether +0.0941 c/L clears a noise band or sits inside one.

**Plots:** `per_fold_delta_bars.png`, `seed_mean_vs_median.png`,
`realised_cpl_by_fold.png`, `tau_sweep.png`, `candidate_over_time.png`,
`external_series_overlay.png`.

![](per_fold_delta_bars.png)
![](realised_cpl_by_fold.png)
![](seed_mean_vs_median.png)
![](tau_sweep.png)
![](candidate_over_time.png)
![](external_series_overlay.png)

## Judgement

**Grading verdict: contradicted.** The predicted signature was a negative
pooled delta concentrated in shock. The pooled delta is positive (+0.0941,
worse) — opposite sign — so this fails the batch's pre-registered
sign+concentration pass criterion on sign alone, regardless of magnitude. The
concentration-in-shock axis called correctly (shock moves more than normal),
but in the wrong direction: shock is where the candidate hurts most, not where
it rescues the raw gap's known shock weakness. And even that pooled shock
number isn't a broad pattern — three of four shock folds are flat or mildly
helpful; fold 13 alone (+1.09, not seed-flagged) accounts for essentially all
of the pooled shock harm.

**not_tested:**

- Whether the original `extra_feature_provider` closure implementation
  (matching the June graduation) reproduces the negative pooled effect on this
  later snapshot. Batch0 used a simpler `add_columns` passthrough — a known
  mechanical difference flagged in the candidate's own `PREDICTED_SIGNATURE`
  ahead of the run. This run cannot distinguish "the effect decayed on newer
  data" from "the passthrough plumbing differs from what actually graduated."
- Whether fold 13 reflects a specific historical event where wholesale-floor
  velocity actively misled (a real, narrow failure mode worth understanding)
  versus a general property of shock regimes — only 4 shock folds exist here,
  and one drives the entire pooled shock number.
- Fold 3's extreme seed instability (~280× cohort median) sits inside the
  "normal" pooled average; whether a wider seed sweep changes the normal-regime
  read at all is untested.
- Whether +0.0941 c/L is distinguishable from pipeline noise — genuinely
  unknown, not just unfavourable, until `fps-3jj.9` lands a noise floor for
  this batch.

**Recommendation:** Not a re-lock candidate on this evidence alone — the
pre-registered sign+concentration criterion fails cleanly on sign, so this
run does not support pulling `tgp_delta_7d` into `FEATURE_COLUMNS` right now.
But a single outlier fold driving the whole effect, plus a known mechanical
difference from the implementation that actually graduated in June, makes this
read more like "needs a matched-implementation re-test" than a clean kill of
the underlying mechanism. Confirms the standing call to hold off on the TGP
re-lock (`#271`) pending more evidence, rather than closing the door on
`tgp_delta_7d` outright.

## Followups

- `fps-3jj.9` (noise floor, P3, dormant) would let a future re-run check
  +0.0941 c/L against an actual noise band instead of reading it in isolation.
- A matched-implementation re-test (`extra_feature_provider` instead of
  `add_columns`) would separate "data vintage decayed the effect" from
  "plumbing differs from June's graduation" — not filed as a bead yet; raise
  with the owner before the generator proposes batch2 candidates against this
  series.
