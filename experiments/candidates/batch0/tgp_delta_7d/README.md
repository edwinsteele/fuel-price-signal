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

## Diagnostic — implementation-mismatch check (2026-08-18, post-dossier)

The `not_tested` item below about the harness's `add_columns`/exact-key-lookup
provider vs June's `.asof`-by-date closure was checked directly:
`diagnostic_asof_provider.py` (this directory) reruns the identical frozen
batch0 snapshot/seeds/fold geometry, but replaces the harness's exact
`(station_code, date)` lookup (822 hits / 88 misses in the run above) with a
date-only `.asof()` lookup matching June's `vel7_provider` exactly — valid
because `tgp_delta_7d` is confirmed market-wide (one value per date, verified
`nunique()==1` across stations before the run).

**Result: bit-identical to the harness's run.** 910/910 hits, 0 misses, pooled
`delta_cpl_held = +0.0941` — same to the decimal, same per-fold table across
all 14 folds, fold 13 unchanged at +1.0923. Full output:
`diagnostic_asof_provider.log`.

**This rules out the implementation-mismatch explanation.** The 88 exact-key
misses in the original run evidently never touched an economically live
decision (LightGBM's missing-value handling made no difference to any
buy/no-buy call on those station-days) — a faithful reproduction of June's own
method gives the same result. The rejection is not a plumbing artifact.

## Diagnostic 2 — baseline composition + data vintage (2026-08-19, bd fps-nor)

`diagnostic_baseline_composition.py` (this directory) reran the frozen batch0
snapshot with three arms at seed 42 and identical fold geometry: `R0_54` (June's
exact 54-column `BASE`), `cand_54` (`BASE + tgp_delta_7d`, June's `.asof`
provider), and `R0_64` (batch0's 64-column baseline, as an in-process anchor).
Outputs: `baseline_composition_{aggregate,deltas,per_window,fills}.csv`,
`diagnostic_baseline_composition.log`.

**Two facts settled before the run.** The TGP series is bit-identical between
June's `AIP_TGP_2026-06-19.xlsx` (put through `realised_tgp.py`'s own
`_load_tgp_pit()`) and batch0's `tgp_delta_7d` column — 3520 overlapping dates,
max|diff| 0.000000. And batch0's 64-column R0 is **not the production baseline**:
`batch_freeze.resolve_baseline_columns()` appended
`discover_brand_feature_columns(df)`, pulling in the 10 Phase 4b brand-trough
columns that were **evaluated and rejected on 2026-06-02** (docs/STATUS.md
§ Phase 4b; AGENTS.md § Canonical feature set). `lgbm_calibrated.joblib` carries
54. So the README's earlier framing — "10 new columns that did not exist in
June" — was wrong twice over: they landed 2026-06-02 (PRs #183/#184), 18 days
*before* the TGP experiment, and they are rejected, not new. Fixed in bd
`fps-sa1`; the general problem is bd `fps-zci`.

**The 2×2 (pooled `cpl_held`, always-buy 193.4148 in every cell):**

| baseline | no tgp | + tgp_delta_7d | delta |
|---|---|---|---|
| 54-col, June vintage | 189.7923 | 189.7529 | **−0.0394** |
| 54-col, batch0 snapshot | 189.6717 | 189.6776 | **+0.0059** |
| 64-col, batch0 snapshot | 189.6600 | 189.7541 | **+0.0941** |

The `R0_64` anchor reproduced batch0's 189.6600 **exactly**, so this process is
a faithful reproduction of the original run and the other cells are comparable.

**Attribution of the −0.0394 → +0.0941 swing (+0.1335 total):**

| source | contribution |
|---|---|
| baseline composition (64 vs 54, same snapshot) | **+0.0882** |
| data vintage (June → batch0 snapshot, same 54 columns) | **+0.0453** |

Both are real; neither alone explains it. Note the composition effect lands
mostly on the *candidate* side: the 10 brand-trough columns move R0 by only
−0.0117 c/L, but move the tgp arm by +0.0765. The vintage shift is largely
common-mode — R0_54 improved 0.1206 and cand_54 improved 0.0753 against June, so
the pairing absorbs most of it and only +0.0453 survives into the delta.

**But the headline is not the attribution — it's the resolution.** On the correct
54-column baseline and current data, `tgp_delta_7d`'s pooled realised delta is
**+0.0059 c/L**. That is assembled from per-fold pooled contributions an order of
magnitude larger that very nearly cancel:

| fold | 8 | 2 | 9 | 13 | 6 | 10 | others |
|---|---|---|---|---|---|---|---|
| contribution (c/L) | +0.654 | +0.288 | +0.210 | −0.209 | −0.208 | −0.046 | \|·\| < 0.03 |

The largest single fold contributes **110× the pooled result**. Across all 14
folds the two arms differ on **11 of 752 fills each way** (1.5%), and the two
*baselines* — which differ by 10 whole feature columns — differ on 6/7 fills.
June's own write-up already recorded that fold 10's entire −0.667 came from
**one** decision flip (station 585, 21 Feb 2024). One flip is worth roughly
0.05 c/L pooled, which is **larger than the effect that graduated this feature
in June**.

**Reading:** −0.0394 (June), +0.0059 (correct baseline, now) and +0.0941 (wrong
baseline) are all within a couple of decision flips of zero and of each other.
The realised arbiter, at 14 folds and ~750 fills, cannot resolve an effect this
small. The June graduation and the batch0 rejection were never in genuine
disagreement about the world — they disagreed about a quantity neither run could
measure.

## Judgement

**Grading verdict: superseded — the pipeline's `contradicted` grade was measured
against the wrong baseline, and the corrected measurement is inert.**

The batch0 run's `rejected` outcome (+0.0941, wrong sign) stands as a factual
record of what that run produced, but it does not mean what it appeared to mean.
Its R0 was production plus the rejected Phase 4b brand-trough group (fps-sa1),
and roughly two-thirds of the apparent sign flip is that baseline defect. On the
correct 54-column baseline against the same frozen snapshot, the pooled realised
delta is **+0.0059 c/L** — not a rejection, not a graduation, just nothing.

The pre-registered sign+concentration criterion cannot be adjudicated on this
evidence either way. Sign is meaningless at +0.006 c/L when a single buy/wait
flip is worth ~0.05, and the concentration read from the original run
(shock +0.272 vs normal +0.009, essentially all of it fold 13) does not survive
the baseline correction: on the 54-column baseline, fold 13 is +0.127 and the
largest contribution is fold 8 (+0.654, a normal fold), with fold 2 (+1.147 per
fold, +0.288 pooled) the next biggest. **The identity of the dominant fold
changes with the baseline** — which is itself the diagnosis, not a new finding
about fold 13.

**not_tested:**

- ~~Whether the original `extra_feature_provider` closure reproduces the negative
  pooled effect~~ — tested 2026-08-18, bit-identical. Ruled out.
- ~~Whether the baseline composition explains the flip~~ — tested 2026-08-19.
  It explains +0.0882 of the +0.1335 swing; data vintage explains the other
  +0.0453. Both real, neither sufficient, and both smaller than the instrument's
  resolution.
- ~~Whether fold 13 reflects a specific historical event~~ — moot. Fold 13's
  dominance was an artifact of the 64-column baseline; it is +0.127 on the
  correct one.
- **What the noise floor actually is.** Still the live question, and now the
  blocking one — `fps-3jj.9`. This run gives a first empirical handle (one
  decision flip ≈ 0.05 c/L pooled; per-fold contributions span −0.21 to +0.65
  for a +0.006 pooled result) but a proper band needs the placebo/shuffle
  distribution, not one paired run.
- **Why the 54-column baseline improved 0.1206 c/L between June's vintage and the
  2026-08-10 snapshot.** `always_cpl` is identical to the decimal (193.414835), so
  the 50 preferred stations' eval-date prices did not change; feature *code*
  barely changed in the window (the only relevant commit, `fc8de9f`, is a
  parity-claimed perf refactor of `lga_leadership` backfill). The plausible
  remaining mechanism is gap-fill adding historical rows for OTHER stations,
  which feeds the LGA/network/stickiness aggregates. Unproven.
- Fold 3's extreme seed instability in the original run (~280× cohort median)
  sits inside the normal-regime average; untouched here (single seed, 42).

## Recommendation

**Hold `#271` — but on the corrected reasoning, not the batch0 reject.**

The case for the re-lock rested on a June realised delta of −0.039 c/L. That
number is smaller than one decision flip in the same backtest, and it does not
reproduce: same 54 columns, same folds, same seed, two months later gives
+0.006. There is no realised evidence that `tgp_delta_7d` earns a place in
`FEATURE_COLUMNS`, and equally none that it does harm.

Do **not** close the track. The WFCV log-loss screen has liked this feature
consistently and independently of the baseline defect (June's velocity redesign;
batch0's Δll_all −0.0106 / Δll_hard25 −0.0498 — both favouring the candidate even
against the 64-column R0). A feature that reproducibly helps the screen while
sitting inside the arbiter's noise floor is exactly the case the project has no
tooling for yet.

**Unblocking step is `fps-3jj.9` (noise floor), not another paired run.** Until
there is a band, another 30-minute arbiter run produces one more number in the
±0.1 c/L cloud and settles nothing. `fps-3jj.9` was filed P3/dormant on the
assumption it was a nice-to-have; this investigation is the argument for
promoting it.

## Followups

- `fps-sa1` — batch_freeze resolved R0 to 64 columns (the acute bug). Fixed,
  PR #309.
- `fps-zci` (P1, design) — single-source the feature-scope contract and
  fingerprint the baseline into every result. The fingerprint is the part that
  would have surfaced this in a day rather than two months.
- `fps-3jj.9` (noise floor) — see above; now the blocking capability for any
  further `tgp_delta_7d` verdict.
- Worth a look eventually: PR #306 (`fps-3i7`) wired brand-trough columns into
  `ModelStrategy.decide()` for the live replay. That work was only necessary
  because R0 wrongly included those columns. The parity is fine to keep, but it
  is a measure of how far a wrong contract propagated before anyone questioned
  it.
