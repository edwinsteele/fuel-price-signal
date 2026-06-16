# Regime-local cycle-length denominator (#254 diagnostic)

- **Date:** 2026-06-16
- **Branch:** main
- **SHA:** 2603452 (at authoring; record `git rev-parse HEAD` of the actual run)
- **Status:** done — **no graduation** (feature is more accurate but economically neutral-to-negative; see Conclusion)

## Hypothesis

`cycle_mean_length` is an expanding all-history mean from 2016, so it averages
**across the COVID structural break** (cycle length steps ~28d → ~41d at
2020-03). The stale denominator inflates `cycle_pct_through` (= `days_since_peak
/ mean_cycle_length`) ~13% on average and worst in **fold 7's era** — the exact
fold where #214/#231 rejected feature A. Replacing the denominator with a
**regime-local, break-floored shrunk median** should improve the corrected phase
axis: **normal-fold median Δlogloss improves AND fold 7 does not regress** (the
falsifiable claim). This is the diagnostic that must graduate before the
production single-source fix (#254 ACs) is worth its feature-regen → retrain →
recalibrate blast radius, and before #253 (A retest) / #237 (non-A sweep) can
trust the ruler.

## How to invoke

```bash
# Cheap sanity check (no LightGBM, ~seconds) — run this first:
PYTHONPATH=. uv run python experiments/2026-06-16_regime_cycle_length/validate.py

# Heavy paired walk-forward CV (3 runs × 14 folds × 5 seeds = 210 fits):
PYTHONPATH=. uv run python experiments/2026-06-16_regime_cycle_length/paired_wfcv.py \
  2>&1 | tee experiments/2026-06-16_regime_cycle_length/run.log
```

`PYTHONPATH=.` makes `fuel_signal` importable; the scripts add their own dir to
`sys.path` for the sibling `cycle_regime` module (the dated dir can't be a
package — its name starts with a digit).

## Setup

- **`cycle_regime.py`** — `RegimeCycleDetector(CycleDetector)`, overriding only
  `_mean_cycle_length`. Production `fuel_signal/cycle.py` is **untouched** (this
  stays a diagnostic until it earns the production change).
- **Cycle features are recomputed live through `cycle.py`** — never read from the
  cached `features.csv`. `recompute_cycle_columns()` builds the metro-average
  series from the DB (read-only), runs both the unpatched `CycleDetector`
  (baseline) and `RegimeCycleDetector` (regime) per unique date, and broadcasts
  by date (cycle state is a per-date quantity shared across stations). The cached
  `cycle_mean_length` / `cycle_pct_through` are overwritten with the freshly
  computed **baseline** so R0 is the honest live post-#250 baseline and R0/R1
  differ in the denominator **only**. The other 52 features (length-independent)
  come from the cache unchanged.
- **Arms:**
  - **R0** baseline — 54 feat, denominator = unpatched expanding mean.
  - **R1** + regime denom — 54 feat, `cycle_mean_length`/`cycle_pct_through`
    swapped to the regime values (same model inputs, corrected values).
  - **R2** + `is_post_covid` — 54 baseline feat + a regime dummy (does handing
    the tree the regime flag substitute for fixing the denominator?).
- **CV:** `iter_folds_with_baseline_fit`, 14 walk-forward folds, seeds
  (42,43,44,45,46). Shock folds {1,4,9,13}. Lib helpers only (no inlined
  scaffolding).

### Estimator (`cycle_regime.py`) — frozen design

- **Break date `2020-03-23`** — frozen literal (the *cause* date, first NSW
  Stage-1 lockdown). Confirmed offline to coincide within one cycle with the
  single L2 break in the metro cycle-length series (mean 27.8d → 40.9d). Never
  re-estimated per fold (PIT trap).
- **Hard floor at the break** — pre-break rows see only pre-break cycles,
  post-break only post-break. A cycle is stamped at its **closing peak** and is
  post-break iff that peak `>= break_date` (puts the 33d cycle closing 2020-03-19
  pre-break, the 54d first-COVID cycle closing 2020-05-12 post-break).
- **Expanding median** of post-break cycle lengths (median, not mean, for
  fat-tail robustness against the 50–68d post-COVID tail).
- **Warm-up = pseudo-count shrinkage toward the pre-COVID median, k=2.**

#### Estimator interpretation note (mean vs median shrinkage)

The issue states two things that are in mild tension: *"expanding **median** … for
fat-tail robustness"* and the pseudo-count shrinkage written in **mean** form,
`(Σ real + k·prior)/(n+k)`. Implementing the literal mean form would re-import
exactly the fat-tail sensitivity the median was chosen to remove. We therefore
implement the **median analog of pseudo-count shrinkage**: augment the post-break
sample with `k=2` pseudo-observations at the pre-COVID-median prior, then take the
median. This honours the stated robustness rationale and reproduces the issue's
"~67% data-driven by the 4th post-break cycle" figure (`n/(n+k) = 4/6`).
**Owner confirmed 2026-06-16: median is the intended estimator.**

## Results

Three layers, each answering a different question. Headline: **the regime
denominator is genuinely more accurate but does not buy realised value — at a
held operating point it is mildly worse.**

### Layer 1 — feature accuracy (validate.py): CONFIRMED more accurate

**`validate.py` (pre-run sanity, SHA 2603452) — PASS:**

| date | base mean | regime | base pct | regime pct |
|---|---|---|---|---|
| 2018-06-30 | 26.68 | 26.00 | 0.450 | 0.462 |
| 2020-04-15 | 27.81 | 27.00 | 0.971 | 1.000 |
| 2020-09-30 | 28.82 | 34.50 | 1.110 | 0.928 |
| 2021-11-05 (fold 1) | 30.92 | 38.00 | 0.550 | 0.447 |
| 2023-06-01 (fold 7 era) | 32.58 | 38.00 | 1.995 | 1.711 |
| 2026-05-30 | 34.86 | 38.00 | 0.402 | 0.368 |

- Regime climbs ~27d → 38d and stabilises (the baseline mean lags to 32–35d).
- **0** large month-on-month downward retractions post-break → PIT-monotone /
  sticky (expanding median).
- **Fold-7 era:** base median 33.0d vs regime 38.0d → baseline inflates
  `pct_through` by **~1.15×** there (matches the ~1.2–1.3× memory claim; tails
  worse). The regime axis pulls the inflated >1 tail rows back toward the body.

### Layer 2 — WFCV log-loss (the screen): flat / fragile

14-fold paired WFCV, 5 seeds, 3 arms (R0 baseline / R1 regime denom / R2
`is_post_covid` dummy):

| arm | normal-fold median Δh25 | Δall | fold 7 Δh25 |
|---|---|---|---|
| R1 regime denom | −0.0128 (fragile — leans on seed-flagged fold 2) | +0.0006 (flat) | +0.0089 |
| R2 dummy | ~0 (inert) | ~0 | +0.0054 |

Both **fail** the pre-registered gate below — **but that gate was mis-specified**:
it inherited the #214/#231 *corner* framing (fold-7 rescue), whereas #254's
purpose is feature accuracy. The real lesson is deeper: **WFCV per-row log-loss
is the wrong arbiter** for a trough-timing feature — its value lands in realised
buyer outcome, which log-loss averages away (precedent #250: log-loss-flat,
realised-positive). See memory `feedback-wfcv-logloss-screen-not-verdict`.

### Layer 3 — realised backtest (the arbiter): neutral-to-negative at held τ

Paired realised-spend backtest, 2025-H2 test window, always-buy CPL 191.78 c/L.
Each arm = full pipeline (features → train_lgbm --no-brand-features → calibrate
→ score_phase2), recomputing cycle features live through the patched/un-patched
`cycle.py` (branch `experiment/254-realised-backtest` = regime; `main` = baseline
lock). No `results.csv` writes (`--skip-results-csv`).

| arm | τ | calib | model CPL | saving |
|---|---|---|---|---|
| regime @ its own τ | 0.20 | raw | 184.75 | +3.66% |
| **regime @ held τ** | **0.25** | raw | **185.83** | **+3.10%** |
| baseline lock @ τ | 0.25 | isotonic | 185.32 | +3.37% |

- **The +0.29pp "win" was the τ move, not the feature.** Held at the baseline's
  τ=0.25, the regime feature is **0.51 c/L worse** (185.83 vs 185.32, −0.27pp).
  The regime model's own τ 0.25→0.20 move buys 1.08 c/L; the feature *costs* 0.51.
- **Operating-point selection dominates this feature ~2×** (a τ step ≈ 1 c/L vs
  the feature's ~0.5 c/L, wrong direction).
- The raw 184.75/+3.66% would have sold an operating-point artifact as a feature
  win — why the held-τ cut is mandatory (lesson for #255).
- *Caveat:* the held-τ baseline (185.32) is the remembered lock (isotonic), not a
  fresh same-conditions run. Cut B (fresh baseline arm, staged on
  `experiment/254-realised-backtest-baseline`) was not run; it would tighten the
  −0.27pp and answer whether baseline also re-optimises to τ=0.20.

## Decision rule (gates) — issue #254

**PRIMARY (falsifiable):** normal-fold **median** Δ(`ll_hard25`) < 0 **AND**
fold-7 Δ(`ll_hard25`) ≤ +0.005 (does not regress). Shock-fold worst case
reported and bounded (the project has no systematic shock layer, so shock folds
do not sink the verdict). `hard25` = top-quartile baseline per-row log-loss per
fold (where a wrong phase axis bites hardest). Machine `GateSpec` recorded
alongside in `meta.json`. Seed-variance gate flags any cell >5× cohort-median
seed std.

## Conclusion

**Abandon — do not graduate #254.** The regime denominator is more *correct* but
correctness does not buy buyer outcome; at a fixed operating point it is mildly
harmful. The production single-source fix (regen → retrain → recalibrate,
disturbing the AC3 lock) is not earned.

**The deeper finding — accuracy ≠ objective, one level down.** Making cycle length
*more accurate* (a larger, regime-correct denominator → smaller `pct_through`)
produced a *worse* CPL. The model never wanted an accurate ruler; it wanted a
decision variable, and the expanding mean's *errors* are economically informative.
This is the same objective-vs-proxy divergence as WFCV-vs-CPL, recursing: the
minimum-cycle-length-error estimator is not the minimum-CPL estimator, because
trough-timing costs are asymmetric (missing a trough > buying early).

Two competing, separable mechanisms for *why* the inaccurate ruler helps (untested):
1. **Asymmetric-cost / trough-timing hedge** — the lagging mean reads the cycle as
   *more overdue than it is*, biasing the model to commit earlier; post-COVID
   (long, fat-tailed, jittery cycles) that early bias is protective.
2. **Regime clock** — the expanding mean drifts with calendar time (24d→35d), so
   it smuggles in a slow regime covariate the flat median throws away.

**Implications:** this *inverts* the #237/#253 "broken ruler" premise — the
"corrected" ruler is economically worse, so retesting on it is not the right test.
And it argues for designing phase features against the CPL objective directly (or
handing the model *both* denominators, whose difference is the elongation signal),
rather than chasing estimator accuracy.

## Follow-up: τ-sweep diagnostics — decisive close (2026-06-16, session 2)

Cheap post-hoc analysis over the saved WFCV row predictions (`rowpreds.parquet`,
R0 vs R1), no retrain. Scripts: `tau_coordinate.py`, `tau_by_fold.py`.

- **Aggregate (14 folds pooled):** the two rulers are near decision-twins.
  Buy-rate-vs-τ curves nearly coincide (≤0.3pp apart); both proxy-economics
  curves peak at **τ\*≈0.21** (R0 0.1334 vs R1 0.1301 c/row — stale a hair ahead).
  So the single-window realised **τ=0.20-vs-0.25 split was mostly operating-point
  noise**, not a structural shift. See `tau_coordinate.png`.
- **Hedge vs clock, resolved by fingerprint:** a real *information* (clock) signal
  would cost the honest ruler on the threshold-free measure (log-loss) — it is
  **flat → not clock**. A cost-preference (hedge) is absorbed once each ruler
  picks its own honest τ — both land ~0.21 → **hedge, already handled by τ**. Both
  homes empty.
- **Per-regime split (the escape hatch):** errors localized to the elongation era
  could cancel in the pool. They don't. **Fold 7 (ELONG)** — where the denominators
  diverge most in *value* — shows the **lowest** decision disagreement of all 14
  folds (**1.3%**); per-fold Δcpr is sign-random scatter (±0.06) with no regime
  pattern. The effect isn't hiding anywhere.

**Refined conclusion.** `cycle_mean_length` is **mid-pack** (mean|SHAP| 0.4414,
importance rank 25/52, r +0.18 w/ target) — *not* "barely used." The model leans on
its slow **drift** (24→38d = a calendar/regime clock), not the
expanding-vs-regime-median distinction the accuracy fix changes. **The regime clock
is real but already captured**; a more "accurate" denominator just swaps one clock
shape for another → wash. So there are now **two distinct reasons "accuracy ≠
objective"**: (1) asymmetric loss [original], (2) the corrected component has low
leverage / is already captured [this].

**Dual-denominator feature: not worth building** — it presumes an *uncaptured*
clock signal the data says is already captured.

**Dormancy + wake-up condition.** Regime-median `pct_through` is a built, cheap,
harmless descriptive covariate — keep it **dormant**, don't engineer around it.
Wake-up requires BOTH gates:
1. **Regime-shaped error** — model loss, stratified by regime, is materially
   elevated in elongated/transition rows vs normal. If not → dormant, full stop.
2. **Oracle existence check** — an oracle regime feature (true cycle length/regime,
   full-info, train-only, leaky) cuts that excess loss **beyond** what the existing
   drift already delivers. Yes → build a PIT-safe handle, *then* a targeted
   phase×regime interaction test. No → dormancy confirmed.

## Followups

- **#254** — close: investigated, not graduating (accuracy is the wrong objective).
- **#255** — realised-backtest harness must hold/scan τ (this experiment's raw
  184.75 nearly read an operating-point artifact as a feature win).
- New design issue (forward direction): decompose why the inaccurate ruler helps
  (hedge vs clock) + CPL-objective / dual-denominator phase feature.
- **#237 / #253** — premise inverted; revisit framing, do not retest on the
  "corrected" ruler.
- The realised backtest here is **single-window**; treat the −0.27pp as indicative,
  not final (Cut B / walk-forward via #255 would harden it).
