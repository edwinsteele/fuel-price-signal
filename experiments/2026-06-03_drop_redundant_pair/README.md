# drop_redundant_pair

- **Date:** 2026-06-03
- **Branch:** main
- **SHA:** 326ab80 (workdir + experiment files)
- **Status:** done (abandoned) — paired walk-forward CV showed regime-sensitive failures; the drop does not graduate

## Hypothesis
A previous SHAP-correlation experiment ([[redundancy_phase4b]]) flagged
`station_price_cents` and `station_minus_last_max_cents` as a very redundant
pair. If they are truly redundant, dropping either one should leave val
log-loss roughly unchanged.

## Setup
- Model: LightGBM (`fuel_signal.train_lgbm.build_pipeline`)
- Features: `data/features.csv` (50-col Phase 4 schema = `FEATURE_COLUMNS + LGA_FEATURE_COLUMNS`)
- Split: built-in `evaluate.split` — single train (1,733,335 rows) / val (59,811 rows)
- Three configs:
  - `baseline` — all 50
  - `drop_price` — drop `station_price_cents` (49 cols)
  - `drop_minus_max` — drop `station_minus_last_max_cents` (49 cols)
- Seeds: 0, 1, 2, 3, 42 (5 seeds × 3 configs = 15 fits)
- Script: `run.py` (this dir). Invoke with `PYTHONPATH=. uv run python experiments/2026-06-03_drop_redundant_pair/run.py`.

## Results

### V1 — single seed (seed=42)

| config | n_features | val_logloss | Δ vs baseline | val_brier |
|---|---|---|---|---|
| baseline | 50 | 0.321798 | — | 0.103535 |
| drop_price | 49 | 0.311959 | −0.00984 | 0.100039 |
| drop_minus_max | 49 | 0.303860 | **−0.01794** | 0.097170 |

Read at the time: "drop_minus_max is the clear winner; drop_price also looks
credible." This read was **wrong on drop_price** — see V2.

### V2 — 5 seeds, paired comparison

Per-config mean ± std across seeds [0, 1, 2, 3, 42]:

| config | val_logloss mean ± std | val_brier mean ± std |
|---|---|---|
| baseline | 0.322919 ± 0.003517 | 0.104398 ± 0.001249 |
| drop_minus_max | 0.311691 ± 0.005013 | 0.099664 ± 0.001671 |
| drop_price | 0.320134 ± 0.010953 | 0.103314 ± 0.003935 |

Paired Δ (config − baseline at the same seed):

| seed | Δ drop_minus_max | Δ drop_price |
|---:|---:|---:|
| 0 | −0.006460 | −0.013791 |
| 1 | −0.008696 | **+0.011910** |
| 2 | −0.013903 | **+0.003510** |
| 3 | −0.009145 | −0.005713 |
| 42 | −0.017938 | −0.009839 |
| **paired mean ± std** | **−0.011228 ± 0.004343** | −0.002785 ± 0.010504 |

Per-seed raw numbers in `results_per_seed.csv`; aggregates in
`results_summary.csv`. Train marginal-rate baseline log-loss on this val
window is 0.628283.

## Conclusion (step 1 outcome)

- **`drop_price` is out.** Sign of the paired Δ flips across seeds (helps on
  3/5, hurts on 2/5). Paired Δ = −0.0028 ± 0.0105 is indistinguishable from
  zero. The V1 Δ = −0.0098 at seed=42 was a lucky-seed coincidence.
- **`drop_minus_max` survives.** All 5 seeds favor the drop. Paired Δ =
  −0.0112 ± 0.0043 — roughly 2.6σ from zero, sign-consistent across seeds.
  Magnitude shrank from V1's −0.0179 (seed=42 was on the lucky end) to
  −0.0112 mean.
- Step 1 took us from "two candidates, drop_minus_max bigger" → "one
  candidate, smaller effect, real but borderline-credible".
- Proceeded to **step 2: paired walk-forward CV** on baseline vs
  drop_minus_max via `fuel_signal.cv_report`. See "Step 2" section below.

## Step 2 — paired walk-forward CV

`run_step2.py` builds two joblib artifacts in this dir (`baseline.joblib`,
`drop_minus_max.joblib`) and calls `fuel_signal.cv_report.run_paired_cv`
across 14 folds spanning 2021-11 → 2025-04. Per-fold results in
`step2_cv_results.csv`; full log in `run_step2.log`.

| fold | val window | n | baseline | model (drop_minus_max) | Δ |
|---:|---|---:|---:|---:|---:|
| 1 | 2021-11-05→2022-02-02 | 47,823 | 0.4019 | 0.4083 | +0.0064 |
| 2 | 2022-02-03→2022-05-03 | 49,966 | 0.2687 | 0.2685 | −0.0002 |
| 3 | 2022-05-04→2022-08-01 | 53,719 | 0.3766 | 0.3124 | −0.0642 |
| 4 | 2022-08-02→2022-10-30 | 58,317 | 0.4205 | 0.4869 | **+0.0664** ⚠ |
| 5 | 2022-10-31→2023-01-28 | 57,613 | 0.2940 | 0.2749 | −0.0191 |
| 6 | 2023-01-29→2023-04-28 | 57,361 | 0.2181 | 0.2167 | −0.0015 |
| 7 | 2023-04-29→2023-07-27 | 57,651 | 0.2512 | 0.2617 | +0.0105 |
| 8 | 2023-07-28→2023-10-25 | 59,700 | 0.3063 | 0.3033 | −0.0030 |
| 9 | 2023-10-26→2024-01-23 | 60,022 | 0.3305 | 0.4335 | **+0.1030** ⚠⚠ |
| 10 | 2024-01-24→2024-04-22 | 58,953 | 0.2543 | 0.2567 | +0.0024 |
| 11 | 2024-04-23→2024-07-21 | 59,790 | 0.3441 | 0.3535 | +0.0093 |
| 12 | 2024-07-22→2024-10-19 | 60,944 | 0.3049 | 0.3454 | +0.0405 |
| 13 | 2024-10-20→2025-01-17 | 59,434 | 0.3001 | 0.2987 | −0.0014 |
| 14 | 2025-01-18→2025-04-17 | 57,318 | 0.3619 | 0.3581 | −0.0038 |

Aggregate: 14 folds, **7/14 wins** (coin flip), median Δ = **+0.0011**, mean
Δ = **+0.0104**, std Δ = 0.0391. **Two folds exceed the +0.05 regression
alert threshold:** fold 4 (+0.066) and fold 9 (+0.103).

### Verdict

**Abandon.** `station_minus_last_max_cents` is regime-sensitive — redundant
on some windows (folds 3, 5: large wins) and load-bearing on others (folds
4, 9: large regressions). Across the 4-year span these cancel and the mean
is slightly worse than keeping it.

Step 1's val window (2025-03-25→2025-06-23) corresponds most closely to
fold 14, where Δ = −0.0038 (essentially flat). Step 1's Δ of −0.0112 was
*more generous* than the recent regime warrants, and the prior-regime story
is much worse. Single-window evaluation could not have caught this — only
paired walk-forward CV did.

### What the staircase taught us

| step | conclusion |
|---|---|
| seed=42 single-val | "drop_minus_max looks like a clear win (Δ=−0.018)" |
| 5-seed paired | "drop_minus_max real but modest (Δ=−0.011 ± 0.004)" |
| 14-fold paired CV | **"don't drop it — fold 9 alone is a +0.103 regression"** |

Each step had a chance to stop the work earlier. Step 1 culled `drop_price`
(a real call — its sign flipped across seeds). Step 2 was the one that
mattered for `drop_minus_max`. The original single-seed result was
misleading on both candidates.

## Methodology notes

### Why paired Δ > pooled std (the right decision rule)

The first version of `run.py` printed a *pooled* cross-config std and used 3×
that as a decision threshold. That was wrong. Pooled std mixes drop_price's
high variance (0.0110) into the comparison for drop_minus_max, and ignores
the paired structure — same seeds produce correlated noise across configs
because the LightGBM bagging RNG draws the same training subsets per seed.

Paired-Δ std for drop_minus_max is **0.0043**, lower than the
independent-noise prediction `sqrt(0.0035² + 0.0050²) ≈ 0.0061`. The shared
seed-noise cancels in the paired subtraction, so the paired comparison is
strictly sharper than comparing means + their own stds.

Future quick-experiment templates should default to paired-Δ reporting and
treat the per-config std as a *secondary* diagnostic.

## Followups
- **Investigate fold 9 (2023-10-26→2024-01-23) specifically.** A +0.103 logloss
  regression on 60k rows is unusual. What is `station_minus_last_max_cents`
  providing in that window that the other 49 features can't reconstruct?
  Likely a regime shift in cycle structure or an LGA-level event.
- **Re-examine the [[redundancy_phase4b]] SHAP analysis assumption.** SHAP
  correlation on a single training run conflates "redundant on this regime"
  with "redundant globally". The phase4b sweep used only the most recent val
  window; the redundancy pairs it surfaced should be revalidated against
  earlier folds before any are proposed for dropping. Filed as design issue
  [#195](../../../../issues/195).
- Cross-reference with [[project_lga_feature_mechanisms]]:
  `station_minus_last_max_cents` may flip mechanism (mean-reversion vs
  cycle-decoder) across regimes, which would explain the fold-by-fold sign
  flip.

## Notes for future fast-iteration experiments
Friction surfaced by this run that we should fix before the next ablation
sweep. All filed as GitHub issues:

1. **No `--drop-feature` / `--seed` flags on `train_lgbm.py`** — had to write
   a custom `run.py`. Issue [#191](../../../../issues/191).
2. **`PYTHONPATH=.` required** to import `fuel_signal` from
   `experiments/<dir>/run.py`. Bundled with #4 below. Issue
   [#194](../../../../issues/194).
3. **sklearn `UserWarning: X does not have valid feature names`** fires every
   `predict_proba` call. Bundled with #2 above. Issue
   [#194](../../../../issues/194).
4. **Loading 683 MB CSV** dominates wall time (~30 s). Parquet cache.
   Issue [#193](../../../../issues/193).
5. **No multi-seed mode by default** — quick ablations should report std so
   we can immediately judge whether Δ exceeds seed noise. Folded into the
   `--seed` work in [#191](../../../../issues/191).
6. **Single val window → paired walk-forward CV** requires trained joblib
   artifacts. A `cv_report --drop-feature X` mode would close the loop.
   Issue [#192](../../../../issues/192).

Surfaced *during* step 1 (added as I worked):

7. **Decision rule used pooled cross-config std** — wrong (see
   "Methodology notes" above). Paired-Δ analysis is the correct primary
   statistic. Future ablation templates should default to this.
