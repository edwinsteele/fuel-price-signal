# Non-parametric phase model — retest of phase-residual diagnosis

- **Date:** 2026-06-05 (parked mid-design), resumed and completed 2026-06-06
- **Branch:** main
- **Status:** **complete — phase-residual concept retired**
- **Predecessor:** `experiments/2026-06-04_cycle_pct_through_interaction/` (step 4 abandoned ablationA with worst-fold +0.101)
- **Verdict:** additive (51-feature) is neutral; ablationA (49-feature, drop siblings, add lookup) regresses mildly on stable folds and catastrophically on fold 9 (+0.066). The non-parametric shape correction improved fold 9 from +0.101 → +0.066 vs the predecessor's linear-interp formula — the shape *was* mis-specified — but the diagonal-projection issue is real and independent of shape. **The phase-residual concept is dead.** See § Outcome for the mechanism-level story (it's not generic "shock fragility" — it's a specific late-descent × cycle-elongation × lookup-tail-noise triple-stack).

## Hypothesis

Step 4 of the predecessor experiment abandoned the engineered feature `station_minus_expected_phase_price = station_price − (last_min + pct × (last_max − last_min))` on the basis that any single residual scalar drops absolute-anchor information the siblings preserve. The end-of-step diagnosis (captured in `project_phase_residual_fragile`) was: *"the issue is fundamental to projecting onto one diagonal, not the linear-interpolation approximation per se."*

This experiment tests the diagnosis directly by replacing the linear-interp formula with a **non-parametric** `E[price | phase]` lookup fit on training data. If shocked-regime folds (1, 4, 9, 13) still regress, the diagonal-projection issue is confirmed and all phase-residual variants are dead. If fold 1 *fixes*, the linear-interp shape was the real problem and there is still a feature-engineering win on the table.

## Major mid-session finding (must read before continuing)

**`cycle_pct_through` is `days_since_last_peak / mean_cycle_length`** (see `fuel_signal/cycle.py:142` and the `CycleState` docstring at line 29-30). Not "fraction of the way between last min and last max" as the step-4 engineered formula implicitly assumed.

Consequences:

1. **pct = 0** → just peaked. **pct ≈ 0.5** → roughly at trough. **pct = 1.0** → next peak / overdue. So the actual price trajectory vs pct is **peak → fall → trough (~0.5) → rise → peak** — non-monotonic.
2. The step-4 formula `expected = last_min + pct × (last_max − last_min)` is **monotonically rising**, i.e. **inverted in shape** relative to the variable it claims to model. The step-4 engineered feature was therefore structurally mis-specified, not merely "linear-approximation crude". Trees still found utility because the residual is a deterministic function of inputs, but it wasn't subtracting out phase the way its name implied.
3. The diagnostic test for this experiment now does **two things at once**: (a) **shape correctness** (replace monotone misspecification with the empirical non-monotonic shape), and (b) **diagonal-projection sufficiency** (even with the right shape, can a single scalar residual carry shock-regime information that the two siblings carry between them?).

Empirical phase shape (full-dataset, normalised price `(station_price − last_min) / (last_max − last_min)`):

| pct | empirical mean norm_price | what step-4 formula predicted |
|---|---|---|
| 0.025 | 0.68 | 0.025 |
| 0.075 (peak) | 0.82 | 0.075 |
| 0.275 | 0.54 | 0.275 |
| 0.525 | 0.26 | 0.525 |
| 0.675 (trough) | 0.18 | 0.675 |
| 0.875 | 0.23 | 0.875 |
| 1.075 | 0.40 | 1.075 |
| 1.375 | 0.44 | 1.375 |

See `phase_shape_vs_linear.png` (empirical curve vs linear-interp diagonal) and `cycle_pct_through_dist.png` (raw pct distribution).

This finding is captured cross-session as `project_cycle_pct_through_semantics`. The existing `project_phase_residual_fragile` memory has been updated to flag that the diagnosis is now under retest with the corrected pct semantics.

## Design decisions so far

### Step 1 — define phase (IN PROGRESS, PARKED)

| sub-decision | state | choice |
|---|---|---|
| Phase variable | done | `cycle_pct_through` raw |
| Bin axis extent | done | extend bins to **pct = 1.5**, clip beyond. Revised from 2.0 after inspecting `phase_shape_empirical.csv`: bins past 1.5 have 3k–13k rows and means swing 0.03 → 0.77 → −0.13 in adjacent bins, larger than the signal we're modelling. Selection-bias-in-tail makes those bins shock-conditional; an extended lookup would inject noise > signal. ~3% of rows clip to bin 1.475 (≈ 0.33), which sits inside the [1.0, 1.5] plateau and acts as a sensible "well into late cycle" shrinkage value. |
| Bin method | done | **equal-width**. Quantile breaks on the 15.4% pct=0 spike (multiple quantile bins collapse onto pct=0). Equal-width keeps the phase axis interpretable and leaves selection-bias-in-tail visible in diagnostics. |
| Bin count | done | **30 bins of width 0.05 over [0, 1.5]**. Resolves the early-peak hump and trough cleanly; per-fold train (~1.4M rows) gives ~30k–60k rows in normal-range bins. |
| pct = 0 spike (15.4% of rows) | done | benign: fresh-peak days. Treat honestly as bin 0. Not a data artefact (only 3/312,319 have station_price == last_min; no degenerate denominators). |
| Tail handling past pct = 1.5 | done | clip to 1.5 |

Distribution facts (full dataset, 2,023,658 rows):

- mean 0.55, median 0.48, 75th pct 0.83, 90th pct 1.18, 99th pct 1.86, max 2.54
- exact zeros: 312,319 (15.4%)
- pct > 1: ~17% of rows
- per-bin sample counts (20 equal-width bins over [0,1]): ~44k–92k except the 0-spike bin at 328k

### Step 2 — define "expected price" (DONE 2026-06-06)

**Choice: (A) `E[norm_price | phase]` where `norm_price = (station − last_min) / (last_max − last_min)`.** Lookup returns the empirical curve as percentages of (last_max − last_min). Per-row un-normalisation: `expected = last_min + lookup(phase) × (last_max − last_min)`.

Why over (B) `E[station − last_min | phase]` in cents:

1. **Cross-station amplitude bias.** Big-cycle stations contribute proportionally more cents to each bin; the bin mean is pulled toward them. Applied to a small-cycle station, the cents value can exceed its own (last_max − last_min), giving expected_price > last_max — nonsensical.
2. **Fold-to-fold instability.** Cycle amplitudes drifted across 2021–2025. (B)'s bin means are amplitude-mix dependent, so the lookup encodes regime drift as phase shape. (A) is amplitude-invariant by construction.
3. **Experiment attribution.** Step-4 formula already implicitly assumed linear amplitude scaling. (A) keeps that assumption and changes only the shape; (B) changes both, confounding any observed result.

(C) Per-station / per-LGA / per-amplitude conditioning — skip for v1.

### Step 3 — per-fold leakage-safe fit (DONE 2026-06-06)

Lookup is refit per fold on that fold's train data, then applied to val. Decisions:

- **Empty val bins → NaN propagation.** LightGBM handles missing values natively (learns a default split direction). On 30 equal-width bins of width 0.05 over 5 years of train data, empty bins should essentially never fire (sparsest tail bin has ~8k+ rows). Picked over nearest-non-empty (small risk of misleading near the noisy tail) and formula fallback (would reintroduce the broken linear-interp shape we're trying to replace).
- **Zero-amplitude rows (last_max == last_min) → filter at fit, NaN at apply.** ~3 rows per 312k spike-subset; vanishingly rare elsewhere. They can't inform the curve and have no swing to project onto.

Implementation in `step4_paired_wfcv.py` (this dir). Uses numpy bincount/digitize for speed.

### Step 4 — paired walk-forward CV (DONE 2026-06-06)

Mirrored `step4_paired_wfcv.py` from the predecessor dir: 14 folds (train_min_days=1825, val_days=90, step_days=90), seed=42, three configs (baseline 50 / additive 51 / ablationA 49), per-fold refit of the lookup before feature compute, ~2 min compute budget. Total wall: 118s.

Per-fold deltas saved to `step4_folds.csv`; aggregates and per-fold lookup means to `step4_meta.json` and `step4_fold_lookups.csv`. Re-aggregation under alternative labelling schemes in `reaggregate_by_label.py` (see Outcome).

## Pre-committed shock-fold taxonomy (BEFORE seeing results)

Per the in-session methodology discussion: report per-fold deltas separately for shock-coded folds vs normal folds. Taxonomy fixed *now* so the interpretation is principled.

| fold | val window | regime | basis |
|---|---|---|---|
| 1 | 2021-11 → 2022-02 | **shock** | late-2021 surge |
| 4 | 2022-08 → 2022-10 | **shock** | Ukraine continuation |
| 9 | 2023-10 → 2024-01 | **shock** | Israel-Gaza onset |
| 13 | 2024-10 → 2025-01 | **shock** | confirmed in step-4 regressions; named exogenous cause to be added if pursued |
| all others (2, 3, 5, 6, 7, 8, 10, 11, 12, 14) | — | **normal** | no named regime shock |

**Reporting protocol for the rerun:**

1. Per-fold delta table (same as step 4).
2. Aggregate: median Δ, mean Δ, helps/hurts ratio.
3. Separated aggregates: median across normal folds, median across shock folds, reported *both* before any judgement.
4. The pre-commit gate (revised mid-session 2026-06-06: shocks are *not* the primary focus of this experiment, so a neutral-to-mild shock regression is tolerable as long as normal folds improve. We have not yet built any systematic shock-period machinery; gating on shock-fold performance would over-penalise a feature whose job is normal-regime modelling):
   - **Helps normals (normal-fold median Δ ≤ −0.005) and shocks neutral-to-mild (no shock fold Δ > +0.02, shock median Δ ≤ +0.01).** → Methodological win; phase-residual concept viable as a normal-regime feature. Document the shock behaviour but don't block on it. Phase-shape correction was the missing piece.
   - **Helps normals but material shock regression (any shock fold Δ > +0.02 or shock median Δ > +0.01).** → Still informative: shape correction helps normal regime but doesn't carry shocks. Tag as "needs paired shock-detector partner before shipping" and feed into the independent shock-feature track in the Followups section.
   - **Regresses across the board (normal-fold median Δ ≥ 0).** → Confirms diagonal-projection diagnosis. Abandon all phase-residual variants permanently.

## Outcome (2026-06-06)

### Headline

The phase-residual concept is dead. **Additive (51 features) is neutral** — the lookup adds essentially nothing on top of the 50 baseline features. **AblationA (49 features, drop siblings + add lookup) regresses mildly on stable folds and catastrophically on fold 9.** The non-parametric shape correction was a real improvement over the predecessor's linear-interp formula (fold 9 went +0.101 → +0.066) but it did not flip the verdict.

### Per-fold result (paired walk-forward CV, seed=42, 14 folds)

| fold | regime (pre-committed) | val_start → val_end | ll_baseline | Δ additive | Δ ablationA |
|---|---|---|---|---|---|
| 1 | shock | 2021-11 → 2022-02 | 0.402 | −0.015 | **+0.044** |
| 2 | normal | 2022-02 → 2022-05 | 0.269 | −0.010 | −0.012 |
| 3 | normal | 2022-05 → 2022-08 | 0.377 | +0.015 | −0.010 |
| 4 | shock | 2022-08 → 2022-10 | 0.421 | −0.013 | −0.025 |
| 5 | normal | 2022-10 → 2023-01 | 0.294 | +0.014 | +0.008 |
| 6 | normal | 2023-01 → 2023-04 | 0.218 | +0.002 | +0.008 |
| 7 | normal | 2023-04 → 2023-07 | 0.251 | +0.006 | +0.012 |
| 8 | normal | 2023-07 → 2023-10 | 0.306 | −0.023 | +0.029 |
| 9 | shock | 2023-10 → 2024-01 | 0.331 | +0.000 | **+0.066** |
| 10 | normal | 2024-01 → 2024-04 | 0.254 | −0.001 | +0.004 |
| 11 | normal | 2024-04 → 2024-07 | 0.344 | +0.011 | −0.014 |
| 12 | normal | 2024-07 → 2024-10 | 0.305 | −0.001 | −0.018 |
| 13 | shock | 2024-10 → 2025-01 | 0.300 | +0.000 | +0.004 |
| 14 | normal | 2025-01 → 2025-04 | 0.362 | −0.002 | +0.005 |

### Aggregates under three labelling schemes

The pre-committed shock taxonomy (`{1, 4, 9, 13}`, macro-event-named) suggested ablationA fails the gate on shock-fold regression. Re-aggregation by empirical fold difficulty (ll_baseline) showed the verdict shifts.

| scheme | shock folds | additive median (normal / shock) | ablationA median (normal / shock) |
|---|---|---|---|
| **1. Pre-committed (macro events)** | {1, 4, 9, 13} | +0.0004 / −0.0066 | **+0.0045 / +0.0238** |
| **2. Top-quartile ll_baseline** | {1, 3, 4, 14} | +0.0001 / −0.0075 | +0.0059 / **−0.0027** |
| **3. Top-third ll_baseline** | {1, 3, 4, 11, 14} | +0.0001 / −0.0018 | +0.0078 / **−0.0105** |

The "shock median" cell is where the schemes diverge: scheme 1 says shocks regress (+0.024 median); schemes 2 & 3 say the hardest folds by baseline difficulty actually *help* (−0.003, −0.011 median). The pre-committed methodology was carrying fold 9 specifically — and fold 9 is mid-pack by ll_baseline (0.331, neither hard nor easy). It's not a "shock fold" in any data-driven sense; the macro-event label (Israel-Gaza onset) is coincident, not causal.

**Use schemes 2/3 as the authoritative read.** Scheme 1 stays in the record as the pre-committed control, methodology weakness acknowledged.

### Why fold 9 is the outlier (see `fold9_inspection.png`, `fold9_summary.csv`)

Fold 9 isn't generically hard — it has a structural anomaly. The first ~5 weeks of the window are a single sustained downtick (peak ~215c → trough ~175c) before recovery. Cycles in the window ran ~50 days vs the dataset-typical ~33, and amplitudes are ~30% larger than surrounding windows (37c vs 27–30c). Concretely:

| metric | full dataset | pre (Jul–Oct '23) | **fold 9** | post (Jan–Apr '24) |
|---|---|---|---|---|
| pct_through p99 | 1.86 | 1.21 | **1.57** | 1.37 |
| frac rows with pct > 1.5 | 3.8% | 0% | **2.2%** | 0% |
| amplitude mean | 30.6c | 27.7c | **37.2c** | 30.3c |

### Mechanism: the triple-stack

The phase-residual failure mode is the intersection of three properties that compound only in fold-9-like windows:

1. **Late descent is the model's hardest prediction zone.** The objective P(min over next H days < today − X) lives at trough-proximity (pct ≈ 0.3–0.6 in typical cycles). Trough timing has ±1–2 day jitter even in stable structure; the model has the least signal exactly where the buy decision matters most.
2. **The lookup is structurally noisy in the pct > 1 tail.** Per `phase_shape_empirical.csv`, bins past 1.5 had 3k–13k rows vs ~80k in the bulk, with mean swings 0.03 → 0.77 → −0.13 between adjacent bins. We clip at 1.5, but the [1.0, 1.5] range still has ~half the bulk's row density and proportionally noisier means. Worse, the tail is selection-bias-conditional on *prior* shock-like episodes — the bins are dominated by training-period anomalies and don't generalise to *new* anomalies.
3. **Cycle elongation pushes trough-proximity into the lookup's tail.** In a typical 33-day cycle, the trough sits at pct ≈ 0.5–0.7 (well-supported). In fold 9's ~50-day cycles, the trough drifts to pct ≈ 1.0–1.5 (selection-biased and noisy).

In any typical fold, (1) and (2) don't intersect — trough-proximity stays in the well-supported lookup region. Additive ≈ neutral, ablationA ≈ flat or mild regression (siblings carry signal the lookup can't reproduce, but the lookup at least isn't actively wrong in the rows that matter).

In fold 9 all three align: more calendar days in the late-descent zone *and* those days carry pct values that overshoot into the noisy tail *and* the model needs maximum quality on exactly those rows *and* ablationA has removed the only stable anchor (the cents-above/below-last_min/max siblings) that doesn't depend on pct. Result: +0.066 ablationA regression, the largest single-fold regression in the run.

The sibling features handle this gracefully because they're pct-invariant — `station_minus_last_min` reads "n cents above the recent trough" whether the cycle ran 25 days or 50. AblationA throws them out; fold 9 makes that throw-out catastrophic.

### Why the additive (51) result is more interesting than it looks

Additive's normal-fold median is +0.0004 — *exactly zero*. Even the shock-fold median is only −0.007. The lookup feature **adds essentially no information on top of the 50 baseline features**. Trees were already constructing phase-residual-like splits from the siblings (e.g. `station_minus_last_max > -k`) and an explicit pre-computed residual gives them no leverage they didn't already have.

This is a bigger finding than "diagonal projection is fragile". It says **the phase-residual concept is redundant with the existing feature set** — the encoding question (linear-interp vs non-parametric vs whatever else) is downstream of the conceptual question of whether projecting onto a single scalar residual carries any signal the siblings don't already carry. Answer: no.

### Pre-commit gate verdict (applied to schemes 2/3 as authoritative)

- **Additive** fails on "helps normals" (normal median ≈ 0, not ≤ −0.005). Drops into gate 3 (regresses across the board) — except it's not really regressing, it's just neutral. **Outcome: neutral additive → not worth shipping; the feature is dominated by what already exists.**
- **AblationA** fails on "helps normals" (normal median +0.006 to +0.008) under all three schemes. Material single-fold regression (fold 9 +0.066) under all three schemes regardless of which bucket fold 9 falls into. **Outcome: drop, do not ship.**

### What this means for the phase-residual concept

Retired. The non-parametric shape correction worked at the level it could (predecessor's fold 9 was +0.101; we got it to +0.066) — confirming the linear-interp formula was structurally mis-specified per [[project-cycle-pct-through-semantics]]. But the deeper diagnosis from [[project-phase-residual-fragile]] survives the retest and gains a more precise mechanism: it's not "shock-fragile", it's **"loses absolute-anchor information exactly when the rest of the model's input is also least reliable (late descent in elongated cycles)"**.

No phase-residual variant — shape-corrected, station-conditioned, or otherwise — should be pursued without an architecture that preserves the absolute anchors. The siblings are non-negotiable for this prediction objective.

## Selection-bias-in-tail finding (important for interpretation)

Observed empirically: bins past pct = 1.5 have 3k–13k rows (vs 50k–90k in the normal range), and the means swing wildly (0.025 → 0.69 → −0.13 in adjacent bins). The population of rows with `pct > 1.5` is *only* those cycles that lasted ≥ 1.5 × mean_cycle_length without re-peaking — a shock-distorted subset, not a representative one.

Implication: the lookup is **most unreliable exactly where we most need it to be reliable.** Fold 1 and fold 9 put a lot of mass at pct > 1. The lookup will plug bin means there that are dominated by *prior* shock episodes in train — which may not generalise to a *new* shock in val. Structurally the same concern that killed the linear-interp version, in a different shape.

This sharpens the prediction: even with the correct shape, the shock-fold regression may survive. The diagnostic is still cleanly informative — just less likely to flip than the "shape correction will fix it" intuition would suggest.

## Followups

- **Phase-residual track:** closed. No further v2 / v3 attempts (station-conditioned lookup, etc.) on the diagonal-projection idea; the diagnosis stands. Clip-point sensitivity (v1 used clip = 1.5; revisit 2.0 was on the table) is also closed — not worth investigating a dead concept.
- **Next investigation — late-descent feature class.** Filed as design issue #206. The model's hardest prediction zone is trough-proximity (pct ≈ 0.3–0.6 typical, drifts into pct ≈ 1.0–1.5 when cycles elongate). Find features that help in *general* late-descent, not shock-specific. Anything that improves trough-proximity prediction will help every fold; shock folds (which are essentially elongated-cycle folds at higher amplitude) become a free win on top. This is strictly better than starting with a shock-regime indicator because shock-specific work is a special case of late-descent work, not the other way around.
- **Methodology lesson:** macro-event-based shock taxonomy was a soft methodology. Empirical labelling (e.g. top-quartile ll_baseline) gave a sharper view in this case. Consider standardising on empirical labels for future paired CV reports unless a specific macro hypothesis is being tested.

