# Non-parametric phase model — retest of phase-residual diagnosis

- **Date:** 2026-06-05 (PARKED mid-design)
- **Branch:** main
- **SHA at park:** 3da7f38
- **Status:** open
- **Predecessor:** `experiments/2026-06-04_cycle_pct_through_interaction/` (step 4 abandoned ablationA with worst-fold +0.101)

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
| Bin axis extent | done | extend bins to **pct = 2.0**, clip beyond |
| Bin method | **pending** | quantile vs equal-width — not chosen |
| Bin count | **pending** | not chosen |
| pct = 0 spike (15.4% of rows) | done | benign: fresh-peak days. Treat honestly as bin 0. Not a data artefact (only 3/312,319 have station_price == last_min; no degenerate denominators). |
| Tail handling past pct = 2.0 | done | clip to 2.0 |

Distribution facts (full dataset, 2,023,658 rows):

- mean 0.55, median 0.48, 75th pct 0.83, 90th pct 1.18, 99th pct 1.86, max 2.54
- exact zeros: 312,319 (15.4%)
- pct > 1: ~17% of rows
- per-bin sample counts (20 equal-width bins over [0,1]): ~44k–92k except the 0-spike bin at 328k

### Step 2 — define "expected price" (NOT STARTED)

The most important design choice. Open question: what does the lookup learn? Candidates:

- **(A) `E[norm_price | phase]` where `norm_price = (station − last_min) / (last_max − last_min)`.** Lookup returns the empirical curve in the table above; we un-normalise per row as `expected = last_min + lookup(phase) × (last_max − last_min)`. Cleanly compatible with the step-4 protocol — same un-normalisation algebra, replacing the diagonal with a learned curve.
- **(B) `E[station − last_min | phase]` in cents.** Avoids the assumption that amplitude scales the curve linearly. But then we conflate cycle amplitude with phase shape in the lookup.
- **(C) Per-station / per-LGA / per-amplitude conditioning.** Adds complexity. Skip for v1.
- **Selection-bias-in-tail risk** (see below) applies to all of these.

Leaning (A) for v1 — most direct apples-to-apples test against the step-4 formula. Confirm before implementing.

### Step 3 — per-fold leakage-safe fit (NOT STARTED)

Lookup must be refit per fold on that fold's train, applied to val. Sketch:

```python
def fit_lookup(train_df, bin_edges):
    train_df = train_df.assign(_norm = (train_df.station_price_cents - train_df.cycle_last_min_cents)
                                       / (train_df.cycle_last_max_cents - train_df.cycle_last_min_cents),
                               _bin  = pd.cut(train_df.cycle_pct_through.clip(upper=2.0), bins=bin_edges))
    return train_df.groupby('_bin', observed=True)['_norm'].mean()

def apply_lookup(df, lookup, bin_edges):
    bins = pd.cut(df.cycle_pct_through.clip(upper=2.0), bins=bin_edges)
    expected_norm = bins.map(lookup)
    expected = df.cycle_last_min_cents + expected_norm * (df.cycle_last_max_cents - df.cycle_last_min_cents)
    return df.station_price_cents - expected  # the new feature
```

Pending: handle val bins with no train support (NaN propagation? fall back to formula? carry the nearest non-empty bin's mean?).

### Step 4 — paired walk-forward CV (NOT STARTED)

Mirror `step4_paired_wfcv.py` from the predecessor dir: 14 folds (train_min_days=1825, val_days=90, step_days=90), seed=42, three configs (baseline 50 / additive 51 / ablationA 49), regression threshold 0.005, ~2 min compute budget. The only change is per-fold refitting of the lookup before computing the feature in train and val.

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
4. The pre-commit gate (decided in-session):
   - **Lookup is neutral in shocks (|Δ| ≤ 0.005) and helps normals.** → Methodological win; phase-residual concept viable. Document, but don't graduate without a deeper pass.
   - **Lookup regresses materially in shocks (Δ > +0.005 in any shock fold) and helps normals.** → Same outcome as step 4; do not ship without a paired shock-detector feature that absorbs the regime signal. Either abandon the phase-residual track or sequence the shock-detector first and re-test.
   - **Lookup regresses across the board.** → Confirms diagonal-projection diagnosis. Abandon all phase-residual variants permanently.

## Selection-bias-in-tail finding (important for interpretation)

Observed empirically: bins past pct = 1.5 have 3k–13k rows (vs 50k–90k in the normal range), and the means swing wildly (0.025 → 0.69 → −0.13 in adjacent bins). The population of rows with `pct > 1.5` is *only* those cycles that lasted ≥ 1.5 × mean_cycle_length without re-peaking — a shock-distorted subset, not a representative one.

Implication: the lookup is **most unreliable exactly where we most need it to be reliable.** Fold 1 and fold 9 put a lot of mass at pct > 1. The lookup will plug bin means there that are dominated by *prior* shock episodes in train — which may not generalise to a *new* shock in val. Structurally the same concern that killed the linear-interp version, in a different shape.

This sharpens the prediction: even with the correct shape, the shock-fold regression may survive. The diagnostic is still cleanly informative — just less likely to flip than the "shape correction will fix it" intuition would suggest.

## Where the conversation left off

Done in this session:

- Confirmed pct=0 spike is fresh-peak days (15.4%, not an artefact).
- Discovered pct_through semantic correction (the big finding).
- Settled tail extension to pct = 2.0.
- Pre-committed the shock-fold taxonomy and reporting gate.
- Discussed and aligned on the portfolio-of-features methodology (shock-regressing features need paired shock-detector before they can ship).

Not done:

- Choose bin method (quantile vs equal-width) and bin count.
- Pick normalisation for the lookup (A vs B in step 2 — leaning A).
- Implement the lookup + paired CV.
- Run + interpret.

## Followups

- Once this completes (either outcome): if the diagnosis confirms, retire all phase-residual ideas; if it flips, design v2 with explicit shock-feature partner before shipping.
- Independent track: build a shock-regime indicator feature (oil-price shock proxy, news-driven cycle-distortion signal, or just `days_since_last_peak > 2 × mean_cycle_length` style flags) that could pair with phase-residual or be useful on its own.

## Restart instructions

See `next_session_prompt.md` for the paste-in restart message.
