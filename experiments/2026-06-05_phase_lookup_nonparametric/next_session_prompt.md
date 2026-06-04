# Next session — paste this into a fresh Claude conversation

> Continuing a learning-style experiment from 2026-06-05. Walk me through step by step, explain choices as we go, pause for questions. Lighter maths; visualisations welcome.

## Where we left off

Mid-design on `experiments/2026-06-05_phase_lookup_nonparametric/` — a retest of the phase-residual diagnosis from the predecessor experiment `experiments/2026-06-04_cycle_pct_through_interaction/`. The README in this dir is the canonical state; read it in full before continuing.

## What was settled last session

1. **The big finding:** `cycle_pct_through` = `days_since_last_peak / mean_cycle_length`. Not "fraction between trough and peak". The actual cycle goes peak → fall → trough (~pct=0.5) → rise → peak. The step-4 linear-interp formula `expected = last_min + pct × (last_max − last_min)` is monotonically rising and therefore *inverted in shape* relative to reality. Captured as memory `project_cycle_pct_through_semantics`.
2. **Tail handling:** extend lookup bins to pct = 2.0, clip beyond.
3. **pct = 0 spike:** 15.4% of rows; benign (fresh-peak days), not an artefact.
4. **Shock-fold taxonomy pre-committed:** folds 1, 4, 9, 13 are shock; 2,3,5,6,7,8,10,11,12,14 are normal. Reporting will separate normal-median from shock-median *before* applying any gate.
5. **Methodology decision:** if the lookup regresses materially in shock folds, do NOT ship without a paired shock-detector feature. Portfolio-of-specialists, not monolithic gate.
6. **Selection-bias-in-tail risk:** pct > 1.5 bins have 3k–13k rows each, dominated by *prior* shock episodes. Tail lookup values are regime-conditional and may not generalise to new shocks.

## What's still pending (resume from here)

Read README.md, then continue Step 1 sub-decisions:

- **Bin method:** quantile vs equal-width. (Quantile gives uniform per-bin sample size; equal-width keeps the phase axis interpretable.)
- **Bin count.** (Bias-variance — too few smooths real structure; too many → noise in sample-poor tail.)

Then Step 2 (normalisation choice — leaning option A in README), Step 3 (per-fold leakage-safe fit), Step 4 (implement and run paired CV, ~2 min budget), Step 5 (interpret per-fold, especially folds 1 and 9).

## Constraints

- Use `fuel_signal.features.load_features()`, not `pd.read_csv`.
- Mirror `experiments/2026-06-04_cycle_pct_through_interaction/step4_paired_wfcv.py` — only the feature recipe (refit-per-fold lookup) changes.
- Log per-step wall times per `feedback_instrument_walltime`.
- Report per-fold deltas with the pre-committed shock-fold taxonomy from README.
- experiments/** is exempt from the PR rule — lab book + results can be committed directly to main.

## Relevant memories

- `project_cycle_pct_through_semantics` — the pct semantics correction (new this session).
- `project_phase_residual_fragile` — the prior diagnosis being retested; updated to flag the retest pending.
- `feedback_regime_segmented_evaluation` — report format.
- `feedback_instrument_walltime` — log per-step wall (step 4 of predecessor took 115s → ~2 min budget here).
- `feedback_teaching_pace` — one focused point per response.
- `feedback_graphs_for_teaching` — announce plots, reach for them proactively.
