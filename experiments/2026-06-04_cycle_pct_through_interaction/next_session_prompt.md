# Next session — paste this into a fresh Claude conversation

> I'd like to do this as a learning exercise. Walk me through it step by step, explain what we're testing at each stage and why, and pause for me to think / ask questions rather than running everything end-to-end.

## Context I'm carrying in

In experiment `experiments/2026-06-04_cycle_pct_through_interaction/` (README there) we tested whether an engineered "phase residual" feature could replace the algebraic siblings `station_minus_last_min_cents` and `station_minus_last_max_cents`. The recipe was:

```
expected_price = last_cycle_min + cycle_pct_through × (last_cycle_max − last_cycle_min)
station_minus_expected_phase_price = station_price − expected_price
```

The step-4 paired walk-forward CV result:
- *Additive* (siblings kept, engineered added): neutral. Median Δ −0.0003, 7/7 split.
- *AblationA* (siblings dropped, engineered keeps the role): regresses in 10/14 folds, mean Δ +0.018, **worst-fold +0.101 (fold 1: late-2021 → Feb-2022)**. Big regressions concentrated in shocked / high-price regimes (Ukraine, Israel-Gaza, late-2021 surge).

My **diagnosis** at the end of step 4 was: the issue is *fundamental to projecting onto one diagonal*, not the linear-interpolation approximation per se. The siblings carry shock-regime information about *absolute* distance from named landmarks (last trough, last peak) that any single residual feature drops.

## What I want to learn by doing this

I want to **test the diagnosis** rather than assume it. If a better phase model (non-parametric instead of linear) still regresses in the same shock regimes, the diagonal-projection issue is confirmed. If it *fixes* the regressions, my diagnosis was wrong and there's a real feature engineering win still on the table.

This is option 3 from the end-of-step-4 recommendations.

## The experiment

Build a non-parametric `E[price | phase]` lookup from training data, then test as a feature.

Sketch of what we'll do — but I want to think through the design choices with you, not have you just code it:

1. **Define "phase" precisely.** It's `cycle_pct_through`. What's its range, distribution, and how should we bin it? (e.g. 20 quantile bins? 50 equal-width? What if a bin has too few rows?)
2. **Define "expected price" precisely.** The mean station_price within each phase bin? Or normalised by something — last_cycle_min, last_cycle_max, station mean? **This is the most important design choice.** Linear interpolation already gave us *normalised* expected price (between this trough and this peak). A naïve unnormalised lookup would conflate phase with absolute price level — definitely wrong. So the lookup probably needs to be something like `E[(price − last_min) / (last_max − last_min) | phase]` and then we un-normalise per row. Help me think this through.
3. **Watch for leakage.** The lookup must come from training data only. When we run paired walk-forward CV the lookup needs to be re-fit per fold on that fold's train. Different from a "computed once on the full features.csv" feature.
4. **Build the feature + retrain.** Same three configs as step 4: baseline / additive / ablation-A-style replacement.
5. **Run paired walk-forward CV (14 folds).** Compare per-fold deltas vs step 4. Specifically: does **fold 1 (Δ was +0.101)** still regress? That's the diagnostic test.
6. **Interpret.**

## Constraints / context to read first

- `experiments/2026-06-04_cycle_pct_through_interaction/README.md` (this experiment's lab book)
- `experiments/2026-06-04_cycle_pct_through_interaction/step4_paired_wfcv.py` (the protocol to mirror, modulo the per-fold lookup recomputation)
- Memory: `project-phase-residual-fragile` (the diagnosis we're testing), `feedback-regime-segmented-evaluation` (median + named regressions is the report format), `feedback-instrument-walltime` (log per-step wall — the step 4 CV took 115s on this machine, so ~2 min budget for the rerun)
- Use `fuel_signal.features.load_features()`, not `pd.read_csv`.

## What I expect (and why I want to be wrong)

My prior is that fold 1 and fold 9 still regress. The "expected price" lookup is still a single scalar that has to subsume both peak-distance and trough-distance information into one residual axis. A non-parametric lookup might handle asymmetric cycle shapes better (good for normal-cycle folds) but won't recover the absolute-distance-from-peak signal that the siblings preserve during shocked windows.

If I'm right, this confirms the diagnosis and we abandon all phase-residual variants. If I'm wrong (fold 1's regression disappears), it means the linear-interpolation crudeness *was* doing the damage and there's a real feature engineering win — which would be the more interesting outcome.

Either way, I'd like to actually *understand* why the per-fold result lands where it does, not just collect the number. So: explain the design choices as we go, pause for my questions, and at the end help me articulate what each fold's outcome tells us.
