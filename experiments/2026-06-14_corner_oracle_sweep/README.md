# Corner oracle sweep — does any OTHER intra-series signal carry shallow-vs-steep? (#237)

- **Date:** 2026-06-14
- **Branch:** main
- **Status:** PARKED — results invalid pending #250 + regime-axis fix; do not read them

> ## ⚠️ PARKED — results sit on a contaminated basis, do not analyse them
>
> The sweep ran and produced `oracle_sweep.png` / `meta.json`, **but the run is
> built on a broken phase axis and must not be read as-is.** Two defects
> surfaced *during* this work and invalidate it:
>
> 1. **Cycle over-segmentation (#250).** `cycle_days_since_peak` whipsaws at
>    boundaries, so the dsp-reset reconstruction in `cycle_shape.label_cycle_shape`
>    finds ~2.6× too many (spurious, short) cycles vs `find_peaks(distance=7,
>    prominence=1.0)`. The oracle `normal/elongated_steep/elongated_shallow`
>    classes are therefore unreliable.
> 2. **Regime-stretched phase axis.** `cycle_mean_length` is an expanding
>    all-history mean, so it lags the early-2020 cycle-length regime shift
>    (~24d → ~38d; see `regime_cycle_length.png`). `cycle_pct_through` — the
>    x-axis of every panel — is inflated ~13% (worse, ~1.2–1.3× in 2022–2024 /
>    fold 7's era). The "predictive zone" doesn't sit at a stable pct.
>
> **This also reframes the premise.** #214/#231 rejected A on this same broken
> ruler, in the worst-affected fold — so "A is exhausted" is no longer safe.
>
> **Rebuild plan when we return to #237:**
> - Reconstruct cycles via `find_peaks` (matches the detector + heatmap), not
>   dsp-resets — once #250's production fix lands, or directly in the diagnostic.
> - Replace the denominator with a **clipped-at-break regime-local** baseline
>   (PELT-confirmed changepoint), and condition on the post-2020 regime.
> - **Add A back into the candidate set** — it was never fairly tested.
>
> The durable artifact from this session is `regime_cycle_length.py` /
> `.png` (evidence behind #250 and the regime fix). The candidate design below
> (the seven signals + rationale) is still good; only the axis is broken.

## Hypothesis

A (`network_px_std`) does **not** carry the shallow-vs-steep-within-elongated
distinction in the late-descent corner — #214 (raw axes) and #231 (explicit
interaction column) both proved that, the second one *backwards* (fold 7
regressed: A's reading in the corner is actively misleading, not merely absent).

But that ruling is narrow — it is about **A specifically**. Before conceding the
corner signal isn't in our price-only data and pivoting to external data (#215's
reserve), check cheaply whether the distinction lives in some **other**
intra-series signal that simply hasn't been surfaced.

## Method — train-only oracle existence check

No model fitting (per `feedback_oracle_diagnostic_pattern`). Mirrors #214's
`phase_oracle_cycles.py`; the cycle-classification logic is now the shared
`experiments/lib/features/cycle_shape.py:label_cycle_shape` helper, so this
sweep classifies cycles **identically** to #214 (the diagnostics are only
comparable if the class definitions match).

1. Restrict to the train segment (`price_date < 2021-11-01`, strictly before
   fold 1's val_start) — val/test stay pristine for the eventual real test.
2. Oracle pre-classification of whole cycles by eventual shape: `normal`
   (length ≤ train-median), `elongated_steep` (> median, descent slope
   ≤ −0.9 c/day), `elongated_shallow` (> median, slope > −0.9).
3. For each candidate, plot its median vs `cycle_pct_through`, one line per class.
4. Read the divergence **in the predictive zone** (pct ≈ 0.15–0.60, the
   late-descent / trough band where the buy/sell objective carries its
   uncertainty) — **not** pct > 1.4, where A finally separated (too late to use).

The script prints, per candidate, the shallow-vs-steep `|Δmedian|` in the
predictive zone vs the late tail, and ranks candidates by predictive-zone
separation. `meta.json` persists these numbers.

## Candidate signals (computed in-script; `fuel_signal/` untouched)

Per `feedback_features_via_experiment_first` — candidates live here; nothing
lands in `features.py` unless a follow-up paired-WFCV graduates it.

| Family | Candidate | Definition |
|---|---|---|
| F1 trough-proximity | `days_since_meaningful_drop` | days since the last ≥0.3c daily fall in Sydney-avg |
| F1 | `down_run_length` | length of the current consecutive-fall run (days) |
| F1 | `px_change_5d` | net 5-calendar-day change in Sydney-avg |
| F2 consensus | `lga_phase_std` | std of `days_since_trough` across 35 LGAs (existing col; = triplet **C**) |
| F2 | `lga_trough_fraction` | share of 35 LGAs with `days_since_trough_entry ≤ 3` |
| F3 triplet | `network_disc_gap` | comp_median − disc_median per date (triplet **B**) |
| F3 | `network_disc_gap_delta_3d` | 3-day delta of the gap |

Three genuinely different signal families: local price kinematics (F1),
cross-section trough consensus (F2), discount-cohort lead (F3). The triplet's B
and C were evaluated population-wide in #206; here they are read **corner-
conditioned**.

## Two-stage discipline (this is hypothesis generation, not a verdict)

- **Oracle miss** (flat across shallow/steep under perfect labels) → drop for
  free; you can't beat the oracle.
- **Oracle hit** → a *lead only*. It must then earn a PIT-safe proxy and pass
  the paired-WFCV gates on the untouched folds (separate follow-up). The oracle
  label never becomes a feature.
- Sweeping several candidates inflates the chance one separates by luck — so the
  untouched-fold paired-WFCV stays the non-negotiable real test; oracle-hit count
  is not evidence.

## Decision rule

- **≥1 candidate separates shallow/steep in the predictive zone** → file a
  follow-up to build its PIT-safe proxy + run paired-WFCV. Intra-series track
  stays alive.
- **No candidate separates across the three families** → real evidence the
  corner signal is not in our price-only data; external data (wholesale /
  Singapore Mogas / retail margin) becomes *earned*. File the external-data
  scoping issue then.

## Run

```bash
PYTHONPATH=. uv run python experiments/2026-06-14_corner_oracle_sweep/oracle_sweep.py \
  2>&1 | tee experiments/2026-06-14_corner_oracle_sweep/run.log
```

## Outputs

- `oracle_sweep.png` — 7-panel grid, candidate median vs phase, one line per class.
- `meta.json` — config + per-candidate predictive-zone/late-tail separation.
- `run.log` — captured stdout (tee'd, gitignored).
- `analysis.md` — written after the run: per-candidate verdict (separates early /
  late / not) + the decision-rule outcome.

## Acceptance criteria (#237)

- [ ] Oracle sweep run train-only over the candidate set; per-candidate divergence plotted + read.
- [ ] Verdict per candidate (separates early / late / not) recorded in `analysis.md`.
- [ ] Either a follow-up issue to build the winning candidate's PIT-safe proxy,
      OR an external-data scoping issue if the sweep is empty.

## Related

- Predecessors: #214 (`2026-06-09_shallow_elongated`, rejected), #231
  (`2026-06-11_interaction_column`, rejected — backwards).
- Planning: #215 (closed) — this resolves "intra-series vs external" honestly.
- Memory: `project_late_descent_investigation` (2026-06-11 update),
  `feedback_oracle_diagnostic_pattern`, `feedback_tree_interaction_limits`.
