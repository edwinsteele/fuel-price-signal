# experiments/lib/features — feature-computation primitives

Sub-package of `experiments/lib/`. Shared primitives for the inside of `compute_features()` / `add_candidate_columns()`. Import with `PYTHONPATH=.`.

Each function carries a PIT-safety contract in its docstring. The guiding rule: a helper is PIT-safe when it never reads a date beyond the one it is being asked about. Violating this introduces look-ahead leakage that inflates CV scores without improving live performance.

## dispersion.py

`cohort_std_by_date(df, mask, value_col, date_col)` — std of `value_col` within `df[mask]` grouped by `date_col`. Returns a Series indexed by date. PIT-safe when `mask` is derived from same-date row attributes (e.g. `stickiness_score` snapshot).

`cohort_agg_diff_by_date(df, mask_a, mask_b, value_col, date_col, agg)` — `agg(cohort_a) − agg(cohort_b)` per date. Generalises the comp-vs-discount median spread (Signal B in `late_descent_triplet/step2`). PIT-safe under the same mask constraint.

## deltas.py

`calendar_aware_delta(per_date_series, lag_days)` — `(value − value_lag_days_prior)` for a per-date Series. Reindexes to a contiguous daily range before shifting; gaps yield NaN instead of silently spanning more than `lag_days` calendar days. Returns a Series on the full daily index; callers join on date. PIT-safe: shift is backward-looking only.

## rolling.py

`rolling_baseline(per_date_series, window_days, closed, min_periods, agg)` — PIT-strict rolling aggregation. Reindexes to a contiguous daily range, then applies `rolling(f"{window_days}D", closed=closed)`. Default `closed='left'` excludes today's value from today's aggregate. Returns a Series on the full daily index; callers join on date. PIT-safe by default; callers that pass `closed='right'` opt into including the current date and must justify it.

## cycle_shape.py

`label_cycle_shape(per_date, ...)` — **ORACLE** classification of network price cycles by eventual shape. Reconstructs cycle boundaries from resets in `cycle_days_since_peak`, summarises each cycle by length + peak→trough descent slope, and labels it `normal` / `elongated_steep` / `elongated_shallow` using the train-median length cutoff and a `-0.9 c/day` shallow threshold. Returns `(per_date, cyc)`: the input frame tagged with `cycle_id` + `cycle_type`, and a per-cycle summary. **Uses future info** (full-cycle length/trough) — valid only for train-only existence diagnostics; the `cycle_type` label must never become a model feature. Extracted from `2026-06-09_shallow_elongated/phase_oracle_cycles.py` (#214) so later oracle diagnostics classify cycles identically. First reused by `2026-06-14_corner_oracle_sweep` (#237).

> ⚠️ **PARKED / KNOWN-FLAWED (#250).** `cycle_days_since_peak` whipsaws at cycle boundaries, so the dsp-reset reconstruction over-segments ~2.6× vs `find_peaks(distance=7, prominence=1.0)`. Do not trust classifications from this until it is rebuilt on `find_peaks`. Kept only as the parked #237 record.

## diagnostics.py

`px_change_lag_diagnostic(df, lag_days, station_col, date_col, value_col)` — per-row price change vs `lag_days` calendar days prior for the same station. Returns a Series aligned to `df.index`. NaN when no observation exists at exactly `(date − lag_days)`. Uses an exact-date self-merge with `validate='m:1'` rather than positional diff; positional diff silently spans data gaps and corrupts cohort definitions that depend on a fixed calendar window.

## Composition example

```python
from experiments.lib.features.deltas import calendar_aware_delta
from experiments.lib.features.diagnostics import px_change_lag_diagnostic
from experiments.lib.features.dispersion import cohort_std_by_date

def compute_features(df):
    df = df.copy()
    df["price_date"] = pd.to_datetime(df["price_date"])

    # Cohort dispersion level
    comp_mask = df["stickiness_score"].abs() <= COMP_BAND_CENTS
    df = df.join(cohort_std_by_date(df, comp_mask).rename("network_px_std"), on="price_date")

    # Calendar-aware 3d delta
    s = df.drop_duplicates("price_date").set_index("price_date")["network_px_std"]
    df = df.join(calendar_aware_delta(s, 3).rename("network_px_std_delta_3d"), on="price_date")

    # Lated cohort diagnostic (not a feature)
    df["_px_5d_change"] = px_change_lag_diagnostic(df, lag_days=5)

    return df
```
