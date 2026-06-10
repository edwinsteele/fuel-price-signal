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
