from __future__ import annotations

import pandas as pd


def calendar_aware_delta(per_date_series: pd.Series, lag_days: int) -> pd.Series:
    """Compute (value − value_lag_days_prior) for a per-date Series.

    Reindexes to a contiguous daily range before shifting so gaps in the
    input series yield NaN instead of silently spanning more than lag_days
    calendar days. Input index must be (or be coercible to) DatetimeIndex.
    Returns a Series on the same full daily index; callers join on date.

    PIT-safe: shift is backward-looking only. Today's value is never used
    to construct a prior-date delta.
    """
    s = per_date_series.sort_index()
    full_idx = pd.date_range(s.index.min(), s.index.max(), freq="D")
    s = s.reindex(full_idx)
    return s - s.shift(lag_days)
