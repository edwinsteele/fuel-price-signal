from __future__ import annotations

import pandas as pd


def rolling_baseline(
    per_date_series: pd.Series,
    window_days: int,
    closed: str = "left",
    min_periods: int = 1,
    agg: str = "median",
) -> pd.Series:
    """PIT-strict rolling aggregation over a per-date Series.

    Reindexes to a contiguous daily range first, then applies
    rolling(f"{window_days}D", closed=closed). closed='left' (the default)
    means today's value does NOT enter today's aggregate — call this before
    joining to df to prevent look-ahead leakage. Returns a Series on the
    full daily index; callers join on date.

    PIT-safe by default: closed='left' excludes the current date. Callers
    that pass closed='right' opt into including the current date and must
    justify it.
    """
    s = per_date_series.sort_index()
    full_idx = pd.date_range(s.index.min(), s.index.max(), freq="D")
    s = s.reindex(full_idx)
    return s.rolling(f"{window_days}D", closed=closed, min_periods=min_periods).agg(agg)
