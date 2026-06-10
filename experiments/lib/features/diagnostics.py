from __future__ import annotations

import pandas as pd


def px_change_lag_diagnostic(
    df: pd.DataFrame,
    lag_days: int,
    station_col: str = "station_code",
    date_col: str = "price_date",
    value_col: str = "station_price_cents",
) -> pd.Series:
    """Per-row price change vs lag_days calendar days prior for the same station.

    Returns a Series aligned to df.index. NaN when no observation exists at
    exactly (date − lag_days) for that station.

    PIT-safe: uses an exact-date self-merge (validate='m:1') rather than
    positional diff. Positional diff silently spans data gaps larger than
    lag_days calendar days, which corrupts any cohort definition that depends
    on knowing the actual price movement over a fixed window.

    validate='m:1' raises if (station_col, date_col) has duplicate rows in df,
    which would otherwise silently row-explode the merge.
    """
    lookup = df[[station_col, date_col, value_col]].rename(
        columns={date_col: "_lookup_date", value_col: "_px_lag_ago"}
    )
    work = df[[station_col]].assign(
        _lookup_date=pd.to_datetime(df[date_col]) - pd.Timedelta(days=lag_days)
    )
    merged = work.merge(
        lookup, on=[station_col, "_lookup_date"], how="left", validate="m:1"
    )
    # Left merge preserves left-row order; restore original index.
    merged.index = df.index
    return (df[value_col] - merged["_px_lag_ago"]).rename(f"_px_{lag_days}d_change")
