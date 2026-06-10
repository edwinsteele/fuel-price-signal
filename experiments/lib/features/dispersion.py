from __future__ import annotations

import pandas as pd


def cohort_std_by_date(
    df: pd.DataFrame,
    mask: pd.Series,
    value_col: str = "station_price_cents",
    date_col: str = "price_date",
) -> pd.Series:
    """Std of value_col within df[mask], grouped by date_col.

    PIT-safe: mask must be derived from same-date row attributes (e.g.
    stickiness_score snapshot), so only contemporaneous rows enter each
    date's std. No future dates are read.
    """
    return df.loc[mask].groupby(date_col)[value_col].std()


def cohort_agg_diff_by_date(
    df: pd.DataFrame,
    mask_a: pd.Series,
    mask_b: pd.Series,
    value_col: str = "station_price_cents",
    date_col: str = "price_date",
    agg: str = "median",
) -> pd.Series:
    """agg(df[mask_a][value_col]) − agg(df[mask_b][value_col]) per date.

    PIT-safe when mask_a and mask_b are derived from same-date row attributes.
    Returns a Series indexed by date; NaN for dates where either cohort is empty.
    """
    a = df.loc[mask_a].groupby(date_col)[value_col].agg(agg)
    b = df.loc[mask_b].groupby(date_col)[value_col].agg(agg)
    return (a - b).rename(f"{value_col}_{agg}_diff")
