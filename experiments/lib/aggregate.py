from __future__ import annotations

import numpy as np
import pandas as pd


def aggregate_with_deltas(
    df_rows: pd.DataFrame,
    cohort_ll_map: dict[str, str],
    baseline_run: str = "R0",
) -> pd.DataFrame:
    """Group by (fold, regime, run); compute mean/median/seedstd per cohort.

    Appends delta_* columns (vs baseline_run) for both mean and median
    aggregations. The _mean_base and _median_base reference columns are kept
    in the returned DataFrame so they end up in fold_run.csv.
    """
    agg_kwargs: dict[str, tuple[str, object]] = {}
    for col in cohort_ll_map.values():
        agg_kwargs[f"{col}_mean"] = (col, "mean")
        agg_kwargs[f"{col}_median"] = (col, "median")
        agg_kwargs[f"{col}_seedstd"] = (col, lambda s: float(np.nanstd(s, ddof=1)))
    fold_run = df_rows.groupby(["fold", "regime", "run"], as_index=False).agg(**agg_kwargs)

    base_rename = {}
    for c in cohort_ll_map.values():
        base_rename[f"{c}_mean"] = f"{c}_mean_base"
        base_rename[f"{c}_median"] = f"{c}_median_base"
    base = fold_run[fold_run["run"] == baseline_run][
        ["fold"]
        + [f"{c}_mean" for c in cohort_ll_map.values()]
        + [f"{c}_median" for c in cohort_ll_map.values()]
    ].rename(columns=base_rename)
    fold_run = fold_run.merge(base, on="fold")
    for c in cohort_ll_map.values():
        fold_run[f"delta_{c}_mean"] = fold_run[f"{c}_mean"] - fold_run[f"{c}_mean_base"]
        fold_run[f"delta_{c}_median"] = fold_run[f"{c}_median"] - fold_run[f"{c}_median_base"]

    return fold_run
