from __future__ import annotations

from collections.abc import Generator

import numpy as np
import pandas as pd

from fuel_signal import evaluate as _ev

from .constants import SEEDS, SHOCK_FOLDS
from .fit import fit_score, per_row_log_loss


def iter_folds_with_baseline_fit(
    df: pd.DataFrame,
    baseline_cols: list[str],
    seed: int = SEEDS[0],
    *,
    train_min_days: int = 1825,
    val_days: int = 90,
    step_days: int = 90,
) -> Generator[
    tuple[int, str, pd.DataFrame, pd.DataFrame, float, np.ndarray, float, np.ndarray],
    None,
    None,
]:
    """Yield one tuple per non-empty fold with the baseline fit already done.

    Yields: (fold_idx, regime, train_df, val_df,
             baseline_ll, baseline_p, baseline_t, baseline_prl)

    The per-fold loop and all experiment-specific cohort masks stay in the
    calling script; this function only encapsulates the
    "fit baseline once per fold, reuse for R0+seed0" pattern.
    """
    folds = list(
        _ev.walk_forward_folds(
            df,
            train_min_days=train_min_days,
            val_days=val_days,
            step_days=step_days,
        )
    )
    print(f"Walk-forward folds: {len(folds)}\n", flush=True)
    for i, (train_df, val_df) in enumerate(folds, start=1):
        if val_df.empty:
            continue
        regime = "shock" if i in SHOCK_FOLDS else "normal"
        y = val_df["label"].to_numpy(dtype=int)
        baseline_ll, baseline_p, baseline_t = fit_score(train_df, val_df, baseline_cols, seed)
        baseline_prl = per_row_log_loss(y, baseline_p)
        yield i, regime, train_df, val_df, baseline_ll, baseline_p, baseline_t, baseline_prl
