from __future__ import annotations

import math
from collections.abc import Generator

import numpy as np
import pandas as pd

from fuel_signal import evaluate as _ev

from .constants import SEEDS, SHOCK_QUANTILE
from .fit import fit_score, per_row_log_loss


def compute_shock_folds(fold_ll: dict[int, float], quantile: float = SHOCK_QUANTILE) -> frozenset[int]:
    """The `ceil(len(fold_ll) * quantile)` folds with the HIGHEST baseline log-loss.

    Replaces the fixed-index `SHOCK_FOLDS` taxonomy (fps-3tu): a macro-event calendar
    drifts silently as more history gets backfilled (fold N's calendar dates shift
    with `walk_forward_folds`'s data-anchored boundaries) and, per the taxonomy's own
    origin document (`experiments/2026-06-05_phase_lookup_nonparametric/README.md` §
    "Aggregates under three labelling schemes"), a named macro event doesn't reliably
    predict where the baseline model actually struggles — that document found the
    2023-10 Israel-Gaza-tagged fold was "mid-pack" by this same statistic and
    recommended empirical (data-driven) labelling as "the authoritative read" going
    forward. This is that scheme, generalised into shared code.

    `fold_ll` is fold_idx -> the baseline's own val-fold log-loss (same quantity
    `iter_folds_with_baseline_fit` already computes as `baseline_ll`) — NOT a
    candidate-vs-baseline delta. quantile=0 -> empty set; quantile>=1 -> every fold.
    """
    if not fold_ll:
        return frozenset()
    quantile = max(0.0, min(1.0, quantile))
    k = math.ceil(len(fold_ll) * quantile)
    if k <= 0:
        return frozenset()
    hardest = sorted(fold_ll, key=lambda i: fold_ll[i], reverse=True)[:k]
    return frozenset(hardest)


def iter_folds_with_baseline_fit(
    df: pd.DataFrame,
    baseline_cols: list[str],
    seed: int = SEEDS[0],
    *,
    train_min_days: int = 1825,
    val_days: int = 90,
    step_days: int = 90,
    shock_quantile: float = SHOCK_QUANTILE,
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

    `regime` is now computed empirically (fps-3tu, `compute_shock_folds` above),
    which needs every fold's baseline log-loss before any fold can be labelled —
    so this fits the baseline for ALL folds first (the `folds = list(...)` line
    below already materialises every fold's train_df/val_df before the loop
    starts, so caching the fit results alongside costs nothing extra in kind,
    only degree), derives the shock-fold set once, then yields.
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

    fits: dict[int, tuple[pd.DataFrame, pd.DataFrame, float, np.ndarray, float, np.ndarray]] = {}
    for i, (train_df, val_df) in enumerate(folds, start=1):
        if val_df.empty:
            continue
        y = val_df["label"].to_numpy(dtype=int)
        baseline_ll, baseline_p, baseline_t = fit_score(train_df, val_df, baseline_cols, seed)
        baseline_prl = per_row_log_loss(y, baseline_p)
        fits[i] = (train_df, val_df, baseline_ll, baseline_p, baseline_t, baseline_prl)

    shock_folds = compute_shock_folds({i: f[2] for i, f in fits.items()}, quantile=shock_quantile)

    for i, (train_df, val_df, baseline_ll, baseline_p, baseline_t, baseline_prl) in fits.items():
        regime = "shock" if i in shock_folds else "normal"
        yield i, regime, train_df, val_df, baseline_ll, baseline_p, baseline_t, baseline_prl
