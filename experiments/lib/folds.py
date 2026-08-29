from __future__ import annotations

import inspect
import math
from collections.abc import Generator

import numpy as np
import pandas as pd

from fuel_signal import evaluate as _ev

from .constants import SEEDS, SHOCK_QUANTILE
from .fit import fit_score, per_row_log_loss

# walk_forward_folds' own defaults, read off the signature rather than restated, so a
# change there can't silently invalidate dependent code — mirrors runner.py's own
# _WFCV_DEFAULTS derivation. Shared here (not duplicated) because fps-3tu's batch-level
# shock-fold cache (experiments.pipeline.shock_folds) needs the same three values in a
# second place, to resolve a caller's possibly-partial outer_fold_params dict to the
# full config before comparing it against what a cache was computed under (review
# finding on PR #350: a hardcoded restatement here would go stale exactly the way
# SHOCK_FOLDS itself did).
DEFAULT_OUTER_FOLD_PARAMS: dict = {
    name: param.default
    for name, param in inspect.signature(_ev.walk_forward_folds).parameters.items()
    if name in ("train_min_days", "val_days", "step_days")
}


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

    Raises ValueError on a NaN value (a degenerate fit — e.g. a single-class val fold)
    rather than silently including or excluding it: `sorted(..., reverse=True)` orders a
    NaN by its POSITION relative to comparisons that involve it, which depends on
    dict/list iteration order, not on any property of the data — the same fold could
    land either side of the cut depending on insertion order alone (review finding on
    PR #350). Consistent with this codebase's rule elsewhere: a value that cannot be
    shown to be well-ordered cannot be trusted to pick a "hardest" fold.
    """
    if not fold_ll:
        return frozenset()
    nan_folds = sorted(i for i, ll in fold_ll.items() if ll != ll)  # ll != ll <=> isnan, no numpy import needed
    if nan_folds:
        raise ValueError(
            f"compute_shock_folds(): fold_ll has a NaN log-loss for fold(s) {nan_folds} — "
            "cannot rank a fold that isn't comparable; sorted() with a NaN present is "
            "order-dependent, not well-defined. Investigate the degenerate fit rather than "
            "silently including or excluding it."
        )
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
    train_min_days: int = DEFAULT_OUTER_FOLD_PARAMS["train_min_days"],
    val_days: int = DEFAULT_OUTER_FOLD_PARAMS["val_days"],
    step_days: int = DEFAULT_OUTER_FOLD_PARAMS["step_days"],
    shock_quantile: float = SHOCK_QUANTILE,
    shock_folds: frozenset[int] | None = None,
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

    `regime` is computed empirically (fps-3tu, `compute_shock_folds` above) by default,
    which needs every fold's baseline log-loss before any fold can be labelled — so this
    fits the baseline for ALL folds first (the `folds = list(...)` line below already
    materialises every fold's train_df/val_df before the loop starts, so caching the fit
    results alongside costs nothing extra in kind, only degree), derives the shock-fold
    set once, then yields.

    `shock_folds`, when given, is used AS-IS instead of derived from this call's own fits
    — the batch-level cache path (fps-1lb, `experiments.pipeline.shock_folds`): the
    empirical set is deterministic given (data, baseline_cols, seed), so every candidate
    in a batch would otherwise redo the identical derivation. `shock_quantile` is ignored
    when this is set.
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

    resolved_shock_folds = (
        shock_folds if shock_folds is not None
        else compute_shock_folds({i: f[2] for i, f in fits.items()}, quantile=shock_quantile)
    )

    for i, (train_df, val_df, baseline_ll, baseline_p, baseline_t, baseline_prl) in fits.items():
        regime = "shock" if i in resolved_shock_folds else "normal"
        yield i, regime, train_df, val_df, baseline_ll, baseline_p, baseline_t, baseline_prl
