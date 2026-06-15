"""Regime-local cycle-length denominator — the #254 diagnostic estimator.

Isolated here (not in the production `fuel_signal/cycle.py`) so the experiment
can prove the candidate on the corrected axis BEFORE it earns a production
change. `paired_wfcv.py` and `validate.py` both import from this module.

Design (frozen in #254 / memory project-cycle-feature-regime-defects):

  * Hard floor at the COVID structural break (2020-03-23). Lookback never
    reaches past the break: pre-break rows see only pre-break cycles, post-break
    rows only post-break cycles.
  * Estimator = expanding *median* of confirmed-peak cycle lengths in
    [max(break, series_start), t). Median (not mean) for fat-tail robustness
    (post-COVID 50-68d outliers).
  * Warm-up = pseudo-count shrinkage toward the pre-COVID median, k=2. The issue
    writes the shrinkage in mean form ((Σ + k·prior)/(n+k)); implemented here as
    the *median analog* — augment the post-break sample with k pseudo-obs at the
    prior, then take the median. This honours the stated "median for fat-tail
    robustness" rationale (a mean would re-import the tail sensitivity the design
    set out to remove) and reproduces the "~67% data-driven by the 4th post-break
    cycle" figure (n/(n+k)=4/6). See README § Estimator interpretation.
  * Cycle assignment: each cycle is stamped at its CLOSING peak (the PIT moment
    its length is known) and is post-break iff that closing peak >= break_date.
    This puts the 33d cycle closing 2020-03-19 pre-break and the 54d first-COVID
    cycle closing 2020-05-12 post-break.

The break date is a frozen literal constant (the *cause* date — first NSW Stage-1
lockdown), never re-estimated per fold (PIT trap). k is fixed, not swept (its
entire warm-up sits in train-only 2020; zero validation leverage).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fuel_signal.cycle import CycleDetector

# Frozen COVID structural break — the *cause* date (first NSW Stage-1 lockdown),
# confirmed offline 2026-06-15 to coincide (within one cycle) with the single
# L2 break in the metro cycle-length series (mean 27.8d -> 40.9d). Never
# re-estimated per fold.
BREAK_DATE = pd.Timestamp("2020-03-23")

# Pseudo-count for warm-up shrinkage toward the pre-COVID median. Fixed, not
# swept (no validation leverage — warm-up is entirely in train-only 2020).
PSEUDO_K = 2


class RegimeCycleDetector(CycleDetector):
    """CycleDetector with the #254 regime-local cycle-length denominator.

    Overrides only ``_mean_cycle_length``; the confirmed-peak boundary logic
    (post-#250 sticky find_peaks confirmation) and every other field of
    ``CycleState`` are inherited unchanged. Because ``cycle_pct_through`` divides
    ``days_since_last_peak`` by this value (cycle.py), swapping the denominator
    here updates the ratio consistently — exactly the single-source-of-truth
    relationship the production fix (#254 AC) will preserve.
    """

    def _mean_cycle_length(  # type: ignore[override]
        self,
        series: pd.Series,
        peak_indices: np.ndarray,
    ) -> float:
        if len(peak_indices) < 2:
            return float("nan")

        peak_dates = series.index[peak_indices].astype("datetime64[ns]")
        # Cycle i closes at peak_dates[i] (the later peak); length is its gap
        # from the previous peak. Mirror the base method's day arithmetic so the
        # ONLY difference vs baseline is mean -> regime-median.
        lengths = np.diff(peak_dates).astype("float64") / 1_000_000_000 / 60 / 60 / 24
        closing = pd.DatetimeIndex(peak_dates[1:])

        as_of = series.index[-1]  # boundary: detect() slices the series to as_of
        pre_mask = np.asarray(closing < BREAK_DATE)
        pre_lengths = lengths[pre_mask]

        if as_of < BREAK_DATE:
            # Pre-break row: expanding median of pre-break cycles only. (Folds
            # start late-2021, so this branch is train-only / low-stakes.)
            if len(pre_lengths) == 0:
                return float("nan")
            return float(round(float(np.median(pre_lengths)), 2))

        # Post-break row: post-break cycles, shrunk toward the pre-COVID prior.
        prior = float(np.median(pre_lengths)) if len(pre_lengths) else float("nan")
        post_lengths = lengths[~pre_mask]
        if len(post_lengths) == 0:
            # Immediately post-break, no closed post-break cycle yet -> prior.
            return float(round(prior, 2)) if not np.isnan(prior) else float("nan")
        if np.isnan(prior):
            sample = post_lengths
        else:
            sample = np.concatenate([post_lengths, np.full(PSEUDO_K, prior)])
        return float(round(float(np.median(sample)), 2))
