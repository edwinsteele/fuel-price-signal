"""Tests for experiments/lib/features — deltas and rolling helpers."""
from __future__ import annotations

import pandas as pd

from experiments.lib.features.deltas import calendar_aware_delta
from experiments.lib.features.rolling import rolling_baseline


def _series(dates: list[str], values: list[float]) -> pd.Series:
    return pd.Series(values, index=pd.to_datetime(dates))


# ---------------------------------------------------------------------------
# calendar_aware_delta — empty-input guard
# ---------------------------------------------------------------------------

def test_calendar_aware_delta_empty_returns_empty():
    s = pd.Series([], dtype=float)
    result = calendar_aware_delta(s, lag_days=3)
    assert result.empty
    assert result.dtype == s.dtype


def test_calendar_aware_delta_empty_does_not_raise():
    s = pd.Series([], index=pd.DatetimeIndex([]), dtype=float)
    calendar_aware_delta(s, lag_days=7)  # must not raise ValueError


# ---------------------------------------------------------------------------
# calendar_aware_delta — normal path
# ---------------------------------------------------------------------------

def test_calendar_aware_delta_basic():
    s = _series(["2024-01-01", "2024-01-02", "2024-01-03"], [10.0, 12.0, 15.0])
    result = calendar_aware_delta(s, lag_days=1)
    assert abs(result.loc[pd.Timestamp("2024-01-02")] - 2.0) < 1e-9
    assert abs(result.loc[pd.Timestamp("2024-01-03")] - 3.0) < 1e-9


def test_calendar_aware_delta_gap_yields_nan():
    """A gap in the input wider than lag_days produces NaN for the missing date."""
    s = _series(["2024-01-01", "2024-01-05"], [10.0, 20.0])
    result = calendar_aware_delta(s, lag_days=1)
    # 01-05 shifted by 1 → 01-04 which was gap-filled with NaN → result is NaN
    assert result.loc[pd.Timestamp("2024-01-05")] != result.loc[pd.Timestamp("2024-01-05")]


# ---------------------------------------------------------------------------
# rolling_baseline — empty-input guard
# ---------------------------------------------------------------------------

def test_rolling_baseline_empty_returns_empty():
    s = pd.Series([], dtype=float)
    result = rolling_baseline(s, window_days=7)
    assert result.empty
    assert result.dtype == s.dtype


def test_rolling_baseline_empty_does_not_raise():
    s = pd.Series([], index=pd.DatetimeIndex([]), dtype=float)
    rolling_baseline(s, window_days=3)  # must not raise ValueError


# ---------------------------------------------------------------------------
# rolling_baseline — normal path
# ---------------------------------------------------------------------------

def test_rolling_baseline_basic():
    s = _series(
        ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
        [10.0, 20.0, 30.0, 40.0],
    )
    result = rolling_baseline(s, window_days=3, closed="left", min_periods=1)
    # At 01-03, closed='left' window [01-01, 01-02] → median(10, 20) = 15
    assert abs(result.loc[pd.Timestamp("2024-01-03")] - 15.0) < 1e-9
