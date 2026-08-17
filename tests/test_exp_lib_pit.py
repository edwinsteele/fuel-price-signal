"""Tests for experiments/lib/pit_test.py — the differential PIT leak test."""
from __future__ import annotations

import pandas as pd
import pytest

from experiments.lib.pit_test import PitLeakError, differential_pit_test


def _panel_frame() -> pd.DataFrame:
    dates = [20260801, 20260802, 20260803, 20260804, 20260805, 20260806, 20260807, 20260808]
    rows = []
    for station in ("A", "B"):
        for i, d in enumerate(dates):
            price = 100.0 + i + (0 if station == "A" else 10)
            rows.append({"station_code": station, "price_date": d, "price_cents": price})
    return pd.DataFrame(rows)


def _pit_safe_rolling_mean(df: pd.DataFrame) -> pd.DataFrame:
    """Backward-looking rolling mean over price_cents, grouped by station — no future reads."""
    out = df.sort_values(["station_code", "price_date"]).copy()
    out["candidate_col"] = (
        out.groupby("station_code")["price_cents"]
        .transform(lambda s: s.rolling(window=3, min_periods=1).mean())
    )
    return out


def _leaky_whole_series_mean(df: pd.DataFrame) -> pd.DataFrame:
    """The canonical LLM-authored leak: a whole-series aggregate with no shift."""
    out = df.copy()
    out["candidate_col"] = out.groupby("station_code")["price_cents"].transform("mean")
    return out


def test_pit_safe_function_passes():
    frame = _panel_frame()
    result = differential_pit_test(_pit_safe_rolling_mean, frame)
    assert "candidate_col" in result.columns


def test_leaky_function_raises_pit_leak_error():
    frame = _panel_frame()
    with pytest.raises(PitLeakError):
        differential_pit_test(_leaky_whole_series_mean, frame)


def test_leaky_series_valued_function_raises():
    frame = _panel_frame()

    def leaky_axis(df: pd.DataFrame) -> pd.Series:
        whole_series_mean = df.groupby("station_code")["price_cents"].transform("mean")
        return (df["price_cents"] > whole_series_mean).rename("axis")

    with pytest.raises(PitLeakError):
        differential_pit_test(leaky_axis, frame)


def test_pit_safe_series_valued_function_passes():
    frame = _panel_frame()

    def safe_axis(df: pd.DataFrame) -> pd.Series:
        return (df["price_date"] % 2 == 0).rename("axis")

    result = differential_pit_test(safe_axis, frame)
    assert result.name == "axis"


def test_requires_at_least_two_distinct_dates():
    frame = _panel_frame()
    frame["price_date"] = 20260801
    with pytest.raises(ValueError):
        differential_pit_test(_pit_safe_rolling_mean, frame)
