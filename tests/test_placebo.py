"""Tests for experiments/pipeline/placebo.py — the placebo-column construction for the
reworked noise floor (fps-awz)."""
from __future__ import annotations

import pandas as pd
import pytest

from experiments.pipeline.placebo import (
    PLACEBO_COLUMN_NAME,
    PLACEBO_SHIFT_DAYS_POOL,
    _effective_shift,
    axis_is_market_wide,
    make_placebo_series,
    select_draws,
)

DATES = [20260801 + i for i in range(8)]
STATIONS = [100, 200, 300]


def _fixture_df() -> pd.DataFrame:
    """3 stations x 8 dates, one market-wide column (constant per date across every
    station) and one station-varying column (distinct per (station, date))."""
    rows = []
    for date_idx, date in enumerate(DATES):
        for station in STATIONS:
            rows.append(
                {
                    "price_date": date,
                    "station_code": station,
                    "mw_col": float(date_idx),  # depends only on date
                    "station_col": float(station + date_idx * 10),  # depends on both
                }
            )
    return pd.DataFrame(rows)


def test_axis_is_market_wide_true_for_a_per_date_only_column():
    df = _fixture_df()
    assert axis_is_market_wide(df, "mw_col") is True


def test_axis_is_market_wide_false_for_a_station_varying_column():
    df = _fixture_df()
    assert axis_is_market_wide(df, "station_col") is False


def test_effective_shift_never_degenerates_to_zero_for_a_prime_pool_member():
    for n_unique in range(2, 12):
        for shift_days in PLACEBO_SHIFT_DAYS_POOL:
            assert _effective_shift(shift_days, n_unique) != 0


def test_effective_shift_raises_on_fewer_than_two_positions():
    with pytest.raises(ValueError, match="distinct date"):
        _effective_shift(97, 1)
    with pytest.raises(ValueError, match="distinct date"):
        _effective_shift(97, 0)


def test_make_placebo_series_on_market_wide_column_stays_constant_per_date():
    df = _fixture_df()
    placebo = make_placebo_series(df, "mw_col", shift_days=97)
    check = df.assign(**{PLACEBO_COLUMN_NAME: placebo})
    # The market-wide invariant must survive the shift — this is the whole point of routing
    # a market-wide column through the date-only branch rather than a within-date shuffle.
    assert axis_is_market_wide(check, PLACEBO_COLUMN_NAME) is True


def test_make_placebo_series_on_market_wide_column_preserves_the_value_multiset():
    df = _fixture_df()
    placebo = make_placebo_series(df, "mw_col", shift_days=97)
    original_per_date = sorted(df.drop_duplicates("price_date")["mw_col"])
    shifted_per_date = sorted(placebo.loc[df.drop_duplicates("price_date").index])
    assert original_per_date == shifted_per_date


def test_make_placebo_series_on_market_wide_column_actually_moves_values():
    df = _fixture_df()
    placebo = make_placebo_series(df, "mw_col", shift_days=97)
    # Not identical to the source column — the shift must have actually reassigned dates.
    assert not placebo.equals(df["mw_col"].rename(PLACEBO_COLUMN_NAME))


def test_make_placebo_series_on_station_varying_column_preserves_each_stations_own_values():
    df = _fixture_df()
    placebo = make_placebo_series(df, "station_col", shift_days=101)
    check = df.assign(**{PLACEBO_COLUMN_NAME: placebo})
    for station, group in check.groupby("station_code"):
        original = sorted(group["station_col"])
        shifted = sorted(group[PLACEBO_COLUMN_NAME])
        assert original == shifted, f"station {station} lost/gained a value under the shift"


def test_make_placebo_series_on_station_varying_column_actually_moves_values():
    df = _fixture_df()
    placebo = make_placebo_series(df, "station_col", shift_days=101)
    assert not placebo.equals(df["station_col"].rename(PLACEBO_COLUMN_NAME))


def test_make_placebo_series_returns_aligned_index():
    df = _fixture_df()
    placebo = make_placebo_series(df, "station_col", shift_days=101)
    assert placebo.index.equals(df.index)
    placebo_mw = make_placebo_series(df, "mw_col", shift_days=101)
    assert placebo_mw.index.equals(df.index)


def test_select_draws_returns_n_draws_distinct_columns():
    baseline_columns = [f"col{i}" for i in range(54)]
    draws = select_draws(baseline_columns, 20)
    assert len(draws) == 20
    columns = [c for c, _ in draws]
    assert len(set(columns)) == 20
    assert all(c in baseline_columns for c in columns)


def test_select_draws_is_deterministic():
    baseline_columns = [f"col{i}" for i in range(54)]
    assert select_draws(baseline_columns, 20) == select_draws(baseline_columns, 20)


def test_select_draws_spreads_across_the_full_column_list():
    baseline_columns = [f"col{i}" for i in range(54)]
    draws = select_draws(baseline_columns, 20)
    indices = sorted(baseline_columns.index(c) for c, _ in draws)
    # Not clustered at the front — the last drawn index should be well past the halfway
    # point of a 54-column list when spreading 20 draws evenly.
    assert indices[-1] >= 27


def test_select_draws_uses_distinct_shift_days():
    baseline_columns = [f"col{i}" for i in range(54)]
    draws = select_draws(baseline_columns, 20)
    shifts = [s for _, s in draws]
    assert len(set(shifts)) == len(shifts)
    assert all(s in PLACEBO_SHIFT_DAYS_POOL for s in shifts)


def test_select_draws_raises_when_n_draws_exceeds_column_count():
    baseline_columns = [f"col{i}" for i in range(5)]
    with pytest.raises(ValueError, match="exceeds the baseline column count"):
        select_draws(baseline_columns, 10)


def test_select_draws_raises_when_n_draws_exceeds_shift_pool():
    baseline_columns = [f"col{i}" for i in range(100)]
    with pytest.raises(ValueError, match="exceeds the shift-day pool size"):
        select_draws(baseline_columns, len(PLACEBO_SHIFT_DAYS_POOL) + 1)


def test_select_draws_raises_on_non_positive_n_draws():
    baseline_columns = [f"col{i}" for i in range(5)]
    with pytest.raises(ValueError, match="n_draws must be >= 1"):
        select_draws(baseline_columns, 0)
