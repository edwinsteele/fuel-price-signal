"""Tests for experiments/pipeline/placebo.py — the placebo-column construction for the
reworked noise floor (fps-awz)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.pipeline.placebo import (
    PLACEBO_COLUMN_NAME,
    PLACEBO_SHIFT_DAYS_POOL,
    _effective_shift,
    axis_is_market_wide,
    candidate_pool,
    make_placebo_series,
    screen_draws,
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


# ── candidate_pool ──────────────────────────────────────────────────────────

def test_candidate_pool_starts_with_select_draws_own_output():
    """The common case (nothing fails screening) must behave identically to plain
    select_draws — candidate_pool only adds a FALLBACK tail beyond it."""
    baseline_columns = [f"col{i}" for i in range(10)]
    pool = candidate_pool(baseline_columns, 6)
    assert pool[:6] == select_draws(baseline_columns, 6)


def test_candidate_pool_fallback_covers_every_remaining_column_exactly_once():
    baseline_columns = [f"col{i}" for i in range(10)]
    pool = candidate_pool(baseline_columns, 6)
    primary_columns = {c for c, _ in pool[:6]}
    fallback_columns = [c for c, _ in pool[6:]]
    assert set(fallback_columns) == set(baseline_columns) - primary_columns
    assert len(fallback_columns) == len(set(fallback_columns))  # no repeats
    assert len(pool) == len(baseline_columns)


def test_candidate_pool_fallback_cycles_the_shift_pool_when_it_runs_short():
    """More remaining columns than PLACEBO_SHIFT_DAYS_POOL entries — the fallback tail
    must still produce one candidate per remaining column (shift reuse is fine; only the
    PRIMARY set needs distinct shifts for diversity)."""
    baseline_columns = [f"col{i}" for i in range(len(PLACEBO_SHIFT_DAYS_POOL) + 10)]
    pool = candidate_pool(baseline_columns, 5)
    assert len(pool) == len(baseline_columns)
    assert sorted(c for c, _ in pool) == sorted(baseline_columns)


# ── screen_draws ────────────────────────────────────────────────────────────

def _diverse_fixture_df(n_dates: int, n_cols: int) -> pd.DataFrame:
    """A longer, market-wide-only fixture whose columns are deliberately diverse (not the
    slow/short-series case) so a reasonable shift decorrelates all of them — used where a
    test needs several REAL candidates that pass screening, not a forced failure."""
    dates = list(range(20260101, 20260101 + n_dates))
    df = pd.DataFrame({"price_date": dates, "station_code": [111] * n_dates})
    for i in range(n_cols):
        df[f"col{i}"] = [float((d * (i + 3) + i) % 23) for d in range(n_dates)]
    return df


def test_screen_draws_happy_path_returns_exactly_n_draws_with_correlations():
    df = _diverse_fixture_df(n_dates=60, n_cols=10)
    baseline_columns = [f"col{i}" for i in range(10)]

    screened = screen_draws(df, baseline_columns, n_draws=5, max_shift_autocorrelation=0.9)

    assert len(screened) == 5
    for source_column, shift_days, corr in screened:
        assert source_column in baseline_columns
        assert shift_days in PLACEBO_SHIFT_DAYS_POOL
        assert pd.notna(corr) and corr <= 0.9


def test_screen_draws_substitutes_a_failing_primary_candidate(monkeypatch):
    """A candidate that fails the correlation screen must be replaced by the next
    candidate in candidate_pool, not abort the whole computation — the fps-awz review
    finding: a slowly-varying column fails at EVERY shift, deterministically, so an
    earlier abort-on-first-failure design could not produce a floor on real batch0 data
    at all."""
    import experiments.pipeline.placebo as placebo_module

    df = _diverse_fixture_df(n_dates=60, n_cols=6)
    baseline_columns = [f"col{i}" for i in range(6)]
    real_make_placebo_series = placebo_module.make_placebo_series

    def _fake(frame, source_column, shift_days):
        if source_column == "col2":
            # A no-op "shift" — guaranteed correlation 1.0, guaranteed to fail any
            # reasonable threshold.
            return frame[source_column].rename(placebo_module.PLACEBO_COLUMN_NAME)
        return real_make_placebo_series(frame, source_column, shift_days)

    monkeypatch.setattr(placebo_module, "make_placebo_series", _fake)

    screened = screen_draws(df, baseline_columns, n_draws=5, max_shift_autocorrelation=0.9)

    columns = {c for c, _, _ in screened}
    assert len(screened) == 5
    assert "col2" not in columns
    # n_draws == n_cols - 1 and exactly one column fails, so the result must be every
    # OTHER column — this holds regardless of primary/fallback index arithmetic.
    assert columns == set(baseline_columns) - {"col2"}


def test_screen_draws_skips_an_all_nan_column():
    df = _fixture_df()
    df["all_nan_col"] = float("nan")
    baseline_columns = ["mw_col", "station_col", "all_nan_col"]

    screened = screen_draws(df, baseline_columns, n_draws=2, max_shift_autocorrelation=0.99)

    assert {c for c, _, _ in screened} == {"mw_col", "station_col"}


def test_screen_draws_uses_pairwise_complete_correlation_not_np_corrcoef():
    """Review finding: np.corrcoef returns NaN the instant ANY element is NaN, which would
    mask a column's real, checkable correlation as unverifiable the moment it has even one
    missing value (e.g. lga_mean_cents/brand_mean_cents on real batch0 data, which carry
    partial NaN on dates with too few classified stations — fuel_signal/features.py's own
    documented behaviour). pd.Series.corr is pairwise-complete and must be used instead."""
    n = 60
    df = pd.DataFrame({
        "price_date": list(range(20260101, 20260101 + n)),
        "station_code": [111] * n,
        "ramp_col": [float(i) for i in range(n)],
    })
    df.loc[df.index[::10], "ramp_col"] = float("nan")  # a few, not all, NaN

    placebo = make_placebo_series(df, "ramp_col", shift_days=101)
    np_corr = np.corrcoef(placebo.to_numpy(dtype=float), df["ramp_col"].to_numpy(dtype=float))[0, 1]
    assert np.isnan(np_corr)  # confirms this fixture actually exercises the discrepancy

    screened = screen_draws(df, ["ramp_col"], n_draws=1, max_shift_autocorrelation=0.99)
    assert len(screened) == 1
    assert pd.notna(screened[0][2])


def test_screen_draws_raises_when_the_whole_pool_cannot_meet_n_draws(monkeypatch):
    import experiments.pipeline.placebo as placebo_module

    df = _diverse_fixture_df(n_dates=60, n_cols=2)
    baseline_columns = ["col0", "col1"]

    def _always_no_op(frame, source_column, shift_days):
        return frame[source_column].rename(placebo_module.PLACEBO_COLUMN_NAME)

    monkeypatch.setattr(placebo_module, "make_placebo_series", _always_no_op)

    with pytest.raises(ValueError, match=r"only found 0/2"):
        screen_draws(df, baseline_columns, n_draws=2, max_shift_autocorrelation=0.9)


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
