"""Tests for fuel_signal.signal — signal primitives, combination, build_signals."""

from __future__ import annotations

import datetime

import pytest
from click.testing import CliRunner

import fuel_signal.db as db
from fuel_signal.cycle import CycleState
from fuel_signal.signal import (
    SignalEvaluation,
    SignalRecommendation,
    _gap_boundaries,
    _station_latest_gradient,
    _station_price_at,
    average_cycle_time_signal,
    average_gradient_after_peak_signal,
    average_near_previous_min_max_signal,
    build_signals,
    combine_signals,
    favourite_station_price_gradient_signal,
)
from fuel_signal.signal import main as signal_cli

# ---------------------------------------------------------------------------
# Synthetic series helpers
# ---------------------------------------------------------------------------

_CYCLE_LENGTH = 46
_RISE_DAYS = 3
_BASE_PRICE = 150.0
_AMPLITUDE = 25.0


def _sawtooth_series(
    n_cycles: float = 4.0,
    cycle_length: int = _CYCLE_LENGTH,
    base_price: float = _BASE_PRICE,
    amplitude: float = _AMPLITUDE,
    start: str = "2020-01-01",
) -> list[tuple[str, float]]:
    total_days = int(n_cycles * cycle_length)
    start_date = datetime.date.fromisoformat(start)
    result = []
    for day in range(total_days):
        pos = day % cycle_length
        if pos < _RISE_DAYS:
            price = base_price + amplitude * (pos / _RISE_DAYS)
        else:
            price = base_price + amplitude * (
                1.0 - (pos - _RISE_DAYS) / (cycle_length - _RISE_DAYS)
            )
        result.append(((start_date + datetime.timedelta(days=day)).isoformat(), price))
    return result


def _state(
    pct: float = 0.5,
    days_since_peak: int = 23,
    mean_cycle: float = 46.0,
    last_min: float = 150.0,
    last_max: float = 175.0,
    gradients: list[float] | None = None,
) -> CycleState:
    return CycleState(
        as_of_date="2024-01-01",
        days_since_last_peak=days_since_peak,
        mean_cycle_length=mean_cycle,
        pct_through_cycle=pct,
        last_cycle_min=last_min,
        last_cycle_max=last_max,
        last_3_gradients=gradients if gradients is not None else [-1.0, -1.0, -1.0],
        peak_count=3,
    )


# ---------------------------------------------------------------------------
# AverageCycleTimeSignal
# ---------------------------------------------------------------------------

def test_cycle_time_buy_when_late():
    ev = average_cycle_time_signal(_state(pct=0.85))
    assert ev.recommendation is SignalRecommendation.BUY


def test_cycle_time_wait_mid():
    ev = average_cycle_time_signal(_state(pct=0.50))
    assert ev.recommendation is SignalRecommendation.WAIT


def test_cycle_time_dont_buy_early():
    ev = average_cycle_time_signal(_state(pct=0.10))
    assert ev.recommendation is SignalRecommendation.DONT_BUY


def test_cycle_time_boundary_at_buy_threshold():
    # > 0.66 is BUY; exactly 0.66 should NOT be BUY (matches original >, not >=)
    assert average_cycle_time_signal(_state(pct=0.66)).recommendation is (
        SignalRecommendation.WAIT
    )
    assert average_cycle_time_signal(_state(pct=0.67)).recommendation is (
        SignalRecommendation.BUY
    )


# ---------------------------------------------------------------------------
# AverageGradientAfterPeakSignal
# ---------------------------------------------------------------------------

def test_gradient_flat_late_in_cycle_buys():
    state = _state(days_since_peak=30, mean_cycle=46.0, gradients=[0.1, -0.2, 0.0])
    ev = average_gradient_after_peak_signal(state)
    assert ev.recommendation is SignalRecommendation.BUY


def test_gradient_flat_early_in_cycle_dont_buys():
    state = _state(days_since_peak=5, mean_cycle=46.0, gradients=[0.1, -0.2, 0.0])
    ev = average_gradient_after_peak_signal(state)
    assert ev.recommendation is SignalRecommendation.DONT_BUY


def test_gradient_not_flat_is_neutral():
    state = _state(days_since_peak=30, gradients=[-1.0, -2.0, -1.5])
    ev = average_gradient_after_peak_signal(state)
    assert ev.recommendation is SignalRecommendation.NEUTRAL


def test_gradient_boundary_just_outside_flat():
    # -0.5 is NOT flat (boundaries are strict <, >)
    state = _state(days_since_peak=30, gradients=[-0.5, 0.0, 0.0])
    assert average_gradient_after_peak_signal(state).recommendation is (
        SignalRecommendation.NEUTRAL
    )


# ---------------------------------------------------------------------------
# AverageNearPreviousMinMaxSignal
# ---------------------------------------------------------------------------

def test_near_min_buys():
    ev = average_near_previous_min_max_signal(_state(), current_price=151.0)
    assert ev.recommendation is SignalRecommendation.BUY


def test_near_max_dont_buys():
    ev = average_near_previous_min_max_signal(_state(), current_price=170.0)
    assert ev.recommendation is SignalRecommendation.DONT_BUY


def test_middle_waits():
    ev = average_near_previous_min_max_signal(_state(), current_price=160.0)
    assert ev.recommendation is SignalRecommendation.WAIT


# ---------------------------------------------------------------------------
# FavouriteServiceStationPriceGradientSignal
# ---------------------------------------------------------------------------

def test_fav_all_big_raisers_dont_buys():
    ev = favourite_station_price_gradient_signal({"A": 12.0, "B": 11.0})
    assert ev.recommendation is SignalRecommendation.DONT_BUY


def test_fav_some_big_raisers_buys():
    ev = favourite_station_price_gradient_signal({"A": 12.0, "B": 0.1})
    assert ev.recommendation is SignalRecommendation.BUY


def test_fav_no_big_raisers_neutral():
    ev = favourite_station_price_gradient_signal({"A": -0.5, "B": 0.1})
    assert ev.recommendation is SignalRecommendation.NEUTRAL


def test_fav_empty_neutral():
    ev = favourite_station_price_gradient_signal({})
    assert ev.recommendation is SignalRecommendation.NEUTRAL


# ---------------------------------------------------------------------------
# combine_signals — threshold boundaries
# ---------------------------------------------------------------------------

def _ev(rec: SignalRecommendation) -> SignalEvaluation:
    return SignalEvaluation("test", rec, "")


def test_combine_buy_when_mean_at_threshold():
    # mean = 0.5 → BUY
    v = combine_signals([_ev(SignalRecommendation.BUY), _ev(SignalRecommendation.WAIT)])
    assert v.long_label == "BUY"


def test_combine_dont_buy_when_mean_at_threshold():
    v = combine_signals(
        [_ev(SignalRecommendation.DONT_BUY), _ev(SignalRecommendation.WAIT)]
    )
    assert v.long_label == "DON'T BUY"


def test_combine_wait_when_mean_in_middle():
    v = combine_signals(
        [_ev(SignalRecommendation.BUY), _ev(SignalRecommendation.DONT_BUY)]
    )
    assert v.long_label == "WAIT"


def test_combine_excludes_neutral():
    # Two BUYs + a NEUTRAL → mean = 1.0 → BUY
    v = combine_signals(
        [
            _ev(SignalRecommendation.BUY),
            _ev(SignalRecommendation.BUY),
            _ev(SignalRecommendation.NEUTRAL),
        ]
    )
    assert v.long_label == "BUY"
    assert v.mean_value == pytest.approx(1.0)


def test_combine_all_neutral_waits():
    v = combine_signals(
        [_ev(SignalRecommendation.NEUTRAL), _ev(SignalRecommendation.NEUTRAL)]
    )
    assert v.long_label == "WAIT"


# ---------------------------------------------------------------------------
# Fixtures for build_signals integration tests
# ---------------------------------------------------------------------------

@pytest.fixture
def signal_db(tmp_path):
    db_path = tmp_path / "signal_test.db"
    conn = db.open_db(db_path)
    db.create_schema(conn)

    conn.execute(
        "INSERT INTO stations"
        " (station_code, address_normalized, suburb, postcode, name, brand)"
        " VALUES (9001, '1 main street springwood', 'Springwood', '2777', 'Shell Springwood', 'Shell')"
    )
    conn.commit()

    series = _sawtooth_series(n_cycles=4.0, start="2020-01-01")
    fid = db.fuel_type_id(conn, "E10")

    conn.executemany(
        "INSERT INTO daily_prices (station_code, fuel_type_id, price_date, price_decicents)"
        " VALUES (9001, ?, ?, ?)",
        [(fid, db._date_to_int(d), round(p * 10)) for d, p in series],
    )

    midpoint_date = series[len(series) // 2][0]
    last_date = series[-1][0]
    source_h = conn.execute("SELECT id FROM price_sources WHERE code = 'h'").fetchone()[0]
    source_s = conn.execute("SELECT id FROM price_sources WHERE code = 's'").fetchone()[0]

    conn.execute(
        "INSERT OR IGNORE INTO prices"
        " (station_code, fuel_type_id, price_date, price_decicents, source_id)"
        " VALUES (9001, ?, ?, 1500, ?)",
        (fid, db._date_to_int(midpoint_date), source_h),
    )
    conn.execute(
        "INSERT OR IGNORE INTO prices"
        " (station_code, fuel_type_id, price_date, price_decicents, source_id)"
        " VALUES (9001, ?, ?, 1600, ?)",
        (fid, db._date_to_int(last_date), source_s),
    )
    conn.commit()

    yield conn, series, db_path
    conn.close()


_PREFERRED = {9001: "Shell Springwood"}


# ---------------------------------------------------------------------------
# build_signals — output format
# ---------------------------------------------------------------------------

def test_as_of_date_in_output(signal_db):
    conn, series, _ = signal_db
    as_of = series[3 * _CYCLE_LENGTH + _CYCLE_LENGTH // 2][0]
    output = build_signals(conn, as_of, preferred_stations=_PREFERRED)
    assert f"[as of {as_of}]" in output


def test_station_label_in_output(signal_db):
    conn, series, _ = signal_db
    as_of = series[3 * _CYCLE_LENGTH + _CYCLE_LENGTH // 2][0]
    output = build_signals(conn, as_of, preferred_stations=_PREFERRED)
    assert "Shell Springwood" in output


def test_price_in_output(signal_db):
    conn, series, _ = signal_db
    as_of = series[3 * _CYCLE_LENGTH + _CYCLE_LENGTH // 2][0]
    output = build_signals(conn, as_of, preferred_stations=_PREFERRED)
    assert "c" in output
    assert "E10 @ Shell Springwood:" in output


def test_per_signal_reasons_in_output(signal_db):
    conn, series, _ = signal_db
    as_of = series[3 * _CYCLE_LENGTH + _CYCLE_LENGTH // 2][0]
    output = build_signals(conn, as_of, preferred_stations=_PREFERRED)
    assert "AverageCycleTimeSignal" in output
    assert "AverageGradientAfterPeakSignal" in output
    assert "AverageNearPreviousMinMaxSignal" in output
    assert "FavouriteServiceStationPriceGradientSignal" in output


# ---------------------------------------------------------------------------
# build_signals — combined verdict
# ---------------------------------------------------------------------------

def test_buy_verdict_near_trough(signal_db):
    """Late in cycle + price near min → BUY."""
    conn, series, _ = signal_db
    as_of = series[180][0]   # day 180: pct ~0.85, price ~152 (near min)
    output = build_signals(conn, as_of, preferred_stations=_PREFERRED)
    assert "BUY " in output


def test_wait_verdict_mid_cycle(signal_db):
    """Mid cycle + price in middle range → WAIT."""
    conn, series, _ = signal_db
    as_of = series[164][0]   # day 164: pct ~0.5, price ~161
    output = build_signals(conn, as_of, preferred_stations=_PREFERRED)
    assert "WAIT" in output


def test_dont_buy_verdict_just_after_peak(signal_db):
    """Early in cycle + price near max → DONT_BUY."""
    conn, series, _ = signal_db
    as_of = series[146][0]   # day 146: pct ~0.11, price ~172 (near max)
    output = build_signals(conn, as_of, preferred_stations=_PREFERRED)
    assert "DONT" in output


def test_day_and_cycle_format(signal_db):
    conn, series, _ = signal_db
    as_of = series[3 * _CYCLE_LENGTH + _CYCLE_LENGTH // 2][0]
    output = build_signals(conn, as_of, preferred_stations=_PREFERRED)
    import re
    assert re.search(r"Day \d+/\d+ of cycle", output)


# ---------------------------------------------------------------------------
# _station_price_at / _station_latest_gradient
# ---------------------------------------------------------------------------

def test_station_price_at_known_date(signal_db):
    conn, series, _ = signal_db
    date_str, expected_price = series[100]
    result = _station_price_at(conn, 9001, date_str)
    assert result is not None
    assert abs(result - expected_price) < 0.2


def test_station_price_at_returns_none_for_unknown_station(signal_db):
    conn, _, _path = signal_db
    assert _station_price_at(conn, 99999, "2020-05-01") is None


def test_station_price_at_before_any_data(signal_db):
    conn, _, _path = signal_db
    assert _station_price_at(conn, 9001, "2010-01-01") is None


def test_station_latest_gradient_negative_in_descent(signal_db):
    conn, series, _ = signal_db
    # Day 164 is mid-descent; gradient should be negative
    as_of = series[164][0]
    g = _station_latest_gradient(conn, 9001, as_of)
    assert g is not None
    assert g < 0


def test_station_latest_gradient_none_with_insufficient_data(signal_db):
    conn, _, _path = signal_db
    assert _station_latest_gradient(conn, 9001, "2010-01-01") is None


# ---------------------------------------------------------------------------
# _gap_boundaries
# ---------------------------------------------------------------------------

def test_gap_boundaries_detected(signal_db):
    conn, series, _ = signal_db
    gap_start, gap_end = _gap_boundaries(conn)
    midpoint = series[len(series) // 2][0]
    last = series[-1][0]
    expected_start = (
        datetime.date.fromisoformat(midpoint) + datetime.timedelta(days=1)
    ).isoformat()
    expected_end = (
        datetime.date.fromisoformat(last) - datetime.timedelta(days=1)
    ).isoformat()
    assert gap_start == expected_start
    assert gap_end == expected_end


def test_gap_boundaries_none_when_no_gap(tmp_path):
    conn = db.open_db(tmp_path / "nogap.db")
    db.create_schema(conn)
    gap_start, gap_end = _gap_boundaries(conn)
    assert gap_start is None
    assert gap_end is None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_gap_warning(signal_db):
    conn, series, db_path = signal_db
    midpoint_date = series[len(series) // 2][0]
    gap_date = (
        datetime.date.fromisoformat(midpoint_date) + datetime.timedelta(days=2)
    ).isoformat()

    runner = CliRunner()
    result = runner.invoke(signal_cli, ["--as-of", gap_date, "--db", str(db_path)])
    assert "WARNING" in result.output


def test_cli_no_warning_outside_gap(signal_db):
    conn, series, db_path = signal_db
    as_of = series[70][0]
    runner = CliRunner()
    result = runner.invoke(signal_cli, ["--as-of", as_of, "--db", str(db_path)])
    assert "WARNING" not in result.output


def test_cli_output_structure(signal_db):
    conn, series, db_path = signal_db
    as_of = series[180][0]

    runner = CliRunner()
    result = runner.invoke(signal_cli, ["--as-of", as_of, "--db", str(db_path)])
    assert result.exit_code == 0
    assert f"[as of {as_of}]" in result.output
    assert ("BUY " in result.output) or ("WAIT" in result.output) or ("DONT" in result.output)
