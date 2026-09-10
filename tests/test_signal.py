"""Tests for fuel_signal.signal — signal primitives, combination, build_signals."""

from __future__ import annotations

import datetime
import re

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
    assert f"E10 - {as_of}" in output


def test_station_label_in_output(signal_db):
    conn, series, _ = signal_db
    as_of = series[3 * _CYCLE_LENGTH + _CYCLE_LENGTH // 2][0]
    output = build_signals(conn, as_of, preferred_stations=_PREFERRED)
    assert "Shell Springwood" in output


def test_price_in_output(signal_db):
    conn, series, _ = signal_db
    as_of = series[3 * _CYCLE_LENGTH + _CYCLE_LENGTH // 2][0]
    output = build_signals(conn, as_of, preferred_stations=_PREFERRED)
    assert re.search(r"Shell Springwood\s+\d+\.\dc", output)


def test_per_signal_reasons_in_output(signal_db):
    conn, series, _ = signal_db
    as_of = series[3 * _CYCLE_LENGTH + _CYCLE_LENGTH // 2][0]
    output = build_signals(conn, as_of, preferred_stations=_PREFERRED, explain=True)
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
    output = build_signals(conn, as_of, preferred_stations=_PREFERRED, explain=True)
    assert "legacy rule signals: BUY" in output


def test_wait_verdict_mid_cycle(signal_db):
    """Mid cycle + price in middle range → WAIT."""
    conn, series, _ = signal_db
    as_of = series[164][0]   # day 164: pct ~0.5, price ~161
    output = build_signals(conn, as_of, preferred_stations=_PREFERRED, explain=True)
    assert "legacy rule signals: WAIT" in output


def test_dont_buy_verdict_just_after_peak(signal_db):
    """Early in cycle + price near max → DONT_BUY."""
    conn, series, _ = signal_db
    as_of = series[146][0]   # day 146: pct ~0.11, price ~172 (near max)
    output = build_signals(conn, as_of, preferred_stations=_PREFERRED, explain=True)
    assert "legacy rule signals: DON'T BUY" in output


def test_day_and_cycle_format(signal_db):
    conn, series, _ = signal_db
    as_of = series[3 * _CYCLE_LENGTH + _CYCLE_LENGTH // 2][0]
    output = build_signals(conn, as_of, preferred_stations=_PREFERRED)
    assert re.search(r"cycle day \d+\+?/\d+", output)


def test_day_number_exceeds_cycle_len_shows_plus_suffix(tmp_path):
    db_path = tmp_path / "long_cycle.db"
    conn = db.open_db(db_path)
    db.create_schema(conn)
    conn.execute(
        "INSERT INTO stations"
        " (station_code, address_normalized, suburb, postcode, name, brand)"
        " VALUES (9001, '1 main street springwood', 'Springwood', '2777',"
        "         'Shell Springwood', 'Shell')"
    )
    conn.commit()
    # 3 complete sawtooth cycles establish mean cycle length ~46 (peaks at days 3, 49, 95).
    # Append 50 flat days at trough price — no new peak, so scipy last peak stays at day 95.
    # At series[-1]: days_since_peak = 92, day_num = 93 > 46 → display must show "Day 46+/46".
    s1 = _sawtooth_series(n_cycles=3.0, start="2020-01-01")
    start = datetime.date(2020, 1, 1)
    tail = [
        ((start + datetime.timedelta(days=len(s1) + i)).isoformat(), 151.0)
        for i in range(50)
    ]
    series = s1 + tail
    fid = db.fuel_type_id(conn, "E10")
    conn.executemany(
        "INSERT INTO daily_prices (station_code, fuel_type_id, price_date, price_decicents)"
        " VALUES (9001, ?, ?, ?)",
        [(fid, db._date_to_int(d), round(p * 10)) for d, p in series],
    )
    conn.commit()

    as_of = series[-1][0]
    output = build_signals(conn, as_of, preferred_stations={9001: "Shell Springwood"})
    assert re.search(r"cycle day \d+\+/\d+", output), (
        f"Expected 'cycle day N+/N' when cycle exceeds mean, got:\n{output}"
    )
    conn.close()


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
    assert f"E10 - {as_of}" in result.output
    assert ("FILL UP" in result.output) or ("WORTH A STOP" in result.output) or (
        "WAIT if you can" in result.output
    ) or ("No price data" in result.output)


# ---------------------------------------------------------------------------
# Decision layer — route awareness, ordering, staleness
# ---------------------------------------------------------------------------

@pytest.fixture
def two_station_db(tmp_path):
    """Two stations on the same synthetic series, priced apart on the last day."""
    db_path = tmp_path / "two_station.db"
    conn = db.open_db(db_path)
    db.create_schema(conn)
    for code, name in ((9001, "Daily Servo"), (9002, "Weekly Servo")):
        conn.execute(
            "INSERT INTO stations"
            " (station_code, address_normalized, suburb, postcode, name, brand)"
            f" VALUES ({code}, '{code} main street', 'Springwood', '2777', ?, 'Shell')",
            (name,),
        )
    series = _sawtooth_series(n_cycles=4.0, start="2020-01-01")
    fid = db.fuel_type_id(conn, "E10")
    rows = []
    for code, offset in ((9001, 0.0), (9002, -8.0)):
        rows += [
            (code, fid, db._date_to_int(d), round((p + offset) * 10)) for d, p in series
        ]
    conn.executemany(
        "INSERT INTO daily_prices (station_code, fuel_type_id, price_date,"
        " price_decicents) VALUES (?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    yield conn, series
    conn.close()


_TWO = {9001: "Daily Servo", 9002: "Weekly Servo"}


def test_off_route_station_is_excluded_from_the_on_route_table(two_station_db, monkeypatch):
    """A station not passed today must not be offered as today's answer."""
    conn, series = two_station_db
    # Weekly Servo is passed Wednesdays only; evaluate on a Monday.
    monkeypatch.setattr(
        "fuel_signal.signal.STATION_ROUTE_DAYS", {9002: frozenset({2})}
    )
    as_of = series[180][0]
    monday = datetime.date(2026, 9, 7)
    assert monday.weekday() == 0
    output = build_signals(
        conn, as_of, preferred_stations=_TWO, today=monday
    )
    # Cheaper, but unreachable → it cannot be the headline recommendation.
    assert "OFF ROUTE" in output
    assert "next pass in 2d, Wed" in output
    headline = [ln for ln in output.splitlines() if "FILL UP" in ln or "WAIT" in ln]
    assert all("Weekly Servo" not in ln for ln in headline)


def test_off_route_station_joins_the_table_on_a_day_it_is_passed(two_station_db, monkeypatch):
    conn, series = two_station_db
    monkeypatch.setattr(
        "fuel_signal.signal.STATION_ROUTE_DAYS", {9002: frozenset({2})}
    )
    as_of = series[180][0]
    wednesday = datetime.date(2026, 9, 9)
    assert wednesday.weekday() == 2
    output = build_signals(conn, as_of, preferred_stations=_TWO, today=wednesday)
    assert "OFF ROUTE" not in output
    # 8c cheaper and reachable → it is the marked pick.
    assert re.search(r"->\s+Weekly Servo", output)


def test_stations_are_ranked_by_price_not_by_probability(two_station_db, monkeypatch):
    """The trap: P(BUY) is station-relative and must never drive the ordering.

    labels.py condition 2 measures cheapness against each station's OWN trailing
    percentile, so the dearest pump routinely carries the highest P(BUY). Sorting
    by it would send the driver to the expensive one.
    """
    conn, series = two_station_db
    monkeypatch.setattr("fuel_signal.signal.STATION_ROUTE_DAYS", {})
    # Give the EXPENSIVE station the high probability.
    monkeypatch.setattr(
        "fuel_signal.signal.model_probabilities",
        lambda *a, **k: {9001: 0.95, 9002: 0.30},
    )
    as_of = series[180][0]
    output = build_signals(
        conn, as_of, preferred_stations=_TWO, today=datetime.date(2026, 9, 7)
    )
    # Table rows only — the headline sentence also names a station.
    table = [ln for ln in output.splitlines() if re.search(r"Servo\s+\d+\.\dc", ln)]
    # Weekly Servo is 8c cheaper, so it leads despite the lower probability.
    assert "Weekly Servo" in table[0]
    assert table[0].lstrip().startswith("->")
    assert "0.30" in table[0]
    assert "Daily Servo" in table[1]
    assert "0.95" in table[1]


def test_stale_prices_are_flagged_with_their_age(signal_db):
    conn, series, _ = signal_db
    # signal_db's newest real `prices` row is the last series date; ask for it so
    # the query is "latest available", not a historical view.
    as_of = series[-1][0]
    later = datetime.date.fromisoformat(as_of) + datetime.timedelta(days=5)
    output = build_signals(conn, as_of, preferred_stations=_PREFERRED, today=later)
    assert "!! 5 days stale" in output


def test_historical_as_of_is_not_reported_as_stale(signal_db):
    """A deliberate --as-of into the past is a view choice, not stale data."""
    conn, series, _ = signal_db
    as_of = series[180][0]           # well before the newest real observation
    output = build_signals(
        conn,
        as_of,
        preferred_stations=_PREFERRED,
        today=datetime.date.fromisoformat(series[-1][0]),
    )
    assert "stale" not in output
    assert "historical view" in output


def test_staleness_is_measured_from_the_last_real_row_not_the_filled_one(signal_db):
    """as_of defaults to daily_prices' MAX, which fill.py may have fabricated.

    Measuring age from as_of would understate it by exactly the forward-filled
    days — the case the banner exists to catch.
    """
    conn, series, _ = signal_db
    fid = db.fuel_type_id(conn, "E10")
    filled = (
        datetime.date.fromisoformat(series[-1][0]) + datetime.timedelta(days=4)
    ).isoformat()
    conn.execute(
        "INSERT INTO daily_prices (station_code, fuel_type_id, price_date,"
        " price_decicents) VALUES (9001, ?, ?, 1600)",
        (fid, db._date_to_int(filled)),
    )
    conn.commit()
    today = datetime.date.fromisoformat(filled) + datetime.timedelta(days=1)
    output = build_signals(conn, filled, preferred_stations=_PREFERRED, today=today)
    # 5 days behind the last REAL row, not the 1 day behind the filled one.
    assert "!! 5 days stale" in output
    assert f"last real reading {series[-1][0]}" in output


def test_same_day_prices_carry_no_staleness_banner(signal_db):
    conn, series, _ = signal_db
    as_of = series[-1][0]
    same_day = datetime.date.fromisoformat(as_of)
    output = build_signals(conn, as_of, preferred_stations=_PREFERRED, today=same_day)
    assert "stale" not in output


def test_missing_model_artifact_degrades_to_prices_and_cycle(signal_db, tmp_path):
    conn, series, _ = signal_db
    as_of = series[180][0]
    output = build_signals(
        conn,
        as_of,
        preferred_stations=_PREFERRED,
        model_path=tmp_path / "definitely-absent.joblib",
        today=datetime.date.fromisoformat(as_of),
    )
    assert "no model artifact" in output
    # The column is gone from the table header; only the explanatory note names it.
    header = [ln for ln in output.splitlines() if "STATION" in ln and "PRICE" in ln]
    assert header and all("P(BUY)" not in ln for ln in header)


def test_diversion_advice_fires_only_when_the_gap_is_worth_it(two_station_db, monkeypatch):
    conn, series = two_station_db
    monkeypatch.setattr(
        "fuel_signal.signal.STATION_ROUTE_DAYS", {9002: frozenset({2})}
    )
    as_of = series[180][0]
    output = build_signals(
        conn, as_of, preferred_stations=_TWO, today=datetime.date(2026, 9, 7)
    )
    # 8c gap clears DIVERSION_WORTH_CENTS (3.0).
    assert "worth timing a fill for" in output
    assert "no reason to divert" not in output


def test_network_drift_survives_a_forward_filled_flat_last_day():
    """A repeated final value must not read as 'flat' — that is the stale case."""
    from fuel_signal.signal import _network_drift

    series = [
        (f"2026-09-{d:02d}", 200.0 - d) for d in range(1, 9)
    ]  # falling 1c/day
    series.append(("2026-09-09", series[-1][1]))  # forward-filled repeat
    drift = _network_drift(series, "2026-09-09")
    assert drift is not None
    assert drift < -0.5, f"expected a clear fall, got {drift}"
