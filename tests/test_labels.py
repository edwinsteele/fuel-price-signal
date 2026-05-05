"""Tests for fuel_signal.labels — label generation and training-row assembly.

Label is BUY=1: fires when price is cheap vs recent history AND no better deal is coming.
Tests use lookback_days=5 (instead of the production default of 90) to keep fixture data
small. Fixture layout per test: 5 lookback days + today + 7 forward days = 13 rows minimum.

Cheap/expensive is controlled by setting past prices high (200c) or low (160c):
    past=[200]*5 → 33rd percentile=200 → today=160 is cheap (160 <= 200)
    past=[160]*5 → 33rd percentile=160 → today=200 is expensive (200 > 160)
"""

import datetime

import pytest
from click.testing import CliRunner

from fuel_signal.db import create_schema, open_db, upsert_daily_prices, upsert_stations
from fuel_signal.labels import assemble_training_rows, compute_label, main

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

LOOKBACK = 5   # small lookback for tests; production default is 90


@pytest.fixture
def conn(tmp_path):
    c = open_db(tmp_path / "test.db")
    create_schema(c)
    yield c
    c.close()


def _d(offset: int) -> str:
    """Return YYYY-MM-DD for today + offset days."""
    return (datetime.date.today() + datetime.timedelta(days=offset)).isoformat()


def _station(conn, station_code: int) -> None:
    upsert_stations(conn, [{
        "station_code": station_code,
        "name": f"Station {station_code}",
        "address": f"{station_code} Main Street, Springwood",
        "suburb": "Springwood",
        "postcode": "2777",
        "brand": "Shell",
    }])


def _prices(conn, station_code: int, day_prices: list[tuple[int, float]]) -> None:
    """Insert (offset, price) pairs into daily_prices for station_code."""
    upsert_daily_prices(conn, [(station_code, "E10", _d(offset), price) for offset, price in day_prices])
    conn.commit()


def _cheap_fixture(conn, station_code: int, today_price: float, forward_prices: list[float]) -> None:
    """Insert 5 past days at 200c (making today_price cheap), today, then forward_prices."""
    past = [(-LOOKBACK - 7 + i, 200.0) for i in range(LOOKBACK)]
    today = [(-7, today_price)]
    fwd = [(-6 + i, p) for i, p in enumerate(forward_prices)]
    _prices(conn, station_code, past + today + fwd)


def _expensive_fixture(conn, station_code: int, today_price: float, forward_prices: list[float]) -> None:
    """Insert 5 past days at 160c (making today_price expensive), today, then forward_prices."""
    past = [(-LOOKBACK - 7 + i, 160.0) for i in range(LOOKBACK)]
    today = [(-7, today_price)]
    fwd = [(-6 + i, p) for i, p in enumerate(forward_prices)]
    _prices(conn, station_code, past + today + fwd)


# ---------------------------------------------------------------------------
# compute_label — two-condition BUY label
# ---------------------------------------------------------------------------

def test_compute_label_buy(conn):
    """Cheap price + no better deal → label=1 (BUY)."""
    _station(conn, 1001)
    # past 5 days at 200c → 33rd pct = 200; today=160 is cheap
    # forward: flat at 162c → future_min=162 >= 160-3=157 → no better deal
    _cheap_fixture(conn, 1001, today_price=160.0, forward_prices=[162, 163, 164, 165, 166, 167, 168])
    label = compute_label(conn, 1001, _d(-7), horizon_days=7, threshold_cents=3.0,
                          lookback_days=LOOKBACK, percentile_pct=33.0)
    assert label == 1


def test_compute_label_wait_drop_coming(conn):
    """Cheap price but a better deal is coming → label=0 (wait for drop)."""
    _station(conn, 1001)
    # past at 200c → today=160 is cheap; but forward drops to 155 → 155 < 160-3=157 → better deal coming
    _cheap_fixture(conn, 1001, today_price=160.0, forward_prices=[158, 156, 155, 156, 157, 158, 159])
    label = compute_label(conn, 1001, _d(-7), horizon_days=7, threshold_cents=3.0,
                          lookback_days=LOOKBACK, percentile_pct=33.0)
    assert label == 0


def test_compute_label_wait_expensive_no_drop(conn):
    """Expensive price with no drop coming (plateau) → label=0.

    This is the key plateau case the two-condition label fixes: the old single-condition
    label would have returned 1 here (no drop predicted), but the price is expensive.
    """
    _station(conn, 1001)
    # past at 160c → 33rd pct = 160; today=200 is expensive (200 > 160)
    # forward stays high → no drop coming
    _expensive_fixture(conn, 1001, today_price=200.0, forward_prices=[200, 200, 200, 200, 200, 200, 200])
    label = compute_label(conn, 1001, _d(-7), horizon_days=7, threshold_cents=3.0,
                          lookback_days=LOOKBACK, percentile_pct=33.0)
    assert label == 0


def test_compute_label_wait_expensive_drop_coming(conn):
    """Expensive price with a drop coming → label=0 (wait on both counts)."""
    _station(conn, 1001)
    # past at 160c → today=200 is expensive; forward drops to 190 → better deal coming
    _expensive_fixture(conn, 1001, today_price=200.0, forward_prices=[198, 195, 192, 190, 191, 192, 193])
    label = compute_label(conn, 1001, _d(-7), horizon_days=7, threshold_cents=3.0,
                          lookback_days=LOOKBACK, percentile_pct=33.0)
    assert label == 0


def test_compute_label_insufficient_forward_data(conn):
    """Fewer than horizon_days forward rows → None."""
    _station(conn, 1001)
    past = [(-LOOKBACK - 7 + i, 200.0) for i in range(LOOKBACK)]
    today = [(-7, 160.0)]
    fwd = [(-6 + i, 162.0) for i in range(5)]  # only 5, need 7
    _prices(conn, 1001, past + today + fwd)
    label = compute_label(conn, 1001, _d(-7), horizon_days=7, threshold_cents=3.0,
                          lookback_days=LOOKBACK, percentile_pct=33.0)
    assert label is None


def test_compute_label_insufficient_lookback(conn):
    """Fewer than lookback_days past rows → None."""
    _station(conn, 1001)
    # Only 3 past days, but lookback_days=5
    past = [(-10 + i, 200.0) for i in range(3)]
    today = [(-7, 160.0)]
    fwd = [(-6 + i, 162.0) for i in range(7)]
    _prices(conn, 1001, past + today + fwd)
    label = compute_label(conn, 1001, _d(-7), horizon_days=7, threshold_cents=3.0,
                          lookback_days=LOOKBACK, percentile_pct=33.0)
    assert label is None


def test_compute_label_missing_today(conn):
    """No price row for the requested date → None."""
    _station(conn, 1001)
    past = [(-LOOKBACK - 7 + i, 200.0) for i in range(LOOKBACK)]
    fwd = [(-6 + i, 162.0) for i in range(7)]
    _prices(conn, 1001, past + fwd)  # no today row
    label = compute_label(conn, 1001, _d(-7), horizon_days=7, threshold_cents=3.0,
                          lookback_days=LOOKBACK, percentile_pct=33.0)
    assert label is None


def test_compute_label_invalid_horizon(conn):
    """horizon_days < 1 raises ValueError."""
    _station(conn, 1001)
    with pytest.raises(ValueError, match="horizon_days"):
        compute_label(conn, 1001, _d(-7), horizon_days=0)


# ---------------------------------------------------------------------------
# assemble_training_rows
# ---------------------------------------------------------------------------

def test_assemble_buy_label(conn):
    """Assembler produces label=1 when price is cheap and no better deal is coming."""
    _station(conn, 1001)
    _cheap_fixture(conn, 1001, today_price=160.0, forward_prices=[162, 163, 164, 165, 166, 167, 168])
    df = assemble_training_rows(conn, horizon_days=7, threshold_cents=3.0,
                                lookback_days=LOOKBACK, station_codes=[1001])
    assert len(df) == 1
    row = df.iloc[0]
    assert row["station_code"] == 1001
    assert row["label"] == 1
    assert row["today_price_cents"] == 160.0
    assert row["future_min_cents"] == 162.0


def test_assemble_plateau_label(conn):
    """Assembler produces label=0 on expensive plateau (the plateau fix)."""
    _station(conn, 1001)
    _expensive_fixture(conn, 1001, today_price=200.0, forward_prices=[200, 200, 200, 200, 200, 200, 200])
    df = assemble_training_rows(conn, horizon_days=7, threshold_cents=3.0,
                                lookback_days=LOOKBACK, station_codes=[1001])
    assert len(df) == 1
    assert df.iloc[0]["label"] == 0


def test_assemble_excludes_insufficient_lookback(conn):
    """Rows without full lookback history are excluded."""
    _station(conn, 1001)
    # Only 3 past days before today, lookback_days=5 → no rows produced
    past = [(-10 + i, 200.0) for i in range(3)]
    today = [(-7, 160.0)]
    fwd = [(-6 + i, 162.0) for i in range(7)]
    _prices(conn, 1001, past + today + fwd)
    df = assemble_training_rows(conn, horizon_days=7, threshold_cents=3.0,
                                lookback_days=LOOKBACK, station_codes=[1001])
    assert df.empty


def test_assemble_excludes_incomplete_horizon(conn):
    """Rows without a full horizon of forward data are excluded."""
    _station(conn, 1001)
    past = [(-LOOKBACK - 7 + i, 200.0) for i in range(LOOKBACK)]
    today = [(-7, 160.0)]
    fwd = [(-6 + i, 162.0) for i in range(5)]  # only 5, need 7
    _prices(conn, 1001, past + today + fwd)
    df = assemble_training_rows(conn, horizon_days=7, threshold_cents=3.0,
                                lookback_days=LOOKBACK, station_codes=[1001])
    assert df.empty


def test_assemble_multi_station(conn):
    """Assembler returns rows from both stations."""
    _station(conn, 1001)
    _station(conn, 1002)
    _cheap_fixture(conn, 1001, today_price=160.0, forward_prices=[162, 163, 164, 165, 166, 167, 168])
    _cheap_fixture(conn, 1002, today_price=160.0, forward_prices=[162, 163, 164, 165, 166, 167, 168])
    df = assemble_training_rows(conn, horizon_days=7, threshold_cents=3.0,
                                lookback_days=LOOKBACK, station_codes=[1001, 1002])
    assert set(df["station_code"]) == {1001, 1002}


def test_assemble_empty_station(conn):
    """Station with no prices yields an empty DataFrame with correct columns."""
    _station(conn, 1001)
    df = assemble_training_rows(conn, horizon_days=7, threshold_cents=3.0,
                                lookback_days=LOOKBACK, station_codes=[1001])
    assert df.empty
    assert list(df.columns) == ["station_code", "price_date", "today_price_cents", "future_min_cents", "label"]


def test_assemble_empty_station_codes_list(conn):
    """station_codes=[] returns empty DataFrame without hitting the DB."""
    df = assemble_training_rows(conn, horizon_days=7, threshold_cents=3.0,
                                lookback_days=LOOKBACK, station_codes=[])
    assert df.empty
    assert list(df.columns) == ["station_code", "price_date", "today_price_cents", "future_min_cents", "label"]


def test_assemble_invalid_horizon(conn):
    """horizon_days < 1 raises ValueError."""
    with pytest.raises(ValueError, match="horizon_days"):
        assemble_training_rows(conn, horizon_days=0)


def test_cli_main_smoke(conn, tmp_path):
    """CLI happy path: writes a CSV and exits 0."""
    _station(conn, 1001)
    _cheap_fixture(conn, 1001, today_price=160.0, forward_prices=[162, 163, 164, 165, 166, 167, 168])
    out_path = tmp_path / "labels.csv"
    result = CliRunner().invoke(main, [
        "--db", str(tmp_path / "test.db"),
        "--output", str(out_path),
        "--horizon", "7",
        "--threshold", "3.0",
        "--lookback", str(LOOKBACK),
        "--percentile", "33.0",
    ])
    assert result.exit_code == 0, result.output
    assert out_path.exists()
