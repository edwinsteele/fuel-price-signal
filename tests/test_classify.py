"""Tests for fuel_signal.classify — station classifier.

Synthetic LGA setup
-------------------
All tests use a single LGA ("TestLGA") with 3–6 stations.
Dates are today-relative so the tests never go stale.

Classification rules:
  median_premium > +100 dc (>+10c) → Sticky
  median_premium < -100 dc (<-10c) → Discount
  else                             → Competitive
"""
from __future__ import annotations

import datetime

import pytest

from fuel_signal.classify import (
    CLASS_COMPETITIVE,
    CLASS_DISCOUNT,
    CLASS_STICKY,
    CLASSIFICATION_WINDOW_DAYS,
    DISCOUNT_THRESHOLD_DC,
    STICKY_THRESHOLD_DC,
    _compute_classes,
    classify_snapshot,
)
from fuel_signal.db import (
    create_schema,
    open_db,
    upsert_daily_prices,
    upsert_stations,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LGA = "TestLGA"
_POSTCODE = "2000"

# Station codes
_SC_COMP1 = 1001
_SC_COMP2 = 1002
_SC_COMP3 = 1003
_SC_STICKY = 2001
_SC_DISCOUNT = 3001


def _today() -> datetime.date:
    return datetime.date.today()


def _iso(d: datetime.date) -> str:
    return d.isoformat()


def _snapshot_date() -> str:
    """Snapshot date = today. Window = [today-45, yesterday]."""
    return _iso(_today())


@pytest.fixture
def conn(tmp_path):
    c = open_db(tmp_path / "test.db")
    create_schema(c)
    yield c
    c.close()


def _add_station(conn, code: int, postcode: str = _POSTCODE, brand: str = "TestBrand") -> None:
    upsert_stations(conn, [{
        "station_code": code,
        "name": f"Station {code}",
        "address": f"{code} Main Street, Sydney",
        "suburb": "Sydney",
        "postcode": postcode,
        "brand": brand,
    }])


def _add_prices(
    conn,
    station_code: int,
    base_price_dc: int,
    start_date: datetime.date,
    n_days: int,
) -> None:
    """Insert n_days of constant prices (in decicents) starting from start_date."""
    rows = [
        (station_code, "E10", (start_date + datetime.timedelta(days=i)).isoformat(), base_price_dc / 10)
        for i in range(n_days)
    ]
    upsert_daily_prices(conn, rows)
    conn.commit()


# ---------------------------------------------------------------------------
# _compute_classes unit tests
# ---------------------------------------------------------------------------

def test_compute_classes_all_competitive():
    """Three stations near the cluster median → all Competitive."""
    data = {
        1: {20240101: 1700, 20240102: 1700},
        2: {20240101: 1700, 20240102: 1700},
        3: {20240101: 1700, 20240102: 1700},
    }
    result = _compute_classes(data, cluster_stations=None)
    assert len(result) == 3
    for sc, (cls, premium) in result.items():
        assert cls == CLASS_COMPETITIVE
        assert premium == 0


def test_compute_classes_sticky():
    """Station above +10c (>100dc) cluster median → Sticky."""
    data = {
        1: {20240101: 1700},
        2: {20240101: 1700},
        3: {20240101: 1700},
        4: {20240101: 1700 + STICKY_THRESHOLD_DC + 1},  # just above threshold
    }
    result = _compute_classes(data, cluster_stations=None)
    assert result[4][0] == CLASS_STICKY
    assert result[4][1] == STICKY_THRESHOLD_DC + 1


def test_compute_classes_discount():
    """Station below -10c (<-100dc) cluster median → Discount."""
    data = {
        1: {20240101: 1700},
        2: {20240101: 1700},
        3: {20240101: 1700},
        4: {20240101: 1700 + DISCOUNT_THRESHOLD_DC - 1},  # just below threshold
    }
    result = _compute_classes(data, cluster_stations=None)
    assert result[4][0] == CLASS_DISCOUNT
    assert result[4][1] == DISCOUNT_THRESHOLD_DC - 1


def test_compute_classes_exactly_at_threshold_is_competitive():
    """Premium exactly at ±100dc → Competitive (band is strict inequality)."""
    data = {
        1: {20240101: 1700},
        2: {20240101: 1700},
        3: {20240101: 1700 + STICKY_THRESHOLD_DC},   # exactly +10c
        4: {20240101: 1700 + DISCOUNT_THRESHOLD_DC},  # exactly -10c
    }
    result = _compute_classes(data, cluster_stations=None)
    assert result[3][0] == CLASS_COMPETITIVE
    assert result[4][0] == CLASS_COMPETITIVE


def test_compute_classes_with_cluster_filter():
    """Cluster reference uses only specified cluster_stations."""
    # cluster = {1, 2} at 1700; station 3 computes premium vs 1700.
    data = {
        1: {20240101: 1700},
        2: {20240101: 1700},
        3: {20240101: 1700 + STICKY_THRESHOLD_DC + 50},  # sticky vs 1700
    }
    result = _compute_classes(data, cluster_stations=frozenset({1, 2}))
    assert result[3][0] == CLASS_STICKY


def test_compute_classes_empty_returns_empty():
    assert _compute_classes({}, cluster_stations=None) == {}


def test_compute_classes_cluster_filter_no_overlap_returns_empty():
    """If cluster_stations don't overlap with data, no cluster → no output."""
    data = {1: {20240101: 1700}}
    result = _compute_classes(data, cluster_stations=frozenset({99}))
    assert result == {}


def test_compute_classes_station_only_on_days_without_cluster():
    """Station with no price on days when the cluster has data → no classification."""
    # cluster only has data on day 1; station only has data on day 2.
    data = {
        1: {20240101: 1700},   # cluster station, day 1
        2: {20240102: 1700},   # station to classify, day 2 (no cluster overlap)
    }
    result = _compute_classes(data, cluster_stations=frozenset({1}))
    # Station 2 has no days with a cluster reference → not in result.
    assert 2 not in result
    assert 1 in result  # station 1 computes vs itself: premium 0 → Competitive


# ---------------------------------------------------------------------------
# classify_snapshot — integration tests
# ---------------------------------------------------------------------------

def test_classify_snapshot_basic(conn):
    """classify_snapshot produces Competitive/Sticky/Discount for known stations."""
    today = _today()
    window_start = today - datetime.timedelta(days=CLASSIFICATION_WINDOW_DAYS)

    _add_station(conn, _SC_COMP1)
    _add_station(conn, _SC_COMP2)
    _add_station(conn, _SC_COMP3)
    _add_station(conn, _SC_STICKY)
    _add_station(conn, _SC_DISCOUNT)

    # Competitive stations at 170.0c (1700 dc)
    for sc in (_SC_COMP1, _SC_COMP2, _SC_COMP3):
        _add_prices(conn, sc, 1700, window_start, CLASSIFICATION_WINDOW_DAYS)

    # Sticky: +12c above cluster (1700 + 120 dc)
    _add_prices(conn, _SC_STICKY, 1700 + 120, window_start, CLASSIFICATION_WINDOW_DAYS)

    # Discount: -12c below cluster (1700 - 120 dc)
    _add_prices(conn, _SC_DISCOUNT, 1700 - 120, window_start, CLASSIFICATION_WINDOW_DAYS)

    n_stations, n_lgas = classify_snapshot(conn, _snapshot_date())
    assert n_stations == 5
    assert n_lgas == 1

    rows = conn.execute(
        "SELECT station_code, class FROM station_class ORDER BY station_code"
    ).fetchall()
    class_by_code = dict(rows)

    assert class_by_code[_SC_COMP1] == CLASS_COMPETITIVE
    assert class_by_code[_SC_COMP2] == CLASS_COMPETITIVE
    assert class_by_code[_SC_COMP3] == CLASS_COMPETITIVE
    assert class_by_code[_SC_STICKY] == CLASS_STICKY
    assert class_by_code[_SC_DISCOUNT] == CLASS_DISCOUNT


def test_classify_snapshot_writes_summary(conn):
    """classify_snapshot writes a classification_summary row per LGA."""
    today = _today()
    window_start = today - datetime.timedelta(days=CLASSIFICATION_WINDOW_DAYS)

    for sc in (_SC_COMP1, _SC_COMP2, _SC_COMP3, _SC_STICKY, _SC_DISCOUNT):
        _add_station(conn, sc)
    for sc in (_SC_COMP1, _SC_COMP2, _SC_COMP3):
        _add_prices(conn, sc, 1700, window_start, CLASSIFICATION_WINDOW_DAYS)
    _add_prices(conn, _SC_STICKY, 1700 + 120, window_start, CLASSIFICATION_WINDOW_DAYS)
    _add_prices(conn, _SC_DISCOUNT, 1700 - 120, window_start, CLASSIFICATION_WINDOW_DAYS)

    classify_snapshot(conn, _snapshot_date())

    row = conn.execute(
        "SELECT n_competitive, n_sticky, n_discount FROM classification_summary"
    ).fetchone()
    assert row is not None
    n_comp, n_sticky, n_disc = row
    assert n_comp == 3
    assert n_sticky == 1
    assert n_disc == 1


def test_classify_snapshot_pit_safety(conn):
    """station_class for snapshot_date D must not use price_date >= D."""
    today = _today()
    snapshot_date = today.isoformat()

    # Stations with prices ONLY on today (i.e. exactly at snapshot_date, not D-1)
    _add_station(conn, _SC_COMP1)
    _add_station(conn, _SC_COMP2)
    _add_station(conn, _SC_STICKY)

    # Only add price on today — should NOT be included in the window [D-45, D-1]
    for sc in (_SC_COMP1, _SC_COMP2, _SC_STICKY):
        upsert_daily_prices(conn, [(sc, "E10", today.isoformat(), 170.0)])
    conn.commit()

    n_stations, _ = classify_snapshot(conn, snapshot_date)
    # No prices in [D-45, D-1], so nothing should be classified
    assert n_stations == 0
    assert conn.execute("SELECT COUNT(*) FROM station_class").fetchone()[0] == 0


def test_classify_snapshot_cold_start_single_observation(conn):
    """Station with only 1 observation in window still gets classified."""
    today = _today()
    yesterday = today - datetime.timedelta(days=1)

    _add_station(conn, _SC_COMP1)
    _add_station(conn, _SC_COMP2)
    _add_station(conn, _SC_COMP3)

    window_start = today - datetime.timedelta(days=CLASSIFICATION_WINDOW_DAYS)

    # COMP1 and COMP2 have full window history
    for sc in (_SC_COMP1, _SC_COMP2):
        _add_prices(conn, sc, 1700, window_start, CLASSIFICATION_WINDOW_DAYS)

    # COMP3 has only 1 observation (yesterday)
    upsert_daily_prices(conn, [(_SC_COMP3, "E10", yesterday.isoformat(), 170.0)])
    conn.commit()

    n_stations, _ = classify_snapshot(conn, today.isoformat())
    assert n_stations == 3  # COMP3 still classified

    rows = conn.execute(
        "SELECT station_code, class FROM station_class ORDER BY station_code"
    ).fetchall()
    assert len(rows) == 3
    code_classes = {sc: cls for sc, cls in rows}
    assert code_classes[_SC_COMP3] == CLASS_COMPETITIVE


def test_classify_snapshot_zero_competitive_writes_summary_no_station_rows(conn):
    """If all stations are Sticky after iter 1, write summary(n_competitive=0) but no station_class rows."""
    today = _today()
    window_start = today - datetime.timedelta(days=CLASSIFICATION_WINDOW_DAYS)

    # Two stations both well above (different enough prices that one will be Sticky)
    # To get zero Competitive: all stations must be Sticky after iter 1.
    # This happens when all stations are at very different prices: each one's premium
    # vs the all-station median is large. With just 2 stations 50c apart, each is
    # 25c from the median → each classified Sticky.
    _add_station(conn, _SC_COMP1)
    _add_station(conn, _SC_STICKY)
    _add_prices(conn, _SC_COMP1, 1500, window_start, CLASSIFICATION_WINDOW_DAYS)  # 150c
    _add_prices(conn, _SC_STICKY, 2100, window_start, CLASSIFICATION_WINDOW_DAYS)  # 210c
    # Cluster median = 1800 dc; premium for 1500 = -300 (Discount); premium for 2100 = +300 (Sticky)
    # After iter 1: COMP1 → Discount, STICKY → Sticky. Zero Competitive → no iter 2.

    n_stations, n_lgas = classify_snapshot(conn, today.isoformat())
    assert n_stations == 0  # no station_class rows
    assert n_lgas == 1

    sc_count = conn.execute("SELECT COUNT(*) FROM station_class").fetchone()[0]
    assert sc_count == 0

    summary_row = conn.execute(
        "SELECT n_competitive, n_sticky, n_discount FROM classification_summary"
    ).fetchone()
    assert summary_row is not None
    assert summary_row[0] == 0  # n_competitive = 0


def test_classify_snapshot_idempotent(conn):
    """Calling classify_snapshot twice for the same date replaces rather than duplicates rows."""
    today = _today()
    window_start = today - datetime.timedelta(days=CLASSIFICATION_WINDOW_DAYS)

    for sc in (_SC_COMP1, _SC_COMP2, _SC_COMP3):
        _add_station(conn, sc)
        _add_prices(conn, sc, 1700, window_start, CLASSIFICATION_WINDOW_DAYS)

    classify_snapshot(conn, today.isoformat())
    classify_snapshot(conn, today.isoformat())

    # Should still be exactly 3 rows (not 6)
    count = conn.execute("SELECT COUNT(*) FROM station_class").fetchone()[0]
    assert count == 3


def test_classify_snapshot_no_data_returns_zero(conn):
    """classify_snapshot on a date with no daily_prices returns (0, 0)."""
    n_stations, n_lgas = classify_snapshot(conn, _snapshot_date())
    assert n_stations == 0
    assert n_lgas == 0


def test_classify_snapshot_iter2_reclassification(conn):
    """Iter 2 uses only Competitive cluster; a borderline station may flip."""
    today = _today()
    window_start = today - datetime.timedelta(days=CLASSIFICATION_WINDOW_DAYS)

    # Three Competitive stations at 170c + one that is 5c above (only just over in iter 1
    # but near the all-station median in iter 2 because the Sticky station pulls the iter-1
    # median up).
    # Station layout:
    #   COMP1, COMP2, COMP3: 170c (1700 dc)
    #   STICKY:              185c (1850 dc)  → +150dc vs all-station median (1762 dc)
    #   BORDERLINE:          172c (1720 dc)  → +20dc vs all-station median → iter-1 Competitive
    #                                        → +20dc vs iter-2 Competitive median (1700) → Competitive
    SC_BORDERLINE = 9001
    for sc in (_SC_COMP1, _SC_COMP2, _SC_COMP3):
        _add_station(conn, sc)
        _add_prices(conn, sc, 1700, window_start, CLASSIFICATION_WINDOW_DAYS)
    _add_station(conn, _SC_STICKY)
    _add_prices(conn, _SC_STICKY, 1850, window_start, CLASSIFICATION_WINDOW_DAYS)
    _add_station(conn, SC_BORDERLINE)
    _add_prices(conn, SC_BORDERLINE, 1720, window_start, CLASSIFICATION_WINDOW_DAYS)

    classify_snapshot(conn, today.isoformat())

    class_map = {sc: cls for sc, cls in conn.execute(
        "SELECT station_code, class FROM station_class"
    ).fetchall()}

    assert class_map[_SC_STICKY] == CLASS_STICKY
    # BORDERLINE: premium vs iter-2 ref (1700 dc) = +20dc → within ±100dc → Competitive
    assert class_map[SC_BORDERLINE] == CLASS_COMPETITIVE
