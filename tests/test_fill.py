"""Tests for fuel_signal.fill — gap detection and daily_prices rebuild."""

import pytest

from fuel_signal.db import (
    create_schema,
    get_daily_prices,
    insert_prices,
    open_db,
    sydney_average_series,
    upsert_stations,
)
from fuel_signal.fill import fill_all, find_daily_gaps

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _station(code: int = 1001) -> dict:
    return {
        "station_code": code,
        "name": f"Station {code}",
        "address": f"{code} Main Street, Springwood",
        "suburb": "Springwood",
        "postcode": "2777",
        "brand": "Shell",
    }


def _price(station_code: int, date: str, cents: float, fuel_code: str = "E10") -> dict:
    return {"station_code": station_code, "fuel_code": fuel_code, "price_date": date, "price_cents": cents}


@pytest.fixture
def conn(tmp_path):
    c = open_db(tmp_path / "test.db")
    create_schema(c)
    yield c
    c.close()


# ---------------------------------------------------------------------------
# find_daily_gaps — unit tests (pure function)
# ---------------------------------------------------------------------------

class TestFindDailyGaps:
    def test_no_gap_consecutive_days(self):
        rows = [("2024-01-01", 150.0), ("2024-01-02", 152.0)]
        assert find_daily_gaps(rows) == []

    def test_single_observation_no_end_date(self):
        rows = [("2024-01-01", 150.0)]
        assert find_daily_gaps(rows) == []

    def test_single_gap(self):
        rows = [("2024-01-01", 150.0), ("2024-01-03", 155.0)]
        gaps = find_daily_gaps(rows)
        assert gaps == [("2024-01-02", 150.0)]

    def test_multi_day_gap(self):
        rows = [("2024-01-01", 150.0), ("2024-01-05", 160.0)]
        gaps = find_daily_gaps(rows)
        assert gaps == [
            ("2024-01-02", 150.0),
            ("2024-01-03", 150.0),
            ("2024-01-04", 150.0),
        ]

    def test_long_gap_row_count(self):
        rows = [("2024-01-01", 150.0), ("2024-02-01", 160.0)]
        gaps = find_daily_gaps(rows)
        assert len(gaps) == 30  # Jan has 31 days; gap is Jan 2–31

    def test_trailing_fill_to_end_date(self):
        rows = [("2024-01-01", 150.0)]
        gaps = find_daily_gaps(rows, end_date="2024-01-04")
        assert gaps == [
            ("2024-01-02", 150.0),
            ("2024-01-03", 150.0),
            ("2024-01-04", 150.0),
        ]

    def test_trailing_fill_uses_last_known_price(self):
        rows = [("2024-01-01", 150.0), ("2024-01-03", 155.0)]
        gaps = find_daily_gaps(rows, end_date="2024-01-05")
        # Gap between observations: Jan 2 @ 150
        # Trailing fill: Jan 4, Jan 5 @ 155
        assert ("2024-01-02", 150.0) in gaps
        assert ("2024-01-04", 155.0) in gaps
        assert ("2024-01-05", 155.0) in gaps

    def test_end_date_before_last_observation_no_trailing(self):
        rows = [("2024-01-01", 150.0), ("2024-01-03", 155.0)]
        gaps = find_daily_gaps(rows, end_date="2024-01-02")
        # end_date is before last observation — no trailing fill, only mid gap
        assert gaps == [("2024-01-02", 150.0)]

    def test_same_day_duplicate_last_price_wins(self):
        # Duplicate date — last price should be used for subsequent fill
        rows = [("2024-01-01", 150.0), ("2024-01-01", 160.0), ("2024-01-03", 170.0)]
        gaps = find_daily_gaps(rows)
        # Fill Jan 2 with 160 (last price seen for Jan 1)
        assert gaps == [("2024-01-02", 160.0)]

    def test_empty_input(self):
        assert find_daily_gaps([]) == []

    def test_gap_carries_correct_price_forward(self):
        rows = [("2024-01-01", 100.0), ("2024-01-04", 200.0), ("2024-01-06", 300.0)]
        gaps = find_daily_gaps(rows)
        date_price = dict(gaps)
        assert date_price["2024-01-02"] == 100.0
        assert date_price["2024-01-03"] == 100.0
        assert date_price["2024-01-05"] == 200.0


# ---------------------------------------------------------------------------
# fill_all — integration tests
# ---------------------------------------------------------------------------

class TestFillAll:
    def test_single_station_gap_filled(self, conn):
        upsert_stations(conn, [_station(1001)])
        insert_prices(conn, [
            _price(1001, "2024-01-01", 150.0),
            _price(1001, "2024-01-04", 155.0),
        ])
        fill_all(conn, end_date="2024-01-04")
        rows = get_daily_prices(conn, 1001)
        dates = [r[0] for r in rows]
        assert dates == ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
        # Gap days carry forward Jan 1 price
        price_map = dict(rows)
        assert price_map["2024-01-02"] == 150.0
        assert price_map["2024-01-03"] == 150.0

    def test_two_stations_independent_fill(self, conn):
        upsert_stations(conn, [_station(1001), _station(1002)])
        insert_prices(conn, [
            _price(1001, "2024-01-01", 150.0),
            _price(1001, "2024-01-03", 155.0),
            _price(1002, "2024-01-02", 160.0),
            _price(1002, "2024-01-04", 165.0),
        ])
        fill_all(conn, end_date="2024-01-04")

        s1 = conn.execute(
            "SELECT COUNT(*) FROM daily_prices WHERE station_code=1001"
        ).fetchone()[0]
        s2 = conn.execute(
            "SELECT COUNT(*) FROM daily_prices WHERE station_code=1002"
        ).fetchone()[0]
        assert s1 == 4   # Jan 1–4 (Jan 2 gap-filled, Jan 4 trail-filled to end_date)
        assert s2 == 3   # Jan 2–4

    def test_trailing_fill_to_end_date(self, conn):
        upsert_stations(conn, [_station(1001)])
        insert_prices(conn, [_price(1001, "2024-01-01", 150.0)])
        fill_all(conn, end_date="2024-01-05")
        count = conn.execute("SELECT COUNT(*) FROM daily_prices WHERE station_code=1001").fetchone()[0]
        assert count == 5  # Jan 1–5

    def test_full_rebuild_on_second_call(self, conn):
        upsert_stations(conn, [_station(1001)])
        insert_prices(conn, [_price(1001, "2024-01-01", 150.0)])
        fill_all(conn, end_date="2024-01-03")
        fill_all(conn, end_date="2024-01-05")  # second call should replace, not accumulate
        count = conn.execute("SELECT COUNT(*) FROM daily_prices WHERE station_code=1001").fetchone()[0]
        assert count == 5

    def test_returns_total_row_count(self, conn):
        upsert_stations(conn, [_station(1001)])
        insert_prices(conn, [
            _price(1001, "2024-01-01", 150.0),
            _price(1001, "2024-01-03", 155.0),
        ])
        total = fill_all(conn, end_date="2024-01-03")
        assert total == 3  # 2 observed + 1 gap


# ---------------------------------------------------------------------------
# sydney_average_series
# ---------------------------------------------------------------------------

class TestSydneyAverageSeries:
    def test_single_station(self, conn):
        upsert_stations(conn, [_station(1001)])
        insert_prices(conn, [
            _price(1001, "2024-01-01", 150.0),
            _price(1001, "2024-01-02", 160.0),
        ])
        fill_all(conn, end_date="2024-01-02")
        series = sydney_average_series(conn)
        assert series == [("2024-01-01", 150.0), ("2024-01-02", 160.0)]

    def test_two_stations_averaged(self, conn):
        upsert_stations(conn, [_station(1001), _station(1002)])
        insert_prices(conn, [
            _price(1001, "2024-01-01", 140.0),
            _price(1002, "2024-01-01", 160.0),
        ])
        fill_all(conn, end_date="2024-01-01")
        series = sydney_average_series(conn)
        assert len(series) == 1
        assert series[0][0] == "2024-01-01"
        assert series[0][1] == pytest.approx(150.0)

    def test_gap_filled_price_included_in_average(self, conn):
        upsert_stations(conn, [_station(1001), _station(1002)])
        insert_prices(conn, [
            _price(1001, "2024-01-01", 100.0),
            _price(1001, "2024-01-03", 120.0),
            _price(1002, "2024-01-01", 200.0),
            _price(1002, "2024-01-03", 200.0),
        ])
        fill_all(conn, end_date="2024-01-03")
        series = dict(sydney_average_series(conn))
        # Jan 2: station 1001 filled at 100.0, station 1002 filled at 200.0 → avg 150.0
        assert series["2024-01-02"] == pytest.approx(150.0)

    def test_empty_daily_prices_returns_empty(self, conn):
        assert sydney_average_series(conn) == []
