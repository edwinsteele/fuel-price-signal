"""Tests for fuel_signal.fill — gap detection and daily_prices rebuild."""

import pytest

from fuel_signal.db import (
    average_price_series,
    create_schema,
    get_daily_prices,
    insert_prices,
    open_db,
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
        # Explicit large max_gap_days to test fill-count arithmetic independent of threshold.
        rows = [("2024-01-01", 150.0), ("2024-02-01", 160.0)]
        gaps = find_daily_gaps(rows, max_gap_days=31)
        assert len(gaps) == 30  # Jan has 31 days; gap is Jan 2–31

    def test_gap_at_threshold_is_filled(self):
        # Gap of exactly max_gap_days → filled.
        rows = [("2024-01-01", 150.0), ("2024-01-29", 160.0)]  # 28 days apart
        gaps = find_daily_gaps(rows, max_gap_days=28)
        assert len(gaps) == 27  # Jan 2–28

    def test_gap_exceeding_threshold_not_filled(self):
        # Gap of max_gap_days + 1 → not filled at all.
        rows = [("2024-01-01", 150.0), ("2024-01-30", 160.0)]  # 29 days apart
        gaps = find_daily_gaps(rows, max_gap_days=28)
        assert gaps == []

    def test_gap_below_threshold_is_filled(self):
        rows = [("2024-01-01", 150.0), ("2024-01-05", 160.0)]  # 4 days apart
        gaps = find_daily_gaps(rows, max_gap_days=28)
        assert len(gaps) == 3  # Jan 2–4

    def test_trailing_fill_beyond_threshold_not_filled(self):
        rows = [("2024-01-01", 150.0)]
        # 90 days beyond last observation → no trailing fill
        gaps = find_daily_gaps(rows, end_date="2024-04-01", max_gap_days=28)
        assert gaps == []

    def test_negative_max_gap_days_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            find_daily_gaps([("2024-01-01", 150.0)], max_gap_days=-1)

    def test_mixed_gaps_only_short_ones_filled(self):
        # Long gap (>28) skipped; short gap (<28) filled; subsequent obs still processed.
        rows = [
            ("2024-01-01", 100.0),
            ("2024-04-01", 110.0),  # 91-day gap → skipped
            ("2024-04-03", 120.0),  # 2-day gap → filled
        ]
        gaps = find_daily_gaps(rows, max_gap_days=28)
        dates = [d for d, _ in gaps]
        assert "2024-04-02" in dates
        # No fill rows from Jan 1 to Apr 1
        assert not any(d < "2024-04-01" for d in dates)

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

    def test_large_gap_not_filled(self, conn):
        # Station 414 scenario: 91-day gap (station closed) → no fill rows in the gap.
        upsert_stations(conn, [_station(1001)])
        insert_prices(conn, [
            _price(1001, "2024-01-01", 150.0),
            _price(1001, "2024-04-01", 160.0),  # 91 days later
        ])
        fill_all(conn, end_date="2024-04-01", max_gap_days=28)
        rows = get_daily_prices(conn, 1001)
        assert len(rows) == 2
        assert rows[0][0] == "2024-01-01"
        assert rows[1][0] == "2024-04-01"

    def test_negative_max_gap_days_raises(self, conn):
        upsert_stations(conn, [_station(1001)])
        insert_prices(conn, [_price(1001, "2024-01-01", 150.0)])
        with pytest.raises(ValueError, match="non-negative"):
            fill_all(conn, end_date="2024-01-05", max_gap_days=-1)

    def test_trailing_gap_beyond_threshold_not_filled(self, conn):
        upsert_stations(conn, [_station(1001)])
        insert_prices(conn, [_price(1001, "2024-01-01", 150.0)])
        fill_all(conn, end_date="2024-04-01", max_gap_days=28)  # 91 days to end
        rows = get_daily_prices(conn, 1001)
        assert len(rows) == 1  # only the observed date


# ---------------------------------------------------------------------------
# average_price_series
# ---------------------------------------------------------------------------

class TestSydneyAverageSeries:
    def test_single_station(self, conn):
        upsert_stations(conn, [_station(1001)])
        insert_prices(conn, [
            _price(1001, "2024-01-01", 150.0),
            _price(1001, "2024-01-02", 160.0),
        ])
        fill_all(conn, end_date="2024-01-02")
        series = average_price_series(conn)
        assert series == [("2024-01-01", 150.0), ("2024-01-02", 160.0)]

    def test_two_stations_averaged(self, conn):
        upsert_stations(conn, [_station(1001), _station(1002)])
        insert_prices(conn, [
            _price(1001, "2024-01-01", 140.0),
            _price(1002, "2024-01-01", 160.0),
        ])
        fill_all(conn, end_date="2024-01-01")
        series = average_price_series(conn)
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
        series = dict(average_price_series(conn))
        # Jan 2: station 1001 filled at 100.0, station 1002 filled at 200.0 → avg 150.0
        assert series["2024-01-02"] == pytest.approx(150.0)

    def test_empty_daily_prices_returns_empty(self, conn):
        assert average_price_series(conn) == []
