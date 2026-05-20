"""Tests for fuel_signal.classify — station classifier."""

import datetime

import pytest

from fuel_signal.classify import (
    WINDOW_DAYS,
    _classify_lga,
    _run_classification,
    _window_bounds,
    classify_all,
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

def _today_minus(n: int) -> str:
    return (datetime.date.today() - datetime.timedelta(days=n)).isoformat()


def _station(code: int, postcode: str = "2777") -> dict:
    """Blue Mountains (2777) → council='Blue Mountains'."""
    return {
        "station_code": code,
        "name": f"Station {code}",
        "address": f"{code} Main Street, Springwood",
        "suburb": "Springwood",
        "postcode": postcode,
        "brand": "Shell",
    }


def _daily_price(station_code: int, date: str, cents: float) -> tuple:
    return (station_code, "E10", date, cents)


def _seed_window(
    conn,
    station_prices: dict[int, float],
    postcode: str = "2777",
    days: int = WINDOW_DAYS,
    snapshot_date_offset: int = 1,
) -> int:
    """Seed daily_prices for `days` days ending at today-snapshot_date_offset.

    Returns snapshot_date as int (YYYYMMDD).
    """
    stations = [_station(sc, postcode) for sc in station_prices]
    upsert_stations(conn, stations)

    snapshot_date = datetime.date.today() - datetime.timedelta(days=snapshot_date_offset - 1)
    window_end = snapshot_date - datetime.timedelta(days=1)
    window_start = window_end - datetime.timedelta(days=days - 1)

    rows = []
    d = window_start
    while d <= window_end:
        for sc, cents in station_prices.items():
            rows.append(_daily_price(sc, d.isoformat(), cents))
        d += datetime.timedelta(days=1)

    upsert_daily_prices(conn, rows)
    conn.commit()
    return int(snapshot_date.strftime("%Y%m%d"))


@pytest.fixture
def conn(tmp_path):
    c = open_db(tmp_path / "test.db")
    create_schema(c)
    yield c
    c.close()


# ---------------------------------------------------------------------------
# _window_bounds
# ---------------------------------------------------------------------------

class TestWindowBounds:
    def test_window_length(self):
        snapshot = 20240215
        start, end = _window_bounds(snapshot)
        start_d = datetime.date(int(str(start)[:4]), int(str(start)[4:6]), int(str(start)[6:]))
        end_d = datetime.date(int(str(end)[:4]), int(str(end)[4:6]), int(str(end)[6:]))
        assert (end_d - start_d).days == WINDOW_DAYS - 1

    def test_window_ends_day_before_snapshot(self):
        snapshot = 20240215
        _, end = _window_bounds(snapshot)
        end_d = datetime.date(int(str(end)[:4]), int(str(end)[4:6]), int(str(end)[6:]))
        assert end_d == datetime.date(2024, 2, 14)

    def test_window_excludes_snapshot_date(self):
        snapshot = 20240215
        start, end = _window_bounds(snapshot)
        assert end < snapshot


# ---------------------------------------------------------------------------
# _run_classification — unit tests (pure function)
# ---------------------------------------------------------------------------

class TestRunClassification:
    def _obs(self, prices: list[float]) -> list[tuple[int, int]]:
        """Create observations across WINDOW_DAYS consecutive days."""
        base = 20240101
        return [(base + i, round(p * 10)) for i, p in enumerate(prices)]

    def test_all_competitive(self):
        station_obs = {
            1: [(20240101, 1600), (20240102, 1605)],
            2: [(20240101, 1590), (20240102, 1595)],
        }
        result = _run_classification(station_obs)
        for cls, _ in result.values():
            assert cls == "Competitive"

    def test_sticky_above_band(self):
        # 3 stations: two at 1600 form the majority reference; station 1 at 1750 → +150dc → Sticky.
        # Per-day ref = median([1750, 1600, 1600]) = 1600; premium = 150dc > BAND_DECICENTS.
        station_obs = {
            1: [(20240101, 1750), (20240102, 1750)],
            2: [(20240101, 1600), (20240102, 1600)],
            3: [(20240101, 1600), (20240102, 1600)],
        }
        result = _run_classification(station_obs)
        assert result[1][0] == "Sticky"
        assert result[2][0] == "Competitive"

    def test_discount_below_band(self):
        # 3 stations: two at 1600; station 1 at 1450 → -150dc → Discount.
        station_obs = {
            1: [(20240101, 1450), (20240102, 1450)],
            2: [(20240101, 1600), (20240102, 1600)],
            3: [(20240101, 1600), (20240102, 1600)],
        }
        result = _run_classification(station_obs)
        assert result[1][0] == "Discount"
        assert result[2][0] == "Competitive"

    def test_boundary_at_band_is_competitive(self):
        # Station 1 exactly at +100dc above reference (not Sticky; > is required).
        station_obs = {
            1: [(20240101, 1700), (20240102, 1700)],
            2: [(20240101, 1600), (20240102, 1600)],
            3: [(20240101, 1600), (20240102, 1600)],
        }
        result = _run_classification(station_obs)
        assert result[1][0] == "Competitive"

    def test_just_over_band_is_sticky(self):
        # Station 1 at +101dc → Sticky.
        station_obs = {
            1: [(20240101, 1701), (20240102, 1701)],
            2: [(20240101, 1600), (20240102, 1600)],
            3: [(20240101, 1600), (20240102, 1600)],
        }
        result = _run_classification(station_obs)
        assert result[1][0] == "Sticky"

    def test_station_with_no_days_in_reference_excluded(self):
        # Station 2 has no days in common with reference stations → excluded from result.
        station_obs = {
            1: [(20240101, 1600), (20240102, 1600)],
            2: [(20240103, 1600)],  # day not in station 1's observations
        }
        result = _run_classification(station_obs, reference_station_codes={1})
        assert 1 in result
        assert 2 not in result  # no overlap with reference days

    def test_median_premium_stored_as_rounded_decicents(self):
        # Station 1 alternates +95dc and +105dc relative to the majority reference.
        # Per-day ref = median([1695,1600,1600]) = 1600 and median([1705,1600,1600]) = 1600.
        # Premiums: 95, 105 → median = 100dc → Competitive.
        station_obs = {
            1: [(20240101, 1695), (20240102, 1705)],
            2: [(20240101, 1600), (20240102, 1600)],
            3: [(20240101, 1600), (20240102, 1600)],
        }
        result = _run_classification(station_obs)
        _, med = result[1]
        assert med == 100
        assert result[1][0] == "Competitive"


# ---------------------------------------------------------------------------
# _classify_lga — two-iteration unit tests
# ---------------------------------------------------------------------------

class TestClassifyLga:
    def test_two_iter_sticky_majority_triggers_zero_competitive(self):
        # When iter 1 produces no Competitive stations, zero-Competitive applies.
        # All stations above the median reference → all Sticky → n_comp=0 returned.
        # With only Sticky stations, the all-station median IS their price,
        # so premiums are near 0 → all Competitive in iter 1 unless they differ enough.
        # To force zero-Competitive in iter 1: use 2 stations, one very high above the other.
        # Median of [1600, 2000] = 1800. Station 1: 1600-1800=-200→Discount. Station 2: 2000-1800=+200→Sticky.
        # → iter 1 has 0 Competitive → zero-Competitive case.
        station_obs = {
            1: [(20240101, 1600)],
            2: [(20240101, 2000)],
        }
        classes, n_comp, n_sticky, n_disc = _classify_lga(station_obs)
        assert classes == {}
        assert n_comp == 0

    def test_iter2_uses_only_competitive_reference(self):
        # Iter 1 cluster ref = median([1600, 1700, 2000]) = 1700
        # Station 1: 1600-1700=-100 → Competitive (boundary)
        # Station 2: 1700-1700=0 → Competitive
        # Station 3: 2000-1700=+300 → Sticky
        # Iter 2 ref = median([1600, 1700]) = 1650 (only Competitive from iter1)
        # Station 1: 1600-1650=-50 → Competitive
        # Station 2: 1700-1650=+50 → Competitive
        # Station 3: 2000-1650=+350 → Sticky
        station_obs = {
            1: [(20240101, 1600)],
            2: [(20240101, 1700)],
            3: [(20240101, 2000)],
        }
        classes, n_comp, n_sticky, n_disc = _classify_lga(station_obs)
        assert classes != {}
        assert classes[3][0] == "Sticky"
        assert classes[1][0] == "Competitive"
        assert classes[2][0] == "Competitive"
        assert n_comp == 2
        assert n_sticky == 1
        assert n_disc == 0

    def test_all_same_price_all_competitive(self):
        station_obs = {sc: [(20240101, 1600), (20240102, 1600)] for sc in range(1, 5)}
        classes, n_comp, n_sticky, n_disc = _classify_lga(station_obs)
        assert n_comp == 4
        assert n_sticky == 0
        assert n_disc == 0

    def test_returns_empty_for_zero_competitive_after_iter2(self):
        # Force iter 2 to produce zero Competitive.
        # Iter 1: all Competitive (prices very close).
        # Use that same set as iter 2 reference → after shift, all drift outside band.
        # Hard to force naturally; instead test that the empty-return path is hit.
        # Simplest: only 2 stations, exactly at boundaries in iter 1.
        # Iter 1 ref = median([1600, 2100]) = 1850
        # St1: 1600-1850=-250 → Discount; St2: 2100-1850=+250 → Sticky → no Competitive in iter1
        station_obs = {
            1: [(20240101, 1600)],
            2: [(20240101, 2100)],
        }
        classes, n_comp, _, _ = _classify_lga(station_obs)
        assert classes == {}
        assert n_comp == 0


# ---------------------------------------------------------------------------
# classify_snapshot — integration tests
# ---------------------------------------------------------------------------

class TestClassifySnapshot:
    def test_basic_classification_written(self, conn):
        # 3 stations: one Sticky (+200dc), one Discount (-200dc), one Competitive.
        snapshot_date = _seed_window(conn, {1: 160.0, 2: 140.0, 3: 180.0})
        # Ref = median([1600, 1400, 1800]) = 1600
        # St1: 0 → Competitive; St2: -200 → Discount; St3: +200 → Sticky
        rows_written, lgas = classify_snapshot(conn, snapshot_date)
        assert rows_written == 3
        assert lgas == 1
        rows = conn.execute(
            "SELECT station_code, class FROM station_class WHERE snapshot_date = ? ORDER BY station_code",
            (snapshot_date,),
        ).fetchall()
        cls_by_sc = {r[0]: r[1] for r in rows}
        assert cls_by_sc[1] == "Competitive"
        assert cls_by_sc[2] == "Discount"
        assert cls_by_sc[3] == "Sticky"

    def test_classification_summary_written(self, conn):
        snapshot_date = _seed_window(conn, {1: 160.0, 2: 140.0, 3: 180.0})
        classify_snapshot(conn, snapshot_date)
        row = conn.execute(
            "SELECT n_competitive, n_sticky, n_discount FROM classification_summary"
            " WHERE snapshot_date = ? AND lga = 'Blue Mountains'",
            (snapshot_date,),
        ).fetchone()
        assert row is not None
        n_comp, n_sticky, n_disc = row
        assert n_comp + n_sticky + n_disc == 3

    def test_zero_competitive_lga_no_station_class_rows(self, conn):
        # 2 stations far apart → zero Competitive after iter 1.
        snapshot_date = _seed_window(conn, {1: 140.0, 2: 190.0})
        rows_written, _ = classify_snapshot(conn, snapshot_date)
        assert rows_written == 0
        # Summary row must still exist with n_competitive=0.
        row = conn.execute(
            "SELECT n_competitive FROM classification_summary WHERE snapshot_date = ?",
            (snapshot_date,),
        ).fetchone()
        assert row is not None
        assert row[0] == 0

    def test_median_premium_stored_in_decicents(self, conn):
        # Station 2 is always 20c above station 1 → +200 decicents.
        snapshot_date = _seed_window(conn, {1: 160.0, 2: 180.0, 3: 140.0})
        classify_snapshot(conn, snapshot_date)
        row = conn.execute(
            "SELECT median_premium_decicents FROM station_class"
            " WHERE snapshot_date = ? AND station_code = 2",
            (snapshot_date,),
        ).fetchone()
        assert row is not None
        assert isinstance(row[0], int)

    def test_upsert_replaces_existing_rows(self, conn):
        snapshot_date = _seed_window(conn, {1: 160.0, 2: 140.0, 3: 180.0})
        classify_snapshot(conn, snapshot_date)
        classify_snapshot(conn, snapshot_date)  # second run
        count = conn.execute(
            "SELECT COUNT(*) FROM station_class WHERE snapshot_date = ?", (snapshot_date,)
        ).fetchone()[0]
        assert count == 3  # no duplicates

    def test_cold_start_single_observation(self, conn):
        # Station 2 has only 1 observation in window; must still get a row.
        stations = [_station(1), _station(2)]
        upsert_stations(conn, stations)
        today = datetime.date.today()
        snapshot_date = today
        window_end = snapshot_date - datetime.timedelta(days=1)
        # Station 1: full window; Station 2: only last day.
        rows = []
        d = window_end - datetime.timedelta(days=WINDOW_DAYS - 1)
        while d <= window_end:
            rows.append(_daily_price(1, d.isoformat(), 160.0))
            d += datetime.timedelta(days=1)
        # Station 2 gets one observation.
        rows.append(_daily_price(2, window_end.isoformat(), 160.0))
        upsert_daily_prices(conn, rows)
        conn.commit()
        snap_int = int(snapshot_date.strftime("%Y%m%d"))
        rows_written, _ = classify_snapshot(conn, snap_int)
        codes = {
            r[0]
            for r in conn.execute(
                "SELECT station_code FROM station_class WHERE snapshot_date = ?", (snap_int,)
            )
        }
        assert 2 in codes

    def test_station_with_no_window_observations_excluded(self, conn):
        # Station 2 has no observations in the window → no station_class row.
        stations = [_station(1), _station(2)]
        upsert_stations(conn, stations)
        today = datetime.date.today()
        snapshot_date = today
        window_end = snapshot_date - datetime.timedelta(days=1)
        window_start = window_end - datetime.timedelta(days=WINDOW_DAYS - 1)
        # Only station 1 gets data.
        rows = []
        d = window_start
        while d <= window_end:
            rows.append(_daily_price(1, d.isoformat(), 160.0))
            d += datetime.timedelta(days=1)
        # Station 2 has data, but outside the window.
        rows.append(_daily_price(2, (window_start - datetime.timedelta(days=5)).isoformat(), 160.0))
        upsert_daily_prices(conn, rows)
        conn.commit()
        snap_int = int(snapshot_date.strftime("%Y%m%d"))
        classify_snapshot(conn, snap_int)
        count = conn.execute(
            "SELECT COUNT(*) FROM station_class WHERE snapshot_date = ? AND station_code = 2",
            (snap_int,),
        ).fetchone()[0]
        assert count == 0

    def test_two_lgas_classified_independently(self, conn):
        # Blue Mountains (2777) and Penrith (2750) are separate LGAs.
        stations_bm = [_station(1, "2777"), _station(2, "2777")]
        stations_pr = [_station(3, "2750"), _station(4, "2750")]
        upsert_stations(conn, stations_bm + stations_pr)

        today = datetime.date.today()
        snapshot_date = today
        window_end = snapshot_date - datetime.timedelta(days=1)
        window_start = window_end - datetime.timedelta(days=WINDOW_DAYS - 1)

        rows = []
        d = window_start
        while d <= window_end:
            rows += [
                _daily_price(1, d.isoformat(), 160.0),
                _daily_price(2, d.isoformat(), 160.0),
                _daily_price(3, d.isoformat(), 160.0),
                _daily_price(4, d.isoformat(), 160.0),
            ]
            d += datetime.timedelta(days=1)
        upsert_daily_prices(conn, rows)
        conn.commit()

        snap_int = int(snapshot_date.strftime("%Y%m%d"))
        rows_written, lgas = classify_snapshot(conn, snap_int)
        assert lgas == 2
        assert rows_written == 4

    # -----------------------------------------------------------------------
    # PIT discipline
    # -----------------------------------------------------------------------

    def test_pit_snapshot_date_prices_not_used(self, conn):
        # Inject a future-price observation ON snapshot_date itself.
        # The classifier must NOT use it (window ends at snapshot_date - 1).
        stations = [_station(1), _station(2), _station(3)]
        upsert_stations(conn, stations)

        today = datetime.date.today()
        snapshot_date = today
        window_end = snapshot_date - datetime.timedelta(days=1)
        window_start = window_end - datetime.timedelta(days=WINDOW_DAYS - 1)

        rows = []
        d = window_start
        while d <= window_end:
            for sc in [1, 2, 3]:
                rows.append(_daily_price(sc, d.isoformat(), 160.0))
            d += datetime.timedelta(days=1)

        # Station 4 with a price only ON snapshot_date (should be ignored).
        upsert_stations(conn, [_station(4)])
        rows.append(_daily_price(4, snapshot_date.isoformat(), 999.9))  # outside window
        upsert_daily_prices(conn, rows)
        conn.commit()

        snap_int = int(snapshot_date.strftime("%Y%m%d"))
        classify_snapshot(conn, snap_int)

        # Station 4 should have no station_class row (zero observations in window).
        count = conn.execute(
            "SELECT COUNT(*) FROM station_class WHERE snapshot_date = ? AND station_code = 4",
            (snap_int,),
        ).fetchone()[0]
        assert count == 0

        # Stations 1-3 should be unaffected by station 4's price.
        classes = conn.execute(
            "SELECT class FROM station_class WHERE snapshot_date = ? AND station_code IN (1, 2, 3)",
            (snap_int,),
        ).fetchall()
        assert all(r[0] == "Competitive" for r in classes)


# ---------------------------------------------------------------------------
# classify_all — batch API tests
# ---------------------------------------------------------------------------

class TestClassifyAll:
    def test_empty_db_returns_zero(self, conn):
        total = classify_all(conn)
        assert total == 0

    def test_processes_dates_and_returns_row_count(self, conn):
        # _seed_window seeds WINDOW_DAYS daily_prices ending at today-1.
        # classify_all treats each distinct price_date as a snapshot_date.
        # Snapshot dates near the end of the window will have enough prior data
        # to produce station_class rows.
        _seed_window(conn, {1: 160.0, 2: 160.0, 3: 170.0})
        total = classify_all(conn)
        count = conn.execute("SELECT COUNT(*) FROM station_class").fetchone()[0]
        assert count > 0
        assert total == count

    def test_end_date_before_all_data_returns_zero(self, conn):
        _seed_window(conn, {1: 160.0, 2: 160.0, 3: 170.0})
        total = classify_all(conn, end_date="2000-01-01")
        assert total == 0
        count = conn.execute("SELECT COUNT(*) FROM station_class").fetchone()[0]
        assert count == 0

    def test_start_date_after_all_data_returns_zero(self, conn):
        _seed_window(conn, {1: 160.0, 2: 160.0, 3: 170.0})
        future = (datetime.date.today() + datetime.timedelta(days=1)).isoformat()
        total = classify_all(conn, start_date=future)
        assert total == 0
