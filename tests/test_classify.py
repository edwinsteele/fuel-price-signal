"""Tests for fuel_signal.classify — station classifier.

Verifies the 1D median-premium classifier with two-pass cluster reference,
PIT discipline, cold-start, and the zero-Competitive edge case.
"""

from datetime import date, timedelta

import pytest

from fuel_signal.classify import (
    CLASS_COMPETITIVE,
    CLASS_DISCOUNT,
    CLASS_STICKY,
    PREMIUM_BAND_CENTS,
    WINDOW_DAYS,
    _classify_lga,
    classify_range,
    classify_snapshot,
)
from fuel_signal.db import (
    create_schema,
    fuel_type_id,
    get_station_class,
    insert_prices,
    open_db,
    upsert_daily_prices,
    upsert_stations,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

# Two different Sydney-metro postcodes that map to two different LGAs.
PC_PARRAMATTA = "2150"  # → "Parramatta"
PC_LIVERPOOL = "2170"   # → "Liverpool"
PC_SYDNEY = "2000"      # → "Sydney"


@pytest.fixture
def conn(tmp_path):
    c = open_db(tmp_path / "classify.db")
    create_schema(c)
    yield c
    c.close()


def _station(code: int, postcode: str = PC_PARRAMATTA, brand: str = "Shell") -> dict:
    return {
        "station_code": code,
        "name": f"Station {code}",
        "address": f"{code} Test Street, Town",
        "suburb": "Town",
        "postcode": postcode,
        "brand": brand,
    }


def _seed_daily_prices(
    conn,
    station_code: int,
    start: str,
    days: int,
    prices: list[float],
    fuel_code: str = "E10",
) -> None:
    """Write `days` consecutive daily_prices rows starting at `start`.

    If `prices` is shorter than days, the last price is repeated (forward-fill).
    """
    start_d = date.fromisoformat(start)
    if len(prices) < days:
        prices = prices + [prices[-1]] * (days - len(prices))
    rows = [
        (
            station_code,
            fuel_code,
            (start_d + timedelta(days=i)).isoformat(),
            prices[i],
        )
        for i in range(days)
    ]
    upsert_daily_prices(conn, rows)
    conn.commit()


def _seed_constant_lga(
    conn,
    station_codes: list[int],
    prices: list[float],
    postcode: str,
    snapshot_date: str,
    window_days: int = WINDOW_DAYS,
) -> None:
    """Create stations in one LGA with one constant price each across the full window."""
    assert len(station_codes) == len(prices)
    upsert_stations(conn, [_station(c, postcode=postcode) for c in station_codes])
    start_d = date.fromisoformat(snapshot_date) - timedelta(days=window_days)
    for code, p in zip(station_codes, prices):
        _seed_daily_prices(conn, code, start_d.isoformat(), window_days, [p])


# ---------------------------------------------------------------------------
# Pure-function unit tests for _classify_lga
# ---------------------------------------------------------------------------

class TestClassifyLga:
    def _const_series(self, codes_prices: dict[int, float], n_days: int = 10) -> dict:
        dates = [(date(2024, 1, 1) + timedelta(days=i)).isoformat() for i in range(n_days)]
        return {code: {d: p for d in dates} for code, p in codes_prices.items()}

    def test_all_within_band_are_competitive(self):
        prices = {1: 150.0, 2: 152.0, 3: 153.0}
        classes, premiums = _classify_lga(self._const_series(prices))
        assert all(c == CLASS_COMPETITIVE for c in classes.values())
        # Cluster (iter 2) = median of competitive iter-1 set ≈ 152.
        assert abs(premiums[2]) < 1e-6

    def test_high_premium_is_sticky(self):
        # Three competitive ones + one well above the band.
        prices = {1: 150.0, 2: 152.0, 3: 154.0, 4: 200.0}
        classes, _ = _classify_lga(self._const_series(prices))
        assert classes[4] == CLASS_STICKY
        assert classes[1] == CLASS_COMPETITIVE
        assert classes[2] == CLASS_COMPETITIVE
        assert classes[3] == CLASS_COMPETITIVE

    def test_low_premium_is_discount(self):
        prices = {1: 150.0, 2: 152.0, 3: 154.0, 4: 120.0}
        classes, _ = _classify_lga(self._const_series(prices))
        assert classes[4] == CLASS_DISCOUNT
        assert classes[1] == CLASS_COMPETITIVE

    def test_band_boundary_above_is_sticky(self):
        # 3 anchors at 150 → cluster ref = 150. Test station at +band + 0.5c
        # has premium = +10.5 > band → Sticky.
        prices = {1: 150.0, 2: 150.0, 3: 150.0, 4: 150.0 + PREMIUM_BAND_CENTS + 0.5}
        classes, _ = _classify_lga(self._const_series(prices))
        assert classes[4] == CLASS_STICKY

    def test_band_boundary_at_band_is_competitive(self):
        # Exactly +band cents (the edge) → Competitive (band uses strict >).
        prices = {1: 150.0, 2: 150.0, 3: 150.0, 4: 150.0 + PREMIUM_BAND_CENTS}
        classes, _ = _classify_lga(self._const_series(prices))
        assert classes[4] == CLASS_COMPETITIVE

    def test_zero_iter1_competitive_falls_back_to_iter1_labels(self):
        # 2-station LGA: prices straddle the iter-1 median by more than the band,
        # so iter 1 yields zero Competitive. Iter 2 has no cluster reference to
        # compute; the function should fall back to iter-1 labels (one Sticky,
        # one Discount) rather than crash.
        prices = {1: 100.0, 2: 200.0}
        classes, _ = _classify_lga(self._const_series(prices))
        assert CLASS_COMPETITIVE not in classes.values()
        assert set(classes.values()) == {CLASS_STICKY, CLASS_DISCOUNT}

    def test_iter2_uses_only_iter1_competitive(self):
        # 3 tightly clustered anchors at [148, 150, 152] (cluster ref = 150 on
        # both iter 1 and iter 2). A Sticky outlier at 180 should be removed
        # from the iter-2 cluster set so the anchors keep their Competitive label
        # even though iter-1 cluster ref was pulled up to 151 by the Sticky.
        prices = {1: 148.0, 2: 150.0, 3: 152.0, 4: 180.0}
        classes, _ = _classify_lga(self._const_series(prices))
        # 1,2,3 must end up Competitive; 4 Sticky.
        assert classes[1] == CLASS_COMPETITIVE
        assert classes[2] == CLASS_COMPETITIVE
        assert classes[3] == CLASS_COMPETITIVE
        assert classes[4] == CLASS_STICKY


# ---------------------------------------------------------------------------
# Integration tests for classify_snapshot
# ---------------------------------------------------------------------------

class TestClassifySnapshot:
    def test_writes_one_row_per_station_per_date(self, conn):
        _seed_constant_lga(
            conn, [101, 102, 103], [150.0, 152.0, 153.0],
            PC_PARRAMATTA, "2024-03-01",
        )
        n_class, n_summary = classify_snapshot(conn, "2024-03-01")
        assert n_class == 3
        assert n_summary == 1
        for code in (101, 102, 103):
            row = get_station_class(conn, code, "2024-03-01")
            assert row is not None
            assert row[0] == CLASS_COMPETITIVE

    def test_sticky_and_discount_assignment(self, conn):
        # Four stations in one LGA: two anchors at 150/152, one sticky, one discount.
        _seed_constant_lga(
            conn, [201, 202, 203, 204],
            [150.0, 152.0, 200.0, 120.0],
            PC_PARRAMATTA, "2024-03-01",
        )
        classify_snapshot(conn, "2024-03-01")
        assert get_station_class(conn, 201, "2024-03-01")[0] == CLASS_COMPETITIVE
        assert get_station_class(conn, 202, "2024-03-01")[0] == CLASS_COMPETITIVE
        assert get_station_class(conn, 203, "2024-03-01")[0] == CLASS_STICKY
        assert get_station_class(conn, 204, "2024-03-01")[0] == CLASS_DISCOUNT

    def test_lga_isolation(self, conn):
        # Two LGAs at different absolute price levels; each should classify
        # internally rather than across.
        _seed_constant_lga(
            conn, [301, 302, 303], [150.0, 152.0, 153.0],
            PC_PARRAMATTA, "2024-03-01",
        )
        _seed_constant_lga(
            conn, [401, 402, 403], [200.0, 202.0, 203.0],
            PC_LIVERPOOL, "2024-03-01",
        )
        classify_snapshot(conn, "2024-03-01")
        # All six classify as Competitive within their own LGA.
        for code in (301, 302, 303, 401, 402, 403):
            assert get_station_class(conn, code, "2024-03-01")[0] == CLASS_COMPETITIVE

        # classification_summary has one row per LGA.
        summary = conn.execute(
            "SELECT lga, n_competitive, n_sticky, n_discount"
            " FROM classification_summary WHERE snapshot_date = 20240301"
            " ORDER BY lga"
        ).fetchall()
        assert len(summary) == 2
        for _, n_c, n_s, n_d in summary:
            assert (n_c, n_s, n_d) == (3, 0, 0)

    def test_cold_start_one_observation_only(self, conn):
        # A station with exactly one observation in the window still gets a row.
        upsert_stations(conn, [_station(c) for c in (501, 502, 503, 504)])
        snap = "2024-03-01"
        start = (date.fromisoformat(snap) - timedelta(days=WINDOW_DAYS)).isoformat()
        for code in (501, 502, 503):
            _seed_daily_prices(conn, code, start, WINDOW_DAYS, [150.0 + code % 5])
        # Station 504 only has the very last day in window.
        _seed_daily_prices(
            conn, 504,
            (date.fromisoformat(snap) - timedelta(days=1)).isoformat(),
            1, [151.0],
        )
        classify_snapshot(conn, snap)
        row = get_station_class(conn, 504, snap)
        assert row is not None
        assert row[0] == CLASS_COMPETITIVE

    def test_pit_window_excludes_snapshot_date_and_later(self, conn):
        # A Sticky-looking station's loud prices land on snapshot_date and later;
        # they must be invisible to the classifier.
        snap = "2024-03-01"
        upsert_stations(conn, [_station(c) for c in (601, 602, 603, 604)])
        window_start = (date.fromisoformat(snap) - timedelta(days=WINDOW_DAYS)).isoformat()

        # Anchors: 3 competitive stations at 150c across full window.
        for code in (601, 602, 603):
            _seed_daily_prices(conn, code, window_start, WINDOW_DAYS, [150.0])
        # Held-out station 604: matches anchors inside window, jumps to 300c
        # ON and AFTER snapshot_date. If the classifier honours PIT, 604 is
        # Competitive on snapshot_date; if it cheats and reads >= snapshot_date,
        # it becomes Sticky.
        _seed_daily_prices(conn, 604, window_start, WINDOW_DAYS, [150.0])
        _seed_daily_prices(
            conn, 604, snap, 5, [300.0, 300.0, 300.0, 300.0, 300.0],
        )

        classify_snapshot(conn, snap)
        row = get_station_class(conn, 604, snap)
        assert row is not None
        assert row[0] == CLASS_COMPETITIVE, (
            "PIT violation: classifier saw price data on/after snapshot_date"
        )

    def test_pit_window_is_45_days(self, conn):
        # Prices older than D-45 must not influence the classification.
        snap = "2024-03-01"
        upsert_stations(conn, [_station(c) for c in (701, 702, 703, 704)])
        window_start = (date.fromisoformat(snap) - timedelta(days=WINDOW_DAYS)).isoformat()

        for code in (701, 702, 703):
            _seed_daily_prices(conn, code, window_start, WINDOW_DAYS, [150.0])
        # Station 704: outside-window prices are wildly high; inside-window matches anchors.
        old_start = (date.fromisoformat(snap) - timedelta(days=WINDOW_DAYS + 30)).isoformat()
        _seed_daily_prices(conn, 704, old_start, 30, [300.0])
        _seed_daily_prices(conn, 704, window_start, WINDOW_DAYS, [150.0])

        classify_snapshot(conn, snap)
        row = get_station_class(conn, 704, snap)
        assert row is not None
        assert row[0] == CLASS_COMPETITIVE

    def test_zero_competitive_lga_drops_station_rows(self, conn):
        # A 2-station LGA where one is way above and one way below the other.
        # Iter-1 cluster ref = median across both → midpoint. One station ends up
        # exactly +X above, one exactly −X below; for X >> 10 both fall outside
        # the band → iter-1 yields zero Competitive → iter-2 cannot run →
        # final classes = (Sticky, Discount). The LGA must produce a summary row
        # but no station_class rows.
        _seed_constant_lga(
            conn, [801, 802], [100.0, 200.0],
            PC_SYDNEY, "2024-03-01",
        )
        n_class, n_summary = classify_snapshot(conn, "2024-03-01")
        assert n_class == 0
        assert n_summary == 1
        # No station_class rows for either station.
        assert get_station_class(conn, 801, "2024-03-01") is None
        assert get_station_class(conn, 802, "2024-03-01") is None
        # The summary row records n_competitive=0 plus the iter-1 Sticky/Discount counts.
        row = conn.execute(
            "SELECT n_competitive, n_sticky, n_discount FROM classification_summary"
            " WHERE snapshot_date = 20240301 AND lga = 'Sydney'"
        ).fetchone()
        assert row is not None
        assert row[0] == 0
        assert row[1] + row[2] == 2

    def test_idempotent_rerun(self, conn):
        _seed_constant_lga(
            conn, [901, 902, 903], [150.0, 151.0, 152.0],
            PC_PARRAMATTA, "2024-03-01",
        )
        classify_snapshot(conn, "2024-03-01")
        n_first = conn.execute(
            "SELECT COUNT(*) FROM station_class WHERE snapshot_date = 20240301"
        ).fetchone()[0]
        classify_snapshot(conn, "2024-03-01")
        n_second = conn.execute(
            "SELECT COUNT(*) FROM station_class WHERE snapshot_date = 20240301"
        ).fetchone()[0]
        assert n_first == n_second == 3

    def test_empty_window_writes_nothing(self, conn):
        # Stations exist but no daily_prices in the window → no rows.
        upsert_stations(conn, [_station(1101)])
        n_class, n_summary = classify_snapshot(conn, "2024-03-01")
        assert n_class == 0
        assert n_summary == 0

    def test_median_premium_persisted_in_decicents(self, conn):
        # Discount station ~30c below cluster — median_premium_decicents ≈ -300.
        _seed_constant_lga(
            conn, [1201, 1202, 1203, 1204],
            [150.0, 152.0, 154.0, 120.0],
            PC_PARRAMATTA, "2024-03-01",
        )
        classify_snapshot(conn, "2024-03-01")
        # Iter-2 cluster ref = median of the three competitive ≈ 152.0
        cls, premium_decicents = get_station_class(conn, 1204, "2024-03-01")
        assert cls == CLASS_DISCOUNT
        # 120 - 152 = -32c → -320 decicents.
        assert premium_decicents == -320

    def test_only_uses_daily_prices_not_raw_prices(self, conn):
        # Put raw prices that would change the classification, but no daily_prices
        # at all → classifier writes nothing.
        upsert_stations(conn, [_station(c) for c in (1301, 1302, 1303)])
        # Pre-create the fuel type id by using daily_prices upsert path elsewhere
        # not needed; insert_prices auto-registers fuel types.
        insert_prices(conn, [
            {"station_code": 1301, "fuel_code": "E10",
             "price_date": "2024-02-15", "price_cents": 150.0},
            {"station_code": 1302, "fuel_code": "E10",
             "price_date": "2024-02-15", "price_cents": 200.0},
            {"station_code": 1303, "fuel_code": "E10",
             "price_date": "2024-02-15", "price_cents": 120.0},
        ])
        n_class, n_summary = classify_snapshot(conn, "2024-03-01")
        # Nothing in daily_prices → empty window → no output.
        assert n_class == 0
        assert n_summary == 0
        # Sanity: the raw prices are there.
        fid = fuel_type_id(conn, "E10")
        assert conn.execute(
            "SELECT COUNT(*) FROM prices WHERE fuel_type_id = ?", (fid,)
        ).fetchone()[0] == 3


class TestClassifyRange:
    def test_writes_rows_for_each_day(self, conn):
        # Stations stable across a 3-day classify window.
        snap_start = "2024-03-01"
        snap_end = "2024-03-03"
        # Seed enough history so the 45d window has data for all three days.
        upsert_stations(conn, [_station(c) for c in (1401, 1402, 1403)])
        seed_start = (date.fromisoformat(snap_start) - timedelta(days=WINDOW_DAYS)).isoformat()
        for code, p in [(1401, 150.0), (1402, 152.0), (1403, 154.0)]:
            _seed_daily_prices(conn, code, seed_start, WINDOW_DAYS + 3, [p])

        n_class, n_summary = classify_range(conn, snap_start, snap_end)
        assert n_class == 9   # 3 stations × 3 snapshot dates
        assert n_summary == 3  # 1 LGA × 3 snapshot dates

    def test_rejects_inverted_range(self, conn):
        with pytest.raises(ValueError, match="must not exceed"):
            classify_range(conn, "2024-03-05", "2024-03-01")


class TestClassifyEdgeCases:
    def test_single_station_lga_is_competitive(self, conn):
        # 1-station LGA: cluster ref = that one station's price → premium = 0 → Competitive.
        _seed_constant_lga(conn, [1501], [150.0], PC_LIVERPOOL, "2024-03-01")
        classify_snapshot(conn, "2024-03-01")
        row = get_station_class(conn, 1501, "2024-03-01")
        assert row is not None
        assert row[0] == CLASS_COMPETITIVE
        # And premium is exactly 0c (= 0 decicents).
        assert row[1] == 0

    def test_invalid_window_days_raises(self, conn):
        with pytest.raises(ValueError, match="positive"):
            classify_snapshot(conn, "2024-03-01", window_days=0)
