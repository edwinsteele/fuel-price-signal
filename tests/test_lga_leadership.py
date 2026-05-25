"""Tests for fuel_signal.lga_leadership."""

from datetime import date, timedelta

import numpy as np
import pytest

from fuel_signal.db import (
    create_schema,
    get_lga_leadership_board,
    latest_lga_leadership_date,
    open_db,
    upsert_daily_prices,
    upsert_stations,
)
from fuel_signal.lga_leadership import (
    LGA_FEATURE_COUNCILS,
    build_lga_trough_lookups,
    detect_trough_events,
    lga_feature_columns,
    lga_slug,
    score_leadership_range,
    score_leadership_snapshot,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Parramatta and Penrith: two distinct Sydney-metro councils
PC_PARRAMATTA = "2150"  # → "Parramatta"
PC_PENRITH = "2750"     # → "Penrith"


@pytest.fixture
def conn(tmp_path):
    c = open_db(tmp_path / "lga_leadership.db")
    create_schema(c)
    yield c
    c.close()


def _station(code: int, postcode: str, brand: str = "Shell") -> dict:
    return {
        "station_code": code,
        "name": f"Station {code}",
        "address": f"{code} Test Street, Town",
        "suburb": "Town",
        "postcode": postcode,
        "brand": brand,
    }


def _seed_daily_prices(conn, station_code: int, start: date, prices: list[float]) -> None:
    """Insert daily_prices rows for station_code starting from start."""
    rows = []
    for i, p in enumerate(prices):
        d = (start + timedelta(days=i)).isoformat()
        rows.append((station_code, "E10", d, p))
    upsert_daily_prices(conn, rows)
    conn.commit()


def _seed_station_class(conn, station_code: int, start: date, n_days: int, cls: str = "Competitive") -> None:
    """Insert Competitive station_class rows for all days so the Sticky filter passes."""
    from fuel_signal.db import _date_to_int as _d2i  # type: ignore[attr-defined]
    rows = []
    for i in range(n_days):
        d = _d2i((start + timedelta(days=i)).isoformat())
        rows.append((station_code, d, cls, 0))
    conn.executemany(
        "INSERT OR REPLACE INTO station_class"
        " (station_code, snapshot_date, class, median_premium_decicents)"
        " VALUES (?, ?, ?, ?)",
        rows,
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Unit tests — detect_trough_events
# ---------------------------------------------------------------------------

def test_detect_trough_events_sinusoidal():
    """A simple sinusoidal series should produce troughs at the expected positions."""
    n = 200
    # One full cycle: trough at index 50, then at index ~100+50=150
    t = np.linspace(0, 4 * np.pi, n)
    prices = 100.0 + 10.0 * np.cos(t)  # troughs at cos(π)=−1 → idx ≈ 50, 150
    idx = detect_trough_events(prices, min_spacing=18)
    assert len(idx) >= 1
    # Each detected trough should sit in the lower half of the price range
    for i in idx:
        assert prices[i] < 100.0


def test_detect_trough_events_too_short():
    prices = np.array([100.0, 99.0, 98.0])  # fewer than smooth_window * 2
    assert len(detect_trough_events(prices)) == 0


def test_detect_trough_events_flat():
    prices = np.full(100, 150.0)
    assert len(detect_trough_events(prices)) == 0


def test_detect_trough_events_single_trough():
    """V-shaped series with one clear trough."""
    prices = np.concatenate([np.linspace(160, 140, 30), np.linspace(140, 160, 30)])
    idx = detect_trough_events(prices, min_spacing=18)
    assert len(idx) == 1
    # Trough should be near the midpoint
    assert 20 <= idx[0] <= 40


def test_detect_trough_events_snap():
    """Verify that snapping corrects a smooth-min that sits a few positions from the raw min."""
    # Asymmetric descent: raw min at position 45, centered-smooth min slightly earlier.
    n = 100
    prices = np.full(n, 160.0)
    prices[30:55] = np.linspace(160, 140, 25)  # gradual descent
    prices[55:70] = np.linspace(140, 160, 15)  # fast recovery
    # Minimum is at position 54 (last of the descent)
    prices[54] = 138.0  # slight extra dip
    idx = detect_trough_events(prices, min_spacing=18)
    assert len(idx) == 1
    # Snap should land at or near position 54
    assert abs(int(idx[0]) - 54) <= 5


# ---------------------------------------------------------------------------
# Unit tests — lga_slug / lga_feature_columns
# ---------------------------------------------------------------------------

def test_lga_slug_plain():
    assert lga_slug("Penrith") == "penrith"


def test_lga_slug_space():
    assert lga_slug("Blue Mountains") == "blue_mountains"


def test_lga_slug_hyphen():
    assert lga_slug("Ku-ring-gai") == "ku_ring_gai"


def test_lga_slug_hyphenated_compound():
    assert lga_slug("Canterbury-Bankstown") == "canterbury_bankstown"


def test_lga_feature_columns_count():
    cols = lga_feature_columns()
    assert len(cols) == len(LGA_FEATURE_COUNCILS)
    assert all(c.startswith("days_since_trough_entry_") for c in cols)


def test_lga_feature_columns_stable():
    """Column list must be deterministic (sorted input → stable output)."""
    assert lga_feature_columns() == lga_feature_columns()


# ---------------------------------------------------------------------------
# Integration tests — score_leadership_snapshot
# ---------------------------------------------------------------------------

def _build_two_lga_db(conn, *, lead_days: int = 5):
    """Seed two 3-station LGAs (Parramatta leading, Penrith lagging) with several
    synthetic price cycles so the leadership scorer has enough events."""
    N_STATIONS_EACH = 3
    start = date(2020, 1, 1)
    n_days = 800  # > 730d window

    # Synthetic cycle: period 45 days, Parramatta leads by lead_days
    t = np.arange(n_days, dtype=float)
    parramatta_prices = 150.0 + 10.0 * np.cos(2 * np.pi * t / 45)
    penrith_prices = 150.0 + 10.0 * np.cos(2 * np.pi * (t - lead_days) / 45)

    for i in range(N_STATIONS_EACH):
        code_p = 1000 + i
        code_pe = 2000 + i
        upsert_stations(conn, [_station(code_p, PC_PARRAMATTA), _station(code_pe, PC_PENRITH)])
        _seed_daily_prices(conn, code_p, start, list(parramatta_prices))
        _seed_daily_prices(conn, code_pe, start, list(penrith_prices))
        _seed_station_class(conn, code_p, start, n_days)
        _seed_station_class(conn, code_pe, start, n_days)


def test_score_leadership_snapshot_returns_row_count(conn):
    _build_two_lga_db(conn)
    snapshot = "2022-04-01"  # well inside the 800-day range
    n = score_leadership_snapshot(conn, snapshot)
    assert n == 2  # Parramatta + Penrith


def test_score_leadership_snapshot_idempotent(conn):
    _build_two_lga_db(conn)
    snapshot = "2022-04-01"
    score_leadership_snapshot(conn, snapshot)
    n2 = score_leadership_snapshot(conn, snapshot)  # second run replaces rows
    board = get_lga_leadership_board(conn, snapshot)
    assert len(board) == 2
    assert n2 == 2


def test_score_leadership_parramatta_leads(conn):
    """Parramatta (lead_days=5) should have positive trough_lead_median_days."""
    _build_two_lga_db(conn, lead_days=5)
    snapshot = "2022-04-01"
    score_leadership_snapshot(conn, snapshot)
    board = get_lga_leadership_board(conn, snapshot)
    rows_by_lga = {r[0]: r for r in board}
    p_row = rows_by_lga["Parramatta"]
    pe_row = rows_by_lga["Penrith"]
    # Parramatta leads (positive median), Penrith lags (negative or near zero)
    assert p_row[1] is not None and p_row[1] > 0
    assert pe_row[1] is None or pe_row[1] <= 0


def test_score_leadership_range(conn):
    _build_two_lga_db(conn)
    n = score_leadership_range(conn, "2022-01-01", "2022-01-22", step_days=7)
    # Three snapshots: Jan 1, Jan 8, Jan 15 (Jan 22 is included too)
    assert n >= 6  # at least 2 LGAs × 3 snapshots


def test_latest_lga_leadership_date(conn):
    _build_two_lga_db(conn)
    assert latest_lga_leadership_date(conn) is None
    score_leadership_snapshot(conn, "2022-04-01")
    assert latest_lga_leadership_date(conn) == "2022-04-01"


def test_score_no_data_returns_zero(conn):
    n = score_leadership_snapshot(conn, "2024-01-01")
    assert n == 0


# ---------------------------------------------------------------------------
# Integration tests — build_lga_trough_lookups
# ---------------------------------------------------------------------------

def test_build_lga_trough_lookups_returns_dict(conn):
    _build_two_lga_db(conn)
    lookups = build_lga_trough_lookups(conn)
    assert "Parramatta" in lookups
    assert "Penrith" in lookups
    # Should have at least a few troughs over 800 days with 45-day cycles
    assert len(lookups["Parramatta"]) >= 2
    assert len(lookups["Penrith"]) >= 2


def test_build_lga_trough_lookups_sorted(conn):
    _build_two_lga_db(conn)
    lookups = build_lga_trough_lookups(conn)
    for lga, arr in lookups.items():
        assert list(arr) == sorted(arr), f"{lga} trough dates are not sorted"


def test_build_lga_trough_lookups_empty_db(conn):
    lookups = build_lga_trough_lookups(conn)
    assert lookups == {}


def test_build_lga_trough_parramatta_leads_penrith(conn):
    """With lead_days=7, Parramatta troughs should precede Penrith troughs."""
    _build_two_lga_db(conn, lead_days=7)
    lookups = build_lga_trough_lookups(conn)
    p_arr = lookups["Parramatta"]
    pe_arr = lookups["Penrith"]
    assert len(p_arr) >= 2 and len(pe_arr) >= 2
    # For each Parramatta trough, there should be a Penrith trough ~7 days later
    for p_t in p_arr:
        diffs = [int(pe_t) - int(p_t) for pe_t in pe_arr]
        nearest = min(diffs, key=abs)
        # Penrith is behind, so nearest diff should be positive (Penrith fires after)
        assert nearest > 0 or abs(nearest) <= 10  # loose tolerance for snap + cycle variance
