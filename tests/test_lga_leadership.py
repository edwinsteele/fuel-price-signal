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
    LGA_LEADERSHIP_EXCLUSIONS,
    build_lga_trough_lookups,
    compute_pit_strict_days_since_trough,
    detect_trough_events,
    lga_feature_columns,
    lga_slug,
    score_leadership_range,
    score_leadership_snapshot,
)
from fuel_signal.postcode_council import SYDNEY_METRO_COUNCILS

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
    # Four snapshots: Jan 1, Jan 8, Jan 15, Jan 22 (end_date inclusive)
    assert n >= 8  # at least 2 LGAs × 4 snapshots


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
    """With lead_days=7, Parramatta troughs should precede Penrith troughs by ~7 days."""
    from fuel_signal.lga_leadership import _int_to_date

    lead_days = 7
    _build_two_lga_db(conn, lead_days=lead_days)
    lookups = build_lga_trough_lookups(conn)
    p_arr = lookups["Parramatta"]
    pe_arr = lookups["Penrith"]
    assert len(p_arr) >= 2 and len(pe_arr) >= 2

    # Use proper date arithmetic (YYYYMMDD integer subtraction is wrong cross-month).
    pe_dates = [_int_to_date(int(t)) for t in pe_arr]
    pair_leads = []
    for p_t in p_arr:
        p_date = _int_to_date(int(p_t))
        diffs = [(pe_d - p_date).days for pe_d in pe_dates]
        pair_leads.append(min(diffs, key=abs))

    median_lead = sorted(pair_leads)[len(pair_leads) // 2]
    # Penrith fires after Parramatta (positive), within a generous tolerance of lead_days
    assert 0 < median_lead <= lead_days + 10, (
        f"Median Penrith lag {median_lead}d outside expected range (0, {lead_days + 10}]"
    )


# ---------------------------------------------------------------------------
# Exclusion mechanism — Central Coast scoped out of leadership only
# ---------------------------------------------------------------------------

def test_excluded_lgas_missing_from_feature_schema():
    """LGA_LEADERSHIP_EXCLUSIONS members must not appear in the feature schema."""
    assert "Central Coast" in LGA_LEADERSHIP_EXCLUSIONS
    assert "Central Coast" in SYDNEY_METRO_COUNCILS
    assert "Central Coast" not in LGA_FEATURE_COUNCILS
    # Schema size must match SYDNEY_METRO_COUNCILS minus exclusions
    assert len(LGA_FEATURE_COUNCILS) == len(SYDNEY_METRO_COUNCILS) - len(LGA_LEADERSHIP_EXCLUSIONS)


def test_excluded_lga_absent_from_scoring(conn):
    """Excluded LGAs must not produce a leadership row.

    Anchor-side exclusion is the same mechanism (_load_lga_sums drops excluded
    rows via NOT IN), so excluded LGAs also can't contribute to other LGAs'
    rest-of-Sydney anchor; this test doesn't separately assert that because
    a 2-LGA fixture has no surviving non-excluded LGA whose anchor could
    differ. The SQL path is shared, so the absence-from-scoring assertion
    indirectly covers absence-from-anchor."""
    # Patch the exclusion set to include 'Penrith' so we can use the existing fixture.
    import fuel_signal.lga_leadership as lga_mod
    original = lga_mod.LGA_LEADERSHIP_EXCLUSIONS
    lga_mod.LGA_LEADERSHIP_EXCLUSIONS = frozenset({"Penrith"})
    try:
        _build_two_lga_db(conn, lead_days=5)
        snapshot = "2022-04-01"
        n = score_leadership_snapshot(conn, snapshot)
        # Only Parramatta should be scored. Penrith excluded.
        assert n == 1
        board = get_lga_leadership_board(conn, snapshot)
        lgas = {r[0] for r in board}
        assert "Parramatta" in lgas
        assert "Penrith" not in lgas
    finally:
        lga_mod.LGA_LEADERSHIP_EXCLUSIONS = original


# ---------------------------------------------------------------------------
# PIT-strict days_since_trough — adding future prices must not change the past
# ---------------------------------------------------------------------------

def test_pit_strict_days_since_immune_to_future_data(tmp_path):
    """For a fixed query date d, compute_pit_strict_days_since_trough must
    return the same value whether the DB ends at d or extends well past d.

    This is the PIT contract: the feature value at d depends only on prices ≤ d.
    """
    from fuel_signal.db import open_db as _open

    # Build a synthetic 800-day cosine cycle for Parramatta (in LGA_FEATURE_COUNCILS).
    start = date(2020, 1, 1)
    n_days_full = 800
    n_days_truncated = 400  # query date sits well inside this window
    query_date = (start + timedelta(days=350)).isoformat()
    t_full = np.arange(n_days_full, dtype=float)
    prices_full = 150.0 + 10.0 * np.cos(2 * np.pi * t_full / 45)

    def _populate(conn, n_days: int) -> None:
        for code in (1000, 1001, 1002):
            upsert_stations(conn, [_station(code, PC_PARRAMATTA)])
            _seed_daily_prices(conn, code, start, list(prices_full[:n_days]))
            _seed_station_class(conn, code, start, n_days)

    db_full = _open(tmp_path / "full.db")
    create_schema(db_full)
    _populate(db_full, n_days_full)

    db_truncated = _open(tmp_path / "trunc.db")
    create_schema(db_truncated)
    _populate(db_truncated, n_days_truncated)

    full = compute_pit_strict_days_since_trough(db_full, [query_date])
    trunc = compute_pit_strict_days_since_trough(db_truncated, [query_date])

    db_full.close()
    db_truncated.close()

    key = (query_date, "Parramatta")
    assert key in full and key in trunc
    assert full[key] == trunc[key], (
        f"days_since drift: full-db={full[key]}, truncated-db={trunc[key]} — "
        f"future data leaked into past detection"
    )


def test_pit_strict_days_since_returns_none_when_too_early(conn):
    """Before enough history accumulates for trough detection, value must be None."""
    _build_two_lga_db(conn, lead_days=5)
    # Day 10 is too early — smoothing window needs at least 14 days of data.
    early_date = "2020-01-10"
    result = compute_pit_strict_days_since_trough(conn, [early_date])
    key = (early_date, "Parramatta")
    assert key in result
    assert result[key] is None
