"""Tests for fuel_signal.brand_leadership."""

from datetime import date, timedelta

import numpy as np
import pytest

from fuel_signal.brand_leadership import (
    brand_feature_columns,
    brand_slug,
    build_brand_trough_lookups,
    compute_pit_strict_days_since_trough_brand,
    qualifying_brands,
)
from fuel_signal.config import MIN_BRAND_SITES
from fuel_signal.db import (
    create_schema,
    open_db,
    upsert_daily_prices,
    upsert_stations,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BRAND_ALPHA = "AlphaFuel"
BRAND_BETA = "BetaGas"
BRAND_TINY = "TinyBrand"   # below MIN_BRAND_SITES — should not qualify


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def conn(tmp_path):
    c = open_db(tmp_path / "brand_leadership.db")
    create_schema(c)
    yield c
    c.close()


def _station(code: int, brand: str, postcode: str = "2000") -> dict:
    return {
        "station_code": code,
        "name": f"Station {code}",
        "address": f"{code} Test Street, Town",
        "suburb": "Town",
        "postcode": postcode,
        "brand": brand,
    }


def _seed_daily_prices(conn, station_code: int, start: date, prices: list[float]) -> None:
    rows = []
    for i, p in enumerate(prices):
        d = (start + timedelta(days=i)).isoformat()
        rows.append((station_code, "E10", d, p))
    upsert_daily_prices(conn, rows)
    conn.commit()


def _seed_station_class(conn, station_code: int, start: date, n_days: int, cls: str = "Competitive") -> None:
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


def _build_two_brand_db(conn, *, lead_days: int = 5, n_alpha: int = MIN_BRAND_SITES, n_beta: int = MIN_BRAND_SITES):
    """Seed two qualifying brands with synthetic cosine cycles.

    AlphaFuel leads BetaGas by lead_days.
    """
    start = date(2020, 1, 1)
    n_days = 600

    t = np.arange(n_days, dtype=float)
    alpha_prices = 150.0 + 10.0 * np.cos(2 * np.pi * t / 45)
    beta_prices = 150.0 + 10.0 * np.cos(2 * np.pi * (t - lead_days) / 45)

    for i in range(n_alpha):
        code = 1000 + i
        upsert_stations(conn, [_station(code, BRAND_ALPHA)])
        _seed_daily_prices(conn, code, start, list(alpha_prices))
        _seed_station_class(conn, code, start, n_days)

    for i in range(n_beta):
        code = 2000 + i
        upsert_stations(conn, [_station(code, BRAND_BETA)])
        _seed_daily_prices(conn, code, start, list(beta_prices))
        _seed_station_class(conn, code, start, n_days)


# ---------------------------------------------------------------------------
# Unit tests — brand_slug
# ---------------------------------------------------------------------------

def test_brand_slug_plain():
    assert brand_slug("Shell") == "shell"


def test_brand_slug_spaces():
    assert brand_slug("EG Ampol") == "eg_ampol"


def test_brand_slug_digits():
    assert brand_slug("7-Eleven") == "7_eleven"


def test_brand_slug_ampersand():
    assert brand_slug("BP & co") == "bp_co"


# ---------------------------------------------------------------------------
# Unit tests — qualifying_brands / brand_feature_columns
# ---------------------------------------------------------------------------

def test_qualifying_brands_empty_db(conn):
    assert qualifying_brands(conn) == []


def test_qualifying_brands_threshold(conn):
    """Only brands with >= MIN_BRAND_SITES stations should qualify."""
    _build_two_brand_db(conn)

    # Add a tiny brand well below the threshold
    for i in range(MIN_BRAND_SITES - 1):
        code = 9000 + i
        upsert_stations(conn, [_station(code, BRAND_TINY)])
        _seed_daily_prices(conn, code, date(2020, 1, 1), [150.0] * 100)

    brands = qualifying_brands(conn)
    assert BRAND_ALPHA in brands
    assert BRAND_BETA in brands
    assert BRAND_TINY not in brands


def test_brand_feature_columns_prefix(conn):
    _build_two_brand_db(conn)
    cols = brand_feature_columns(conn)
    assert all(c.startswith("days_since_trough_entry_") for c in cols)
    assert len(cols) == 2
    # Columns are ordered by brand slug (alphabetically)
    slugs = [c.replace("days_since_trough_entry_", "") for c in cols]
    assert slugs == sorted(slugs)


def test_brand_feature_columns_empty_when_no_qualifying_brands(conn):
    """When no brand meets the threshold, brand_feature_columns returns []."""
    for i in range(MIN_BRAND_SITES - 1):
        code = 9000 + i
        upsert_stations(conn, [_station(code, BRAND_TINY)])
        _seed_daily_prices(conn, code, date(2020, 1, 1), [150.0] * 100)
    assert brand_feature_columns(conn) == []


def test_brand_feature_columns_sorted(conn):
    """Column list must be deterministic (sorted brand names → stable output)."""
    _build_two_brand_db(conn)
    assert brand_feature_columns(conn) == brand_feature_columns(conn)


# ---------------------------------------------------------------------------
# Integration tests — build_brand_trough_lookups
# ---------------------------------------------------------------------------

def test_build_brand_trough_lookups_returns_arrays(conn):
    _build_two_brand_db(conn)
    brands = qualifying_brands(conn)
    lookups = build_brand_trough_lookups(conn, brands)
    assert BRAND_ALPHA in lookups
    assert BRAND_BETA in lookups
    # 600 days at 45-day cycle → ~13 cycles, expect several troughs
    assert len(lookups[BRAND_ALPHA]) >= 2
    assert len(lookups[BRAND_BETA]) >= 2


def test_build_brand_trough_lookups_sorted(conn):
    _build_two_brand_db(conn)
    brands = qualifying_brands(conn)
    lookups = build_brand_trough_lookups(conn, brands)
    for brand, arr in lookups.items():
        assert list(arr) == sorted(arr), f"{brand} trough dates are not sorted"


def test_build_brand_trough_lookups_empty_db(conn):
    lookups = build_brand_trough_lookups(conn, [])
    assert lookups == {}


def test_sticky_stations_excluded_from_brand_series(conn):
    """Brand trough detection excludes Sticky stations; their prices should not shift the median."""
    start = date(2020, 1, 1)
    n_days = 600
    t = np.arange(n_days, dtype=float)
    cycle_prices = 150.0 + 10.0 * np.cos(2 * np.pi * t / 45)

    # Competitive stations
    for i in range(MIN_BRAND_SITES):
        code = 1000 + i
        upsert_stations(conn, [_station(code, BRAND_ALPHA)])
        _seed_daily_prices(conn, code, start, list(cycle_prices))
        _seed_station_class(conn, code, start, n_days, cls="Competitive")

    # One Sticky station with a flat high price that would flatten the cycle
    sticky_code = 9999
    upsert_stations(conn, [_station(sticky_code, BRAND_ALPHA)])
    _seed_daily_prices(conn, sticky_code, start, [999.0] * n_days)  # extreme high price
    _seed_station_class(conn, sticky_code, start, n_days, cls="Sticky")

    brands = qualifying_brands(conn)
    lookups_with_sticky = build_brand_trough_lookups(conn, brands)

    # If Sticky were included, median of [cycle_prices * N_COMPETITIVE, 999] would shift.
    # Since it's excluded, the trough detection should still find multiple troughs.
    assert len(lookups_with_sticky.get(BRAND_ALPHA, [])) >= 2


# ---------------------------------------------------------------------------
# PIT-strict days_since_trough — future data must not change past
# ---------------------------------------------------------------------------

def test_pit_strict_brand_immune_to_future_data(tmp_path):
    """For a fixed query date d, result must be identical whether DB ends at d or extends past it."""
    start = date(2020, 1, 1)
    n_days_full = 600
    n_days_truncated = 300
    query_date = (start + timedelta(days=250)).isoformat()

    t_full = np.arange(n_days_full, dtype=float)
    prices_full = 150.0 + 10.0 * np.cos(2 * np.pi * t_full / 45)

    def _populate(conn, n_days: int) -> None:
        for i in range(MIN_BRAND_SITES):
            code = 1000 + i
            upsert_stations(conn, [_station(code, BRAND_ALPHA)])
            _seed_daily_prices(conn, code, start, list(prices_full[:n_days]))
            _seed_station_class(conn, code, start, n_days)

    db_full = open_db(tmp_path / "full.db")
    create_schema(db_full)
    _populate(db_full, n_days_full)

    db_trunc = open_db(tmp_path / "trunc.db")
    create_schema(db_trunc)
    _populate(db_trunc, n_days_truncated)

    full_result = compute_pit_strict_days_since_trough_brand(
        db_full, [query_date], [BRAND_ALPHA]
    )
    trunc_result = compute_pit_strict_days_since_trough_brand(
        db_trunc, [query_date], [BRAND_ALPHA]
    )

    db_full.close()
    db_trunc.close()

    key = (query_date, BRAND_ALPHA)
    assert key in full_result and key in trunc_result
    assert full_result[key] == trunc_result[key], (
        f"PIT drift: full={full_result[key]}, trunc={trunc_result[key]}"
    )


def test_pit_strict_returns_none_when_too_early(conn):
    """Before enough history for trough detection, value must be None."""
    _build_two_brand_db(conn)
    early = "2020-01-10"
    result = compute_pit_strict_days_since_trough_brand(conn, [early], [BRAND_ALPHA])
    assert result.get((early, BRAND_ALPHA)) is None


def test_pit_strict_empty_brand_list(conn):
    _build_two_brand_db(conn)
    result = compute_pit_strict_days_since_trough_brand(conn, ["2020-06-01"], [])
    assert result == {}


def test_pit_strict_missing_brand_returns_none(conn):
    """A brand with no DB data must map to None for all query dates."""
    _build_two_brand_db(conn)
    result = compute_pit_strict_days_since_trough_brand(
        conn, ["2021-01-01"], ["NoSuchBrand"]
    )
    assert result.get(("2021-01-01", "NoSuchBrand")) is None
