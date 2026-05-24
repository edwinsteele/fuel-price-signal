"""Tests for fuel_signal.features — feature pipeline and PIT validation.

Synthetic series strategy
--------------------------
A sawtooth price series with known cycle parameters is used throughout.
Each cycle: 3-day sharp rise to peak, then 43-day linear decline to trough.
Total cycle length = 46 days, amplitude = 25c, base = 150c.

Three full cycles gives peaks at approximately days 2, 48, 94 —
enough for CycleDetector to confirm >= 2 peaks.
"""

from __future__ import annotations

import datetime

import pytest

from fuel_signal.db import (
    create_schema,
    open_db,
    upsert_daily_prices,
    upsert_station_class_rows,
    upsert_stations,
)
from fuel_signal.features import (
    FEATURE_COLUMNS,
    MIN_TRAINING_ROWS_PER_STATION,
    assemble_feature_rows,
    compute_features,
)

# ---------------------------------------------------------------------------
# Synthetic series helpers
# ---------------------------------------------------------------------------

_CYCLE_LENGTH = 46
_RISE_DAYS = 3
_BASE_PRICE = 150.0
_AMPLITUDE = 25.0


def _sawtooth_series(
    n_cycles: float = 3.0,
    start: str = "2020-01-01",
) -> list[tuple[str, float]]:
    """Return [(date_str, price_cents)] with n_cycles full sawtooth cycles."""
    total_days = int(n_cycles * _CYCLE_LENGTH)
    start_date = datetime.date.fromisoformat(start)
    result = []
    for day in range(total_days):
        pos = day % _CYCLE_LENGTH
        if pos < _RISE_DAYS:
            price = _BASE_PRICE + _AMPLITUDE * (pos / _RISE_DAYS)
        else:
            price = _BASE_PRICE + _AMPLITUDE * (
                1.0 - (pos - _RISE_DAYS) / (_CYCLE_LENGTH - _RISE_DAYS)
            )
        result.append(((start_date + datetime.timedelta(days=day)).isoformat(), round(price, 1)))
    return result


STATION_A = 1001
STATION_B = 1002

_START = "2020-01-01"
_3_CYCLES = _sawtooth_series(3.0, start=_START)  # 138 days
_5_CYCLES = _sawtooth_series(5.0, start=_START)  # 230 days
_16_CYCLES = _sawtooth_series(16.0, start=_START)  # 736 days, > 365 label rows


def _date_at_day(day: int, start: str = _START) -> str:
    return (datetime.date.fromisoformat(start) + datetime.timedelta(days=day)).isoformat()


# Day 120 is well into cycle 3, past all three confirmed peaks (2, 48, 94).
_DATE_D = _date_at_day(120)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def conn(tmp_path):
    c = open_db(tmp_path / "test.db")
    create_schema(c)
    yield c
    c.close()


def _add_station(conn, station_code: int) -> None:
    upsert_stations(conn, [{
        "station_code": station_code,
        "name": f"Station {station_code}",
        "address": f"{station_code} Main Street, Springwood",
        "suburb": "Springwood",
        "postcode": "2777",
        "brand": "Shell",
    }])


def _add_station_in_lga(conn, station_code: int, lga: str, brand: str = "Shell") -> None:
    upsert_stations(conn, [{
        "station_code": station_code,
        "name": f"Station {station_code}",
        "address": f"{station_code} Main Street, Springwood",
        "suburb": "Springwood",
        "postcode": "2777",
        "brand": brand,
    }])
    conn.execute(
        "UPDATE stations SET council = ?, brand = ? WHERE station_code = ?",
        (lga, brand, station_code),
    )
    conn.commit()


def _set_station_class(
    conn, station_code: int, date_d: str, cls: str, premium: int = 0
) -> None:
    upsert_station_class_rows(conn, [(station_code, date_d, cls, premium)])
    conn.commit()


def _add_prices(conn, station_code: int, series: list[tuple[str, float]]) -> None:
    upsert_daily_prices(conn, [(station_code, "E10", d, p) for d, p in series])
    conn.commit()


# ---------------------------------------------------------------------------
# compute_features — None-return cases
# ---------------------------------------------------------------------------

def test_no_station_price_returns_none(conn):
    """Returns None when station has no price on date_d."""
    _add_station(conn, STATION_A)
    # Prices exist but not on _DATE_D (series ends at day 119)
    _add_prices(conn, STATION_A, _3_CYCLES[: 120])  # days 0..119
    assert compute_features(conn, STATION_A, _DATE_D) is None


def test_insufficient_cycle_data_returns_none(conn):
    """Returns None when CycleDetector.detect returns None (< 2 peaks)."""
    _add_station(conn, STATION_A)
    # Only ~30 days of data — not enough for two peaks (first peak is around day 2,
    # but a second peak requires another 46 days minimum).
    short_series = _sawtooth_series(0.7, start=_START)  # ~32 days, only 1 peak
    _add_prices(conn, STATION_A, short_series)
    date_d = _date_at_day(len(short_series) - 1)
    assert compute_features(conn, STATION_A, date_d) is None


# ---------------------------------------------------------------------------
# compute_features — success path and feature contract
# ---------------------------------------------------------------------------

def test_feature_keys_match_documented_columns(conn):
    """Feature dict keys match FEATURE_COLUMNS exactly (catches accidental renames)."""
    _add_station(conn, STATION_A)
    _add_prices(conn, STATION_A, _3_CYCLES)
    features = compute_features(conn, STATION_A, _DATE_D)
    assert features is not None
    assert set(features.keys()) == set(FEATURE_COLUMNS)


def test_station_price_matches_daily_prices(conn):
    """station_price_cents reflects the actual price on date_d."""
    _add_station(conn, STATION_A)
    _add_prices(conn, STATION_A, _3_CYCLES)
    expected_price = dict(_3_CYCLES)[_DATE_D]
    features = compute_features(conn, STATION_A, _DATE_D)
    assert features is not None
    assert abs(features["station_price_cents"] - expected_price) < 0.05


def test_derived_features_are_consistent(conn):
    """station_minus_last_min/max are consistent with the other features."""
    _add_station(conn, STATION_A)
    _add_prices(conn, STATION_A, _3_CYCLES)
    features = compute_features(conn, STATION_A, _DATE_D)
    assert features is not None
    price = features["station_price_cents"]
    assert abs(features["station_minus_last_min_cents"] - (price - features["cycle_last_min_cents"])) < 1e-9
    assert abs(features["station_minus_last_max_cents"] - (price - features["cycle_last_max_cents"])) < 1e-9


def test_single_station_sydney_avg_delta_is_zero(conn):
    """With one station, station_minus_sydney_avg_cents must be 0."""
    _add_station(conn, STATION_A)
    _add_prices(conn, STATION_A, _3_CYCLES)
    features = compute_features(conn, STATION_A, _DATE_D)
    assert features is not None
    assert abs(features["station_minus_sydney_avg_cents"]) < 1e-9


# ---------------------------------------------------------------------------
# PIT (point-in-time) safety validation
# ---------------------------------------------------------------------------

def test_pit_safety(tmp_path):
    """Features computed with future data present must equal features without it.

    Procedure:
      1. Build a full DB with 5 cycles of prices (230 days).
      2. Compute f1 at DATE_D (day 120) — full DB; future data exists past day 120.
      3. Build a truncated DB with data only up to DATE_D (days 0..120 inclusive).
      4. Compute f2 at DATE_D from the truncated DB.
      5. Assert f1 == f2 within float tolerance.

    Any difference would mean a feature has a forward-looking dependency.
    """
    # --- Full DB (5 cycles = 230 days) ---
    conn_full = open_db(tmp_path / "full.db")
    create_schema(conn_full)
    _add_station(conn_full, STATION_A)
    _add_prices(conn_full, STATION_A, _5_CYCLES)

    f1 = compute_features(conn_full, STATION_A, _DATE_D)
    conn_full.close()

    # --- Truncated DB (days 0..120 inclusive) ---
    conn_trunc = open_db(tmp_path / "trunc.db")
    create_schema(conn_trunc)
    _add_station(conn_trunc, STATION_A)
    truncated = [row for row in _5_CYCLES if row[0] <= _DATE_D]
    _add_prices(conn_trunc, STATION_A, truncated)

    f2 = compute_features(conn_trunc, STATION_A, _DATE_D)
    conn_trunc.close()

    assert f1 is not None, "Full-DB features should not be None"
    assert f2 is not None, "Truncated-DB features should not be None"

    for key in FEATURE_COLUMNS:
        v1, v2 = f1[key], f2[key]
        if v1 is None and v2 is None:
            continue  # both absent (no classification data) — PIT-safe
        assert v1 is not None and v2 is not None, (
            f"PIT violation: feature '{key}' is None in one DB but not the other "
            f"(full={v1}, truncated={v2})"
        )
        assert abs(v1 - v2) < 1e-6, (
            f"PIT violation: feature '{key}' differs between full and truncated DB "
            f"(full={v1}, truncated={v2})"
        )


# ---------------------------------------------------------------------------
# Pre-built CycleDetector passthrough
# ---------------------------------------------------------------------------

def test_prebuilt_cycle_detector_matches_standalone(conn):
    """Passing a pre-built CycleDetector gives the same result as building inline."""
    from fuel_signal.cycle import CycleDetector
    from fuel_signal.db import average_price_series

    _add_station(conn, STATION_A)
    _add_prices(conn, STATION_A, _3_CYCLES)

    f_standalone = compute_features(conn, STATION_A, _DATE_D)
    cd = CycleDetector(average_price_series(conn))
    f_batched = compute_features(conn, STATION_A, _DATE_D, cycle_detector=cd)

    assert f_standalone is not None
    assert f_batched is not None
    for key in FEATURE_COLUMNS:
        v1, v2 = f_standalone[key], f_batched[key]
        if v1 is None and v2 is None:
            continue
        assert v1 is not None and v2 is not None, (
            f"Feature '{key}' is None in one path but not the other"
        )
        assert abs(v1 - v2) < 1e-9


# ---------------------------------------------------------------------------
# assemble_feature_rows
# ---------------------------------------------------------------------------

def test_assembler_drops_none_rows(conn):
    """Rows where compute_features returns None are excluded from the output."""
    _add_station(conn, STATION_A)
    _add_station(conn, STATION_B)

    # STATION_A: full 3 cycles — will produce label + feature rows
    _add_prices(conn, STATION_A, _3_CYCLES)

    # STATION_B: 30 days only — too short for cycle detection; labels will be
    # produced (assemble_training_rows requires only forward data, not cycle data)
    # but compute_features will return None for all STATION_B rows.
    short = _sawtooth_series(0.7, start=_START)
    _add_prices(conn, STATION_B, short)

    # min_rows_per_station=0 bypasses the row-count filter; this test targets
    # the None-row dropping path only.
    df = assemble_feature_rows(conn, station_codes=[STATION_A, STATION_B], min_rows_per_station=0)
    # All rows in output must be for STATION_A only
    assert len(df) > 0
    assert set(df["station_code"].unique()) == {STATION_A}


def test_assembler_columns(conn):
    """Output DataFrame has label columns followed by all FEATURE_COLUMNS."""
    _add_station(conn, STATION_A)
    _add_prices(conn, STATION_A, _3_CYCLES)

    df = assemble_feature_rows(conn, station_codes=[STATION_A], min_rows_per_station=0)
    assert len(df) > 0
    for col in FEATURE_COLUMNS:
        assert col in df.columns, f"Missing feature column: {col}"
    # Label columns present
    for col in ("station_code", "price_date", "today_price_cents", "future_min_cents", "label"):
        assert col in df.columns


def test_assembler_empty_station_list(conn):
    """Empty station_codes returns an empty DataFrame with correct columns."""
    df = assemble_feature_rows(conn, station_codes=[])
    assert len(df) == 0
    assert set(FEATURE_COLUMNS).issubset(set(df.columns))


def test_assembler_excludes_stations_below_min_rows(conn):
    """Stations with fewer label rows than min_rows_per_station are excluded.

    STATION_A gets 3 cycles (138 days) → ~40 label rows with default params.
    STATION_B gets 5 cycles (230 days) → ~133 label rows.
    Setting min_rows_per_station=100 should admit only STATION_B.
    """
    _add_station(conn, STATION_A)
    _add_station(conn, STATION_B)
    _add_prices(conn, STATION_A, _3_CYCLES)   # ~40 label rows → below threshold
    _add_prices(conn, STATION_B, _5_CYCLES)   # ~133 label rows → above threshold

    df = assemble_feature_rows(
        conn,
        station_codes=[STATION_A, STATION_B],
        min_rows_per_station=100,
    )
    assert len(df) > 0
    assert set(df["station_code"].unique()) == {STATION_B}


def test_assembler_default_min_rows_enforces_365(conn):
    """Default min_rows_per_station applies MIN_TRAINING_ROWS_PER_STATION (365).

    STATION_A gets 3 cycles (138 days) → label-row count well below 365.
    STATION_B gets 16 cycles (736 days) → label-row count well above 365.
    Calling without min_rows_per_station must rely on the default and admit
    only STATION_B, pinning the constant + default-path behaviour.
    """
    assert MIN_TRAINING_ROWS_PER_STATION == 365

    _add_station(conn, STATION_A)
    _add_station(conn, STATION_B)
    _add_prices(conn, STATION_A, _3_CYCLES)
    _add_prices(conn, STATION_B, _16_CYCLES)

    df = assemble_feature_rows(conn, station_codes=[STATION_A, STATION_B])
    assert len(df) > 0
    assert set(df["station_code"].unique()) == {STATION_B}
    assert (df["station_code"] == STATION_B).sum() >= MIN_TRAINING_ROWS_PER_STATION


def test_assembler_rejects_negative_min_rows(conn):
    """assemble_feature_rows raises ValueError for negative min_rows_per_station."""
    import pytest

    with pytest.raises(ValueError, match="min_rows_per_station must be >= 0"):
        assemble_feature_rows(conn, min_rows_per_station=-1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_writes_csv(conn, tmp_path):
    """CLI happy path: exits 0 and writes a CSV with expected columns."""
    from click.testing import CliRunner

    from fuel_signal.features import main

    _add_station(conn, STATION_A)
    _add_prices(conn, STATION_A, _3_CYCLES)

    out_csv = tmp_path / "features.csv"
    result = CliRunner().invoke(main, [
        "--db", str(tmp_path / "test.db"),
        "--output", str(out_csv),
        "--min-rows", "0",
    ])
    assert result.exit_code == 0, result.output
    assert out_csv.exists()

    import pandas as pd
    df = pd.read_csv(out_csv)
    assert len(df) > 0
    for col in FEATURE_COLUMNS:
        assert col in df.columns


# ---------------------------------------------------------------------------
# LGA mean and brand mean features
# ---------------------------------------------------------------------------

# Synthetic station codes for LGA/brand tests
LGA_A = 2001   # Competitive, LGA "Alpha"
LGA_B = 2002   # Competitive, LGA "Alpha"
LGA_C = 2003   # Competitive, LGA "Alpha"
LGA_STICKY = 2004  # Sticky, LGA "Alpha"
LGA_DISC = 2005    # Discount, LGA "Alpha"
LGA_OTHER = 2006   # Competitive, LGA "Beta"

BRAND_CHEAP = 3001  # Cheap brand, 3 stations
BRAND_CHEAP2 = 3002
BRAND_CHEAP3 = 3003
BRAND_STICKY = 3004  # Sticky, same brand

# Dates within the 3-cycle sawtooth series (days 0..137); use day 120 from _DATE_D.
_LGA_DATE = _DATE_D  # same date used by existing cycle tests


def _setup_lga_alpha_3comp(conn) -> None:
    """Three Competitive stations in LGA 'Alpha' with known prices on _LGA_DATE."""
    for code, offset in [(LGA_A, 0.0), (LGA_B, 5.0), (LGA_C, -5.0)]:
        _add_station_in_lga(conn, code, "Alpha")
        series = [(d, p + offset) for d, p in _3_CYCLES]
        _add_prices(conn, code, series)
        _set_station_class(conn, code, _LGA_DATE, "Competitive")


def test_lga_mean_excludes_sticky(conn):
    """LGA mean excludes Sticky stations; reflects only Competitive+Discount prices."""
    _setup_lga_alpha_3comp(conn)

    # Add a Sticky station in the same LGA with a much higher price (+50c)
    _add_station_in_lga(conn, LGA_STICKY, "Alpha")
    sticky_series = [(d, p + 50.0) for d, p in _3_CYCLES]
    _add_prices(conn, LGA_STICKY, sticky_series)
    _set_station_class(conn, LGA_STICKY, _LGA_DATE, "Sticky")

    from fuel_signal import db as _db
    from fuel_signal.features import _lga_mean_on_date
    fid = _db.fuel_type_id(conn, "E10")

    mean_with_sticky_excluded = _lga_mean_on_date(conn, _LGA_DATE, "Alpha", fid)
    assert mean_with_sticky_excluded is not None

    # Naive mean (all stations) would be pulled upward by the Sticky station.
    # The classifier-filtered mean must equal AVG of the 3 Competitive prices.
    price_map = dict(_3_CYCLES)
    comp_avg = (price_map[_LGA_DATE] + price_map[_LGA_DATE] + 5.0 +
                price_map[_LGA_DATE] - 5.0) / 3
    assert abs(mean_with_sticky_excluded - comp_avg) < 0.1


def test_lga_mean_includes_discount(conn):
    """Discount stations are included in the LGA mean (blended policy)."""
    _setup_lga_alpha_3comp(conn)

    # Replace one Competitive classification with Discount
    _set_station_class(conn, LGA_C, _LGA_DATE, "Discount")

    from fuel_signal import db as _db
    from fuel_signal.features import _lga_mean_on_date
    fid = _db.fuel_type_id(conn, "E10")

    mean = _lga_mean_on_date(conn, _LGA_DATE, "Alpha", fid)
    assert mean is not None

    # All three stations (1 Competitive + 1 Discount + 1 Competitive) included.
    price_map = dict(_3_CYCLES)
    expected = (price_map[_LGA_DATE] + price_map[_LGA_DATE] + 5.0 +
                price_map[_LGA_DATE] - 5.0) / 3
    assert abs(mean - expected) < 0.1


def test_lga_mean_null_floor(conn):
    """LGA mean is NULL when fewer than 3 non-Sticky stations contribute."""
    # Only 2 non-Sticky stations
    for code, offset in [(LGA_A, 0.0), (LGA_B, 5.0)]:
        _add_station_in_lga(conn, code, "Alpha")
        _add_prices(conn, code, [(d, p + offset) for d, p in _3_CYCLES])
        _set_station_class(conn, code, _LGA_DATE, "Competitive")

    # Third station present but classified Sticky
    _add_station_in_lga(conn, LGA_STICKY, "Alpha")
    _add_prices(conn, LGA_STICKY, [(d, p + 30.0) for d, p in _3_CYCLES])
    _set_station_class(conn, LGA_STICKY, _LGA_DATE, "Sticky")

    from fuel_signal import db as _db
    from fuel_signal.features import _lga_mean_on_date
    fid = _db.fuel_type_id(conn, "E10")

    assert _lga_mean_on_date(conn, _LGA_DATE, "Alpha", fid) is None


def test_lga_mean_zero_competitive_gap(conn):
    """LGA mean is NULL when no station_class rows exist for that LGA/date."""
    _setup_lga_alpha_3comp(conn)
    # Delete all station_class rows → zero-Competitive gap
    conn.execute("DELETE FROM station_class")
    conn.commit()

    from fuel_signal import db as _db
    from fuel_signal.features import _lga_mean_on_date
    fid = _db.fuel_type_id(conn, "E10")

    assert _lga_mean_on_date(conn, _LGA_DATE, "Alpha", fid) is None


def test_brand_mean_excludes_sticky(conn):
    """Brand mean excludes Sticky stations (Sydney-wide)."""
    # 3 Competitive stations of brand "TestBrand"
    for code, offset in [(BRAND_CHEAP, 0.0), (BRAND_CHEAP2, 3.0), (BRAND_CHEAP3, -3.0)]:
        _add_station_in_lga(conn, code, "Alpha", brand="TestBrand")
        _add_prices(conn, code, [(d, p + offset) for d, p in _3_CYCLES])
        _set_station_class(conn, code, _LGA_DATE, "Competitive")

    # 1 Sticky station of same brand (+40c)
    _add_station_in_lga(conn, BRAND_STICKY, "Beta", brand="TestBrand")
    _add_prices(conn, BRAND_STICKY, [(d, p + 40.0) for d, p in _3_CYCLES])
    _set_station_class(conn, BRAND_STICKY, _LGA_DATE, "Sticky")

    from fuel_signal import db as _db
    from fuel_signal.features import _brand_mean_on_date
    fid = _db.fuel_type_id(conn, "E10")

    mean = _brand_mean_on_date(conn, _LGA_DATE, "TestBrand", fid)
    assert mean is not None

    price_map = dict(_3_CYCLES)
    comp_avg = (price_map[_LGA_DATE] + price_map[_LGA_DATE] + 3.0 +
                price_map[_LGA_DATE] - 3.0) / 3
    assert abs(mean - comp_avg) < 0.1


def test_brand_mean_is_sydney_wide(conn):
    """Brand mean aggregates across all LGAs, not per-LGA-Brand."""
    # 2 Competitive stations in LGA "Alpha" + 1 in LGA "Beta" — same brand.
    # If the mean were per-LGA-Brand, "Alpha" would have 2 (below floor) → NULL.
    # Sydney-wide has 3 → not NULL.
    for code, lga in [(BRAND_CHEAP, "Alpha"), (BRAND_CHEAP2, "Alpha"), (BRAND_CHEAP3, "Beta")]:
        _add_station_in_lga(conn, code, lga, brand="TestBrand")
        _add_prices(conn, code, _3_CYCLES)
        _set_station_class(conn, code, _LGA_DATE, "Competitive")

    from fuel_signal import db as _db
    from fuel_signal.features import _brand_mean_on_date
    fid = _db.fuel_type_id(conn, "E10")

    mean = _brand_mean_on_date(conn, _LGA_DATE, "TestBrand", fid)
    assert mean is not None, "Sydney-wide brand mean with 3 stations should not be NULL"


def test_snapshot_invariant_all_competitive(conn):
    """All-Competitive LGA: classifier-filtered mean equals naive all-station mean.

    Equivalent to the 'Blue Mountains' snapshot fixture from the issue spec.
    An LGA with no Sticky/Discount stations must be unchanged after classification
    filtering is applied.
    """
    _setup_lga_alpha_3comp(conn)

    from fuel_signal import db as _db
    from fuel_signal.features import _lga_mean_on_date

    fid = _db.fuel_type_id(conn, "E10")
    filtered_mean = _lga_mean_on_date(conn, _LGA_DATE, "Alpha", fid)

    # Naive mean: AVG of all stations in LGA Alpha (no filter)
    row = conn.execute(
        "SELECT AVG(dp.price_decicents)"
        " FROM daily_prices dp JOIN stations s ON dp.station_code = s.station_code"
        " WHERE s.council = 'Alpha' AND dp.fuel_type_id = ? AND dp.price_date = ?",
        (fid, int(_LGA_DATE.replace("-", ""))),
    ).fetchone()
    naive_mean = row[0] / 10 if row and row[0] is not None else None

    assert filtered_mean is not None
    assert naive_mean is not None
    assert abs(filtered_mean - naive_mean) < 1e-9, (
        f"All-Competitive LGA mean drifted after classifier filter: "
        f"filtered={filtered_mean}, naive={naive_mean}"
    )


def test_competitive_anchor_within_3c(conn):
    """LGA with Sticky stations: mean stays within 3c of the Competitive anchor.

    Analogous to the Central Coast acceptance criterion from the issue spec.
    """
    # 3 Competitive stations at base price, 2 Sticky at base+30c
    price_map = dict(_3_CYCLES)
    base = price_map[_LGA_DATE]
    for code in [LGA_A, LGA_B, LGA_C]:
        _add_station_in_lga(conn, code, "Alpha")
        _add_prices(conn, code, _3_CYCLES)
        _set_station_class(conn, code, _LGA_DATE, "Competitive")

    for code in [LGA_STICKY, LGA_OTHER]:
        _add_station_in_lga(conn, code, "Alpha")
        _add_prices(conn, code, [(d, p + 30.0) for d, p in _3_CYCLES])
        _set_station_class(conn, code, _LGA_DATE, "Sticky")

    from fuel_signal import db as _db
    from fuel_signal.features import _lga_mean_on_date
    fid = _db.fuel_type_id(conn, "E10")

    lga_mean = _lga_mean_on_date(conn, _LGA_DATE, "Alpha", fid)
    competitive_anchor = base  # all 3 Competitive stations have the same base price

    assert lga_mean is not None
    assert abs(lga_mean - competitive_anchor) <= 3.0, (
        f"LGA mean {lga_mean:.2f}c is not within ±3c of competitive anchor {competitive_anchor:.2f}c"
    )


def test_assembler_includes_lga_brand_columns(conn):
    """assemble_feature_rows output contains the new lga/brand feature columns."""
    _add_station(conn, STATION_A)
    _add_prices(conn, STATION_A, _3_CYCLES)

    df = assemble_feature_rows(conn, station_codes=[STATION_A], min_rows_per_station=0)
    assert len(df) > 0
    for col in ("lga_mean_cents", "station_minus_lga_mean_cents",
                "brand_mean_cents", "station_minus_brand_mean_cents"):
        assert col in df.columns, f"Missing column: {col}"


def test_assembler_lga_mean_matches_compute_features(conn):
    """assemble_feature_rows lga_mean_cents matches compute_features for same row.

    Verifies the bulk-cache path and per-row path produce the same result for
    a station with classification data.
    """
    _setup_lga_alpha_3comp(conn)
    # Reuse STATION_A infrastructure for the cycle detector (same sawtooth series)
    _add_station(conn, STATION_A)
    _add_prices(conn, STATION_A, _3_CYCLES)
    # Set LGA and classification on STATION_A too
    conn.execute(
        "UPDATE stations SET council = 'Alpha', brand = 'Shell' WHERE station_code = ?",
        (STATION_A,),
    )
    _set_station_class(conn, STATION_A, _LGA_DATE, "Competitive")
    conn.commit()

    per_row = compute_features(conn, STATION_A, _LGA_DATE)
    assert per_row is not None

    df = assemble_feature_rows(conn, station_codes=[STATION_A], min_rows_per_station=0)
    matching = df[df["price_date"] == _LGA_DATE]
    assert len(matching) == 1

    row = matching.iloc[0]
    if per_row["lga_mean_cents"] is None:
        assert row["lga_mean_cents"] != row["lga_mean_cents"]  # NaN
    else:
        assert abs(row["lga_mean_cents"] - per_row["lga_mean_cents"]) < 0.1


# ---------------------------------------------------------------------------
# stickiness_score feature
# ---------------------------------------------------------------------------

def test_stickiness_score_nan_when_no_classification(conn):
    """stickiness_score is None when station has no station_class row for date."""
    _add_station(conn, STATION_A)
    _add_prices(conn, STATION_A, _3_CYCLES)
    features = compute_features(conn, STATION_A, _DATE_D)
    assert features is not None
    assert features["stickiness_score"] is None


def test_stickiness_score_value(conn):
    """stickiness_score equals median_premium_decicents / 10 for a present (station, date)."""
    _add_station(conn, STATION_A)
    _add_prices(conn, STATION_A, _3_CYCLES)
    _set_station_class(conn, STATION_A, _DATE_D, "Competitive", 150)  # 150 decicents = 15.0 cents
    features = compute_features(conn, STATION_A, _DATE_D)
    assert features is not None
    assert abs(features["stickiness_score"] - 15.0) < 1e-9


def test_stickiness_score_pit_safety(tmp_path):
    """stickiness_score cannot read station_class rows where snapshot_date > target_date."""
    conn = open_db(tmp_path / "pit.db")
    create_schema(conn)
    _add_station(conn, STATION_A)
    _add_prices(conn, STATION_A, _3_CYCLES)

    future_date = _date_at_day(130)  # after _DATE_D (day 120)
    _set_station_class(conn, STATION_A, future_date, "Sticky", 200)

    features = compute_features(conn, STATION_A, _DATE_D)
    assert features is not None
    assert features["stickiness_score"] is None, (
        "stickiness_score must be None at _DATE_D when only a future station_class row exists"
    )
    conn.close()


def test_assembler_stickiness_score_matches_compute_features(conn):
    """assemble_feature_rows stickiness_score matches compute_features for same row."""
    _add_station(conn, STATION_A)
    _add_prices(conn, STATION_A, _3_CYCLES)
    _set_station_class(conn, STATION_A, _DATE_D, "Competitive", 120)  # 12.0 cents

    per_row = compute_features(conn, STATION_A, _DATE_D)
    assert per_row is not None
    assert abs(per_row["stickiness_score"] - 12.0) < 1e-9

    df = assemble_feature_rows(conn, station_codes=[STATION_A], min_rows_per_station=0)
    matching = df[df["price_date"] == _DATE_D]
    assert len(matching) == 1
    assert abs(matching.iloc[0]["stickiness_score"] - 12.0) < 1e-9


def test_assembler_brand_mean_matches_compute_features(conn):
    """assemble_feature_rows brand_mean_cents matches compute_features for same row.

    Mirrors test_assembler_lga_mean_matches_compute_features but for the brand
    bulk-cache path.
    """
    # Three Competitive stations of brand "TestBrand" (clears the 3-station floor)
    for code, offset in [(BRAND_CHEAP, 0.0), (BRAND_CHEAP2, 3.0), (BRAND_CHEAP3, -3.0)]:
        _add_station_in_lga(conn, code, "Alpha", brand="TestBrand")
        _add_prices(conn, code, [(d, p + offset) for d, p in _3_CYCLES])
        _set_station_class(conn, code, _LGA_DATE, "Competitive")

    # Also need STATION_A for the cycle detector series (sydney avg)
    _add_station(conn, STATION_A)
    _add_prices(conn, STATION_A, _3_CYCLES)
    conn.execute(
        "UPDATE stations SET council = 'Alpha', brand = 'TestBrand' WHERE station_code = ?",
        (STATION_A,),
    )
    _set_station_class(conn, STATION_A, _LGA_DATE, "Competitive")
    conn.commit()

    per_row = compute_features(conn, STATION_A, _LGA_DATE)
    assert per_row is not None

    df = assemble_feature_rows(conn, station_codes=[STATION_A], min_rows_per_station=0)
    matching = df[df["price_date"] == _LGA_DATE]
    assert len(matching) == 1

    row = matching.iloc[0]
    if per_row["brand_mean_cents"] is None:
        assert row["brand_mean_cents"] != row["brand_mean_cents"]  # NaN
    else:
        assert abs(row["brand_mean_cents"] - per_row["brand_mean_cents"]) < 0.1
    if per_row["station_minus_brand_mean_cents"] is None:
        assert row["station_minus_brand_mean_cents"] != row["station_minus_brand_mean_cents"]
    else:
        assert abs(row["station_minus_brand_mean_cents"] - per_row["station_minus_brand_mean_cents"]) < 0.1
