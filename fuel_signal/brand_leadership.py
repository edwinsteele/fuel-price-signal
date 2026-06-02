"""Brand-axis trough event feature lookups (Phase 5).

Mirrors lga_leadership.py for the brand dimension.  For each qualifying brand
(≥ MIN_BRAND_SITES distinct stations), emits one feature column
days_since_trough_entry_<brand_slug> broadcast per date across all rows.

Trough detection reuses detect_trough_events from lga_leadership.  Series is
the brand's daily median price (non-Sticky stations, ≥ MIN_STATION_FLOOR per
day).  PIT contract is identical to the LGA version: centered smoothing at
label date d only sees prices ≤ d.
"""

from __future__ import annotations

import logging
import re
import sqlite3

import numpy as np

from fuel_signal.config import MIN_BRAND_SITES
from fuel_signal.dates import date_to_int as _date_to_int
from fuel_signal.dates import int_to_date as _int_to_date
from fuel_signal.db import distinct_brands, fuel_type_id
from fuel_signal.lga_leadership import (
    MIN_STATION_FLOOR,
    TROUGH_SMOOTH_WINDOW,
    detect_trough_events,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def brand_slug(name: str) -> str:
    """'7-Eleven' → '7_eleven', 'EG Ampol' → 'eg_ampol'"""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def qualifying_brands(
    conn: sqlite3.Connection,
    min_sites: int = MIN_BRAND_SITES,
) -> list[str]:
    """Return sorted list of brand names with ≥ min_sites distinct stations."""
    return distinct_brands(conn, min_stations=min_sites)


def brand_feature_columns(
    conn: sqlite3.Connection,
    min_sites: int = MIN_BRAND_SITES,
) -> list[str]:
    """Return ordered column names for all qualifying brand trough features."""
    return [
        f"days_since_trough_entry_{brand_slug(b)}"
        for b in qualifying_brands(conn, min_sites)
    ]


# ---------------------------------------------------------------------------
# Brand series loader
# ---------------------------------------------------------------------------

def _load_brand_series(
    conn: sqlite3.Connection,
    fid: int,
    qualifying_brand_list: list[str],
) -> dict[str, dict[int, float]]:
    """Return {brand: {date_int: median_decicents}} for qualifying brands.

    Only non-Sticky stations are included (same exclusion as LGA mean).
    Days where fewer than MIN_STATION_FLOOR non-Sticky prices are available
    for a brand are omitted — their median would be unreliable.

    PIT-safety: station_class JOIN uses price_date = snapshot_date, so each
    day's Sticky classification uses only data available on that day.
    """
    if not qualifying_brand_list:
        return {}

    brand_ph = ", ".join(["?"] * len(qualifying_brand_list))
    sql = (
        "SELECT dp.price_date, s.brand, dp.price_decicents"
        " FROM daily_prices dp"
        " JOIN stations s ON dp.station_code = s.station_code"
        " JOIN station_class sc ON dp.station_code = sc.station_code"
        "   AND dp.price_date = sc.snapshot_date"
        f" WHERE dp.fuel_type_id = ? AND sc.class != 'Sticky'"
        f"   AND s.brand IN ({brand_ph})"
    )

    # Accumulate all non-Sticky prices per (date, brand)
    price_lists: dict[tuple[int, str], list[float]] = {}
    for date_int, brand, price_dc in conn.execute(sql, [fid, *qualifying_brand_list]):
        key = (int(date_int), str(brand))
        if key not in price_lists:
            price_lists[key] = []
        price_lists[key].append(float(price_dc))

    brand_series: dict[str, dict[int, float]] = {}
    for (date_int, brand), prices in price_lists.items():
        if len(prices) < MIN_STATION_FLOOR:
            continue
        if brand not in brand_series:
            brand_series[brand] = {}
        brand_series[brand][date_int] = float(np.median(prices))

    return brand_series


# ---------------------------------------------------------------------------
# Feature lookup builder
# ---------------------------------------------------------------------------

def build_brand_trough_lookups(
    conn: sqlite3.Connection,
    qualifying_brand_list: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Build per-brand sorted arrays of trough-entry date integers (YYYYMMDD).

    NOT PIT-safe — use compute_pit_strict_days_since_trough_brand for features.
    Returns {brand_name: np.ndarray[int]} — empty arrays where no troughs detected.
    """
    if qualifying_brand_list is None:
        qualifying_brand_list = qualifying_brands(conn)

    fid = fuel_type_id(conn, "E10")
    brand_series = _load_brand_series(conn, fid, qualifying_brand_list)

    lookups: dict[str, np.ndarray] = {}
    for brand, series_dict in brand_series.items():
        date_ints = sorted(series_dict.keys())
        prices = np.array([series_dict[d] for d in date_ints], dtype=float)
        trough_idx = detect_trough_events(prices)
        lookups[brand] = np.array([date_ints[i] for i in trough_idx], dtype=int)
        logger.debug(
            "Trough lookup brand=%s: %d dates, %d trough events",
            brand, len(date_ints), len(trough_idx),
        )

    return lookups


def compute_pit_strict_days_since_trough_brand(
    conn: sqlite3.Connection,
    label_date_strs: list[str],
    qualifying_brand_list: list[str],
) -> dict[tuple[str, str], int | None]:
    """PIT-safe days_since_trough_entry per (label_date, brand_name).

    For each unique label_date d and each brand in qualifying_brand_list,
    runs detect_trough_events on that brand's prices restricted to [≤ d].
    The most recent trough in the restricted detection gives days_since.

    Contract mirrors compute_pit_strict_days_since_trough in lga_leadership:
    centered smoothing and snap-to-argmin only see data available on or before d,
    so the recorded trough date never depends on future prices.

    Returns {(label_date_str, brand_name): days_since}.  None where the
    restricted detection finds no troughs or the brand has no data on/before d.
    """
    if not qualifying_brand_list:
        return {}

    fid = fuel_type_id(conn, "E10")
    brand_series = _load_brand_series(conn, fid, qualifying_brand_list)

    brand_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for brand, series_dict in brand_series.items():
        date_ints = sorted(series_dict.keys())
        brand_arrays[brand] = (
            np.array(date_ints, dtype=int),
            np.array([series_dict[d] for d in date_ints], dtype=float),
        )

    label_date_ints: dict[str, int] = {d: _date_to_int(d) for d in label_date_strs}
    label_date_objs = {d: _int_to_date(label_date_ints[d]) for d in label_date_strs}

    result: dict[tuple[str, str], int | None] = {}

    for brand in qualifying_brand_list:
        if brand not in brand_arrays:
            for d_str in label_date_strs:
                result[(d_str, brand)] = None
            continue
        dates_arr, prices_arr = brand_arrays[brand]

        for d_str in label_date_strs:
            d_int = label_date_ints[d_str]
            cutoff = int(np.searchsorted(dates_arr, d_int, side="right"))
            if cutoff < TROUGH_SMOOTH_WINDOW * 2:
                result[(d_str, brand)] = None
                continue
            trough_idx = detect_trough_events(prices_arr[:cutoff])
            if len(trough_idx) == 0:
                result[(d_str, brand)] = None
                continue
            last_trough_date_int = int(dates_arr[trough_idx[-1]])
            result[(d_str, brand)] = (
                label_date_objs[d_str] - _int_to_date(last_trough_date_int)
            ).days

    return result
