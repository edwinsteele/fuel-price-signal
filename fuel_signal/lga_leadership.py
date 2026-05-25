"""LGA leadership scoring and per-LGA trough event feature lookups (Phase 4).

Two outputs:
1. lga_leadership table — offline leadership scores at weekly snapshots.
   Mirrors the station_class pattern: PIT-safe, 730d trailing window,
   rest-of-Sydney weighted-mean anchor (excludes L when scoring L).

2. build_lga_trough_lookups() — for features.py.
   Returns per-LGA arrays of trough-entry date integers used to compute
   days_since_trough_entry_<lga_slug> over the full training history.
"""

from __future__ import annotations

import logging
import pathlib
import re
import sqlite3
from datetime import date, timedelta

import click
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from fuel_signal.db import (
    DEFAULT_DB_PATH,
    create_schema,
    delete_lga_leadership_for_date,
    fuel_type_id,
    open_db,
    upsert_lga_leadership_rows,
)
from fuel_signal.postcode_council import SYDNEY_METRO_COUNCILS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TROUGH_SMOOTH_WINDOW: int = 7
TROUGH_MIN_SPACING: int = 18
TROUGH_SNAP_RADIUS: int = 5
LEADERSHIP_WINDOW_DAYS: int = 730
EVENT_MATCH_MAX_LAG: int = 60
MIN_STATION_FLOOR: int = 3

# Stable sorted list of all Sydney metro LGAs — defines the feature schema.
LGA_FEATURE_COUNCILS: list[str] = sorted(SYDNEY_METRO_COUNCILS)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def lga_slug(name: str) -> str:
    """'Canterbury-Bankstown' → 'canterbury_bankstown'"""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def lga_feature_columns() -> list[str]:
    """Return ordered list of days_since_trough_entry_<lga> column names."""
    return [f"days_since_trough_entry_{lga_slug(lga)}" for lga in LGA_FEATURE_COUNCILS]


def _date_to_int(s: str) -> int:
    return int(s[:10].replace("-", ""))


def _date_from_int(v: int) -> str:
    s = str(v)
    return f"{s[:4]}-{s[4:6]}-{s[6:]}"


def _int_to_date(v: int) -> date:
    s = str(v)
    return date(int(s[:4]), int(s[4:6]), int(s[6:]))


# ---------------------------------------------------------------------------
# Trough detection
# ---------------------------------------------------------------------------

def detect_trough_events(
    prices: np.ndarray,
    min_spacing: int = TROUGH_MIN_SPACING,
    smooth_window: int = TROUGH_SMOOTH_WINDOW,
    snap_radius: int = TROUGH_SNAP_RADIUS,
) -> np.ndarray:
    """Return sorted array of trough-entry indices into prices.

    Method: centered rolling mean → find_peaks on negated series → snap each
    event to the raw argmin within ±snap_radius.  Matches the method used in
    experiments/trough_weakness/run.py (min_spacing is 18d here vs 10d there).

    PIT-safe: caller passes series truncated to ≤ snapshot_date, so centered
    smoothing only reads available data (min_periods=1 handles the trailing edge).
    Returns empty array when fewer than smooth_window*2 observations.
    """
    if len(prices) < smooth_window * 2:
        return np.array([], dtype=int)

    smooth = (
        pd.Series(prices)
        .rolling(smooth_window, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )
    trough_idx, _ = find_peaks(-smooth, distance=min_spacing)
    if len(trough_idx) == 0:
        return np.array([], dtype=int)

    snapped = np.empty_like(trough_idx)
    for i, t in enumerate(trough_idx):
        lo = max(0, t - snap_radius)
        hi = min(len(prices), t + snap_radius + 1)
        snapped[i] = lo + int(np.argmin(prices[lo:hi]))
    return np.unique(snapped)


# ---------------------------------------------------------------------------
# Event matching
# ---------------------------------------------------------------------------

def _match_events(
    lga_dates: list[date],
    anchor_dates: list[date],
    max_lag: int = EVENT_MATCH_MAX_LAG,
) -> list[int]:
    """Match each anchor trough event to the nearest LGA trough event.

    Returns lead_days for each matched pair.  Positive = LGA fires first (leads).
    Only pairs within ±max_lag days are considered.
    """
    leads = []
    for a in anchor_dates:
        best: int | None = None
        best_abs = max_lag + 1
        for lg in lga_dates:
            lead = (a - lg).days
            if abs(lead) <= max_lag and abs(lead) < best_abs:
                best = lead
                best_abs = abs(lead)
        if best is not None:
            leads.append(best)
    return leads


# ---------------------------------------------------------------------------
# LGA daily mean query helpers
# ---------------------------------------------------------------------------

def _load_lga_sums(
    conn: sqlite3.Connection,
    fid: int,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[tuple[int, str], tuple[float, int]]:
    """Return {(date_int, lga): (sum_decicents, n_stations)} for non-Sticky ≥-floor LGAs.

    Sticky exclusion is PIT-safe: joins station_class on price_date = snapshot_date,
    so each day uses that day's classification.
    """
    where = [
        "dp.fuel_type_id = ?",
        "sc.class != 'Sticky'",
        "s.council IS NOT NULL",
    ]
    params: list = [fid]

    if start_date is not None:
        where.append("dp.price_date >= ?")
        params.append(_date_to_int(start_date))
    if end_date is not None:
        where.append("dp.price_date <= ?")
        params.append(_date_to_int(end_date))

    params.append(MIN_STATION_FLOOR)

    sql = (
        "SELECT dp.price_date, s.council, SUM(dp.price_decicents), COUNT(*)"
        " FROM daily_prices dp"
        " JOIN stations s ON dp.station_code = s.station_code"
        " JOIN station_class sc ON dp.station_code = sc.station_code"
        "   AND dp.price_date = sc.snapshot_date"
        f" WHERE {' AND '.join(where)}"
        " GROUP BY dp.price_date, s.council"
        " HAVING COUNT(*) >= ?"
    )
    return {
        (int(r[0]), str(r[1])): (float(r[2]), int(r[3]))
        for r in conn.execute(sql, params)
    }


# ---------------------------------------------------------------------------
# Feature lookup builder (for features.py)
# ---------------------------------------------------------------------------

def build_lga_trough_lookups(
    conn: sqlite3.Connection,
) -> dict[str, np.ndarray]:
    """Build per-LGA sorted arrays of trough-entry date integers (YYYYMMDD).

    Covers the full available history with PIT-safe Sticky exclusion.
    Called once per features.py batch; results used for O(log n) date lookups.

    Returns {lga_name: np.ndarray[int]} — empty arrays where no troughs detected.
    """
    fid = fuel_type_id(conn, "E10")
    sums = _load_lga_sums(conn, fid)

    by_lga: dict[str, dict[int, float]] = {}
    for (date_int, lga), (s, n) in sums.items():
        if lga not in by_lga:
            by_lga[lga] = {}
        by_lga[lga][date_int] = s / n

    lookups: dict[str, np.ndarray] = {}
    for lga, series_dict in by_lga.items():
        date_ints = sorted(series_dict.keys())
        prices = np.array([series_dict[d] for d in date_ints], dtype=float)
        trough_idx = detect_trough_events(prices)
        lookups[lga] = np.array([date_ints[i] for i in trough_idx], dtype=int)
        logger.debug(
            "Trough lookup %s: %d dates, %d trough events", lga, len(date_ints), len(trough_idx)
        )

    return lookups


# ---------------------------------------------------------------------------
# Leadership scoring (for lga_leadership table)
# ---------------------------------------------------------------------------

def score_leadership_snapshot(
    conn: sqlite3.Connection,
    snapshot_date: str,
    fuel_code: str = "E10",
    window_days: int = LEADERSHIP_WINDOW_DAYS,
) -> int:
    """Compute and store lga_leadership rows for snapshot_date.

    Window: [snapshot_date - window_days, snapshot_date - 1].
    Existing rows for snapshot_date are replaced (idempotent).
    Returns number of LGA rows written.
    """
    if window_days <= 0:
        raise ValueError("window_days must be positive")

    snap = date.fromisoformat(snapshot_date)
    start = (snap - timedelta(days=window_days)).isoformat()
    end = (snap - timedelta(days=1)).isoformat()

    fid = fuel_type_id(conn, fuel_code)
    sums = _load_lga_sums(conn, fid, start_date=start, end_date=end)

    delete_lga_leadership_for_date(conn, snapshot_date)

    if not sums:
        logger.warning("score_leadership_snapshot %s: no data in window [%s, %s]", snapshot_date, start, end)
        conn.commit()
        return 0

    # Build per-date Sydney totals
    sydney_sum: dict[int, float] = {}
    sydney_n: dict[int, int] = {}
    for (date_int, lga), (s, n) in sums.items():
        sydney_sum[date_int] = sydney_sum.get(date_int, 0.0) + s
        sydney_n[date_int] = sydney_n.get(date_int, 0) + n

    # Per-LGA series (mean_decicents by date_int)
    by_lga: dict[str, dict[int, tuple[float, int]]] = {}
    for (date_int, lga), (s, n) in sums.items():
        if lga not in by_lga:
            by_lga[lga] = {}
        by_lga[lga][date_int] = (s, n)

    rows = []
    for lga, lga_data in by_lga.items():
        # LGA daily mean series
        lga_dates = sorted(lga_data.keys())
        lga_prices = np.array([lga_data[d][0] / lga_data[d][1] for d in lga_dates], dtype=float)

        # Rest-of-Sydney anchor: subtract this LGA's contribution
        anchor_dict: dict[int, float] = {}
        for d in sorted(sydney_sum.keys()):
            lga_s, lga_n_ = lga_data.get(d, (0.0, 0))
            a_s = sydney_sum[d] - lga_s
            a_n = sydney_n[d] - lga_n_
            if a_n >= MIN_STATION_FLOOR:
                anchor_dict[d] = a_s / a_n

        anchor_dates = sorted(anchor_dict.keys())
        anchor_prices = np.array([anchor_dict[d] for d in anchor_dates], dtype=float)

        # Trough detection
        lga_trough_idx = detect_trough_events(lga_prices)
        anchor_trough_idx = detect_trough_events(anchor_prices)

        n_lga_events = len(lga_trough_idx)
        n_anchor_events = len(anchor_trough_idx)

        lga_trough_dates = [_int_to_date(lga_dates[i]) for i in lga_trough_idx]
        anchor_trough_dates = [_int_to_date(anchor_dates[i]) for i in anchor_trough_idx]

        if n_anchor_events == 0:
            rows.append((lga, snapshot_date, None, None, 0.0, None, None, None, n_lga_events))
            continue

        leads = _match_events(lga_trough_dates, anchor_trough_dates)
        match_fraction = len(leads) / n_anchor_events

        if len(leads) == 0:
            rows.append((lga, snapshot_date, None, None, 0.0, None, None, None, n_lga_events))
            continue

        trough_median = float(np.median(leads))
        if len(leads) >= 2:
            std = float(np.std(leads))
            trough_consistency: float | None = (1.0 / std) if std > 0 else None
        else:
            trough_consistency = None

        rows.append((
            lga, snapshot_date,
            trough_median, trough_consistency, match_fraction,
            None, None, None,
            n_lga_events,
        ))

    upsert_lga_leadership_rows(conn, rows)
    conn.commit()
    logger.info(
        "score_leadership_snapshot %s: %d LGA rows written", snapshot_date, len(rows)
    )
    return len(rows)


def score_leadership_range(
    conn: sqlite3.Connection,
    start_date: str,
    end_date: str,
    step_days: int = 7,
    fuel_code: str = "E10",
    window_days: int = LEADERSHIP_WINDOW_DAYS,
) -> int:
    """Run score_leadership_snapshot for weekly snapshots in [start_date, end_date].

    Returns total LGA rows written.
    """
    d = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    if d > end:
        raise ValueError(f"start_date {start_date} must not exceed end_date {end_date}")

    total = 0
    while d <= end:
        total += score_leadership_snapshot(
            conn, d.isoformat(), fuel_code=fuel_code, window_days=window_days
        )
        d += timedelta(days=step_days)
    return total


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command("lga-leadership")
@click.option("--db", "db_path", default=str(DEFAULT_DB_PATH), show_default=True)
@click.option(
    "--snapshot-date",
    default=None,
    help="Single snapshot date (YYYY-MM-DD). Defaults to today.",
)
@click.option("--start-date", default=None, help="If set, score weekly from start-date to snapshot-date.")
@click.option("--window-days", default=LEADERSHIP_WINDOW_DAYS, show_default=True)
@click.option("--step-days", default=7, show_default=True, help="Days between weekly snapshots.")
@click.option("-v", "--verbose", is_flag=True)
def main(
    db_path: str,
    snapshot_date: str | None,
    start_date: str | None,
    window_days: int,
    step_days: int,
    verbose: bool,
) -> None:
    """Populate lga_leadership table at weekly snapshot intervals."""
    import logging as _logging
    _logging.basicConfig(
        level=_logging.DEBUG if verbose else _logging.INFO,
        format="%(levelname)s %(message)s",
    )

    end_date = snapshot_date or date.today().isoformat()
    conn = open_db(pathlib.Path(db_path))
    create_schema(conn)

    if start_date:
        total = score_leadership_range(
            conn, start_date, end_date, step_days=step_days, window_days=window_days
        )
        logger.info("lga-leadership [%s..%s step=%dd]: %d rows written", start_date, end_date, step_days, total)
    else:
        n = score_leadership_snapshot(conn, end_date, window_days=window_days)
        logger.info("lga-leadership %s: %d rows written", end_date, n)

    conn.close()


if __name__ == "__main__":
    main()
