"""Per-station, time-varying classifier — Competitive | Sticky | Discount.

For each snapshot_date D, classifies stations by their 45-day median price
premium vs the LGA cluster reference. Two iterations converge to a stable
cluster: iter 1 takes the median across all LGA stations as cluster reference;
iter 2 takes the median across iter-1 Competitive stations only.

Writes station_class (per station per day) and classification_summary
(per LGA per day) tables. PIT-safe: classification at D uses only
daily_prices rows in [D-45, D-1].
"""

import logging
import pathlib
import sqlite3
from collections import defaultdict
from datetime import date, timedelta
from statistics import median

import click

from fuel_signal.db import (
    DEFAULT_DB_PATH,
    create_schema,
    daily_prices_in_window,
    delete_classification_summary_for_date,
    delete_station_class_for_date,
    open_db,
    upsert_classification_summary_rows,
    upsert_station_class_rows,
)

logger = logging.getLogger(__name__)

# One full NSW cycle (empirical mean ≈ 45 days). PIT-safe by construction:
# window ends at snapshot_date - 1.
WINDOW_DAYS: int = 45

# Stations whose median premium falls outside ±PREMIUM_BAND_CENTS cents of the
# LGA cluster reference are labelled Sticky (above) or Discount (below).
PREMIUM_BAND_CENTS: float = 10.0

CLASS_COMPETITIVE = "Competitive"
CLASS_STICKY = "Sticky"
CLASS_DISCOUNT = "Discount"


def _band_class(premium_cents: float) -> str:
    if premium_cents > PREMIUM_BAND_CENTS:
        return CLASS_STICKY
    if premium_cents < -PREMIUM_BAND_CENTS:
        return CLASS_DISCOUNT
    return CLASS_COMPETITIVE


def _median_premiums(
    station_prices: dict[int, dict[str, float]],
    cluster_ref_by_date: dict[str, float],
) -> dict[int, float]:
    """Return {station_code: median premium} across days the station observed
    and the cluster reference is defined."""
    out: dict[int, float] = {}
    for code, prices_by_date in station_prices.items():
        premiums = [
            p - cluster_ref_by_date[d]
            for d, p in prices_by_date.items()
            if d in cluster_ref_by_date
        ]
        if premiums:
            out[code] = median(premiums)
    return out


def _classify_lga(
    station_prices: dict[int, dict[str, float]],
) -> tuple[dict[int, str], dict[int, float]]:
    """Two-pass classification for a single LGA's stations.

    station_prices: {station_code: {date_str: price_cents}}.
    Returns ({station_code: class}, {station_code: median_premium_cents})
    based on the iter-2 cluster reference. Stations with no overlap with the
    cluster reference are absent from both dicts.
    """
    # Iter 1: cluster reference = median across all LGA stations on each date.
    by_date_iter1: defaultdict[str, list[float]] = defaultdict(list)
    for prices_by_date in station_prices.values():
        for d, p in prices_by_date.items():
            by_date_iter1[d].append(p)
    cluster_ref_1 = {d: median(ps) for d, ps in by_date_iter1.items()}

    premiums_1 = _median_premiums(station_prices, cluster_ref_1)
    iter1_classes = {code: _band_class(pm) for code, pm in premiums_1.items()}
    competitive_1 = {c for c, cls in iter1_classes.items() if cls == CLASS_COMPETITIVE}

    # If iter 1 has no Competitive stations, iter 2 has no cluster reference to
    # compute. Keep iter-1 results; the downstream "drop LGAs with zero
    # Competitive" rule will catch this case.
    if not competitive_1:
        return iter1_classes, premiums_1

    # Iter 2: cluster reference = median across iter-1 Competitive stations.
    by_date_iter2: defaultdict[str, list[float]] = defaultdict(list)
    for code in competitive_1:
        for d, p in station_prices[code].items():
            by_date_iter2[d].append(p)
    cluster_ref_2 = {d: median(ps) for d, ps in by_date_iter2.items()}

    premiums_2 = _median_premiums(station_prices, cluster_ref_2)
    iter2_classes = {code: _band_class(pm) for code, pm in premiums_2.items()}
    return iter2_classes, premiums_2


def classify_snapshot(
    conn: sqlite3.Connection,
    snapshot_date: str,
    fuel_code: str = "E10",
    window_days: int = WINDOW_DAYS,
) -> tuple[int, int]:
    """Classify all stations as of snapshot_date and persist results.

    Window: [snapshot_date - window_days, snapshot_date - 1] inclusive.
    Existing rows for snapshot_date are removed before the new ones are written
    (idempotent re-runs).

    Returns (n_station_class_rows_written, n_summary_rows_written).
    """
    if window_days <= 0:
        raise ValueError("window_days must be positive")

    snap = date.fromisoformat(snapshot_date)
    start = (snap - timedelta(days=window_days)).isoformat()
    end = (snap - timedelta(days=1)).isoformat()

    rows = daily_prices_in_window(conn, start, end, fuel_code=fuel_code)

    # Idempotent: wipe previous rows for this date even if window is empty.
    delete_station_class_for_date(conn, snapshot_date)
    delete_classification_summary_for_date(conn, snapshot_date)

    if not rows:
        logger.warning(
            "classify_snapshot %s: no daily_prices in window [%s, %s]",
            snapshot_date, start, end,
        )
        conn.commit()
        return 0, 0

    # by_lga[lga][station_code][date_str] = price_cents
    by_lga: defaultdict[str, defaultdict[int, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for station_code, lga, d, price in rows:
        by_lga[lga][station_code][d] = price

    class_rows: list[tuple[int, str, str, int]] = []
    summary_rows: list[tuple[str, str, int, int, int]] = []

    for lga, station_prices in by_lga.items():
        # Cast nested defaultdict to plain dict for hashable/iterable safety downstream.
        classes, premiums = _classify_lga({c: dict(d) for c, d in station_prices.items()})

        n_competitive = sum(1 for cls in classes.values() if cls == CLASS_COMPETITIVE)
        n_sticky = sum(1 for cls in classes.values() if cls == CLASS_STICKY)
        n_discount = sum(1 for cls in classes.values() if cls == CLASS_DISCOUNT)
        summary_rows.append((snapshot_date, lga, n_competitive, n_sticky, n_discount))

        # Zero-Competitive LGA: emit no station_class rows, but the summary row
        # records the failure so downstream code can detect the gap.
        if n_competitive == 0:
            continue

        for code, cls in classes.items():
            class_rows.append((code, snapshot_date, cls, round(premiums[code] * 10)))

    upsert_station_class_rows(conn, class_rows)
    upsert_classification_summary_rows(conn, summary_rows)
    conn.commit()

    logger.info(
        "classify_snapshot %s: %d station_class rows across %d LGAs (%d summary rows)",
        snapshot_date, len(class_rows), len(by_lga), len(summary_rows),
    )
    return len(class_rows), len(summary_rows)


def classify_range(
    conn: sqlite3.Connection,
    start_date: str,
    end_date: str,
    fuel_code: str = "E10",
    window_days: int = WINDOW_DAYS,
) -> tuple[int, int]:
    """Run classify_snapshot for every date in [start_date, end_date] inclusive.

    Iterates chronologically so that a future feature pass reading station_class
    in date order sees only PIT-consistent rows.
    """
    d = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    if d > end:
        raise ValueError(f"start_date {start_date} must not exceed end_date {end_date}")

    total_class = total_summary = 0
    while d <= end:
        n_c, n_s = classify_snapshot(
            conn, d.isoformat(), fuel_code=fuel_code, window_days=window_days,
        )
        total_class += n_c
        total_summary += n_s
        d += timedelta(days=1)
    return total_class, total_summary


@click.command("classify")
@click.option(
    "--db",
    "db_path",
    default=str(DEFAULT_DB_PATH),
    show_default=True,
    help="Path to SQLite database.",
)
@click.option(
    "--snapshot-date",
    default=None,
    help="Snapshot date (YYYY-MM-DD). Defaults to today.",
)
@click.option(
    "--start-date",
    default=None,
    help="If set, classify every date in [start-date, snapshot-date] inclusive.",
)
@click.option(
    "--window-days",
    default=WINDOW_DAYS,
    show_default=True,
    help="Window length in days, ending at snapshot_date - 1.",
)
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
def main(
    db_path: str,
    snapshot_date: str | None,
    start_date: str | None,
    window_days: int,
    verbose: bool,
) -> None:
    """Run the station classifier and populate station_class + classification_summary."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    end_date = snapshot_date or date.today().isoformat()
    conn = open_db(pathlib.Path(db_path))
    create_schema(conn)

    if start_date:
        n_c, n_s = classify_range(conn, start_date, end_date, window_days=window_days)
        logger.info(
            "classify [%s..%s]: %d station_class rows, %d summary rows",
            start_date, end_date, n_c, n_s,
        )
    else:
        classify_snapshot(conn, end_date, window_days=window_days)

    conn.close()


if __name__ == "__main__":
    main()
