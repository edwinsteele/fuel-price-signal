"""Per-station, time-varying price-class classifier.

Classifies each station as Competitive, Sticky, or Discount based on its
median price premium vs the LGA cluster reference over a rolling 45-day window.

Algorithm (per LGA, per snapshot_date D):
    Iter 1: cluster reference = daily median across ALL stations in the LGA.
    Iter 2: cluster reference = daily median across iter-1 Competitive stations only.
    Classify each station by its median premium vs the iter-2 reference.

If iter 2 yields zero Competitive stations for an LGA, no station_class rows are
written for that LGA-date, but a classification_summary row with n_competitive=0 is.

PIT safety: snapshot_date D only reads price_date in [D-45, D-1].

Usage::

    uv run python -m fuel_signal.classify [--start-date DATE] [--end-date DATE] [--db PATH]
"""
from __future__ import annotations

import datetime
import logging
import pathlib
import statistics

import click

from fuel_signal import db as _db

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASSIFICATION_WINDOW_DAYS: int = 45
STICKY_THRESHOLD_DC: int = 100    # +10.0c in decicents
DISCOUNT_THRESHOLD_DC: int = -100  # -10.0c in decicents

CLASS_COMPETITIVE = "Competitive"
CLASS_STICKY = "Sticky"
CLASS_DISCOUNT = "Discount"


# ---------------------------------------------------------------------------
# Core classifier
# ---------------------------------------------------------------------------

def _compute_classes(
    station_prices: dict[int, dict[int, int]],
    cluster_stations: frozenset[int] | None,
) -> dict[int, tuple[str, int]]:
    """Classify stations by their median premium vs a cluster daily median.

    station_prices: {station_code: {date_int: price_decicents}}
    cluster_stations: stations used to compute the cluster reference;
                      None means use all stations.

    Returns {station_code: (class_str, median_premium_decicents)}.
    Only stations with ≥1 observation are returned.
    Stations not in cluster_stations but in station_prices are still classified.
    """
    if not station_prices:
        return {}

    if cluster_stations is None:
        ref_codes = set(station_prices.keys())
    else:
        ref_codes = cluster_stations & set(station_prices.keys())

    if not ref_codes:
        return {}

    # Collect all dates from ref stations (cluster reference is only meaningful
    # on days where at least one cluster station reported).
    all_dates: set[int] = set()
    for sc in ref_codes:
        all_dates.update(station_prices[sc].keys())

    # Daily cluster median (computed over ref_codes only).
    cluster_median: dict[int, float] = {}
    for date in all_dates:
        day_prices = [station_prices[sc][date] for sc in ref_codes if date in station_prices[sc]]
        if day_prices:
            cluster_median[date] = statistics.median(day_prices)

    # Per-station median premium.
    result: dict[int, tuple[str, int]] = {}
    for sc, prices_by_date in station_prices.items():
        premiums = [
            prices_by_date[date] - cluster_median[date]
            for date in prices_by_date
            if date in cluster_median
        ]
        if not premiums:
            continue
        median_premium = int(statistics.median(premiums))
        if median_premium > STICKY_THRESHOLD_DC:
            cls = CLASS_STICKY
        elif median_premium < DISCOUNT_THRESHOLD_DC:
            cls = CLASS_DISCOUNT
        else:
            cls = CLASS_COMPETITIVE
        result[sc] = (cls, median_premium)

    return result


# ---------------------------------------------------------------------------
# Snapshot materialisation
# ---------------------------------------------------------------------------

def classify_snapshot(conn, snapshot_date: str) -> tuple[int, int]:
    """Classify all stations for the given snapshot_date and write to DB.

    Returns (total_stations_classified, total_lgas_processed).
    snapshot_date: YYYY-MM-DD. Uses price_date in [D-45, D-1] — PIT-safe.
    """
    snapshot_dt = datetime.date.fromisoformat(snapshot_date)
    window_end_dt = snapshot_dt - datetime.timedelta(days=1)
    window_start_dt = snapshot_dt - datetime.timedelta(days=CLASSIFICATION_WINDOW_DAYS)

    window_start_int = _db._date_to_int(window_start_dt.isoformat())
    window_end_int = _db._date_to_int(window_end_dt.isoformat())
    snapshot_date_int = _db._date_to_int(snapshot_date)

    fid = _db.fuel_type_id(conn, "E10")

    rows = conn.execute(
        "SELECT dp.station_code, s.council, dp.price_date, dp.price_decicents"
        " FROM daily_prices dp JOIN stations s USING(station_code)"
        " WHERE dp.fuel_type_id = ? AND dp.price_date >= ? AND dp.price_date <= ?"
        "   AND s.council IS NOT NULL",
        (fid, window_start_int, window_end_int),
    ).fetchall()

    # Group: {lga: {station_code: {date_int: price_dc}}}
    lga_data: dict[str, dict[int, dict[int, int]]] = {}
    for sc, council, price_date, price_dc in rows:
        lga_data.setdefault(council, {}).setdefault(sc, {})[price_date] = price_dc

    station_class_rows: list[tuple[int, int, str, int]] = []
    summary_rows: list[tuple[int, str, int, int, int]] = []
    total_classified = 0

    for lga, station_data in lga_data.items():
        # Iter 1: all-station cluster reference.
        iter1 = _compute_classes(station_data, cluster_stations=None)
        if not iter1:
            summary_rows.append((snapshot_date_int, lga, 0, 0, 0))
            continue

        # Iter 2: Competitive-only cluster reference.
        competitive_after_iter1 = frozenset(
            sc for sc, (cls, _) in iter1.items() if cls == CLASS_COMPETITIVE
        )
        if not competitive_after_iter1:
            n_sticky = sum(1 for cls, _ in iter1.values() if cls == CLASS_STICKY)
            n_discount = sum(1 for cls, _ in iter1.values() if cls == CLASS_DISCOUNT)
            summary_rows.append((snapshot_date_int, lga, 0, n_sticky, n_discount))
            logger.debug("LGA %s on %s: zero Competitive after iter 1 — no station_class rows", lga, snapshot_date)
            continue

        iter2 = _compute_classes(station_data, cluster_stations=competitive_after_iter1)
        if not iter2:
            summary_rows.append((snapshot_date_int, lga, 0, 0, 0))
            continue

        n_competitive = n_sticky = n_discount = 0
        for sc, (cls, median_premium) in iter2.items():
            if cls == CLASS_COMPETITIVE:
                n_competitive += 1
            elif cls == CLASS_STICKY:
                n_sticky += 1
            else:
                n_discount += 1
            station_class_rows.append((sc, snapshot_date_int, cls, median_premium))

        summary_rows.append((snapshot_date_int, lga, n_competitive, n_sticky, n_discount))
        total_classified += n_competitive + n_sticky + n_discount

    _db.upsert_station_classes(conn, station_class_rows)
    _db.upsert_classification_summaries(conn, summary_rows)
    conn.commit()
    return total_classified, len(lga_data)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

@click.command("classify")
@click.option(
    "--db",
    "db_path",
    default=str(_db.DEFAULT_DB_PATH),
    show_default=True,
    help="Path to SQLite database.",
)
@click.option(
    "--start-date",
    default=None,
    help="First snapshot_date to classify (YYYY-MM-DD). Default: earliest daily_prices date + window.",
)
@click.option(
    "--end-date",
    default=None,
    help="Last snapshot_date to classify (YYYY-MM-DD). Default: yesterday.",
)
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
def main(db_path: str, start_date: str | None, end_date: str | None, verbose: bool) -> None:
    """Classify stations as Competitive/Sticky/Discount and materialise daily snapshots."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    path = pathlib.Path(db_path)
    if not path.exists():
        raise click.ClickException(
            f"Database not found: {db_path}. Run 'uv run python -m fuel_signal.db' first."
        )

    conn = _db.open_db(path)
    _db.create_schema(conn)

    fid = _db.fuel_type_id(conn, "E10")

    if start_date is None:
        row = conn.execute(
            "SELECT MIN(price_date) FROM daily_prices WHERE fuel_type_id = ?", (fid,)
        ).fetchone()
        if row is None or row[0] is None:
            raise click.ClickException("No data in daily_prices. Run fill.py first.")
        earliest_str = _db._date_from_int(row[0])
        earliest_dt = datetime.date.fromisoformat(earliest_str)
        start_dt = earliest_dt + datetime.timedelta(days=CLASSIFICATION_WINDOW_DAYS)
    else:
        start_dt = datetime.date.fromisoformat(start_date)

    if end_date is None:
        end_dt = datetime.date.today() - datetime.timedelta(days=1)
    else:
        end_dt = datetime.date.fromisoformat(end_date)

    if start_dt > end_dt:
        logger.info("Nothing to classify (start %s > end %s).", start_dt, end_dt)
        conn.close()
        return

    total_days = (end_dt - start_dt).days + 1
    logger.info("Classifying %d snapshot dates (%s … %s).", total_days, start_dt, end_dt)

    classified = lgas = 0
    cur_dt = start_dt
    while cur_dt <= end_dt:
        n_sc, n_lga = classify_snapshot(conn, cur_dt.isoformat())
        classified += n_sc
        lgas += n_lga
        if verbose:
            logger.debug("%s: %d stations across %d LGAs", cur_dt, n_sc, n_lga)
        cur_dt += datetime.timedelta(days=1)

    logger.info(
        "Done. %d station-date rows written across %d LGA-date rows.",
        classified, lgas,
    )
    conn.close()


if __name__ == "__main__":
    main()
