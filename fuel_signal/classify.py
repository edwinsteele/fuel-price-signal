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
from collections import defaultdict, deque
from datetime import date, timedelta
from statistics import median

import click

from fuel_signal.dates import date_from_int, date_to_int
from fuel_signal.db import (
    DEFAULT_DB_PATH,
    create_schema,
    daily_prices_in_window,
    delete_classification_summary_for_date,
    delete_station_class_for_date,
    fuel_type_id,
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


def _write_classify_snapshot(
    conn: sqlite3.Connection,
    snapshot_date: str,
    by_lga: dict[str, dict[int, dict[str, float]]],
    window: tuple[str, str],
    commit: bool = True,
) -> tuple[int, int]:
    """Classify and persist rows from a pre-grouped by_lga dict.

    Shared by classify_snapshot (direct query) and classify_range (streamed
    sliding-window query) so both paths compute and write identically.
    `window` is (start_date, end_date), used only for the empty-window log.
    `commit` is False in classify_range, which batches commits across the
    whole range instead of once per snapshot.
    """
    delete_station_class_for_date(conn, snapshot_date)
    delete_classification_summary_for_date(conn, snapshot_date)

    if not by_lga:
        logger.warning(
            "classify_snapshot %s: no daily_prices in window [%s, %s]",
            snapshot_date, window[0], window[1],
        )
        if commit:
            conn.commit()
        return 0, 0

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
    if commit:
        conn.commit()

    logger.info(
        "classify_snapshot %s: %d station_class rows across %d LGAs (%d summary rows)",
        snapshot_date, len(class_rows), len(by_lga), len(summary_rows),
    )
    return len(class_rows), len(summary_rows)


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

    # by_lga[lga][station_code][date_str] = price_cents
    by_lga: defaultdict[str, defaultdict[int, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for station_code, lga, d, price in rows:
        by_lga[lga][station_code][d] = price

    return _write_classify_snapshot(conn, snapshot_date, by_lga, window=(start, end))


def _day_bucket_stream(
    conn: sqlite3.Connection,
    start_date: str,
    end_date: str,
    fuel_code: str = "E10",
    window_days: int = WINDOW_DAYS,
):
    """Yield (snapshot_date, deque of (date_int, date_str, [(station_code, lga,
    price_cents)])) for each date in [start_date, end_date], sliding a
    window_days window across a single ORDER BY price_date scan.

    Resident set is O(window_days) day-buckets, not O(history): a naive
    full-range load via daily_prices_in_window is 540 MB resident / 881 MB
    peak for a decade of E10 rows, which would OOM Viking (see AGENTS.md §
    "Backfill (--start-date) paths: load once, slice in memory"). Yields the
    deque itself rather than a flattened row list — flattening rebuilds
    ~31k tuples per snapshot and gives back most of the win, so consumers
    must iterate the buckets directly.
    """
    fid = fuel_type_id(conn, fuel_code)
    scan_lo = date_to_int(
        (date.fromisoformat(start_date) - timedelta(days=window_days)).isoformat()
    )
    scan_hi = date_to_int((date.fromisoformat(end_date) - timedelta(days=1)).isoformat())
    cur = conn.execute(
        "SELECT dp.price_date, dp.station_code, s.council, dp.price_decicents"
        " FROM daily_prices dp JOIN stations s USING(station_code)"
        " WHERE dp.fuel_type_id = ? AND s.council IS NOT NULL"
        "   AND dp.price_date >= ? AND dp.price_date <= ?"
        " ORDER BY dp.price_date",
        (fid, scan_lo, scan_hi),
    )

    buf: deque[tuple[int, str, list[tuple[int, str, float]]]] = deque()
    pending = cur.fetchone()
    d, end = date.fromisoformat(start_date), date.fromisoformat(end_date)
    while d <= end:
        hi = date_to_int((d - timedelta(days=1)).isoformat())
        lo = date_to_int((d - timedelta(days=window_days)).isoformat())
        while pending is not None and pending[0] <= hi:
            day_int = pending[0]
            day_str = date_from_int(day_int)
            day: list[tuple[int, str, float]] = []
            while pending is not None and pending[0] == day_int:
                day.append((pending[1], pending[2], pending[3] / 10))
                pending = cur.fetchone()
            buf.append((day_int, day_str, day))
        while buf and buf[0][0] < lo:
            buf.popleft()
        yield d.isoformat(), buf
        d += timedelta(days=1)


def classify_range(
    conn: sqlite3.Connection,
    start_date: str,
    end_date: str,
    fuel_code: str = "E10",
    window_days: int = WINDOW_DAYS,
) -> tuple[int, int]:
    """Run classification for every date in [start_date, end_date] inclusive.

    Produces the same rows as calling classify_snapshot per snapshot, but
    scans daily_prices once in the ORDER BY price_date order the schema
    already delivers and slides a deque of day-buckets across it, instead of
    re-querying the ~98%-overlapping window per snapshot (see AGENTS.md §
    "Backfill (--start-date) paths: load once, slice in memory").
    classify_snapshot (the daily path) is unchanged. Commits are batched
    across the whole range rather than once per snapshot.

    Iterates chronologically so that a future feature pass reading station_class
    in date order sees only PIT-consistent rows.
    """
    d = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    if d > end:
        raise ValueError(f"start_date {start_date} must not exceed end_date {end_date}")
    if window_days <= 0:
        raise ValueError("window_days must be positive")

    total_class = total_summary = 0
    for snapshot_date, buf in _day_bucket_stream(
        conn, start_date, end_date, fuel_code=fuel_code, window_days=window_days
    ):
        snap = date.fromisoformat(snapshot_date)
        start = (snap - timedelta(days=window_days)).isoformat()
        window_end = (snap - timedelta(days=1)).isoformat()

        by_lga: defaultdict[str, defaultdict[int, dict[str, float]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        for _date_int, day_str, day in buf:
            for station_code, lga, price in day:
                by_lga[lga][station_code][day_str] = price

        n_c, n_s = _write_classify_snapshot(
            conn, snapshot_date, by_lga, window=(start, window_end), commit=False
        )
        total_class += n_c
        total_summary += n_s

    conn.commit()
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
