"""Feature pipeline for the ML price-movement model.

All features are computed using only data with price_date <= date_d
(point-in-time safe). The CycleDetector.detect() method slices its internal
series to date_d, so building one detector from the full series and calling
detect(date_d) per row is both correct and fast.

Usage
-----
Standalone (builds CycleDetector internally — fine for one-off calls)::

    conn = open_db(...)
    features = compute_features(conn, station_code=182, date_d="2024-06-15")

Batched (pre-build CycleDetector once for a large loop — see CLAUDE.md perf note)::

    from fuel_signal.db import average_price_series
    from fuel_signal.cycle import CycleDetector
    cd = CycleDetector(average_price_series(conn))
    for date_d in dates:
        features = compute_features(conn, station_code, date_d, cycle_detector=cd)
"""

from __future__ import annotations

import pathlib
import sqlite3

import click
import pandas as pd

from fuel_signal import db as _db
from fuel_signal.cycle import CycleDetector
from fuel_signal.labels import assemble_training_rows

# Canonical ordered list of feature column names.
FEATURE_COLUMNS: list[str] = [
    "cycle_pct_through",
    "cycle_days_since_peak",
    "cycle_mean_length",
    "cycle_last_min_cents",
    "cycle_last_max_cents",
    "cycle_peak_count",
    "station_price_cents",
    "station_minus_last_min_cents",
    "station_minus_last_max_cents",
    "station_minus_sydney_avg_cents",
]


def _date_to_int(s: str) -> int:
    return int(s[:10].replace("-", ""))


def _station_price_on_date(
    conn: sqlite3.Connection,
    station_code: int,
    date_d: str,
    fuel_type_id: int,
) -> float | None:
    row = conn.execute(
        "SELECT price_decicents FROM daily_prices"
        " WHERE station_code = ? AND fuel_type_id = ? AND price_date = ?",
        (station_code, fuel_type_id, _date_to_int(date_d)),
    ).fetchone()
    return row[0] / 10 if row else None


def _sydney_avg_on_date(
    conn: sqlite3.Connection,
    date_d: str,
    fuel_type_id: int,
) -> float | None:
    # Averages over all stations in daily_prices — intentionally unfiltered because
    # the DB contains only Sydney metro stations by design (filtered at load time).
    row = conn.execute(
        "SELECT AVG(price_decicents) FROM daily_prices"
        " WHERE fuel_type_id = ? AND price_date = ?",
        (fuel_type_id, _date_to_int(date_d)),
    ).fetchone()
    if row is None or row[0] is None:
        return None
    return row[0] / 10


def compute_features(
    conn: sqlite3.Connection,
    station_code: int,
    date_d: str,
    cycle_detector: CycleDetector | None = None,
) -> dict[str, float] | None:
    """Return a feature dict for (station, date), or None if insufficient data.

    If cycle_detector is None, build one from average_price_series(conn).
    detect(date_d) slices the series to date_d internally — PIT-safe regardless
    of how much data the detector was built with.

    For batched callers: pre-build one CycleDetector and pass it in.
    Building per-row costs 3650x on a full backtest (CLAUDE.md perf note).

    Returns None when:
    - Station has no price on date_d in daily_prices
    - CycleDetector.detect(date_d) returns None (fewer than 2 peaks)
    - Sydney average is absent on date_d (data gap)
    """
    fid = _db.fuel_type_id(conn, "E10")

    station_price = _station_price_on_date(conn, station_code, date_d, fid)
    if station_price is None:
        return None

    if cycle_detector is None:
        cycle_detector = CycleDetector(_db.average_price_series(conn))

    state = cycle_detector.detect(date_d)
    if state is None:
        return None

    sydney_avg = _sydney_avg_on_date(conn, date_d, fid)
    if sydney_avg is None:
        return None

    return {
        "cycle_pct_through": state.pct_through_cycle,
        "cycle_days_since_peak": float(state.days_since_last_peak),
        "cycle_mean_length": state.mean_cycle_length,
        "cycle_last_min_cents": state.last_cycle_min,
        "cycle_last_max_cents": state.last_cycle_max,
        "cycle_peak_count": float(state.peak_count),
        "station_price_cents": station_price,
        "station_minus_last_min_cents": station_price - state.last_cycle_min,
        "station_minus_last_max_cents": station_price - state.last_cycle_max,
        "station_minus_sydney_avg_cents": station_price - sydney_avg,
    }


def assemble_feature_rows(
    conn: sqlite3.Connection,
    horizon_days: int = 7,
    threshold_cents: float = 3.0,
    lookback_days: int = 90,
    percentile_pct: float = 33.0,
    station_codes: list[int] | None = None,
) -> pd.DataFrame:
    """Build labels (via labels.assemble_training_rows) and join feature columns.

    Returns the labels DataFrame plus one column per feature in FEATURE_COLUMNS.
    Rows where compute_features returns None are dropped.
    CycleDetector is built once from the full average series — detect() slices
    per row, so PIT-safety is preserved.
    """
    label_df = assemble_training_rows(
        conn,
        horizon_days=horizon_days,
        threshold_cents=threshold_cents,
        lookback_days=lookback_days,
        percentile_pct=percentile_pct,
        station_codes=station_codes,
    )
    all_cols = list(label_df.columns) + FEATURE_COLUMNS
    if label_df.empty:
        return pd.DataFrame(columns=all_cols)

    cd = CycleDetector(_db.average_price_series(conn))

    records = []
    for row_dict in label_df.to_dict("records"):
        features = compute_features(conn, row_dict["station_code"], row_dict["price_date"], cd)
        if features is None:
            continue
        records.append({**row_dict, **features})

    if not records:
        return pd.DataFrame(columns=all_cols)
    return pd.DataFrame(records, columns=all_cols)


@click.command("features")
@click.option(
    "--output",
    default="data/features.csv",
    show_default=True,
    help="Output CSV path.",
)
@click.option("--horizon", type=click.IntRange(min=1), default=7, show_default=True, help="Forward horizon in days.")
@click.option(
    "--threshold", type=click.FloatRange(min=0.0), default=3.0, show_default=True,
    help="Minimum price drop (cents) to label as 1.",
)
@click.option(
    "--lookback", type=click.IntRange(min=1), default=90, show_default=True,
    help="Past days for price percentile (~2 cycles).",
)
@click.option(
    "--percentile", type=click.FloatRange(min=0.0, max=100.0), default=33.0, show_default=True,
    help="Percentile gate for 'price is cheap' condition.",
)
@click.option(
    "--db",
    "db_path",
    default=str(_db.DEFAULT_DB_PATH),
    show_default=True,
    help="Path to SQLite DB.",
)
def main(output: str, horizon: int, threshold: float, lookback: int, percentile: float, db_path: str) -> None:
    """Assemble ML training rows with cycle features joined to labels."""
    path = pathlib.Path(db_path)
    if not path.exists():
        raise click.ClickException(
            f"Database not found: {db_path}. Run 'uv run python -m fuel_signal.db' first."
        )

    conn = _db.open_db(path)
    df = assemble_feature_rows(
        conn,
        horizon_days=horizon,
        threshold_cents=threshold,
        lookback_days=lookback,
        percentile_pct=percentile,
    )
    conn.close()

    out_path = pathlib.Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    click.echo(f"Wrote {len(df):,} rows ({int(df['label'].sum()):,} positive) to {out_path}")


if __name__ == "__main__":
    main()
