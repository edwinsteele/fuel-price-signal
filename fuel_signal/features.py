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
from fuel_signal.cycle import CycleDetector, CycleState
from fuel_signal.labels import assemble_training_rows

# Minimum label rows a station must have to be included in the training dataset.
# Roughly one year of daily observations. Stations below this threshold are
# too new to have survived a full price cycle and produce uninformative label patterns.
MIN_TRAINING_ROWS_PER_STATION: int = 365

# Stations excluded from training due to confirmed data-gap distortion.
# Issue #29: both stations went offline during high-price years, so their
# rolling P33 was computed against a cheap-only price history. This causes
# both label conditions to fire almost constantly (positive rate 0.72–0.84),
# producing misleading training signal that the min-rows filter alone won't catch.
#   20528 — Speedway William Street, Granville: median 116.9c, positive rate 0.84
#   20133 — Metro Condell Park West: median 143.7c, positive rate 0.72
EXCLUDED_STATION_CODES: frozenset[int] = frozenset({20133, 20528})

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


def _date_from_int(v: int) -> str:
    s = str(v)
    return f"{s[:4]}-{s[4:6]}-{s[6:]}"


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


def _build_feature_dict(
    state: CycleState,
    station_price: float,
    sydney_avg: float,
) -> dict[str, float]:
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
    For very large batches use assemble_feature_rows, which caches all three
    inputs across stations sharing a date.

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

    return _build_feature_dict(state, station_price, sydney_avg)


def assemble_feature_rows(
    conn: sqlite3.Connection,
    horizon_days: int = 7,
    threshold_cents: float = 3.0,
    lookback_days: int = 90,
    percentile_pct: float = 33.0,
    station_codes: list[int] | None = None,
    min_rows_per_station: int = MIN_TRAINING_ROWS_PER_STATION,
) -> pd.DataFrame:
    """Build labels (via labels.assemble_training_rows) and join feature columns.

    Returns the labels DataFrame plus one column per feature in FEATURE_COLUMNS.
    Rows where compute_features returns None are dropped.
    CycleDetector is built once from the full average series — detect() slices
    per row, so PIT-safety is preserved.

    Stations in EXCLUDED_STATION_CODES are always removed (data-gap distortion).
    Stations with fewer than min_rows_per_station label rows are also removed
    (too-new stations haven't survived a full price cycle).
    """
    if min_rows_per_station < 0:
        raise ValueError("min_rows_per_station must be >= 0")

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

    if EXCLUDED_STATION_CODES:
        label_df = label_df[~label_df["station_code"].isin(EXCLUDED_STATION_CODES)]

    if min_rows_per_station > 0:
        counts = label_df.groupby("station_code")["label"].count()
        eligible = counts[counts >= min_rows_per_station].index
        label_df = label_df[label_df["station_code"].isin(eligible)]

    if label_df.empty:
        return pd.DataFrame(columns=all_cols)

    fid = _db.fuel_type_id(conn, "E10")

    # Cache 1: Sydney avg by date. average_price_series IS the GROUP BY query
    # the per-row path runs, so reusing its result is bit-for-bit identical.
    sydney_series = _db.average_price_series(conn)
    sydney_avg_by_date: dict[str, float] = dict(sydney_series)

    cd = CycleDetector(sydney_series)

    # Cache 2: cycle state by date. detect() is pure in (cd._series, date),
    # and cd._series is set in __init__ and never mutated, so a single call per
    # unique date is correct.
    cycle_state_by_date: dict[str, CycleState | None] = {
        d: cd.detect(d) for d in label_df["price_date"].unique()
    }

    # Cache 3: station price by (station_code, date_iso). One bulk SELECT
    # replaces ~2M point-lookups. price_date is INTEGER YYYYMMDD in the DB but
    # ISO string in label_df; convert once at load time so the lookup key
    # matches.
    station_price_by_key: dict[tuple[int, str], float] = {
        (sc, _date_from_int(date_int)): decicents / 10
        for sc, date_int, decicents in conn.execute(
            "SELECT station_code, price_date, price_decicents FROM daily_prices"
            " WHERE fuel_type_id = ?",
            (fid,),
        )
    }

    # Every (station_code, price_date) in label_df came from daily_prices, and
    # every label date is in cd._series for the same reason — so cache misses
    # on station_price / sydney_avg are upstream bugs, not data conditions.
    # Letting dict[key] raise KeyError surfaces them rather than silently
    # dropping rows the per-row path would have kept.
    records = []
    for row_dict in label_df.to_dict("records"):
        date_d: str = row_dict["price_date"]
        state = cycle_state_by_date[date_d]
        if state is None:
            continue
        station_price = station_price_by_key[(row_dict["station_code"], date_d)]
        sydney_avg = sydney_avg_by_date[date_d]
        records.append({**row_dict, **_build_feature_dict(state, station_price, sydney_avg)})

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
    "--min-rows", "min_rows", type=click.IntRange(min=0), default=MIN_TRAINING_ROWS_PER_STATION,
    show_default=True,
    help="Minimum label rows per station to include in training set (0 = no filter).",
)
@click.option(
    "--db",
    "db_path",
    default=str(_db.DEFAULT_DB_PATH),
    show_default=True,
    help="Path to SQLite DB.",
)
def main(  # noqa: PLR0913
    output: str, horizon: int, threshold: float, lookback: int,
    percentile: float, min_rows: int, db_path: str,
) -> None:
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
        min_rows_per_station=min_rows,
    )
    conn.close()

    out_path = pathlib.Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    click.echo(f"Wrote {len(df):,} rows ({int(df['label'].sum()):,} positive) to {out_path}")


if __name__ == "__main__":
    main()
