"""One-line buy/don't-buy signal for E10 fuel.

Combines four signals (cycle phase, gradient flatline, price vs last cycle
min/max, preferred-station price-rise detection) into a single BUY / WAIT /
DONT_BUY verdict by averaging the numeric signal values.

Usage:
    uv run fuel-signal signal
    uv run fuel-signal signal --as-of 2026-02-15
    uv run fuel-signal signal --db /path/to/fuel_signal.db
"""

from __future__ import annotations

import datetime
import logging
import pathlib
import sqlite3
import statistics
from dataclasses import dataclass
from enum import Enum

import click
import numpy as np

import fuel_signal.db as db
from fuel_signal.config import PREFERRED_STATIONS
from fuel_signal.cycle import CycleDetector, CycleState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Signal primitives (ported from ff-aws-backend recommendations.py)
# ---------------------------------------------------------------------------

_NEUTRAL_SENTINEL = float("inf")


class SignalRecommendation(Enum):
    BUY = 1.0
    WAIT = 0.0
    DONT_BUY = -1.0
    NEUTRAL = _NEUTRAL_SENTINEL


@dataclass
class SignalEvaluation:
    name: str
    recommendation: SignalRecommendation
    description: str


# Cycle-time thresholds — match the original ff-aws-backend constants
_PCT_BUY = 0.66
_PCT_WAIT = 0.33

# Flatline (gradient) thresholds
_FLATLINE_MIN = -0.5
_FLATLINE_MAX = 0.5

# Near min/max thresholds (cents)
_NEAR_MIN_DELTA = 5.0
_NEAR_MAX_DELTA = 10.0

# Preferred-station big-rise gradient threshold (cents/day)
_BIG_RAISE_THRESHOLD = 10.0


def average_cycle_time_signal(state: CycleState) -> SignalEvaluation:
    """BUY if late in cycle, WAIT mid-cycle, DONT_BUY early."""
    pct = state.pct_through_cycle
    debug = (
        f"({pct:.0%} through cycle; day {state.days_since_last_peak}"
        f" / {state.mean_cycle_length:.1f})"
    )
    if pct > _PCT_BUY:
        return SignalEvaluation(
            "AverageCycleTimeSignal",
            SignalRecommendation.BUY,
            f"cycle ending soon {debug}",
        )
    if pct > _PCT_WAIT:
        return SignalEvaluation(
            "AverageCycleTimeSignal",
            SignalRecommendation.WAIT,
            f"mid cycle {debug}",
        )
    return SignalEvaluation(
        "AverageCycleTimeSignal",
        SignalRecommendation.DONT_BUY,
        f"early in cycle {debug}",
    )


def average_gradient_after_peak_signal(state: CycleState) -> SignalEvaluation:
    """Flatline detection: flat + late → BUY, flat + early → DONT_BUY, else NEUTRAL."""
    has_flatlined = all(
        _FLATLINE_MIN < g < _FLATLINE_MAX for g in state.last_3_gradients
    )
    debug = f"(last 3 gradients: {state.last_3_gradients})"
    if not has_flatlined:
        return SignalEvaluation(
            "AverageGradientAfterPeakSignal",
            SignalRecommendation.NEUTRAL,
            f"price has not flatlined {debug}",
        )
    if state.days_since_last_peak > state.mean_cycle_length / 2:
        return SignalEvaluation(
            "AverageGradientAfterPeakSignal",
            SignalRecommendation.BUY,
            f"price has flatlined after initial drop {debug}",
        )
    return SignalEvaluation(
        "AverageGradientAfterPeakSignal",
        SignalRecommendation.DONT_BUY,
        f"price is at peak {debug}",
    )


def average_near_previous_min_max_signal(
    state: CycleState, current_price: float
) -> SignalEvaluation:
    """BUY if price near last cycle min, DONT_BUY if near max, else WAIT."""
    debug = (
        f"(current {current_price:.1f}c; last cycle min {state.last_cycle_min:.1f}c,"
        f" max {state.last_cycle_max:.1f}c)"
    )
    if current_price < state.last_cycle_min + _NEAR_MIN_DELTA:
        return SignalEvaluation(
            "AverageNearPreviousMinMaxSignal",
            SignalRecommendation.BUY,
            f"price close to low in last cycle {debug}",
        )
    if current_price > state.last_cycle_max - _NEAR_MAX_DELTA:
        return SignalEvaluation(
            "AverageNearPreviousMinMaxSignal",
            SignalRecommendation.DONT_BUY,
            f"price close to high in last cycle {debug}",
        )
    return SignalEvaluation(
        "AverageNearPreviousMinMaxSignal",
        SignalRecommendation.WAIT,
        f"price in middle of last cycle {debug}",
    )


def favourite_station_price_gradient_signal(
    station_latest_gradients: dict[str, float],
) -> SignalEvaluation:
    """Detect big price rises across preferred stations.

    station_latest_gradients: {label: latest gradient cents/day} for each
    preferred station with enough recent data. Stations with insufficient data
    should be omitted by the caller.
    """
    if not station_latest_gradients:
        return SignalEvaluation(
            "FavouriteServiceStationPriceGradientSignal",
            SignalRecommendation.NEUTRAL,
            "insufficient data on preferred stations",
        )
    big_raisers = {
        label: g
        for label, g in station_latest_gradients.items()
        if g >= _BIG_RAISE_THRESHOLD
    }
    non_raisers = {
        label: g
        for label, g in station_latest_gradients.items()
        if g < _BIG_RAISE_THRESHOLD
    }

    def fmt(d: dict[str, float]) -> str:
        return ", ".join(f"{k} @ {v:.1f}" for k, v in d.items()) if d else "none"

    debug = f"(big raisers: {fmt(big_raisers)}; non-raisers: {fmt(non_raisers)})"

    if big_raisers and not non_raisers:
        return SignalEvaluation(
            "FavouriteServiceStationPriceGradientSignal",
            SignalRecommendation.DONT_BUY,
            f"all preferred stations have raised prices {debug}",
        )
    if big_raisers:
        return SignalEvaluation(
            "FavouriteServiceStationPriceGradientSignal",
            SignalRecommendation.BUY,
            f"some preferred stations raising prices {debug}",
        )
    return SignalEvaluation(
        "FavouriteServiceStationPriceGradientSignal",
        SignalRecommendation.NEUTRAL,
        f"no preferred stations raising sharply {debug}",
    )


# ---------------------------------------------------------------------------
# Combination
# ---------------------------------------------------------------------------

_BUY_THRESHOLD = 0.5
_DONT_BUY_THRESHOLD = -0.5


@dataclass
class CombinedVerdict:
    label: str          # "BUY ", "WAIT", "DONT"
    long_label: str     # "BUY", "WAIT", "DON'T BUY"
    mean_value: float   # NaN when all signals were NEUTRAL


def combine_signals(evaluations: list[SignalEvaluation]) -> CombinedVerdict:
    """Average directional signal values; NEUTRAL signals excluded."""
    directional = [
        e.recommendation.value
        for e in evaluations
        if e.recommendation is not SignalRecommendation.NEUTRAL
    ]
    if not directional:
        return CombinedVerdict("WAIT", "WAIT", float("nan"))
    mean = statistics.mean(directional)
    if mean >= _BUY_THRESHOLD:
        return CombinedVerdict("BUY ", "BUY", mean)
    if mean <= _DONT_BUY_THRESHOLD:
        return CombinedVerdict("DONT", "DON'T BUY", mean)
    return CombinedVerdict("WAIT", "WAIT", mean)


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _gap_boundaries(conn: sqlite3.Connection) -> tuple[str | None, str | None]:
    """Return (gap_start, gap_end) YYYY-MM-DD derived from DB source boundaries."""
    fid = db.fuel_type_id(conn, "E10")
    row_h = conn.execute(
        """SELECT MAX(p.price_date)
           FROM prices p
           JOIN price_sources ps ON p.source_id = ps.id
           WHERE p.fuel_type_id = ? AND ps.code = 'h'""",
        (fid,),
    ).fetchone()
    row_s = conn.execute(
        """SELECT MIN(p.price_date)
           FROM prices p
           JOIN price_sources ps ON p.source_id = ps.id
           WHERE p.fuel_type_id = ? AND ps.code = 's'""",
        (fid,),
    ).fetchone()

    if not row_h or row_h[0] is None or not row_s or row_s[0] is None:
        return None, None

    last_hist = db._date_from_int(row_h[0])
    first_snap = db._date_from_int(row_s[0])

    gap_start = (
        datetime.date.fromisoformat(last_hist) + datetime.timedelta(days=1)
    ).isoformat()
    gap_end = (
        datetime.date.fromisoformat(first_snap) - datetime.timedelta(days=1)
    ).isoformat()

    if gap_start > gap_end:
        return None, None

    return gap_start, gap_end


def _latest_daily_date(conn: sqlite3.Connection) -> str:
    row = conn.execute("SELECT MAX(price_date) FROM daily_prices").fetchone()
    if not row or row[0] is None:
        raise click.ClickException(
            "No data in daily_prices. Run 'uv run python -m fuel_signal.db' and fill.py first."
        )
    return db._date_from_int(row[0])


def _station_price_at(
    conn: sqlite3.Connection, station_code: int, as_of_date: str
) -> float | None:
    """Most recent E10 price (cents) at or before as_of_date."""
    fid = db.fuel_type_id(conn, "E10")
    as_of_int = db._date_to_int(as_of_date)
    row = conn.execute(
        """SELECT price_decicents FROM daily_prices
           WHERE station_code = ? AND fuel_type_id = ? AND price_date <= ?
           ORDER BY price_date DESC LIMIT 1""",
        (station_code, fid, as_of_int),
    ).fetchone()
    return row[0] / 10 if row else None


def _station_latest_gradient(
    conn: sqlite3.Connection,
    station_code: int,
    as_of_date: str,
    window: int = 4,
) -> float | None:
    """Latest np.gradient value of the last *window* daily prices at or before as_of_date."""
    fid = db.fuel_type_id(conn, "E10")
    as_of_int = db._date_to_int(as_of_date)
    rows = conn.execute(
        """SELECT price_decicents FROM daily_prices
           WHERE station_code = ? AND fuel_type_id = ? AND price_date <= ?
           ORDER BY price_date DESC LIMIT ?""",
        (station_code, fid, as_of_int, window),
    ).fetchall()
    if len(rows) < 2:
        return None
    prices = np.array([r[0] / 10 for r in reversed(rows)], dtype=float)
    return float(np.gradient(prices)[-1])


# ---------------------------------------------------------------------------
# Core signal logic
# ---------------------------------------------------------------------------

def evaluate_all_signals(
    state: CycleState,
    avg_current_price: float,
    station_latest_gradients: dict[str, float],
) -> list[SignalEvaluation]:
    return [
        average_cycle_time_signal(state),
        average_gradient_after_peak_signal(state),
        average_near_previous_min_max_signal(state, avg_current_price),
        favourite_station_price_gradient_signal(station_latest_gradients),
    ]


def build_signals(
    conn: sqlite3.Connection,
    as_of_date: str,
    preferred_stations: dict[int, str] | None = None,
) -> str:
    """Build signal output for the given date.

    Combines four signals into a single verdict and shows per-station prices
    plus the per-signal reasoning underneath.
    """
    stations = preferred_stations if preferred_stations is not None else PREFERRED_STATIONS

    series = db.average_price_series(conn)
    if not series:
        raise click.ClickException("No average price series available in daily_prices.")

    state = CycleDetector(series).detect(as_of_date)
    if state is None:
        raise click.ClickException(
            f"Cycle detection returned no result for {as_of_date} — insufficient data."
        )

    avg_current_price = next(
        (p for d, p in reversed(series) if d <= as_of_date), series[-1][1]
    )

    station_gradients: dict[str, float] = {}
    for station_code, label in stations.items():
        g = _station_latest_gradient(conn, station_code, as_of_date)
        if g is not None:
            station_gradients[label] = g

    evaluations = evaluate_all_signals(state, avg_current_price, station_gradients)
    verdict = combine_signals(evaluations)

    day_num = state.days_since_last_peak + 1          # 1-indexed
    cycle_len = round(state.mean_cycle_length)

    lines = [f"[as of {as_of_date}]"]
    for station_code, label in stations.items():
        price = _station_price_at(conn, station_code, as_of_date)
        price_str = f"{price:.1f}c" if price is not None else "no data"
        lines.append(
            f"{verdict.label} | Day {day_num}/{cycle_len} of cycle"
            f" | E10 @ {label}: {price_str}"
        )

    mean_str = "n/a" if np.isnan(verdict.mean_value) else f"{verdict.mean_value:+.2f}"
    lines.append(f"Combined: {verdict.long_label} (mean signal {mean_str})")
    for ev in evaluations:
        lines.append(f"  {ev.name}: {ev.recommendation.name} — {ev.description}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command("signal")
@click.option(
    "--as-of",
    "as_of",
    default=None,
    metavar="DATE",
    help="Date to evaluate (YYYY-MM-DD). Defaults to latest date in daily_prices.",
)
@click.option(
    "--db",
    "db_path",
    default=str(db.DEFAULT_DB_PATH),
    show_default=True,
    help="Path to SQLite database.",
)
def main(as_of: str | None, db_path: str) -> None:
    """Output a buy/don't-buy E10 signal."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    path = pathlib.Path(db_path)
    if not path.exists():
        raise click.ClickException(
            f"Database not found: {db_path}. Run 'uv run python -m fuel_signal.db' first."
        )
    conn = db.open_db(path)

    try:
        as_of_date = as_of or _latest_daily_date(conn)

        try:
            datetime.date.fromisoformat(as_of_date)
        except ValueError:
            raise click.BadParameter(
                f"must be YYYY-MM-DD, got {as_of_date!r}", param_hint="--as-of"
            )

        gap_start, gap_end = _gap_boundaries(conn)
        if gap_start and gap_end and gap_start <= as_of_date <= gap_end:
            click.echo(
                f"WARNING: {as_of_date} falls in the forward-fill gap "
                f"({gap_start} to {gap_end}). Prices are fabricated — signal is unreliable.",
                err=True,
            )

        output = build_signals(conn, as_of_date)
        click.echo(output)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
