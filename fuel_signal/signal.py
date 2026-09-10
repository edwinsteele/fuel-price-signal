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
from fuel_signal.config import PREFERRED_STATIONS, STATION_ROUTE_DAYS
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


# ---------------------------------------------------------------------------
# Live decision layer
# ---------------------------------------------------------------------------
#
# labels.py deliberately keeps tank level, refuelling cadence and station
# availability OUT of the training target ("these are real factors — they should
# be applied as filters on top of P(BUY) at decision time"). This section is that
# decision layer: it takes the model's per-station P(BUY) and answers the two
# questions actually asked at 7am — *which* station, and *how much*.

DEFAULT_MODEL_PATH = pathlib.Path("data/models/lgbm_calibrated.joblib")

# The locked operating point. Sourced here rather than re-derived: see
# docs/STATUS.md § Current production model, which is canonical for tau.
MODEL_THRESHOLD = 0.25

# History the bounded feature caches need behind as_of. Generous on purpose —
# it must clear the 3-day calendar lag on the delta features by a wide margin.
MODEL_LOOKBACK_DAYS = 120

# Network drift (cents/day, over a week) past which waiting is worth something.
FALLING_CENTS_PER_DAY = -0.5

# How much cheaper an off-route station must be before a detour is worth it.
DIVERSION_WORTH_CENTS = 3.0

# Window for measuring which way the network is moving.
_DRIFT_WINDOW_DAYS = 7

_WEEKDAY_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


@dataclass
class StationView:
    """One preferred station as the morning table sees it."""

    code: int
    label: str
    price: float | None
    prob: float | None
    # None → on the daily commute. Otherwise the weekdays it is reachable.
    route_days: frozenset[int] | None

    def reachable_on(self, day: datetime.date) -> bool:
        return self.route_days is None or day.weekday() in self.route_days

    def next_reachable(self, day: datetime.date) -> tuple[datetime.date, int] | None:
        """(date, days_ahead) of the next day this station is passed, or None if daily."""
        if self.route_days is None:
            return None
        for ahead in range(0, 8):
            candidate = day + datetime.timedelta(days=ahead)
            if candidate.weekday() in self.route_days:
                return candidate, ahead
        return None

    def route_label(self) -> str:
        if self.route_days is None:
            return "daily"
        return "/".join(_WEEKDAY_NAMES[d] for d in sorted(self.route_days))


def _last_real_price_date(conn: sqlite3.Connection) -> str | None:
    """Newest E10 date in `prices` — real observations only.

    Deliberately not `daily_prices`: fill.py forward-fills that table, so its
    MAX is a fabricated day whenever the snapshot job has missed a run. Telling
    the two apart is the whole point of the staleness banner.
    """
    fid = db.fuel_type_id(conn, "E10")
    row = conn.execute(
        "SELECT MAX(price_date) FROM prices WHERE fuel_type_id = ?", (fid,)
    ).fetchone()
    return db._date_from_int(row[0]) if row and row[0] is not None else None


def _network_drift(series: list[tuple[str, float]], as_of: str) -> float | None:
    """Cents/day change in the network average over the trailing week.

    Preferred over CycleState.last_3_gradients for the brim/bridge call: a single
    forward-filled final day drives that last gradient to exactly 0.0 (it does
    today), which would read as "flat" on precisely the days the data is stalest.
    A week-long difference is not moved by one repeated value.
    """
    at_or_before = [(d, p) for d, p in series if d <= as_of]
    if not at_or_before:
        return None
    today_date, today_price = at_or_before[-1]
    cutoff = (
        datetime.date.fromisoformat(today_date)
        - datetime.timedelta(days=_DRIFT_WINDOW_DAYS)
    ).isoformat()
    earlier = [(d, p) for d, p in at_or_before if d <= cutoff]
    if not earlier:
        return None
    then_date, then_price = earlier[-1]
    span = (
        datetime.date.fromisoformat(today_date) - datetime.date.fromisoformat(then_date)
    ).days
    if span <= 0:
        return None
    return (today_price - then_price) / span


def model_probabilities(
    conn: sqlite3.Connection,
    as_of: str,
    station_codes: list[int],
    model_path: pathlib.Path,
) -> dict[int, float]:
    """P(BUY) per station from the locked artifact, or {} if it cannot be scored.

    Imports backtest lazily — it pulls in LightGBM, which costs several seconds
    and is pure waste on the rule-based fallback path.
    """
    if not model_path.exists():
        logger.info("model artifact %s not found; falling back to rules", model_path)
        return {}
    try:
        from fuel_signal.backtest import ModelStrategy, load_history

        since = (
            datetime.date.fromisoformat(as_of)
            - datetime.timedelta(days=MODEL_LOOKBACK_DAYS)
        ).isoformat()
        history = load_history(
            conn, station_codes, eval_dates=[as_of], since_date=since
        )
        strategy = ModelStrategy(model_path=model_path, threshold=MODEL_THRESHOLD)
        scored: dict[int, float] = {}
        for code in station_codes:
            prob = strategy.probability(as_of, code, history)
            if prob is not None:
                scored[code] = prob
        return scored
    except Exception as exc:  # noqa: BLE001 — the CLI must still print a table
        logger.warning("model scoring failed (%s); falling back to rules", exc)
        return {}


def _fill_advice(
    best: StationView,
    drift: float | None,
    threshold: float,
) -> tuple[str, str]:
    """(headline, reason) — the buy/wait call and how much to put in.

    Two separate questions. Whether to buy at all is the model's (that is what it
    was trained and backtested for). How much is the network's: a falling network
    means cheaper fuel is coming, so bridge to it; a flat or rising one means
    today is as good as it gets, so brim.
    """
    buying = best.prob is not None and best.prob >= threshold
    falling = drift is not None and drift < FALLING_CENTS_PER_DAY

    if buying and not falling:
        return (
            f"FILL UP — {best.label} @ {best.price:.1f}c, brim it.",
            "Model says buy and the network is not falling: no cheaper fuel in sight.",
        )
    if buying and falling:
        return (
            f"WORTH A STOP — {best.label} @ {best.price:.1f}c, but bridge, don't brim.",
            f"Good local price, but the network is falling {abs(drift):.1f}c/day — "
            "buy enough to get by and keep some tank for later.",
        )
    if falling:
        return (
            "WAIT if you can.",
            f"Network falling {abs(drift):.1f}c/day and no station clears the buy "
            f"bar. If you must fill: {best.label} @ {best.price:.1f}c, minimum only.",
        )
    return (
        "WAIT if you can.",
        f"No station clears the buy bar today. If you must fill: {best.label} "
        f"@ {best.price:.1f}c, minimum only.",
    )


def _freshness_note(as_of: str, last_real: str | None, today: datetime.date) -> str:
    """Banner text about how much to trust the date being shown.

    Two different situations that must not be conflated:

    * The DB has newer real data than the date asked for — the caller passed a
      historical ``--as-of`` on purpose. Nothing is stale; say which vintage the
      DB is at and move on.
    * The newest real observation is itself behind today — the snapshot job has
      missed runs and the prices really are old. That is the warning.

    Age is measured from the last REAL observation, not from ``as_of``: `as_of`
    defaults to the newest row in `daily_prices`, which fill.py forward-fills, so
    measuring against it understates the age by exactly the fabricated days.
    """
    if last_real is not None and last_real > as_of:
        return f"  (historical view; DB holds data to {last_real})"
    newest = last_real or as_of
    age = (today - datetime.date.fromisoformat(newest)).days
    if age <= 0:
        return ""
    plural = "day" if age == 1 else "days"
    real = f", last real reading {newest}" if newest != as_of else ""
    return f"  !! {age} {plural} stale{real}"


def build_signals(
    conn: sqlite3.Connection,
    as_of_date: str,
    preferred_stations: dict[int, str] | None = None,
    *,
    model_path: pathlib.Path | None = None,
    today: datetime.date | None = None,
    explain: bool = False,
) -> str:
    """Render the morning decision table for the given date.

    Ranks stations by today's price among those actually reachable today, calls
    the fill size, and reports off-route stations with the arithmetic needed to
    decide whether they are worth a detour.

    Station ordering is by PRICE, never by P(BUY). The label's cheapness test is
    each station's OWN trailing percentile (labels.py condition 2), so P(BUY) is
    station-relative and is not comparable across stations — sorting by it points
    at whichever pump has fallen furthest against its own history, which is
    routinely the dearest one on the list.
    """
    stations = (
        preferred_stations if preferred_stations is not None else PREFERRED_STATIONS
    )
    today = today or datetime.date.today()
    model_path = model_path or DEFAULT_MODEL_PATH

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
    drift = _network_drift(series, as_of_date)
    probs = model_probabilities(conn, as_of_date, list(stations), model_path)

    views = [
        StationView(
            code=code,
            label=label,
            price=_station_price_at(conn, code, as_of_date),
            prob=probs.get(code),
            route_days=STATION_ROUTE_DAYS.get(code),
        )
        for code, label in stations.items()
    ]

    on_route = sorted(
        (v for v in views if v.reachable_on(today) and v.price is not None),
        key=lambda v: v.price,
    )
    off_route = sorted(
        (v for v in views if not v.reachable_on(today) and v.price is not None),
        key=lambda v: v.price,
    )
    unpriced = [v for v in views if v.price is None]

    # --- header -----------------------------------------------------------
    day_num = state.days_since_last_peak + 1          # 1-indexed
    cycle_len = round(state.mean_cycle_length)
    day_str = f"{cycle_len}+" if day_num > cycle_len else str(day_num)
    if drift is None:
        drift_str = "drift unknown"
    elif drift > 0.05:
        drift_str = f"rising {drift:.1f}c/day"
    elif drift < -0.05:
        drift_str = f"falling {abs(drift):.1f}c/day"
    else:
        drift_str = "flat"

    lines = [
        f"E10 - {as_of_date}{_freshness_note(as_of_date, _last_real_price_date(conn), today)}",
        f"Network {avg_current_price:.1f}c, {drift_str}"
        f"  |  cycle day {day_str}/{cycle_len}"
        f"  |  last cycle {state.last_cycle_min:.1f}-{state.last_cycle_max:.1f}c",
        "",
    ]

    # --- the call ---------------------------------------------------------
    if on_route:
        headline, reason = _fill_advice(on_route[0], drift, MODEL_THRESHOLD)
        lines += [f"  {headline}", f"  {reason}", ""]
    elif off_route:
        # Priced stations exist, just none reachable today — a routing answer.
        nearest = off_route[0]
        nxt = nearest.next_reachable(today)
        assert nxt is not None and nxt[1] >= 1   # off-route ⇒ not today
        when = f"in {nxt[1]}d"
        lines += [
            "  Nothing on your route today.",
            f"  Cheapest preferred station is {nearest.label} @ "
            f"{nearest.price:.1f}c, next passed {when}.",
            "",
        ]
    else:
        # No station has a price at all — a data answer, not a routing one.
        lines += [
            "  No price data for any preferred station on this date.",
            "",
        ]

    # --- tables -----------------------------------------------------------
    scored = bool(probs)
    head = f"  {'':3}{'STATION':<24}{'PRICE':>8}{'VS NET':>9}"
    head += f"{'P(BUY)':>9}" if scored else ""

    def row(v: StationView, marker: str) -> str:
        cells = f"  {marker:<3}{v.label:<24}{v.price:>7.1f}c{v.price - avg_current_price:>+9.1f}"
        if scored:
            cells += f"{v.prob:>9.2f}" if v.prob is not None else f"{'-':>9}"
        return cells

    if on_route:
        lines.append(head)
        for i, v in enumerate(on_route):
            lines.append(row(v, "->" if i == 0 else ""))

    if off_route:
        lines.append("")
        if not on_route:
            # No on-route table printed above, so this section carries the header.
            lines.append(head)
        for v in off_route:
            # A station lands in off_route precisely because it is NOT reachable
            # today, so next_reachable is always >= 1 day out and non-None (a
            # station with no route_days is never off-route in the first place).
            nxt = v.next_reachable(today)
            assert nxt is not None and nxt[1] >= 1
            when = f"in {nxt[1]}d, {_WEEKDAY_NAMES[nxt[0].weekday()]}"
            lines.append(f"  OFF ROUTE ({v.route_label()}) - next pass {when}")
            lines.append(row(v, ""))
            if on_route:
                gap = v.price - on_route[0].price
                if gap <= -DIVERSION_WORTH_CENTS:
                    lines.append(
                        f"     {abs(gap):.1f}c cheaper than {on_route[0].label}"
                        f" - worth timing a fill for {when}."
                    )
                else:
                    lines.append(
                        f"     {gap:+.1f}c vs {on_route[0].label} - no reason to divert."
                    )

    for v in unpriced:
        lines.append(f"  {v.label}: no price data")

    if not scored:
        lines += [
            "",
            "  (no model artifact - prices and cycle only; run train_lgbm + calibrate"
            " for P(BUY))",
        ]

    if explain:
        station_gradients: dict[str, float] = {}
        for code, label in stations.items():
            g = _station_latest_gradient(conn, code, as_of_date)
            if g is not None:
                station_gradients[label] = g
        evaluations = evaluate_all_signals(state, avg_current_price, station_gradients)
        verdict = combine_signals(evaluations)
        mean_str = (
            "n/a" if np.isnan(verdict.mean_value) else f"{verdict.mean_value:+.2f}"
        )
        lines += ["", f"  legacy rule signals: {verdict.long_label} (mean {mean_str})"]
        for ev in evaluations:
            lines.append(f"    {ev.name}: {ev.recommendation.name} - {ev.description}")

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
@click.option(
    "--model",
    "model_path",
    type=pathlib.Path,
    default=DEFAULT_MODEL_PATH,
    show_default=True,
    help="Calibrated model artifact. Falls back to prices+cycle only if absent.",
)
@click.option(
    "--explain",
    is_flag=True,
    help="Also print the legacy four-rule breakdown.",
)
def main(
    as_of: str | None, db_path: str, model_path: pathlib.Path, explain: bool
) -> None:
    """Output the morning E10 decision table."""
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

        output = build_signals(
            conn, as_of_date, model_path=model_path, explain=explain
        )
        click.echo(output)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
