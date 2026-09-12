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
import zoneinfo
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


def _station_price_on(
    conn: sqlite3.Connection, station_code: int, as_of_date: str
) -> float | None:
    """E10 price (cents) for this station ON as_of_date, or None.

    Exact date, deliberately — NOT `_station_price_at`'s "at or before". Ranking
    stations means a stale carried price can win the comparison and send the
    driver to a station that is not reporting: BP Springwood has a 268-day hole in
    `daily_prices` (2022-08-31 -> 2023-05-26), and mid-gap the "at or before"
    lookup happily returns 185.9c from up to 134 days earlier.

    Exact-date is the right test because `daily_prices` is the FILLED table:
    fill.py has already applied MAX_GAP_FILL_DAYS all-or-nothing per gap when
    writing it, so a row existing on this date *is* fill.py's ruling that the
    date is legitimately covered. That keeps accepted forward fills and drops
    exactly the spans fill.py refused — no second copy of the cap logic, which
    `PriceHistory.station_price_at` warns is easy to get subtly wrong.
    """
    fid = db.fuel_type_id(conn, "E10")
    row = conn.execute(
        """SELECT price_decicents FROM daily_prices
           WHERE station_code = ? AND fuel_type_id = ? AND price_date = ?""",
        (station_code, fid, db._date_to_int(as_of_date)),
    ).fetchone()
    return row[0] / 10 if row else None


def _station_price_at(
    conn: sqlite3.Connection, station_code: int, as_of_date: str
) -> float | None:
    """Most recent E10 price (cents) at or before as_of_date.

    Retained for the legacy rule path; prefer `_station_price_on` for anything
    that ranks or recommends a station.
    """
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

# Network drift (cents/day over a week) below which the tool says "cheaper fuel is
# coming, bridge" rather than "today is as good as it gets, brim".
#
# UNVALIDATED, AND SITTING IN THE WORST PLACE. This was reasoned, not measured:
# roughly 3.5 c/L over a week felt like the point where waiting earns its
# inconvenience. Measured against 3287 days of the Sydney average afterwards, the
# drift distribution has p50 = -0.47 — so this cut lands within 0.03 of the MEDIAN
# and splits days 49/51. That is the densest part of the distribution and hence the
# least stable place to cut: the verdict flips on noise, and the sensitivity is
# steep (a cut of -0.75 bridges on 39.9% of days, -0.25 on 55.6%).
#
# A distribution-derived cut would be defensible — p25 = -1.12 gives "genuinely
# falling" on ~30% of days rather than "very slightly below average" on half. The
# principled fix is to score brim-vs-bridge policies against realised CPL with the
# tank engine in backtest.py, which is what every other lock parameter in this
# project had to clear. Do not tune this by eye; see
# docs/memory/brim-bridge-threshold-unvalidated.md.
FALLING_CENTS_PER_DAY = -0.5

# How much cheaper an off-route station must be before a detour is worth it.
DIVERSION_WORTH_CENTS = 3.0

# Window for measuring which way the network is moving.
_DRIFT_WINDOW_DAYS = 7

# How far behind the last real observation normally sits, in days, before
# anything is wrong. `daily-snapshot.yml` runs at 10:00 UTC — 8pm AEST / 9pm
# AEDT — so the snapshot for day D is collected on the evening of D and a run
# during the morning decision window on D+1 legitimately sees D as the newest
# real reading. Warning at age 1 would fire every single morning of normal
# operation, and a banner that is always on cannot signal an outage (Codex
# review, PR #410). Age 2+ means a run was actually missed.
EXPECTED_LAG_DAYS = 1

_WEEKDAY_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")

# Every date question this module asks is a Sydney question: which weekday the
# driver is passing which station, and how old NSW price data (stamped in local
# dates) is. `date.today()` answers in the HOST's zone, so on a UTC box — CI, or
# a server — the Sydney morning window before 10:00 AEST / 11:00 AEDT still reads
# as yesterday. At 07:00 Wednesday in Sydney a UTC host computes Tuesday, and a
# Wed/Sun station reads OFF ROUTE on exactly the morning it is reachable.
_SYDNEY = "Australia/Sydney"


def _today_in_sydney() -> datetime.date:
    """Today's date in Sydney, whatever the host clock is set to."""
    try:
        tz = zoneinfo.ZoneInfo(_SYDNEY)
    except zoneinfo.ZoneInfoNotFoundError as exc:   # pragma: no cover - bare container
        # Deliberately fatal rather than falling back to date.today(): a silent
        # fallback reintroduces exactly the off-by-one-day routing bug this
        # exists to prevent, and it would do so invisibly.
        raise click.ClickException(
            f"No timezone database for {_SYDNEY} ({exc}). Install system tzdata "
            "(or `uv pip install tzdata`) — routing needs the Sydney date."
        ) from exc
    return datetime.datetime.now(tz).date()


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
        from fuel_signal.features import DELTA_LAG_DAYS

        as_of_date = datetime.date.fromisoformat(as_of)
        since = (as_of_date - datetime.timedelta(days=MODEL_LOOKBACK_DAYS)).isoformat()
        # The lag date must be in eval_dates, not just within since_date's window.
        # load_history builds lga_phase_std ONLY for eval_dates, and _calendar_delta
        # then needs both `as_of` and `as_of - DELTA_LAG_DAYS` present to produce
        # lga_phase_std_delta_3d — the last column of the locked 54-feat set. With
        # eval_dates=[as_of] the lag date is absent, the delta comes back None, and
        # every live score silently feeds the model NaN for a graduated feature
        # (LightGBM accepts NaN natively, so nothing raises).
        lag = (as_of_date - datetime.timedelta(days=DELTA_LAG_DAYS)).isoformat()
        history = load_history(
            conn, station_codes, eval_dates=[lag, as_of], since_date=since
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
    on_route: list[StationView],
    drift: float | None,
    threshold: float,
    *,
    scored: bool,
    rule_verdict: CombinedVerdict,
) -> tuple[str, str]:
    """(headline, reason) — the buy/wait call and how much to put in.

    Two separate questions. Whether to buy at all is the model's (that is what it
    was trained and backtested for). How much is the network's: a falling network
    means cheaper fuel is coming, so bridge to it; a flat or rising one means
    today is as good as it gets, so brim.

    Two things this must not do:

    * **Manufacture a model verdict with no model.** With the artifact absent
      every ``prob`` is None, and treating that as "fails the threshold" prints a
      confident WAIT sourced from nothing. Fall back to the legacy rules and say
      so, so the reader knows which signal is talking.
    * **Contradict its own table.** The verdict is about the station you would
      actually drive to (the cheapest reachable one), but a dearer station may
      well carry a higher P(BUY) — that is normal, since the probability is
      measured against each station's own history, not across stations. Saying
      "nothing clears the buy bar" while the table shows something that does is
      a straight falsehood, so check before claiming it.
    """
    best = on_route[0]
    falling = drift is not None and drift < FALLING_CENTS_PER_DAY
    bridge_note = (
        f" Network falling {abs(drift):.1f}c/day, so bridge rather than brim."
        if falling
        else ""
    )

    if not scored:
        verdict = rule_verdict.long_label
        if verdict == "BUY":
            return (
                f"FILL UP (rules) — {best.label} @ {best.price:.1f}c"
                f"{', bridge' if falling else ', brim it'}.",
                "No model artifact, so this is the legacy rule signal, not the "
                f"trained model.{bridge_note}",
            )
        return (
            f"WAIT if you can (rules) — legacy signal says {verdict}.",
            "No model artifact, so there is no trained-model opinion today. "
            f"If you must fill: {best.label} @ {best.price:.1f}c.{bridge_note}",
        )

    buying = best.prob is not None and best.prob >= threshold
    if buying and not falling:
        return (
            f"FILL UP — {best.label} @ {best.price:.1f}c, brim it.",
            "Model says buy and the network is not falling: no cheaper fuel in sight.",
        )
    if buying:
        return (
            f"WORTH A STOP — {best.label} @ {best.price:.1f}c, but bridge, don't brim.",
            f"Good local price, but the network is falling {abs(drift):.1f}c/day — "
            "buy enough to get by and keep some tank for later.",
        )

    # Not buying at the cheapest. Does anything else reachable clear the bar?
    others = [
        v
        for v in on_route[1:]
        if v.prob is not None and v.prob >= threshold
    ]
    if others:
        alt = others[0]
        if best.prob is None:
            return (
                "WAIT if you can.",
                f"{best.label} @ {best.price:.1f}c is the cheapest on route but hasn't "
                f"been scored. {alt.label} clears the buy bar instead, at "
                f"+{alt.price - best.price:.1f}c — P(BUY) is measured per station, so "
                "it is not a reason to drive there.",
            )
        return (
            "WAIT if you can.",
            f"{best.label} @ {best.price:.1f}c is the cheapest on route but doesn't "
            f"clear its own buy bar. {alt.label} does, at +{alt.price - best.price:.1f}c "
            "— P(BUY) is measured per station, so it is not a reason to drive there.",
        )
    return (
        "WAIT if you can.",
        f"Nothing on route clears the buy bar today. If you must fill: "
        f"{best.label} @ {best.price:.1f}c, minimum only.{bridge_note}",
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
    if age <= EXPECTED_LAG_DAYS:
        return ""
    plural = "day" if age == 1 else "days"
    real = f", last real reading {newest}" if newest != as_of else ""
    return f"  !! {age} {plural} stale{real}"


@dataclass
class SignalPayload:
    """The expensive, day-stable half of the signal.

    **The invariant, precisely:** every field on this class must be a function
    of ``(as_of_date, database)`` and *nothing else* — in particular, nothing
    derived from the wall clock ("now"), directly or transitively. That is the
    specific, checkable property that makes it safe to precompute once (this is
    the ~29s `load_history` cost — see `backtest.py:load_history`) and reuse
    across reads, e.g. a nightly cache read the next morning. A field that
    reads the clock is unsafe to cache no matter how correct it looks at the
    moment it was computed. If you add a field, ask which side of that line it
    is on — if it reads the clock, it belongs in `render_text`'s arguments
    instead, not here.

    Current fields obey it: per-station price and P(BUY), cycle state, network
    average and drift, the freshness *fact* (last real observation date — not
    its age, see below), and the rule-based verdict are all pure functions of
    ``as_of_date`` and the DB.

    Deliberately excluded: station reachability, diversion arithmetic, and
    freshness *age* — those are cheap, but depend on ``routing_day``/``now``,
    which can differ between when a cached payload was built and when it is
    read (station 261 is `frozenset({2, 6})`: a payload built 22:00 Tuesday
    and read 07:00 Wednesday must not report Wednesday's reachable station as
    off-route; likewise the staleness banner's *age* must be measured against
    the real current date at read time, not baked in — only the observation
    date it is measured from lives here). `render_text` takes
    ``routing_day``/``now`` fresh at read time to compute that half instead.
    See `test_rendering_the_same_payload_at_different_now_moves_only_the_freshness_banner`
    for the mechanical check of this invariant.
    """

    as_of_date: str
    preferred_stations: dict[int, str]
    state: CycleState
    avg_current_price: float
    drift: float | None
    station_prices: dict[int, float | None]
    station_probs: dict[int, float]
    rule_verdict: CombinedVerdict
    evaluations: list[SignalEvaluation]
    last_real_price_date: str | None


def compute_signal(
    conn: sqlite3.Connection,
    as_of_date: str,
    preferred_stations: dict[int, str] | None = None,
    *,
    model_path: pathlib.Path | None = None,
) -> SignalPayload:
    """Compute the expensive, day-stable half of the signal — see `SignalPayload`.

    No formatting concerns: this returns structured data for `render_text` (or
    any other consumer — an API endpoint, a nightly cache job) to work from.
    """
    stations = (
        preferred_stations if preferred_stations is not None else PREFERRED_STATIONS
    )
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

    station_prices = {
        code: _station_price_on(conn, code, as_of_date) for code in stations
    }

    # Always computed: it is the fallback verdict when no model artifact exists
    # (needing no model itself), and --explain reuses it in render_text below.
    station_gradients: dict[str, float] = {}
    for code, label in stations.items():
        g = _station_latest_gradient(conn, code, as_of_date)
        if g is not None:
            station_gradients[label] = g
    evaluations = evaluate_all_signals(state, avg_current_price, station_gradients)
    rule_verdict = combine_signals(evaluations)

    return SignalPayload(
        as_of_date=as_of_date,
        preferred_stations=dict(stations),
        state=state,
        avg_current_price=avg_current_price,
        drift=drift,
        station_prices=station_prices,
        station_probs=probs,
        rule_verdict=rule_verdict,
        evaluations=evaluations,
        last_real_price_date=_last_real_price_date(conn),
    )


def render_text(
    payload: SignalPayload,
    *,
    routing_day: datetime.date | None = None,
    now: datetime.date | None = None,
    explain: bool = False,
) -> str:
    """Render the morning decision table from a `SignalPayload`.

    Ranks stations by today's price among those actually reachable today, calls
    the fill size, and reports off-route stations with the arithmetic needed to
    decide whether they are worth a detour. This is the cheap, day-sensitive
    half of the signal — the on/off-route partition, diversion arithmetic, and
    freshness age all need ``routing_day``/``now`` as they stand at READ time,
    not whenever `compute_signal` happened to run.

    Two separate dates, deliberately not one (Codex review, PR #410):

    * ``routing_day`` — whose weekday decides which stations are reachable.
      Defaults to ``payload.as_of_date``'s own day so the output is a pure
      function of the inputs: a historical ``--as-of`` must render identically
      whenever it is run, or `signal-regression.yml`'s fixed historical
      invocations diff against the wall clock instead of against the code. The
      CLI overrides it with the current Sydney date for live use, where "what
      day is it" really is today's question and ``as_of`` may be several days
      stale.
    * ``now`` — the reference for how old the data is. Always the real current
      date; a historical ``as_of`` does not make the DB fresher.

    Collapsing these into one date is the bug that made a historical signal move
    the Wed/Sun station on and off route depending on the hour it was run.

    Station ordering is by PRICE, never by P(BUY). The label's cheapness test is
    each station's OWN trailing percentile (labels.py condition 2), so P(BUY) is
    station-relative and is not comparable across stations — sorting by it points
    at whichever pump has fallen furthest against its own history, which is
    routinely the dearest one on the list.
    """
    as_of_date = payload.as_of_date
    state = payload.state
    avg_current_price = payload.avg_current_price
    drift = payload.drift
    probs = payload.station_probs
    rule_verdict = payload.rule_verdict
    evaluations = payload.evaluations

    now = now or _today_in_sydney()
    today = routing_day or datetime.date.fromisoformat(as_of_date)

    views = [
        StationView(
            code=code,
            label=label,
            price=payload.station_prices.get(code),
            prob=probs.get(code),
            route_days=STATION_ROUTE_DAYS.get(code),
        )
        for code, label in payload.preferred_stations.items()
    ]

    reachable = [v for v in views if v.reachable_on(today)]
    on_route = sorted(
        (v for v in reachable if v.price is not None),
        key=lambda v: v.price,
    )
    # Reachable today but no price yet — a data problem, not a routing one
    # (#418). Partitioning on price first used to fold these into `unpriced`
    # and empty `on_route`, which made a station that genuinely IS on the
    # route report as "Nothing on your route today."
    on_route_unpriced = [v for v in reachable if v.price is None]
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
        f"E10 - {as_of_date}{_freshness_note(as_of_date, payload.last_real_price_date, now)}",
        f"Network {avg_current_price:.1f}c, {drift_str}"
        f"  |  cycle day {day_str}/{cycle_len}"
        f"  |  last cycle {state.last_cycle_min:.1f}-{state.last_cycle_max:.1f}c",
        "",
    ]

    # --- the call ---------------------------------------------------------
    if on_route:
        headline, reason = _fill_advice(
            on_route,
            drift,
            MODEL_THRESHOLD,
            scored=bool(probs),
            rule_verdict=rule_verdict,
        )
        lines += [f"  {headline}", f"  {reason}", ""]
    elif on_route_unpriced:
        # Something IS reachable today; it simply has no price yet.
        names = ", ".join(v.label for v in on_route_unpriced)
        lines += [
            f"  {names} on route today, but no price data yet.",
            "",
        ]
    elif off_route:
        # Nothing reachable today at all — priced stations exist, just off
        # route. A routing answer, distinct from the case above.
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
            day_name = _WEEKDAY_NAMES[nxt[0].weekday()]
            when = f"in {nxt[1]}d, {day_name}"           # "next pass in 3d, Sun"
            on_day = f"{day_name} ({nxt[1]}d away)"      # "...a fill for Sun (3d away)"
            lines.append(f"  OFF ROUTE ({v.route_label()}) - next pass {when}")
            lines.append(row(v, ""))
            if on_route:
                gap = v.price - on_route[0].price
                if gap <= -DIVERSION_WORTH_CENTS:
                    lines.append(
                        f"     {abs(gap):.1f}c cheaper than {on_route[0].label}"
                        f" - worth timing a fill for {on_day}."
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
        mean_str = (
            "n/a"
            if np.isnan(rule_verdict.mean_value)
            else f"{rule_verdict.mean_value:+.2f}"
        )
        lines += [
            "",
            f"  legacy rule signals: {rule_verdict.long_label} (mean {mean_str})",
        ]
        for ev in evaluations:
            lines.append(f"    {ev.name}: {ev.recommendation.name} - {ev.description}")

    return "\n".join(lines)


def build_signals(
    conn: sqlite3.Connection,
    as_of_date: str,
    preferred_stations: dict[int, str] | None = None,
    *,
    model_path: pathlib.Path | None = None,
    routing_day: datetime.date | None = None,
    now: datetime.date | None = None,
    explain: bool = False,
) -> str:
    """Compute and render the morning decision table in one call.

    A thin compatibility wrapper: `render_text(compute_signal(...))` is the
    real split (#415) — call those two directly for a consumer that wants to
    reuse the expensive half (`compute_signal`) across multiple cheap reads
    (`render_text`), e.g. a nightly cache job feeding a live API. This wrapper
    exists for callers, like the CLI below, that just want the one-shot string.
    """
    payload = compute_signal(
        conn, as_of_date, preferred_stations, model_path=model_path
    )
    return render_text(payload, routing_day=routing_day, now=now, explain=explain)


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

        # Live use (no --as-of): route by today, since as_of may be stale by days.
        # Explicit --as-of: leave routing_day None so build_signals derives it
        # from that date and the render is reproducible.
        output = build_signals(
            conn,
            as_of_date,
            model_path=model_path,
            routing_day=None if as_of else _today_in_sydney(),
            explain=explain,
        )
        click.echo(output)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
