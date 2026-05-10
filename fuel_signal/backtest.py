"""Backtest engine: replay historical prices through a purchasing strategy.

Loads the full price series ONCE from the DB, then evaluates strategies
in-memory at arbitrary historical dates — no per-date DB round-trips.

CycleDetector.detect(as_of_date) slices its internal pd.Series to the given
date (PIT-safe per CLAUDE.md), so the same detector object is safe to reuse
across all evaluation dates in a backtest run.

Usage:
    uv run python -m fuel_signal.backtest --station 414 --start 2023-01-01 --end 2024-12-31
    uv run python -m fuel_signal.backtest --preferred --strategy rule_based
    uv run python -m fuel_signal.backtest \\
        --station 414 --strategy model \\
        --model-path data/models/logreg.joblib --threshold 0.40
"""

from __future__ import annotations

import bisect
import datetime
import math
import pathlib
import sqlite3
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import click
import numpy as np

import fuel_signal.db as db
from fuel_signal.config import PREFERRED_STATIONS
from fuel_signal.cycle import CycleDetector, CycleState
from fuel_signal.features import FEATURE_COLUMNS, _build_feature_dict
from fuel_signal.signal import combine_signals, evaluate_all_signals

# ---------------------------------------------------------------------------
# Pre-loaded price data (loaded once; strategies query in-memory)
# ---------------------------------------------------------------------------

@dataclass
class PriceHistory:
    """Pre-loaded daily price data for all backtest dates.

    The CycleDetector is built once in __post_init__ from the full avg_series.
    detect(as_of) slices in-memory, so PIT-safety is preserved across the
    entire backtest run without rebuilding the detector per evaluation date.
    """

    avg_series: list[tuple[str, float]]                 # [(date_str, cents), ...] sorted
    station_prices: dict[int, list[tuple[str, float]]]  # station_code → [(date_str, cents), ...]

    def __post_init__(self) -> None:
        self._avg_by_date: dict[str, float] = dict(self.avg_series)
        self._avg_dates: list[str] = [d for d, _ in self.avg_series]
        self._detector: CycleDetector = CycleDetector(self.avg_series)
        self._station_dates: dict[int, list[str]] = {
            code: [d for d, _ in prices]
            for code, prices in self.station_prices.items()
        }

    def cycle_state(self, as_of: str) -> CycleState | None:
        return self._detector.detect(as_of)

    def avg_price_at(self, as_of: str) -> float | None:
        """Latest Sydney average price on or before as_of."""
        idx = bisect.bisect_right(self._avg_dates, as_of) - 1
        return self.avg_series[idx][1] if idx >= 0 else None

    def station_price_at(self, station_code: int, as_of: str) -> float | None:
        """Latest E10 price (cents) at station on or before as_of."""
        dates = self._station_dates.get(station_code)
        prices = self.station_prices.get(station_code)
        if not dates or not prices:
            return None
        idx = bisect.bisect_right(dates, as_of) - 1
        return prices[idx][1] if idx >= 0 else None

    def station_gradient_at(self, station_code: int, as_of: str, window: int = 4) -> float | None:
        """Latest np.gradient of the last `window` daily prices at or before as_of."""
        dates = self._station_dates.get(station_code)
        prices = self.station_prices.get(station_code)
        if not dates or not prices:
            return None
        idx = bisect.bisect_right(dates, as_of)
        recent = prices[max(0, idx - window):idx]
        if len(recent) < 2:
            return None
        vals = np.array([p for _, p in recent], dtype=float)
        return float(np.gradient(vals)[-1])


# ---------------------------------------------------------------------------
# Strategy protocol + concrete implementations
# ---------------------------------------------------------------------------

@runtime_checkable
class Strategy(Protocol):
    name: str

    def decide(self, as_of: str, station_code: int, history: PriceHistory) -> bool:
        """Return True to fill up (BUY), False to defer (WAIT / DONT_BUY)."""
        ...


@dataclass
class AlwaysBuyStrategy:
    name: str = "always_buy"

    def decide(self, as_of: str, station_code: int, history: PriceHistory) -> bool:
        return True


@dataclass
class RuleBasedSignalStrategy:
    """Wraps the four-signal combine logic from signal.py."""

    name: str = "rule_based"

    def decide(self, as_of: str, station_code: int, history: PriceHistory) -> bool:
        state = history.cycle_state(as_of)
        if state is None:
            return True  # insufficient data → default buy
        avg_price = history.avg_price_at(as_of)
        if avg_price is None:
            return True
        gradient = history.station_gradient_at(station_code, as_of)
        station_gradients: dict[str, float] = (
            {"station": gradient} if gradient is not None else {}
        )
        evaluations = evaluate_all_signals(state, avg_price, station_gradients)
        verdict = combine_signals(evaluations)
        return verdict.label.strip() == "BUY"


@dataclass
class ModelStrategy:
    """Loads a sklearn pipeline and decides via P(BUY) ≥ threshold."""

    model_path: pathlib.Path
    threshold: float = 0.40

    def __post_init__(self) -> None:
        import joblib  # defer import so non-ML callers don't pay the cost
        self.name: str = f"model(τ={self.threshold})"
        self._pipeline = joblib.load(self.model_path)

    def decide(self, as_of: str, station_code: int, history: PriceHistory) -> bool:
        state = history.cycle_state(as_of)
        if state is None:
            return True
        station_price = history.station_price_at(station_code, as_of)
        if station_price is None:
            return True
        avg_price = history.avg_price_at(as_of)
        if avg_price is None:
            return True
        features = _build_feature_dict(state, station_price, avg_price)
        X = np.array([[features[col] for col in FEATURE_COLUMNS]])
        prob = float(self._pipeline.predict_proba(X)[0][1])
        return prob >= self.threshold


# ---------------------------------------------------------------------------
# Tank model parameters
# ---------------------------------------------------------------------------

@dataclass
class TankParams:
    """Vehicle and refuelling behaviour parameters."""

    tank_size_litres: float = 50.0
    daily_consumption_litres: float = 50.0 / 14   # empties in 14 days
    evaluation_interval_days: int = 7              # how often signal is checked
    floor_fraction: float = 0.10                   # emergency half-fill threshold


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    strategy_name: str
    station_code: int
    start_date: str
    end_date: str
    total_spend_cents: float
    total_litres: float
    fill_events: int
    realised_cpl: float           # cents per litre
    always_buy_cpl: float | None = None
    savings_vs_always_buy_pct: float | None = None

    def set_baseline(self, always_buy_cpl: float) -> None:
        """Compute savings percentage relative to the always-buy CPL."""
        self.always_buy_cpl = always_buy_cpl
        if always_buy_cpl > 0 and not math.isnan(self.realised_cpl):
            self.savings_vs_always_buy_pct = (
                (always_buy_cpl - self.realised_cpl) / always_buy_cpl * 100
            )


# ---------------------------------------------------------------------------
# Engine helpers
# ---------------------------------------------------------------------------

def _evaluation_dates(start_date: str, end_date: str, interval_days: int) -> list[str]:
    """Return ISO date strings at interval_days spacing from start_date to end_date inclusive."""
    current = datetime.date.fromisoformat(start_date)
    end = datetime.date.fromisoformat(end_date)
    dates: list[str] = []
    while current <= end:
        dates.append(current.isoformat())
        current += datetime.timedelta(days=interval_days)
    return dates


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

def run_backtest(
    history: PriceHistory,
    strategy: Strategy,
    station_code: int,
    start_date: str,
    end_date: str,
    tank: TankParams | None = None,
) -> BacktestResult:
    """Replay strategy over [start_date, end_date] and return spend metrics.

    Tank starts at 50% full. Between evaluations it depletes at
    tank.daily_consumption_litres per day. On each evaluation date the
    strategy decides whether to fill up (True) or wait (False). If the
    strategy waits but the tank is below tank.floor_fraction, an emergency
    half-fill is triggered to prevent running dry before the next evaluation.
    """
    if tank is None:
        tank = TankParams()

    eval_dates = _evaluation_dates(start_date, end_date, tank.evaluation_interval_days)
    if not eval_dates:
        return BacktestResult(
            strategy_name=strategy.name,
            station_code=station_code,
            start_date=start_date,
            end_date=end_date,
            total_spend_cents=0.0,
            total_litres=0.0,
            fill_events=0,
            realised_cpl=float("nan"),
        )

    tank_level = tank.tank_size_litres * 0.5  # start at 50%
    total_spend = 0.0
    total_litres = 0.0
    fill_events = 0

    for i, as_of in enumerate(eval_dates):
        if i > 0:
            tank_level = max(
                0.0,
                tank_level - tank.daily_consumption_litres * tank.evaluation_interval_days,
            )

        price = history.station_price_at(station_code, as_of)
        if price is None:
            continue

        if strategy.decide(as_of, station_code, history):
            litres = tank.tank_size_litres - tank_level
            if litres > 0:
                total_spend += litres * price
                total_litres += litres
                fill_events += 1
                tank_level = tank.tank_size_litres
        elif tank_level / tank.tank_size_litres < tank.floor_fraction:
            # Emergency half-fill to avoid running dry before next evaluation
            target = tank.tank_size_litres * 0.5
            litres = max(0.0, target - tank_level)
            if litres > 0:
                total_spend += litres * price
                total_litres += litres
                fill_events += 1
                tank_level = target

    realised_cpl = total_spend / total_litres if total_litres > 0 else float("nan")
    return BacktestResult(
        strategy_name=strategy.name,
        station_code=station_code,
        start_date=start_date,
        end_date=end_date,
        total_spend_cents=total_spend,
        total_litres=total_litres,
        fill_events=fill_events,
        realised_cpl=realised_cpl,
    )


# ---------------------------------------------------------------------------
# DB loader
# ---------------------------------------------------------------------------

def load_history(
    conn: sqlite3.Connection,
    station_codes: list[int],
) -> PriceHistory:
    """Load avg series and per-station prices from DB once.

    Pass the returned PriceHistory to run_backtest; strategies access it
    in-memory without further DB queries.
    """
    avg_series = db.average_price_series(conn)
    station_prices: dict[int, list[tuple[str, float]]] = {}
    for code in station_codes:
        prices = db.get_daily_prices(conn, code)
        if prices:
            station_prices[code] = prices
    return PriceHistory(avg_series=avg_series, station_prices=station_prices)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_results_table(results: list[BacktestResult]) -> None:
    click.echo(f"\n{'Strategy':<22} {'CPL (c/L)':>10} {'vs AlwaysBuy':>13} {'Fills':>6} {'Litres':>8}")
    click.echo("-" * 63)
    for r in results:
        cpl = f"{r.realised_cpl:10.1f}" if not math.isnan(r.realised_cpl) else "       n/a"
        savings = (
            f"{r.savings_vs_always_buy_pct:+12.1f}%"
            if r.savings_vs_always_buy_pct is not None
            else "          n/a"
        )
        click.echo(f"{r.strategy_name:<22} {cpl} {savings} {r.fill_events:6d} {r.total_litres:8.0f}L")


@click.command("backtest")
@click.option(
    "--station", "station_codes", type=int, multiple=True,
    help="Station code(s) to run. Repeatable.",
)
@click.option("--preferred", is_flag=True, help="Run on all preferred stations from config.")
@click.option("--start", "start_date", required=True, metavar="DATE", help="Start date YYYY-MM-DD.")
@click.option("--end", "end_date", required=True, metavar="DATE", help="End date YYYY-MM-DD.")
@click.option(
    "--strategy",
    "strategy_name",
    type=click.Choice(["always_buy", "rule_based", "model", "all"]),
    default="all",
    show_default=True,
    help="Strategy to run ('all' compares all available strategies).",
)
@click.option(
    "--model-path", "model_path", type=pathlib.Path, default=None,
    help="Path to joblib model pipeline (required for --strategy model/all).",
)
@click.option(
    "--threshold", type=float, default=0.40, show_default=True,
    help="Model probability threshold for BUY decision.",
)
@click.option("--tank-size", "tank_size", type=float, default=50.0, show_default=True, help="Tank capacity in litres.")
@click.option(
    "--daily-use", "daily_use", type=float, default=round(50.0 / 14, 3), show_default=True,
    help="Daily fuel consumption in litres.",
)
@click.option(
    "--eval-interval", "eval_interval", type=int, default=7, show_default=True,
    help="Days between signal evaluations.",
)
@click.option(
    "--db", "db_path", default=str(db.DEFAULT_DB_PATH), show_default=True,
    help="Path to SQLite database.",
)
def main(  # noqa: PLR0913
    station_codes: tuple[int, ...],
    preferred: bool,
    start_date: str,
    end_date: str,
    strategy_name: str,
    model_path: pathlib.Path | None,
    threshold: float,
    tank_size: float,
    daily_use: float,
    eval_interval: int,
    db_path: str,
) -> None:
    """Replay purchasing strategies over historical prices and compare spend."""
    if not station_codes and not preferred:
        raise click.UsageError("Specify --station CODE or --preferred.")

    codes: list[int] = list(station_codes)
    if preferred:
        codes.extend(c for c in PREFERRED_STATIONS if c not in codes)
    if not codes:
        raise click.UsageError("No stations to run — check config.PREFERRED_STATIONS.")

    path = pathlib.Path(db_path)
    if not path.exists():
        raise click.ClickException(
            f"Database not found: {db_path}. Run 'uv run python -m fuel_signal.db' first."
        )

    conn = db.open_db(path)
    try:
        history = load_history(conn, codes)
    finally:
        conn.close()

    tank = TankParams(
        tank_size_litres=tank_size,
        daily_consumption_litres=daily_use,
        evaluation_interval_days=eval_interval,
    )

    strategies: list[Strategy] = [AlwaysBuyStrategy()]
    if strategy_name in ("rule_based", "all"):
        strategies.append(RuleBasedSignalStrategy())
    if strategy_name in ("model", "all") and model_path is not None:
        strategies.append(ModelStrategy(model_path=model_path, threshold=threshold))

    for station_code in codes:
        label = PREFERRED_STATIONS.get(station_code, str(station_code))
        click.echo(f"\n=== Station {station_code} ({label}) — {start_date} to {end_date} ===")

        always_buy_cpl: float | None = None
        results: list[BacktestResult] = []
        for strategy in strategies:
            result = run_backtest(history, strategy, station_code, start_date, end_date, tank)
            if isinstance(strategy, AlwaysBuyStrategy):
                always_buy_cpl = result.realised_cpl
            elif always_buy_cpl is not None and not math.isnan(always_buy_cpl):
                result.set_baseline(always_buy_cpl)
            results.append(result)

        _print_results_table(results)


if __name__ == "__main__":
    main()
