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
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import click
import numpy as np

import fuel_signal.db as db
from fuel_signal.config import PREFERRED_STATIONS
from fuel_signal.cycle import CycleDetector, CycleState
from fuel_signal.dates import date_from_int as _date_from_int
from fuel_signal.features import (
    DELTA_LAG_DAYS,
    FEATURE_COLUMNS,
    _build_feature_dict,
    _calendar_delta,
    _lga_phase_std_per_date,
    _network_px_std_per_date,
)
from fuel_signal.lga_leadership import (
    LGA_FEATURE_COUNCILS,
    compute_pit_strict_days_since_trough,
    lga_slug,
)
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

    The optional dicts (station_lga_brand, lga_mean_by_key, brand_mean_by_key,
    stickiness_by_key, lga_days_since_by_key, network_px_std_by_date,
    network_px_std_delta_3d_by_date, lga_phase_std_by_date,
    lga_phase_std_delta_3d_by_date) are populated by load_history and consumed by
    ModelStrategy.decide to supply Phase 4 features. Tests that construct
    PriceHistory directly without a DB can leave them empty (default).
    """

    avg_series: list[tuple[str, float]]                 # [(date_str, cents), ...] sorted
    station_prices: dict[int, list[tuple[str, float]]]  # station_code → [(date_str, cents), ...]
    station_lga_brand: dict[int, tuple[str | None, str | None]] = field(default_factory=dict)
    lga_mean_by_key: dict[tuple[str, str], float] = field(default_factory=dict)
    brand_mean_by_key: dict[tuple[str, str], float] = field(default_factory=dict)
    stickiness_by_key: dict[tuple[int, str], float] = field(default_factory=dict)
    lga_days_since_by_key: dict[tuple[str, str], int | None] = field(default_factory=dict)
    network_px_std_by_date: dict[str, float] = field(default_factory=dict)
    network_px_std_delta_3d_by_date: dict[str, float] = field(default_factory=dict)
    lga_phase_std_by_date: dict[str, float] = field(default_factory=dict)
    lga_phase_std_delta_3d_by_date: dict[str, float] = field(default_factory=dict)
    # Cycle-detector factory: builds the detector from avg_series. Defaults to the
    # production CycleDetector; an experiment can inject an alternate cycle-feature
    # source (e.g. a regime-local detector) in-process, no production-code branch
    # needed. The detector must expose detect(as_of) -> CycleState | None.
    detector_factory: Callable[[list[tuple[str, float]]], CycleDetector] = CycleDetector

    def __post_init__(self) -> None:
        self._avg_by_date: dict[str, float] = dict(self.avg_series)
        self._avg_dates: list[str] = [d for d, _ in self.avg_series]
        self._detector: CycleDetector = self.detector_factory(self.avg_series)
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

    def lga_mean_at(self, station_code: int, as_of: str) -> float | None:
        lga, _ = self.station_lga_brand.get(station_code, (None, None))
        if lga is None:
            return None
        return self.lga_mean_by_key.get((as_of, lga))

    def brand_mean_at(self, station_code: int, as_of: str) -> float | None:
        _, brand = self.station_lga_brand.get(station_code, (None, None))
        if brand is None:
            return None
        return self.brand_mean_by_key.get((as_of, brand))

    def stickiness_score_at(self, station_code: int, as_of: str) -> float | None:
        return self.stickiness_by_key.get((station_code, as_of))

    def lga_days_since_at(self, as_of: str, lga: str) -> float | None:
        val = self.lga_days_since_by_key.get((as_of, lga))
        return float(val) if val is not None else None

    def network_px_std_at(self, as_of: str) -> float | None:
        return self.network_px_std_by_date.get(as_of)

    def network_px_std_delta_3d_at(self, as_of: str) -> float | None:
        return self.network_px_std_delta_3d_by_date.get(as_of)

    def lga_phase_std_at(self, as_of: str) -> float | None:
        return self.lga_phase_std_by_date.get(as_of)

    def lga_phase_std_delta_3d_at(self, as_of: str) -> float | None:
        return self.lga_phase_std_delta_3d_by_date.get(as_of)


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
    """Decides via P(BUY) ≥ threshold.

    Source is either a joblib artifact (``model_path``) or an already-fitted
    in-memory pipeline (``pipeline`` + ``feature_columns``). The in-memory path
    lets an experiment harness score a freshly trained/calibrated model without
    writing a joblib to ``data/models/`` (no production-artifact side effects).
    Exactly one of ``model_path`` / ``pipeline`` must be given.
    """

    model_path: pathlib.Path | None = None
    threshold: float = 0.40
    pipeline: Any = None
    feature_columns: list[str] | None = None

    def __post_init__(self) -> None:
        self.name: str = f"model(τ={self.threshold})"
        if self.pipeline is not None:
            if self.model_path is not None:
                raise ValueError("ModelStrategy: pass exactly one of model_path / pipeline, not both.")
            if not hasattr(self.pipeline, "predict_proba"):
                raise ValueError("ModelStrategy(pipeline=...) needs a predict_proba interface.")
            self._pipeline = self.pipeline
            self._feature_columns = list(self.feature_columns or FEATURE_COLUMNS)
            return
        if self.model_path is None:
            raise ValueError("ModelStrategy requires either model_path or pipeline.")
        import joblib  # defer import so non-ML callers don't pay the cost
        loaded = joblib.load(self.model_path)
        if isinstance(loaded, dict) and loaded.get("calibrated"):
            # Calibrated artifact stores sklearn primitives to avoid __main__ pickle issues.
            # Reconstruct _CalibratedPipeline here where fuel_signal.calibrate is importable.
            from fuel_signal.calibrate import _CalibratedPipeline
            self._feature_columns: list[str] = list(
                loaded.get("feature_columns", FEATURE_COLUMNS)
            )
            self._pipeline = _CalibratedPipeline(
                loaded["base_pipeline"], loaded["calibrator"], loaded["calibration_method"],
                self._feature_columns,
            )
        elif isinstance(loaded, dict):
            self._pipeline = loaded["pipeline"]
            self._feature_columns = list(loaded.get("feature_columns", FEATURE_COLUMNS))
        else:
            self._pipeline = loaded
            self._feature_columns = list(FEATURE_COLUMNS)

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
        lga_mean = history.lga_mean_at(station_code, as_of)
        brand_mean = history.brand_mean_at(station_code, as_of)
        stickiness = history.stickiness_score_at(station_code, as_of)
        features: dict[str, float | None] = _build_feature_dict(
            state, station_price, avg_price, lga_mean, brand_mean, stickiness
        )
        for lga in LGA_FEATURE_COUNCILS:
            features[f"days_since_trough_entry_{lga_slug(lga)}"] = (
                history.lga_days_since_at(as_of, lga)
            )
        features["network_px_std"] = history.network_px_std_at(as_of)
        features["network_px_std_delta_3d"] = history.network_px_std_delta_3d_at(as_of)
        features["lga_phase_std"] = history.lga_phase_std_at(as_of)
        features["lga_phase_std_delta_3d"] = history.lga_phase_std_delta_3d_at(as_of)
        X = np.array([[features[col] for col in self._feature_columns]], dtype=float)
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
    if interval_days <= 0:
        raise ValueError(f"interval_days must be > 0, got {interval_days}")
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
    eval_dates: list[str] | None = None,
    detector_factory: Callable[[list[tuple[str, float]]], CycleDetector] = CycleDetector,
) -> PriceHistory:
    """Load avg series, per-station prices, and Phase 4 feature caches from DB once.

    Pass the returned PriceHistory to run_backtest; strategies access it
    in-memory without further DB queries.

    eval_dates: if provided, PIT-strict days_since_trough_entry_<lga> features are
    pre-computed for exactly those dates (one detect_trough_events call per date×LGA).
    When None or empty, lga_days_since_by_key is empty and those features default to
    NaN during ModelStrategy.decide (acceptable for Phase 2 models; degrades Phase 4).
    """
    avg_series = db.average_price_series(conn)
    station_prices: dict[int, list[tuple[str, float]]] = {}
    for code in station_codes:
        prices = db.get_daily_prices(conn, code)
        if prices:
            station_prices[code] = prices

    if not station_codes:
        return PriceHistory(
            avg_series=avg_series,
            station_prices=station_prices,
            detector_factory=detector_factory,
        )

    fid = db.fuel_type_id(conn, "E10")
    _sc_ph = ", ".join(["?"] * len(station_codes))

    station_lga_brand: dict[int, tuple[str | None, str | None]] = {
        sc: (council, brand)
        for sc, council, brand in conn.execute(
            f"SELECT station_code, council, brand FROM stations"
            f" WHERE station_code IN ({_sc_ph})",
            station_codes,
        )
    }

    lga_mean_by_key: dict[tuple[str, str], float] = {
        (_date_from_int(date_int), lga): avg_decicents / 10
        for date_int, lga, avg_decicents in conn.execute(
            "SELECT dp.price_date, s.council, AVG(dp.price_decicents)"
            " FROM daily_prices dp"
            " JOIN stations s ON dp.station_code = s.station_code"
            " JOIN station_class sc ON dp.station_code = sc.station_code"
            "   AND dp.price_date = sc.snapshot_date"
            " WHERE dp.fuel_type_id = ? AND sc.class != 'Sticky'"
            "   AND s.council IS NOT NULL"
            " GROUP BY dp.price_date, s.council"
            " HAVING COUNT(*) >= 3",
            (fid,),
        )
    }

    brand_mean_by_key: dict[tuple[str, str], float] = {
        (_date_from_int(date_int), brand): avg_decicents / 10
        for date_int, brand, avg_decicents in conn.execute(
            "SELECT dp.price_date, s.brand, AVG(dp.price_decicents)"
            " FROM daily_prices dp"
            " JOIN stations s ON dp.station_code = s.station_code"
            " JOIN station_class sc ON dp.station_code = sc.station_code"
            "   AND dp.price_date = sc.snapshot_date"
            " WHERE dp.fuel_type_id = ? AND sc.class != 'Sticky'"
            "   AND s.brand IS NOT NULL"
            " GROUP BY dp.price_date, s.brand"
            " HAVING COUNT(*) >= 3",
            (fid,),
        )
    }

    stickiness_by_key: dict[tuple[int, str], float] = {
        (sc, _date_from_int(date_int)): decicents / 10
        for sc, date_int, decicents in conn.execute(
            "SELECT station_code, snapshot_date, median_premium_decicents"
            " FROM station_class"
            f" WHERE station_code IN ({_sc_ph})"
            "   AND median_premium_decicents IS NOT NULL",
            station_codes,
        )
    }

    lga_days_since_by_key: dict[tuple[str, str], int | None] = (
        compute_pit_strict_days_since_trough(conn, eval_dates)
        if eval_dates
        else {}
    )

    avg_date_strs: list[str] = [d for d, _ in avg_series]
    network_px_std_by_date = _network_px_std_per_date(conn, fid)
    network_px_std_delta_3d_by_date = _calendar_delta(
        network_px_std_by_date, avg_date_strs, DELTA_LAG_DAYS
    )
    lga_phase_std_by_date = _lga_phase_std_per_date(
        lga_days_since_by_key, eval_dates or []
    )
    lga_phase_std_delta_3d_by_date = _calendar_delta(
        lga_phase_std_by_date, eval_dates or [], DELTA_LAG_DAYS
    )

    return PriceHistory(
        avg_series=avg_series,
        station_prices=station_prices,
        station_lga_brand=station_lga_brand,
        lga_mean_by_key=lga_mean_by_key,
        brand_mean_by_key=brand_mean_by_key,
        stickiness_by_key=stickiness_by_key,
        lga_days_since_by_key=lga_days_since_by_key,
        network_px_std_by_date=network_px_std_by_date,
        network_px_std_delta_3d_by_date=network_px_std_delta_3d_by_date,
        lga_phase_std_by_date=lga_phase_std_by_date,
        lga_phase_std_delta_3d_by_date=lga_phase_std_delta_3d_by_date,
        detector_factory=detector_factory,
    )


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
    if tank_size <= 0:
        raise click.UsageError("--tank-size must be > 0.")
    if daily_use < 0:
        raise click.UsageError("--daily-use must be >= 0.")
    if eval_interval <= 0:
        raise click.UsageError("--eval-interval must be > 0.")
    if strategy_name == "model" and model_path is None:
        raise click.UsageError("--model-path is required when --strategy is 'model'.")
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

    eval_dates = _evaluation_dates(start_date, end_date, eval_interval)
    conn = db.open_db(path)
    try:
        history = load_history(conn, codes, eval_dates=eval_dates)
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
