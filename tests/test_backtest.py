"""Tests for fuel_signal.backtest engine.

Synthetic price series are built entirely in-memory — no DB required.
All date arithmetic uses absolute dates in the past (not rolling
today-relative dates) because the synthetic data has no relationship
to real prices and no rolling-window freshness constraints.
"""

from __future__ import annotations

import datetime
import math
import sqlite3

from fuel_signal.backtest import (
    AlwaysBuyStrategy,
    BacktestResult,
    FillRecord,
    ModelStrategy,
    PriceHistory,
    RuleBasedSignalStrategy,
    TankParams,
    _evaluation_dates,
    format_tank_params,
    main,
    run_backtest,
    run_oracle_backtest,
    tank_params_fields,
    validate_never_dry,
)
from fuel_signal.features import FEATURE_COLUMNS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dates_from(start: str, n: int) -> list[str]:
    """Return n consecutive ISO date strings starting at start."""
    d = datetime.date.fromisoformat(start)
    return [(d + datetime.timedelta(days=i)).isoformat() for i in range(n)]


def _constant_history(
    station_code: int,
    start: str,
    n: int,
    price_cents: float,
) -> PriceHistory:
    """PriceHistory with a constant price at one station (avg == station price)."""
    dates = _dates_from(start, n)
    series = [(d, price_cents) for d in dates]
    return PriceHistory(avg_series=series, station_prices={station_code: series})


def _square_wave_history(
    station_code: int,
    start: str,
    n_cycles: int,
    high_cents: float = 200.0,
    low_cents: float = 150.0,
    half_period: int = 7,
) -> PriceHistory:
    """Alternating high/low square wave; each phase is `half_period` days long.

    The phase boundary aligns with eval_interval=half_period so that
    evaluations fall exactly on high or low days — no partial phases.
    """
    period = 2 * half_period
    total_days = n_cycles * period
    dates = _dates_from(start, total_days)
    prices = [
        (d, high_cents if (i % period) < half_period else low_cents)
        for i, d in enumerate(dates)
    ]
    return PriceHistory(avg_series=prices, station_prices={station_code: prices})


# ---------------------------------------------------------------------------
# _evaluation_dates
# ---------------------------------------------------------------------------

def test_evaluation_dates_basic():
    dates = _evaluation_dates("2024-01-01", "2024-01-22", 7)
    assert dates == ["2024-01-01", "2024-01-08", "2024-01-15", "2024-01-22"]


def test_evaluation_dates_empty_when_start_after_end():
    assert _evaluation_dates("2024-01-10", "2024-01-01", 7) == []


def test_evaluation_dates_single_day():
    assert _evaluation_dates("2024-01-01", "2024-01-01", 7) == ["2024-01-01"]


def test_evaluation_dates_stops_at_end():
    # interval of 7 days from Jan 1: Jan 1, 8, 15, 22 → Jan 25 is after last
    dates = _evaluation_dates("2024-01-01", "2024-01-25", 7)
    assert dates[-1] == "2024-01-22"
    assert "2024-01-29" not in dates


def test_evaluation_dates_raises_on_non_positive_interval():
    import pytest
    with pytest.raises(ValueError, match="interval_days must be > 0"):
        _evaluation_dates("2024-01-01", "2024-01-22", 0)
    with pytest.raises(ValueError, match="interval_days must be > 0"):
        _evaluation_dates("2024-01-01", "2024-01-22", -1)


# ---------------------------------------------------------------------------
# AlwaysBuyStrategy on constant prices
# ---------------------------------------------------------------------------

TANK = TankParams(
    tank_size_litres=50.0,
    daily_consumption_litres=50.0 / 14,
    evaluation_interval_days=7,
    floor_fraction=0.10,
)


def test_always_buy_cpl_equals_constant_price():
    """CPL == constant price when all prices are identical."""
    history = _constant_history(1, "2020-01-01", 90, 180.0)
    result = run_backtest(history, AlwaysBuyStrategy(), 1, "2020-01-01", "2020-03-31", TANK)
    assert not math.isnan(result.realised_cpl)
    assert abs(result.realised_cpl - 180.0) < 0.01


def test_always_buy_spend_equals_litres_times_constant_price():
    """total_spend == total_litres × price for constant prices (mean × fills)."""
    history = _constant_history(2, "2020-01-01", 30, 165.0)
    result = run_backtest(history, AlwaysBuyStrategy(), 2, "2020-01-01", "2020-01-28", TANK)
    assert abs(result.total_spend_cents - result.total_litres * 165.0) < 0.01


def test_always_buy_fill_count():
    """AlwaysBuy fills on every evaluation date when price data is available."""
    # 29 days, eval every 7 → evaluations on days 0, 7, 14, 21, 28 → 5 fills
    history = _constant_history(3, "2020-01-01", 29, 150.0)
    result = run_backtest(history, AlwaysBuyStrategy(), 3, "2020-01-01", "2020-01-29", TANK)
    assert result.fill_events == 5


def test_always_buy_total_litres_positive():
    history = _constant_history(4, "2020-01-01", 14, 170.0)
    result = run_backtest(history, AlwaysBuyStrategy(), 4, "2020-01-01", "2020-01-14", TANK)
    assert result.total_litres > 0


# ---------------------------------------------------------------------------
# Tank mechanics
# ---------------------------------------------------------------------------

def test_tank_starts_half_full():
    """On the very first evaluation the engine fills from 50% to 100% (25L for 50L tank)."""
    history = _constant_history(10, "2020-01-01", 2, 160.0)
    tank = TankParams(tank_size_litres=50.0, daily_consumption_litres=0.0, evaluation_interval_days=1)
    result = run_backtest(history, AlwaysBuyStrategy(), 10, "2020-01-01", "2020-01-01", tank)
    assert abs(result.total_litres - 25.0) < 0.01


def test_tank_depletion_between_fills():
    """Second fill is smaller when tank depleted correctly between evaluations."""
    # With 0 depletion, each fill tops up 25L. With depletion, second fill is same 25L
    # (since AlwaysBuy fills to 100% each time and depletion = 25L over 7 days).
    history = _constant_history(11, "2020-01-01", 14, 180.0)
    tank = TankParams(tank_size_litres=50.0, daily_consumption_litres=50.0 / 14, evaluation_interval_days=7)
    result = run_backtest(history, AlwaysBuyStrategy(), 11, "2020-01-01", "2020-01-14", tank)
    # 2 fills of 25L each = 50L total
    assert abs(result.total_litres - 50.0) < 0.5


def test_emergency_fill_fires_when_tank_near_empty():
    """When strategy says WAIT but tank hits floor, emergency half-fill triggers."""

    class AlwaysWaitStrategy:
        name = "always_wait"

        def decide(self, *_):
            return False

    # Floor = 10% = 5L. Daily use = 50/14 ≈ 3.57 L/day.
    # After 7 days: 25L depletes to 0L (below floor). Emergency fill should fire.
    history = _constant_history(20, "2020-01-01", 15, 200.0)
    tank = TankParams(
        tank_size_litres=50.0,
        daily_consumption_litres=50.0 / 14,
        evaluation_interval_days=7,
        floor_fraction=0.10,
    )
    result = run_backtest(
        history, AlwaysWaitStrategy(), 20, "2020-01-01", "2020-01-14", tank
    )
    # Day 0: tank=25L, WAIT, 25L > 5L → no emergency.
    # Day 7: tank=0L, WAIT, 0L < 5L → emergency half-fill.
    assert result.fill_events == 1


def test_no_emergency_when_tank_above_floor():
    """No fill when strategy says WAIT and tank is comfortably above floor."""

    class AlwaysWaitStrategy:
        name = "always_wait"

        def decide(self, *_):
            return False

    # Zero depletion → tank stays at 50% → well above floor=10% → never fills.
    history = _constant_history(21, "2020-01-01", 14, 180.0)
    tank = TankParams(
        tank_size_litres=50.0,
        daily_consumption_litres=0.0,
        evaluation_interval_days=7,
        floor_fraction=0.10,
    )
    result = run_backtest(
        history, AlwaysWaitStrategy(), 21, "2020-01-01", "2020-01-14", tank
    )
    assert result.fill_events == 0


def test_waiting_strategy_cannot_strand_tank_at_3d_cadence():
    """Regression for fps-5mn: an always-wait strategy on the default tank at
    3-day cadence used to run dry (fps-fii found 109 dry events / 389L under the
    old floor-only emergency rule). With the fix, run_backtest must complete
    without raising the never-dry check, and every emergency fill must actually
    cover the next interval (no fill of 0 litres leaving a level below the next
    depletion)."""

    class AlwaysWaitStrategy:
        name = "always_wait"

        def decide(self, *_):
            return False

    history = _constant_history(60, "2020-01-01", 400, 190.0)
    tank = TankParams(evaluation_interval_days=3)
    result = run_backtest(
        history, AlwaysWaitStrategy(), 60, "2020-01-01", "2020-12-01", tank,
        collect_fills=True,
    )
    # No exception raised above == the never-dry check never fired. Every fill
    # on an always-wait strategy must be an emergency half-fill that actually
    # bought litres (a 0-litre "emergency" would leave the tank exactly as
    # stranded as before the fix).
    assert result.fill_events > 0
    assert all(f.emergency and f.litres > 0 for f in result.fills)


def test_run_backtest_raises_runtimeerror_on_unsafe_tank_with_adversarial_strategy():
    """A tank config validate_never_dry() flags must raise — not silently clamp
    — when an always-wait strategy actually drives it dry. Regression for a
    CodeRabbit finding on the initial fps-5mn fix: a bare ``assert`` is stripped
    under ``python -O``, which would let the clamp mask this exact failure."""
    import pytest

    class AlwaysWaitStrategy:
        name = "always_wait"

        def decide(self, *_):
            return False

    tank = TankParams(
        tank_size_litres=50.0, daily_consumption_litres=30.0,
        evaluation_interval_days=1, floor_fraction=0.10,
    )
    assert validate_never_dry(tank) != []  # this is exactly what it should flag
    history = _constant_history(71, "2020-01-01", 10, 180.0)
    with pytest.raises(RuntimeError, match="tank ran dry"):
        run_backtest(history, AlwaysWaitStrategy(), 71, "2020-01-01", "2020-01-10", tank)


def test_final_evaluation_does_not_force_unnecessary_emergency_fill():
    """The depletion-survival trigger only protects a NEXT evaluation — the
    final eval date has none, so a wait there must not force an emergency buy
    purely because tank_level < depletion (only the floor-comfort trigger still
    applies on the last date). Regression for a CodeRabbit finding: forcing a
    buy on the terminal date inflates spend for a future evaluation that never
    happens. Config: 2 eval dates 1 day apart, D=15L, floor*size=5L; the second
    (and final) date lands at level=10L — above the floor, below D."""

    class AlwaysWaitStrategy:
        name = "always_wait"

        def decide(self, *_):
            return False

    history = _constant_history(72, "2020-01-01", 5, 180.0)
    tank = TankParams(
        tank_size_litres=50.0, daily_consumption_litres=15.0,
        evaluation_interval_days=1, floor_fraction=0.10,
    )
    result = run_backtest(
        history, AlwaysWaitStrategy(), 72, "2020-01-01", "2020-01-02", tank,
    )
    assert result.fill_events == 0


# ---------------------------------------------------------------------------
# The declared cadence lock (fps-oqz)
# ---------------------------------------------------------------------------

def test_canonical_cadence_is_daily():
    """The decision cadence is a declared lock parameter, not a knob
    (docs/CONVENTIONS.md § The decision cadence is a lock parameter). Re-locked
    7d -> 1d on 2026-08-22. Changing this literal is a re-lock, not a tweak: it
    silently redefines every realised CPL and headroom figure the project quotes,
    so it must come with a recorded rationale and a stated before/after."""
    assert TankParams().evaluation_interval_days == 1
    assert format_tank_params(TankParams()) == "50/3.571/1d/10%"


def test_cli_eval_interval_default_tracks_tankparams():
    """The CLI must not carry its own copy of the cadence. It did (a hard-coded
    default=7) and was missed by the fps-oqz re-lock's own change list, so a
    --eval-interval-less CLI run would have kept using 7 while every other caller
    moved to 1 — the two-defaults-disagree failure the tank_params stamp exists to
    make visible. Pinned to the dataclass so the next re-lock cannot repeat it."""
    param = next(p for p in main.params if p.name == "eval_interval")
    assert param.default == TankParams().evaluation_interval_days


def test_cli_daily_use_default_tracks_tankparams():
    """The CLI must not carry its own copy of daily consumption. It did (a
    rounded literal round(50.0 / 14, 3) = 3.571), which stamp-collided with
    TankParams.daily_consumption_litres (3.5714285714285716) under
    format_tank_params's :.3f rendering while consuming marginally
    differently (fps-q3p) — the same two-defaults-disagree failure the
    tank_params stamp exists to make visible. Pinned to the dataclass so the
    next re-lock cannot repeat it."""
    param = next(p for p in main.params if p.name == "daily_use")
    assert param.default == TankParams().daily_consumption_litres


# ---------------------------------------------------------------------------
# validate_never_dry (fps-5mn)
# ---------------------------------------------------------------------------

def test_validate_never_dry_default_tank_safe_at_canonical_cadences():
    """Default tank (50L, 50/14 L/day) is never-dry for every cadence 1-7 days —
    the fix closes the gap that made 3-6d unsafe under the old floor-only rule."""
    for cadence in range(1, 8):
        tank = TankParams(evaluation_interval_days=cadence)
        assert validate_never_dry(tank) == [], f"cadence {cadence}d should be safe"


def test_validate_never_dry_default_tank_unsafe_beyond_7d():
    """Beyond 7 days, one interval's depletion exceeds the emergency half-fill
    target (0.5 * tank_size), so even the fixed emergency rule can't rescue it."""
    for cadence in range(8, 15):
        tank = TankParams(evaluation_interval_days=cadence)
        assert validate_never_dry(tank) != [], f"cadence {cadence}d should be unsafe"


def test_validate_never_dry_rejects_old_readme_example():
    """README.md used to document --tank-size 60 --daily-use 4.5 --eval-interval 7.
    D=31.5L, and the 50%-full starting level (30L) is already inside the gap: it
    is above the floor (6L) yet cannot survive one interval, and the emergency
    half-fill target (30L) equals the starting level so it buys nothing. This
    stays a regression test for that exact known-bad config."""
    bad_tank = TankParams(
        tank_size_litres=60.0, daily_consumption_litres=4.5, evaluation_interval_days=7,
    )
    violations = validate_never_dry(bad_tank)
    assert violations != []


def test_validate_never_dry_catches_starting_level_not_just_wait_chain():
    """A config broken at the very first decision (starting level == emergency
    target, which buys 0 litres) must be caught even though no multi-step wait
    chain is ever explored — this is what a lattice-only check that started from
    an assumed-safe state, rather than the true 50%-full start, would miss."""
    bad_tank = TankParams(
        tank_size_litres=60.0, daily_consumption_litres=4.5, evaluation_interval_days=7,
    )
    violations = validate_never_dry(bad_tank)
    start_level = 0.5 * bad_tank.tank_size_litres
    assert any(abs(v.level - start_level) < 1e-6 for v in violations)


def test_validate_never_dry_accepts_replacement_readme_config():
    """The README's replacement example (--tank-size 60 --daily-use 4.0
    --eval-interval 7) must actually pass the validator it's meant to satisfy."""
    tank = TankParams(
        tank_size_litres=60.0, daily_consumption_litres=4.0, evaluation_interval_days=7,
    )
    assert validate_never_dry(tank) == []


def test_cli_rejects_unsafe_tank_config_before_touching_db():
    """The old README example (--tank-size 60 --daily-use 4.5 --eval-interval 7)
    must be rejected with a UsageError up front — before the DB-existence check —
    so a bad tank config can't silently produce a wrong CPL. A nonexistent
    --db path proves the rejection happens before any DB access is attempted."""
    from click.testing import CliRunner

    from fuel_signal.backtest import main

    result = CliRunner().invoke(main, [
        "--preferred", "--strategy", "rule_based",
        "--start", "2023-01-01", "--end", "2023-01-31",
        "--tank-size", "60", "--daily-use", "4.5", "--eval-interval", "7",
        "--db", "/nonexistent/path/does-not-exist.db",
    ])
    assert result.exit_code != 0
    assert "can run dry" in result.output


# ---------------------------------------------------------------------------
# Per-fill ledger (collect_fills) — #259 realised-gate capability
# ---------------------------------------------------------------------------

def test_collect_fills_off_by_default():
    """Default run records no ledger (zero behaviour change for existing callers)."""
    history = _constant_history(40, "2020-01-01", 29, 150.0)
    result = run_backtest(history, AlwaysBuyStrategy(), 40, "2020-01-01", "2020-01-29", TANK)
    assert result.fills == []


def test_collect_fills_one_record_per_fill_event():
    """With collect_fills, len(fills) == fill_events and each row is a FillRecord."""
    history = _constant_history(41, "2020-01-01", 29, 150.0)
    result = run_backtest(
        history, AlwaysBuyStrategy(), 41, "2020-01-01", "2020-01-29", TANK, collect_fills=True
    )
    assert result.fill_events == 5
    assert len(result.fills) == 5
    assert all(isinstance(f, FillRecord) for f in result.fills)


def test_collect_fills_ledger_reconciles_with_pooled_totals():
    """Per-fill spend/litres sum back to the pooled totals (no double counting)."""
    history = _square_wave_history(42, "2020-01-01", 6, high_cents=200.0, low_cents=140.0)
    result = run_backtest(
        history, AlwaysBuyStrategy(), 42, "2020-01-07", "2020-03-15", TANK, collect_fills=True
    )
    assert abs(sum(f.spend_cents for f in result.fills) - result.total_spend_cents) < 1e-6
    assert abs(sum(f.litres for f in result.fills) - result.total_litres) < 1e-6
    # spend_cents is price × litres for every row.
    for f in result.fills:
        assert abs(f.spend_cents - f.price * f.litres) < 1e-9
        assert f.station_code == 42


def test_collect_fills_marks_emergency():
    """A floor-triggered half-fill is flagged emergency=True."""

    class AlwaysWaitStrategy:
        name = "always_wait"

        def decide(self, *_):
            return False

    history = _constant_history(43, "2020-01-01", 15, 200.0)
    tank = TankParams(
        tank_size_litres=50.0,
        daily_consumption_litres=50.0 / 14,
        evaluation_interval_days=7,
        floor_fraction=0.10,
    )
    result = run_backtest(
        history, AlwaysWaitStrategy(), 43, "2020-01-01", "2020-01-14", tank, collect_fills=True
    )
    assert len(result.fills) == 1
    assert result.fills[0].emergency is True


# ---------------------------------------------------------------------------
# No price data for station
# ---------------------------------------------------------------------------

def test_no_station_data_returns_zero_fills_nan_cpl():
    """Missing station data produces fill_events=0 and NaN CPL."""
    history = PriceHistory(
        avg_series=[("2020-01-01", 180.0)],
        station_prices={},
    )
    result = run_backtest(history, AlwaysBuyStrategy(), 999, "2020-01-01", "2020-01-07", TANK)
    assert result.fill_events == 0
    assert math.isnan(result.realised_cpl)


def test_empty_date_range_returns_nan_cpl():
    history = _constant_history(30, "2020-01-01", 5, 180.0)
    result = run_backtest(history, AlwaysBuyStrategy(), 30, "2020-01-10", "2020-01-01", TANK)
    assert result.fill_events == 0
    assert math.isnan(result.realised_cpl)


# ---------------------------------------------------------------------------
# BacktestResult.set_baseline
# ---------------------------------------------------------------------------

def test_set_baseline_computes_savings_pct():
    result = BacktestResult(
        strategy_name="test",
        station_code=1,
        start_date="2020-01-01",
        end_date="2020-12-31",
        total_spend_cents=1000.0,
        total_litres=10.0,
        fill_events=5,
        realised_cpl=100.0,
    )
    result.set_baseline(always_buy_cpl=120.0)
    assert result.always_buy_cpl == 120.0
    # savings = (120 − 100) / 120 × 100 ≈ 16.67 %
    assert result.savings_vs_always_buy_pct is not None
    assert abs(result.savings_vs_always_buy_pct - 16.667) < 0.01


def test_set_baseline_negative_savings_when_strategy_worse():
    result = BacktestResult(
        strategy_name="bad",
        station_code=1,
        start_date="2020-01-01",
        end_date="2020-12-31",
        total_spend_cents=2000.0,
        total_litres=10.0,
        fill_events=5,
        realised_cpl=200.0,
    )
    result.set_baseline(always_buy_cpl=180.0)
    assert result.savings_vs_always_buy_pct is not None
    assert result.savings_vs_always_buy_pct < 0


# ---------------------------------------------------------------------------
# PriceHistory helper methods
# ---------------------------------------------------------------------------

def test_price_history_avg_price_at_returns_latest_on_or_before():
    series = [("2020-01-01", 160.0), ("2020-01-02", 165.0), ("2020-01-05", 170.0)]
    h = PriceHistory(avg_series=series, station_prices={})
    assert h.avg_price_at("2020-01-03") == 165.0  # latest on or before Jan 3
    assert h.avg_price_at("2020-01-05") == 170.0
    assert h.avg_price_at("2019-12-31") is None  # before all data


def test_price_history_station_price_at_returns_latest_on_or_before():
    prices = [("2020-01-01", 155.0), ("2020-01-03", 160.0)]
    h = PriceHistory(
        avg_series=[("2020-01-01", 155.0)],
        station_prices={42: prices},
    )
    assert h.station_price_at(42, "2020-01-02") == 155.0
    assert h.station_price_at(42, "2020-01-03") == 160.0
    assert h.station_price_at(99, "2020-01-01") is None  # unknown station


def test_price_history_gradient_returns_none_when_insufficient_data():
    prices = [("2020-01-01", 155.0)]  # only 1 point
    h = PriceHistory(avg_series=prices, station_prices={1: prices})
    assert h.station_gradient_at(1, "2020-01-01") is None


def test_price_history_gradient_computed_from_recent_window():
    # Flat series → gradient ≈ 0
    prices = [(f"2020-01-{i + 1:02d}", 180.0) for i in range(10)]
    h = PriceHistory(avg_series=prices, station_prices={1: prices})
    grad = h.station_gradient_at(1, "2020-01-10", window=4)
    assert grad is not None
    assert abs(grad) < 0.01


# ---------------------------------------------------------------------------
# Ordering: low-price oracle vs AlwaysBuy
#
# On an aligned square wave (high=200c, low=150c, half_period=7d, eval_interval=7d),
# a strategy that always buys at low prices pays less than AlwaysBuy which pays the mean.
# ---------------------------------------------------------------------------

class _LowPriceOracleStrategy:
    """Buys only when price is at or below `threshold_cents`."""

    name = "low_price_oracle"

    def __init__(self, threshold_cents: float) -> None:
        self._threshold = threshold_cents

    def decide(self, as_of: str, station_code: int, history: PriceHistory) -> bool:
        price = history.station_price_at(station_code, as_of)
        return price is not None and price <= self._threshold


def test_low_price_oracle_cpl_below_always_buy():
    """Oracle that buys only at the cheap phase pays less per litre than AlwaysBuy."""
    # 10 cycles × 14 days = 140 days. Eval every 7d aligns with phase boundaries.
    # Day 0,14,28... → high (200c); day 7,21,35... → low (150c).
    history = _square_wave_history(50, "2020-01-01", n_cycles=10, high_cents=200.0, low_cents=150.0, half_period=7)
    tank = TankParams(
        tank_size_litres=50.0,
        daily_consumption_litres=50.0 / 14,
        evaluation_interval_days=7,
        floor_fraction=0.10,
    )
    always_buy = run_backtest(history, AlwaysBuyStrategy(), 50, "2020-01-07", "2020-09-16", tank)
    oracle = run_backtest(history, _LowPriceOracleStrategy(150.0), 50, "2020-01-07", "2020-09-16", tank)

    assert not math.isnan(always_buy.realised_cpl)
    assert not math.isnan(oracle.realised_cpl)
    assert oracle.realised_cpl < always_buy.realised_cpl


# ---------------------------------------------------------------------------
# Perfect-foresight oracle (issue #262)
# ---------------------------------------------------------------------------

class _ReplayStrategy:
    """Buys (chosen-fill) on exactly the given set of dates; waits otherwise.

    Replaying the oracle's non-emergency fill dates through run_backtest must
    reproduce the oracle's spend/litres — the emergency rule fills the rest.
    """

    name = "replay"

    def __init__(self, buy_dates: set[str]) -> None:
        self._buy_dates = buy_dates

    def decide(self, as_of: str, station_code: int, history: PriceHistory) -> bool:
        return as_of in self._buy_dates


_ORACLE_TANK = TankParams(
    tank_size_litres=50.0,
    daily_consumption_litres=50.0 / 14,
    evaluation_interval_days=7,
    floor_fraction=0.10,
)


def test_oracle_plan_replays_identically_through_run_backtest():
    """The DP's accounting equals the engine's: replaying the oracle's chosen
    fills through run_backtest reproduces its spend, litres and CPL exactly."""
    history = _square_wave_history(
        50, "2020-01-01", n_cycles=10, high_cents=200.0, low_cents=150.0, half_period=7
    )
    oracle = run_oracle_backtest(
        history, 50, "2020-01-07", "2020-09-16", _ORACLE_TANK, collect_fills=True
    )
    chosen = {f.date for f in oracle.fills if not f.emergency}
    replay = run_backtest(
        history, _ReplayStrategy(chosen), 50, "2020-01-07", "2020-09-16", _ORACLE_TANK,
        collect_fills=True,
    )
    assert math.isclose(replay.total_spend_cents, oracle.total_spend_cents, rel_tol=1e-9)
    assert math.isclose(replay.total_litres, oracle.total_litres, rel_tol=1e-9)
    assert math.isclose(replay.realised_cpl, oracle.realised_cpl, rel_tol=1e-9)


def test_oracle_no_cheaper_than_any_feasible_strategy():
    """Oracle CPL is the floor: ≤ always-buy and ≤ a low-price threshold rule."""
    history = _square_wave_history(
        50, "2020-01-01", n_cycles=10, high_cents=200.0, low_cents=150.0, half_period=7
    )
    args = (50, "2020-01-07", "2020-09-16", _ORACLE_TANK)
    oracle = run_oracle_backtest(history, *args)
    always = run_backtest(history, AlwaysBuyStrategy(), *args)
    low = run_backtest(history, _LowPriceOracleStrategy(150.0), *args)

    assert not math.isnan(oracle.realised_cpl)
    assert oracle.realised_cpl <= always.realised_cpl + 1e-9
    assert oracle.realised_cpl <= low.realised_cpl + 1e-9
    # Genuine headroom exists on a square wave (the oracle skips full-fills at peaks).
    assert oracle.realised_cpl < always.realised_cpl


def test_oracle_equals_constant_price_when_no_edge():
    """On a flat price path there is nothing to exploit → oracle CPL == that price."""
    history = _constant_history(7, "2020-01-01", 200, price_cents=171.3)
    oracle = run_oracle_backtest(history, 7, "2020-01-07", "2020-06-01", _ORACLE_TANK)
    assert math.isclose(oracle.realised_cpl, 171.3, rel_tol=1e-9)


def test_oracle_empty_window_returns_nan():
    history = _constant_history(7, "2020-01-01", 30, price_cents=180.0)
    oracle = run_oracle_backtest(history, 7, "2020-01-10", "2020-01-01", _ORACLE_TANK)
    assert math.isnan(oracle.realised_cpl)
    assert oracle.fill_events == 0


def test_oracle_fills_only_collected_when_requested():
    history = _square_wave_history(50, "2020-01-01", n_cycles=6, half_period=7)
    args = (50, "2020-01-07", "2020-05-01", _ORACLE_TANK)
    assert run_oracle_backtest(history, *args).fills == []
    assert len(run_oracle_backtest(history, *args, collect_fills=True).fills) > 0


def test_oracle_stays_ceiling_and_faithful_in_gap_tank():
    """Tank config where the engine could run dry on a wait (floor·size < D and a
    reachable level lands in the (floor·size, D) gap). The oracle is held to
    never-dry plans there, but it must still (a) be ≤ always-buy — which is itself
    never-dry — and (b) return an engine-faithful plan."""
    # size 50, floor 5; D = 30/day × 1 = 30 ⇒ gap (5, 30); start level 25 ∈ gap.
    gap_tank = TankParams(
        tank_size_litres=50.0,
        daily_consumption_litres=30.0,
        evaluation_interval_days=1,
        floor_fraction=0.10,
    )
    history = _square_wave_history(
        50, "2020-01-01", n_cycles=20, high_cents=210.0, low_cents=150.0, half_period=1
    )
    args = (50, "2020-01-02", "2020-02-20", gap_tank)
    oracle = run_oracle_backtest(history, *args, collect_fills=True)
    always = run_backtest(history, AlwaysBuyStrategy(), *args)
    assert not math.isnan(oracle.realised_cpl)
    assert oracle.realised_cpl <= always.realised_cpl + 1e-9
    chosen = {f.date for f in oracle.fills if not f.emergency}
    replay = run_backtest(history, _ReplayStrategy(chosen), *args)
    assert math.isclose(replay.total_spend_cents, oracle.total_spend_cents, rel_tol=1e-9)
    assert math.isclose(replay.realised_cpl, oracle.realised_cpl, rel_tol=1e-9)


def test_oracle_handles_leading_no_data_prefix():
    """When the window opens before the station has any price, those eval dates
    are None (skipped). The never-dry gate still holds, so the returned plan is
    engine-faithful and conservation is intact (min-spend == min-CPL)."""
    # Gentle tank so a short no-data prefix doesn't drain the tank.
    gentle = TankParams(
        tank_size_litres=50.0,
        daily_consumption_litres=50.0 / 70,
        evaluation_interval_days=7,
        floor_fraction=0.10,
    )
    avg = [(d, 180.0) for d in _dates_from("2020-01-01", 200)]
    # Station prices begin 21 days in → the first ~3 eval dates are None.
    station = [(d, p) for d, p in _square_wave_history(
        9, "2020-01-22", n_cycles=12, high_cents=200.0, low_cents=150.0, half_period=7
    ).station_prices[9]]
    history = PriceHistory(avg_series=avg, station_prices={9: station})
    args = (9, "2020-01-01", "2020-06-01", gentle)
    oracle = run_oracle_backtest(history, *args, collect_fills=True)
    assert not math.isnan(oracle.realised_cpl)
    chosen = {f.date for f in oracle.fills if not f.emergency}
    replay = run_backtest(history, _ReplayStrategy(chosen), *args)
    assert math.isclose(replay.total_spend_cents, oracle.total_spend_cents, rel_tol=1e-9)
    assert math.isclose(replay.total_litres, oracle.total_litres, rel_tol=1e-9)
    assert math.isclose(replay.realised_cpl, oracle.realised_cpl, rel_tol=1e-9)


def test_oracle_prefers_deferring_to_a_known_cheaper_day():
    """With a gentle tank (long deferral horizon) the oracle reaches the global
    minimum price, so its CPL tracks the cheap days, not the path mean."""
    # D = (50/70)*7 = 5 L/step from a 50 L tank → can defer ~8 steps before the floor.
    gentle = TankParams(
        tank_size_litres=50.0,
        daily_consumption_litres=50.0 / 70,
        evaluation_interval_days=7,
        floor_fraction=0.10,
    )
    history = _square_wave_history(
        20, "2020-01-01", n_cycles=8, high_cents=220.0, low_cents=140.0, half_period=7
    )
    oracle = run_oracle_backtest(history, 20, "2020-01-01", "2020-07-01", gentle)
    # A long horizon lets it concentrate buying at the 140c lows.
    assert oracle.realised_cpl < 160.0


# ---------------------------------------------------------------------------
# RuleBasedSignalStrategy — smoke test (valid output, no crash)
# ---------------------------------------------------------------------------

def test_rule_based_produces_valid_result():
    """RuleBasedSignalStrategy runs without error and returns a BacktestResult."""
    # 6 cycles × 45 days = 270 days. CycleDetector needs ≥2 peaks to function.
    import math as _math

    n = 270
    period = 45
    start = "2018-01-01"
    dates = _dates_from(start, n)
    prices = [
        (d, 180.0 + 20.0 * _math.sin(2 * _math.pi * i / period))
        for i, d in enumerate(dates)
    ]
    history = PriceHistory(avg_series=prices, station_prices={100: prices})
    tank = TankParams(
        tank_size_litres=50.0,
        daily_consumption_litres=50.0 / 14,
        evaluation_interval_days=7,
    )
    result = run_backtest(history, RuleBasedSignalStrategy(), 100, "2018-03-01", "2018-10-01", tank)
    assert isinstance(result, BacktestResult)
    assert result.fill_events >= 0
    assert not math.isnan(result.realised_cpl)
    assert result.total_litres > 0


# ---------------------------------------------------------------------------
# ModelStrategy — regression: decide must not raise with Phase 4 feature caches
# ---------------------------------------------------------------------------

def test_model_strategy_decide_with_phase4_features(tmp_path):
    """ModelStrategy.decide runs without TypeError when PriceHistory carries Phase 4 caches.

    Regression for #174: _build_feature_dict requires 6 args but decide was passing 3.
    """
    import math as _math

    import joblib
    import numpy as np
    from sklearn.dummy import DummyClassifier

    # Minimal in-memory model that returns fixed probabilities regardless of input.
    clf = DummyClassifier(strategy="most_frequent")
    clf.fit(np.zeros((2, len(FEATURE_COLUMNS))), [0, 1])
    model_path = tmp_path / "dummy_model.joblib"
    joblib.dump(clf, model_path)

    # Sinusoidal series long enough for CycleDetector to find ≥2 peaks.
    n = 270
    period = 45
    start = "2018-01-01"
    dates = _dates_from(start, n)
    prices = [
        (d, 180.0 + 20.0 * _math.sin(2 * _math.pi * i / period))
        for i, d in enumerate(dates)
    ]
    station_code = 999

    history = PriceHistory(
        avg_series=prices,
        station_prices={station_code: prices},
        station_lga_brand={station_code: ("Penrith", "BP")},
        lga_mean_by_key={(d, "Penrith"): 175.0 for d, _ in prices},
        brand_mean_by_key={(d, "BP"): 178.0 for d, _ in prices},
        stickiness_by_key={(station_code, d): 5.0 for d, _ in prices},
    )

    strategy = ModelStrategy(model_path=model_path, threshold=0.40)
    result = strategy.decide("2018-09-01", station_code, history)
    assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# PriceHistory — network_px_std_at PIT safety
# ---------------------------------------------------------------------------

def _make_db_with_prices(tmp_path, suffix: str, prices_by_date: dict[str, float]) -> sqlite3.Connection:
    """Return an open sqlite3 connection with stations + daily_prices + station_class rows.

    Two competitive stations (|median_premium| = 0) contribute equal prices on each date.
    """
    from fuel_signal.db import create_schema, upsert_daily_prices, upsert_station_class_rows, upsert_stations

    db_path = tmp_path / f"test_{suffix}.db"
    conn = sqlite3.connect(str(db_path))
    create_schema(conn)

    for sc in (1001, 1002):
        upsert_stations(conn, [{
            "station_code": sc,
            "name": f"Station {sc}",
            "address": "1 Test St",
            "suburb": "Testville",
            "postcode": "2000",
            "brand": "TestBrand",
        }])

    rows_prices = []
    rows_class = []
    for date_str, price in prices_by_date.items():
        for sc in (1001, 1002):
            rows_prices.append((sc, "E10", date_str, price))
            rows_class.append((sc, date_str, "Competitive", 0))

    upsert_daily_prices(conn, rows_prices)
    upsert_station_class_rows(conn, rows_class)
    conn.commit()
    return conn


def test_network_px_std_at_pit_safe(tmp_path):
    """network_px_std_at(D) is identical whether DB ends at D or extends beyond D.

    PIT safety comes from _network_px_std_per_date joining station_class on
    snapshot_date = price_date, so future station_class rows don't affect D.
    """
    from fuel_signal.db import fuel_type_id
    from fuel_signal.features import _network_px_std_per_date

    date_d = "2021-03-01"
    prices_short = {date_d: 150.0}
    prices_long = {date_d: 150.0, "2021-03-02": 160.0, "2021-03-03": 170.0}

    conn_short = _make_db_with_prices(tmp_path, "short", prices_short)
    conn_long = _make_db_with_prices(tmp_path, "long", prices_long)

    fid_short = fuel_type_id(conn_short, "E10")
    fid_long = fuel_type_id(conn_long, "E10")

    std_short = _network_px_std_per_date(conn_short, fid_short).get(date_d)
    std_long = _network_px_std_per_date(conn_long, fid_long).get(date_d)

    # Both stations report the same price so std == 0.0. Asserting the concrete
    # value as well as equality between the two DBs guards against regressions
    # in _network_px_std_per_date, not just PIT-safety violations.
    assert std_short == std_long == 0.0

    conn_short.close()
    conn_long.close()


# ---------------------------------------------------------------------------
# ModelStrategy — decide with 54-feat calibrated artifact
# ---------------------------------------------------------------------------

def test_model_strategy_decide_with_54feat_calibrated_artifact(tmp_path):
    """ModelStrategy.decide works end-to-end with a calibrated artifact carrying all 54 features.

    Regression for #218: the 4 new network-dispersion columns must be present in
    the feature dict before np.array indexing by self._feature_columns.
    """
    import math as _math

    import joblib
    import numpy as np
    from sklearn.dummy import DummyClassifier
    from sklearn.isotonic import IsotonicRegression

    n_feat = len(FEATURE_COLUMNS)
    X_dummy = np.zeros((2, n_feat))
    base_clf = DummyClassifier(strategy="most_frequent")
    base_clf.fit(X_dummy, [0, 1])

    # Isotonic calibrator needs to be fitted on 1-D probabilities.
    raw_probs = base_clf.predict_proba(X_dummy)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_probs, [0, 1])

    calibrated_artifact = {
        "calibrated": True,
        "base_pipeline": base_clf,
        "calibrator": iso,
        "calibration_method": "isotonic",
        "feature_columns": list(FEATURE_COLUMNS),
    }
    model_path = tmp_path / "calibrated_54feat.joblib"
    joblib.dump(calibrated_artifact, model_path)

    n = 270
    period = 45
    start = "2018-01-01"
    dates = _dates_from(start, n)
    prices = [
        (d, 180.0 + 20.0 * _math.sin(2 * _math.pi * i / period))
        for i, d in enumerate(dates)
    ]
    station_code = 888

    history = PriceHistory(
        avg_series=prices,
        station_prices={station_code: prices},
        station_lga_brand={station_code: ("Penrith", "BP")},
        lga_mean_by_key={(d, "Penrith"): 175.0 for d, _ in prices},
        brand_mean_by_key={(d, "BP"): 178.0 for d, _ in prices},
        stickiness_by_key={(station_code, d): 3.0 for d, _ in prices},
        network_px_std_by_date={d: 4.5 for d, _ in prices},
        network_px_std_delta_3d_by_date={d: 0.2 for d, _ in prices},
        lga_phase_std_by_date={d: 6.0 for d, _ in prices},
        lga_phase_std_delta_3d_by_date={d: -0.1 for d, _ in prices},
    )

    strategy = ModelStrategy(model_path=model_path, threshold=0.40)
    result = strategy.decide("2018-09-01", station_code, history)
    assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# ModelStrategy — decide with brand-trough columns (fps-3i7)
# ---------------------------------------------------------------------------

def test_model_strategy_decide_with_brand_trough_columns(tmp_path):
    """ModelStrategy.decide populates days_since_trough_entry_<brand> natively.

    Regression for fps-3i7 (the batch0 pipeline abort): decide() looped
    LGA_FEATURE_COUNCILS to populate LGA trough columns but had no equivalent
    loop for brand trough columns, so any feature_columns set including one
    (e.g. batch_freeze.resolve_baseline_columns()'s discover_brand_feature_columns
    output) raised "feature ... is in feature_columns but no value was
    produced for it" on the very first decide() call — candidate-independent,
    since the realised backtest's baseline arm carries no extra_feature_provider.
    """
    import math as _math

    import joblib
    import numpy as np
    from sklearn.dummy import DummyClassifier

    feature_columns = [*FEATURE_COLUMNS, "days_since_trough_entry_bp"]
    clf = DummyClassifier(strategy="most_frequent")
    clf.fit(np.zeros((2, len(feature_columns))), [0, 1])
    model_path = tmp_path / "brand_trough_model.joblib"
    joblib.dump({"pipeline": clf, "feature_columns": feature_columns}, model_path)

    n = 270
    period = 45
    dates = _dates_from("2018-01-01", n)
    prices = [
        (d, 180.0 + 20.0 * _math.sin(2 * _math.pi * i / period))
        for i, d in enumerate(dates)
    ]
    station_code = 777

    history = PriceHistory(
        avg_series=prices,
        station_prices={station_code: prices},
        station_lga_brand={station_code: (None, "BP")},
        brand_mean_by_key={(d, "BP"): 178.0 for d, _ in prices},
        brand_days_since_by_key={(d, "BP"): 12 for d, _ in prices},
        qualifying_brands=["BP"],
    )

    strategy = ModelStrategy(model_path=model_path, threshold=0.40)
    result = strategy.decide("2018-09-01", station_code, history)
    assert isinstance(result, bool)


def test_model_strategy_decide_raises_when_brand_trough_column_unfed(tmp_path):
    """decide() still raises its named-feature ValueError when a declared
    feature_columns entry genuinely has no source — brand-trough support
    must not silently swallow a real config error the way a bare KeyError
    would.
    """
    import math as _math

    import joblib
    import numpy as np
    import pytest
    from sklearn.dummy import DummyClassifier

    feature_columns = [*FEATURE_COLUMNS, "days_since_trough_entry_bp"]
    clf = DummyClassifier(strategy="most_frequent")
    clf.fit(np.zeros((2, len(feature_columns))), [0, 1])
    model_path = tmp_path / "brand_trough_model_unfed.joblib"
    joblib.dump({"pipeline": clf, "feature_columns": feature_columns}, model_path)

    n = 270
    period = 45
    dates = _dates_from("2018-01-01", n)
    prices = [
        (d, 180.0 + 20.0 * _math.sin(2 * _math.pi * i / period))
        for i, d in enumerate(dates)
    ]
    station_code = 778

    # qualifying_brands is empty, so decide() never adds a
    # days_since_trough_entry_bp key to the feature dict at all.
    history = PriceHistory(
        avg_series=prices,
        station_prices={station_code: prices},
        station_lga_brand={station_code: (None, "BP")},
        brand_mean_by_key={(d, "BP"): 178.0 for d, _ in prices},
    )

    strategy = ModelStrategy(model_path=model_path, threshold=0.40)
    with pytest.raises(ValueError, match="days_since_trough_entry_bp.*no value was produced"):
        strategy.decide("2018-09-01", station_code, history)


# ---------------------------------------------------------------------------
# Injection seams (#255): in-memory ModelStrategy + custom detector factory
# ---------------------------------------------------------------------------

def _calibrated_pipeline_and_history(prices, station_code):
    """Build an in-memory calibrated pipeline + a populated PriceHistory."""
    import numpy as np
    from sklearn.dummy import DummyClassifier
    from sklearn.isotonic import IsotonicRegression

    from fuel_signal.calibrate import _CalibratedPipeline

    n_feat = len(FEATURE_COLUMNS)
    X_dummy = np.zeros((2, n_feat))
    base_clf = DummyClassifier(strategy="most_frequent").fit(X_dummy, [0, 1])
    iso = IsotonicRegression(out_of_bounds="clip").fit(base_clf.predict_proba(X_dummy)[:, 1], [0, 1])
    pipeline = _CalibratedPipeline(base_clf, iso, "isotonic", list(FEATURE_COLUMNS))

    history = PriceHistory(
        avg_series=prices,
        station_prices={station_code: prices},
        station_lga_brand={station_code: ("Penrith", "BP")},
        lga_mean_by_key={(d, "Penrith"): 175.0 for d, _ in prices},
        brand_mean_by_key={(d, "BP"): 178.0 for d, _ in prices},
        stickiness_by_key={(station_code, d): 3.0 for d, _ in prices},
        network_px_std_by_date={d: 4.5 for d, _ in prices},
        network_px_std_delta_3d_by_date={d: 0.2 for d, _ in prices},
        lga_phase_std_by_date={d: 6.0 for d, _ in prices},
        lga_phase_std_delta_3d_by_date={d: -0.1 for d, _ in prices},
    )
    return pipeline, history


def test_model_strategy_in_memory_pipeline_decides():
    """ModelStrategy(pipeline=...) scores without loading a joblib from disk."""
    n, period, station_code = 270, 45, 889
    dates = _dates_from("2018-01-01", n)
    prices = [(d, 180.0 + 20.0 * math.sin(2 * math.pi * i / period)) for i, d in enumerate(dates)]

    pipeline, history = _calibrated_pipeline_and_history(prices, station_code)
    strategy = ModelStrategy(pipeline=pipeline, feature_columns=list(FEATURE_COLUMNS), threshold=0.40)
    assert strategy.model_path is None
    assert isinstance(strategy.decide("2018-09-01", station_code, history), bool)


def test_model_strategy_requires_path_or_pipeline():
    """Constructing with neither model_path nor pipeline is an error."""
    import pytest

    with pytest.raises(ValueError, match="either model_path or pipeline"):
        ModelStrategy()


def test_model_strategy_rejects_both_path_and_pipeline():
    """Passing both sources is an error, not a silent ignore of model_path."""
    import pytest

    class _Stub:
        def predict_proba(self, X):  # pragma: no cover - never reached
            raise AssertionError

    with pytest.raises(ValueError, match="exactly one"):
        ModelStrategy(model_path="x.joblib", pipeline=_Stub(), feature_columns=["a"])


def test_price_history_uses_injected_detector_factory():
    """PriceHistory builds its detector via the injected factory, not always CycleDetector."""
    series = [("2020-01-01", 150.0), ("2020-01-02", 151.0)]
    sentinel = object()
    calls: list[object] = []

    class _StubDetector:
        def __init__(self, avg_series):
            calls.append(avg_series)

        def detect(self, as_of):
            return sentinel

    history = PriceHistory(
        avg_series=series, station_prices={}, detector_factory=_StubDetector
    )
    assert calls == [series]  # factory received avg_series exactly once
    assert history.cycle_state("2020-01-02") is sentinel


def test_extra_feature_provider_injects_added_column_into_vector():
    """extra_feature_provider value reaches the model for an ADDED feature column.

    The #268 seam: a candidate feature with no PriceHistory source (here a stand-in
    "tgp_delta_7d") is supplied live by the provider and must land in the vector at
    its feature_columns index. Uses a capturing pipeline to read the exact X built,
    and asserts the provider was called with (as_of, station_code, station_price).
    """
    import numpy as np

    n, period, station_code = 270, 45, 890
    dates = _dates_from("2018-01-01", n)
    prices = [(d, 180.0 + 20.0 * math.sin(2 * math.pi * i / period)) for i, d in enumerate(dates)]
    _, history = _calibrated_pipeline_and_history(prices, station_code)

    captured: dict[str, object] = {}

    class _CapturingPipeline:
        def predict_proba(self, X):
            captured["X"] = np.asarray(X)
            return np.array([[0.5, 0.5]])

    calls: list[tuple] = []

    def provider(as_of, sc, station_price):
        calls.append((as_of, sc, station_price))
        return {"tgp_delta_7d": -3.25}

    feature_columns = list(FEATURE_COLUMNS) + ["tgp_delta_7d"]
    strategy = ModelStrategy(
        pipeline=_CapturingPipeline(), feature_columns=feature_columns,
        threshold=0.40, extra_feature_provider=provider,
    )
    as_of = "2018-09-01"
    strategy.decide(as_of, station_code, history)

    # Provider invoked once with the live decision context (station_price matches).
    assert len(calls) == 1
    assert calls[0][0] == as_of and calls[0][1] == station_code
    assert calls[0][2] == history.station_price_at(station_code, as_of)
    # The injected value occupies the added column's slot in the live vector.
    assert captured["X"][0, feature_columns.index("tgp_delta_7d")] == -3.25


def test_extra_feature_provider_rejects_core_key_shadowing():
    """A provider that returns a core feature key raises rather than silently overwrite."""
    import pytest

    n, period, station_code = 270, 45, 892
    dates = _dates_from("2018-01-01", n)
    prices = [(d, 180.0 + 20.0 * math.sin(2 * math.pi * i / period)) for i, d in enumerate(dates)]
    pipeline, history = _calibrated_pipeline_and_history(prices, station_code)

    def shadowing_provider(as_of, sc, station_price):
        return {"station_price_cents": 1.0}  # collides with a core feature key

    strategy = ModelStrategy(
        pipeline=pipeline, feature_columns=list(FEATURE_COLUMNS), threshold=0.40,
        extra_feature_provider=shadowing_provider,
    )
    with pytest.raises(ValueError, match="shadows core feature"):
        strategy.decide("2018-09-01", station_code, history)


def test_extra_feature_provider_missing_column_raises_clear_error():
    """A provider that omits a column listed in feature_columns gets a named error."""
    import pytest

    n, period, station_code = 270, 45, 893
    dates = _dates_from("2018-01-01", n)
    prices = [(d, 180.0 + 20.0 * math.sin(2 * math.pi * i / period)) for i, d in enumerate(dates)]
    pipeline, history = _calibrated_pipeline_and_history(prices, station_code)

    def forgetful_provider(as_of, sc, station_price):
        return {}  # should have supplied "tgp_delta_7d"

    strategy = ModelStrategy(
        pipeline=pipeline, feature_columns=list(FEATURE_COLUMNS) + ["tgp_delta_7d"],
        threshold=0.40, extra_feature_provider=forgetful_provider,
    )
    with pytest.raises(ValueError, match="tgp_delta_7d.*no value was produced"):
        strategy.decide("2018-09-01", station_code, history)


def test_model_strategy_rejects_empty_feature_columns():
    """feature_columns=[] is a misconfiguration, not a silent swap for FEATURE_COLUMNS."""
    import pytest

    class _Stub:
        def predict_proba(self, X):  # pragma: no cover - never reached
            raise AssertionError

    with pytest.raises(ValueError, match="empty"):
        ModelStrategy(pipeline=_Stub(), feature_columns=[])


def test_decide_without_provider_is_unchanged():
    """extra_feature_provider=None (default) is a no-op — baseline path unchanged."""
    n, period, station_code = 270, 45, 891
    dates = _dates_from("2018-01-01", n)
    prices = [(d, 180.0 + 20.0 * math.sin(2 * math.pi * i / period)) for i, d in enumerate(dates)]
    pipeline, history = _calibrated_pipeline_and_history(prices, station_code)

    strategy = ModelStrategy(pipeline=pipeline, feature_columns=list(FEATURE_COLUMNS), threshold=0.40)
    assert strategy.extra_feature_provider is None
    assert isinstance(strategy.decide("2018-09-01", station_code, history), bool)


# ---------------------------------------------------------------------------
# require_tank_stamp (fps-15c) — the shared cadence-stamping guard every
# CPL-writing artifact routes through, generalising log_experiment's original
# realised_spend_cpl-without-tank check (fps-xx1).
# ---------------------------------------------------------------------------

def test_require_tank_stamp_raises_when_tank_is_none():
    import pytest

    from fuel_signal.backtest import require_tank_stamp

    with pytest.raises(ValueError, match="realised_spend_cpl was passed without tank"):
        require_tank_stamp(None, what="realised_spend_cpl")


def test_require_tank_stamp_returns_format_tank_params_when_tank_given():
    from fuel_signal.backtest import format_tank_params, require_tank_stamp

    tank = TankParams(evaluation_interval_days=1)
    assert require_tank_stamp(tank, what="anything") == format_tank_params(tank) == "50/3.571/1d/10%"


# ---------------------------------------------------------------------------
# TankParams derived quantities (fps-o0h) — one owner for what a tank can do,
# instead of restating the arithmetic at each call site.
# ---------------------------------------------------------------------------

def test_tank_params_derived_quantities_at_locked_config():
    """fps-6yi's locked tank (50/3.571.../1d/10%): full_to_empty_days = 14,
    max_feasible_wait_days = 0.9 * 50 / (50/14) = 12.6 (fps-2js's worked example),
    depletion_litres and run_dry_gap match the plain arithmetic every call site used
    to restate."""
    import pytest

    tank = TankParams()
    assert tank.full_to_empty_days == pytest.approx(14.0)
    assert tank.max_feasible_wait_days == pytest.approx(12.6)
    assert tank.depletion_litres == pytest.approx(tank.daily_consumption_litres)
    assert tank.run_dry_gap == pytest.approx(5.0)


def test_tank_params_feasible_wait_days_is_the_per_fill_bound():
    """regret_horizon_days' docstring: a fill's genuinely feasible wait is
    ((1 - floor) * size - litres) / daily, strictly below max_feasible_wait_days
    for any real (positive-litres) fill."""
    import pytest

    tank = TankParams()
    assert tank.feasible_wait_days(0.0) == pytest.approx(tank.max_feasible_wait_days)
    # A 20L fill uses up 20 / (50/14) = 5.6 empty-days of headroom.
    assert tank.feasible_wait_days(20.0) == pytest.approx(12.6 - 20.0 / (50.0 / 14))
    assert tank.feasible_wait_days(20.0) < tank.max_feasible_wait_days


def test_tank_params_fields_round_trips_exactly():
    """tank_params_fields carries the exact floats — no display rounding — unlike
    format_tank_params's 3dp daily-consumption stamp."""
    tank = TankParams(daily_consumption_litres=40.0 / 14)
    fields = tank_params_fields(tank)
    assert fields["daily_consumption_litres"] == 40.0 / 14
    rebuilt = TankParams(
        tank_size_litres=fields["tank_size_litres"],
        daily_consumption_litres=fields["daily_consumption_litres"],
        evaluation_interval_days=fields["evaluation_interval_days"],
        floor_fraction=fields["floor_fraction"],
    )
    assert rebuilt == tank
