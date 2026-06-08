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
    ModelStrategy,
    PriceHistory,
    RuleBasedSignalStrategy,
    TankParams,
    _evaluation_dates,
    run_backtest,
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
