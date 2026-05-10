"""Tests for fuel_signal.backtest_phase2.

run_tau_sweep is integration-only (requires a real joblib model) and is not
tested here. The remaining public functions — aggregate_backtest,
pick_spend_optimal_tau, and patch_results_csv — are unit-tested with
synthetic data and temp files; no real DB or model is required.
"""

from __future__ import annotations

import csv
import datetime
import math
import pathlib

import pytest

from fuel_signal.backtest import AlwaysBuyStrategy, PriceHistory, TankParams
from fuel_signal.backtest_phase2 import (
    aggregate_backtest,
    patch_results_csv,
    pick_spend_optimal_tau,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dates_from(start: str, n: int) -> list[str]:
    d = datetime.date.fromisoformat(start)
    return [(d + datetime.timedelta(days=i)).isoformat() for i in range(n)]


def _constant_history(codes: list[int], start: str, n: int, price: float) -> PriceHistory:
    dates = _dates_from(start, n)
    series = [(d, price) for d in dates]
    return PriceHistory(
        avg_series=series,
        station_prices={code: series for code in codes},
    )


_RESULTS_HEADER = [
    "timestamp", "git_sha", "name", "features",
    "train_start", "train_end", "val_start", "val_end", "test_start", "test_end",
    "holdout_logloss", "holdout_brier",
    "realised_spend_cpl", "realised_savings_vs_always_buy_pct", "notes",
]


def _write_results_csv(path: pathlib.Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_RESULTS_HEADER)
        writer.writeheader()
        writer.writerows(rows)


def _read_results_csv(path: pathlib.Path) -> list[dict]:
    with path.open("r", newline="") as fh:
        return list(csv.DictReader(fh))


def _base_row(name: str, notes: str) -> dict:
    return {
        "timestamp": "2026-01-01T00:00:00+00:00",
        "git_sha": "abc1234",
        "name": name,
        "features": "",
        "train_start": "2016-08-01",
        "train_end": "2025-03-17",
        "val_start": "2025-03-25",
        "val_end": "2025-06-23",
        "test_start": "2025-07-01",
        "test_end": "2025-12-31",
        "holdout_logloss": "0.400000",
        "holdout_brier": "0.130000",
        "realised_spend_cpl": "",
        "realised_savings_vs_always_buy_pct": "",
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# aggregate_backtest
# ---------------------------------------------------------------------------

def test_aggregate_backtest_single_station_constant_price():
    history = _constant_history([10], "2024-01-01", 91, 180.0)
    tank = TankParams(
        tank_size_litres=50.0,
        daily_consumption_litres=50.0 / 14,
        evaluation_interval_days=7,
    )
    agg = aggregate_backtest(
        history, AlwaysBuyStrategy(), [10], "2024-01-01", "2024-03-31", tank
    )
    assert not math.isnan(agg["cpl"])
    assert abs(agg["cpl"] - 180.0) < 0.01
    assert agg["fill_events"] > 0
    assert agg["total_litres"] > 0


def test_aggregate_backtest_two_stations_same_price():
    history = _constant_history([10, 20], "2024-01-01", 91, 150.0)
    tank = TankParams(
        tank_size_litres=50.0,
        daily_consumption_litres=50.0 / 14,
        evaluation_interval_days=7,
    )
    agg = aggregate_backtest(
        history, AlwaysBuyStrategy(), [10, 20], "2024-01-01", "2024-03-31", tank
    )
    assert abs(agg["cpl"] - 150.0) < 0.01


def test_aggregate_backtest_missing_station_skipped():
    """Station with no data is silently skipped; result uses remaining stations."""
    history = _constant_history([10], "2024-01-01", 91, 160.0)
    tank = TankParams(
        tank_size_litres=50.0,
        daily_consumption_litres=50.0 / 14,
        evaluation_interval_days=7,
    )
    agg = aggregate_backtest(
        history, AlwaysBuyStrategy(), [10, 99], "2024-01-01", "2024-03-31", tank
    )
    assert not math.isnan(agg["cpl"])
    assert abs(agg["cpl"] - 160.0) < 0.01


def test_aggregate_backtest_all_stations_missing():
    history = _constant_history([], "2024-01-01", 91, 160.0)
    tank = TankParams()
    agg = aggregate_backtest(
        history, AlwaysBuyStrategy(), [99], "2024-01-01", "2024-03-31", tank
    )
    assert math.isnan(agg["cpl"])
    assert agg["fill_events"] == 0


# ---------------------------------------------------------------------------
# pick_spend_optimal_tau
# ---------------------------------------------------------------------------

def test_pick_spend_optimal_tau_basic():
    sweep = [
        {"tau": 0.30, "cpl": 180.0},
        {"tau": 0.35, "cpl": 175.0},
        {"tau": 0.40, "cpl": 177.0},
    ]
    assert pick_spend_optimal_tau(sweep) == 0.35


def test_pick_spend_optimal_tau_nan_skipped():
    sweep = [
        {"tau": 0.30, "cpl": float("nan")},
        {"tau": 0.35, "cpl": 175.0},
        {"tau": 0.40, "cpl": 180.0},
    ]
    assert pick_spend_optimal_tau(sweep) == 0.35


def test_pick_spend_optimal_tau_all_nan_raises():
    with pytest.raises(ValueError):
        pick_spend_optimal_tau([{"tau": 0.30, "cpl": float("nan")}])


def test_pick_spend_optimal_tau_empty_raises():
    with pytest.raises(ValueError):
        pick_spend_optimal_tau([])


# ---------------------------------------------------------------------------
# patch_results_csv
# ---------------------------------------------------------------------------

def test_patch_results_csv_patches_both_rows(tmp_path):
    csv_path = tmp_path / "results.csv"
    rows = [
        _base_row("marginal_rate_baseline", "constant predictor"),
        _base_row("logreg_cycle_features", "tau=0.40; criterion=max_expected"),
        _base_row("logreg_cycle_features", "tau=0.35; criterion=max_expected"),
    ]
    _write_results_csv(csv_path, rows)

    patched_b, patched_p = patch_results_csv(csv_path, 175.0, 172.0, 1.71)

    assert patched_b
    assert patched_p
    result = _read_results_csv(csv_path)
    assert result[0]["realised_spend_cpl"] == "175.00"
    assert result[0]["realised_savings_vs_always_buy_pct"] == "0.00"
    assert result[1]["realised_spend_cpl"] == "172.00"
    assert result[1]["realised_savings_vs_always_buy_pct"] == "1.71"
    # tau=0.35 row should not be patched
    assert result[2]["realised_spend_cpl"] == ""


def test_patch_results_csv_missing_baseline(tmp_path):
    csv_path = tmp_path / "results.csv"
    _write_results_csv(
        csv_path,
        [_base_row("logreg_cycle_features", "tau=0.40; criterion=max_expected")],
    )
    patched_b, patched_p = patch_results_csv(csv_path, 175.0, 172.0, 1.71)
    assert not patched_b
    assert patched_p


def test_patch_results_csv_missing_phase2(tmp_path):
    csv_path = tmp_path / "results.csv"
    _write_results_csv(
        csv_path,
        [_base_row("marginal_rate_baseline", "constant predictor")],
    )
    patched_b, patched_p = patch_results_csv(csv_path, 175.0, 172.0, 1.71)
    assert patched_b
    assert not patched_p


def test_patch_results_csv_does_not_match_tau_035(tmp_path):
    """Only the tau=0.40 row should be identified as Phase 2."""
    csv_path = tmp_path / "results.csv"
    rows = [
        _base_row("marginal_rate_baseline", "constant predictor"),
        _base_row("logreg_cycle_features", "tau=0.35; criterion=max_expected"),
    ]
    _write_results_csv(csv_path, rows)
    patched_b, patched_p = patch_results_csv(csv_path, 175.0, 172.0, 1.71)
    assert patched_b
    assert not patched_p  # tau=0.35 is not the Phase 2 row


def test_patch_results_csv_preserves_other_fields(tmp_path):
    csv_path = tmp_path / "results.csv"
    _write_results_csv(
        csv_path,
        [_base_row("marginal_rate_baseline", "constant predictor")],
    )
    patch_results_csv(csv_path, 175.0, 172.0, 1.71)
    result = _read_results_csv(csv_path)
    assert result[0]["holdout_logloss"] == "0.400000"
    assert result[0]["notes"] == "constant predictor"
    assert result[0]["git_sha"] == "abc1234"


def test_patch_results_csv_atomic(tmp_path):
    """After patch, only the original file path should exist (no temp file left)."""
    csv_path = tmp_path / "results.csv"
    _write_results_csv(
        csv_path,
        [_base_row("marginal_rate_baseline", "constant predictor")],
    )
    patch_results_csv(csv_path, 175.0, 172.0, 1.71)
    remaining = list(tmp_path.iterdir())
    assert len(remaining) == 1
    assert remaining[0] == csv_path


def test_patch_results_csv_only_patches_first_matching_row(tmp_path):
    """If two rows match, only the first is patched."""
    csv_path = tmp_path / "results.csv"
    rows = [
        _base_row("marginal_rate_baseline", "constant predictor"),
        _base_row("marginal_rate_baseline", "constant predictor v2"),
    ]
    _write_results_csv(csv_path, rows)
    patch_results_csv(csv_path, 175.0, 172.0, 1.71)
    result = _read_results_csv(csv_path)
    assert result[0]["realised_spend_cpl"] == "175.00"
    assert result[1]["realised_spend_cpl"] == ""  # second match left alone
