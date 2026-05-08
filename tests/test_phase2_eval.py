"""Tests for fuel_signal.score_phase2 — threshold sweep, tau selection, test scoring.

Synthetic data strategy: same approach as test_train_logreg — a DataFrame with
deterministic linear signal spanning train/val/test windows so train_and_evaluate
reliably beats baseline and score_test gets real predictions to work with.

Tests do NOT touch the real DB or real features.csv.
"""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import pytest

import fuel_signal.evaluate as _ev
from fuel_signal.features import FEATURE_COLUMNS
from fuel_signal.score_phase2 import (
    _TAU_STEP,
    _TAUS,
    pick_tau,
    score_test,
    threshold_sweep,
)
from fuel_signal.train_logreg import train_and_evaluate

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _date_range(start: str, n_days: int) -> list[str]:
    d0 = datetime.date.fromisoformat(start)
    return [(d0 + datetime.timedelta(days=i)).isoformat() for i in range(n_days)]


def _synthetic_df(seed: int = 0) -> pd.DataFrame:
    """Feature frame spanning train + val + test with a learnable linear signal."""
    rng = np.random.default_rng(seed)
    train_dates = _date_range("2018-01-01", 800)
    val_dates = _date_range("2025-04-01", 60)
    test_dates = _date_range("2025-08-01", 60)
    all_dates = train_dates + val_dates + test_dates
    n = len(all_dates)

    X = rng.normal(size=(n, len(FEATURE_COLUMNS)))
    logits = 1.5 * X[:, 0] - 1.0 * X[:, 1] - 0.5
    probs = 1.0 / (1.0 + np.exp(-logits))
    labels = (rng.uniform(size=n) < probs).astype(int)

    rows = {col: X[:, i] for i, col in enumerate(FEATURE_COLUMNS)}
    rows["price_date"] = all_dates
    rows["label"] = labels
    rows["station_code"] = np.arange(n) % 10
    rows["today_price_cents"] = 160.0
    rows["future_min_cents"] = 159.0
    return pd.DataFrame(rows)


def _random_sweep(seed: int = 7, n: int = 200) -> list[dict]:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n).astype(float)
    p = rng.uniform(size=n)
    return threshold_sweep(y, p)


# ---------------------------------------------------------------------------
# threshold_sweep — structure
# ---------------------------------------------------------------------------

def test_threshold_sweep_returns_expected_keys():
    rows = _random_sweep()
    required = {"tau", "buy_rate", "precision", "recall", "f1", "expected_cents_per_row",
                "tp", "fp", "fn", "tn"}
    for r in rows:
        assert required.issubset(r.keys())


def test_threshold_sweep_tau_grid_matches_module_constant():
    """Sweep covers every τ in _TAUS (no extra, no missing)."""
    rows = _random_sweep()
    taus_in_rows = [r["tau"] for r in rows]
    expected = [round(float(t), 4) for t in sorted(_TAUS)]
    assert taus_in_rows == expected


def test_threshold_sweep_counts_sum_to_n():
    """tp + fp + fn + tn must equal n for every row."""
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, size=300).astype(float)
    p = rng.uniform(size=300)
    n = len(y)
    for r in threshold_sweep(y, p):
        assert r["tp"] + r["fp"] + r["fn"] + r["tn"] == n


def test_threshold_sweep_precision_recall_f1_consistent():
    """P/R/F1 within each row are internally consistent."""
    for r in _random_sweep():
        if r["precision"] > 0 or r["recall"] > 0:
            expected_f1 = (
                2 * r["precision"] * r["recall"] / (r["precision"] + r["recall"])
                if (r["precision"] + r["recall"]) > 0
                else 0.0
            )
            assert r["f1"] == pytest.approx(expected_f1, abs=1e-4)


# ---------------------------------------------------------------------------
# threshold_sweep — monotonicity invariants
# ---------------------------------------------------------------------------

def test_threshold_sweep_buy_rate_monotone_decreasing():
    """Higher τ → fewer predictions above threshold → buy_rate non-increasing."""
    rows = _random_sweep()
    buy_rates = [r["buy_rate"] for r in rows]
    for i in range(len(buy_rates) - 1):
        assert buy_rates[i] >= buy_rates[i + 1], (
            f"buy_rate not monotone at τ={rows[i]['tau']:.2f}→{rows[i+1]['tau']:.2f}: "
            f"{buy_rates[i]:.4f} < {buy_rates[i+1]:.4f}"
        )


def test_threshold_sweep_recall_monotone_decreasing():
    """Higher τ → fewer TP captured → recall non-increasing."""
    rows = _random_sweep()
    recalls = [r["recall"] for r in rows]
    for i in range(len(recalls) - 1):
        assert recalls[i] >= recalls[i + 1], (
            f"recall not monotone at τ={rows[i]['tau']:.2f}→{rows[i+1]['tau']:.2f}: "
            f"{recalls[i]:.4f} < {recalls[i+1]:.4f}"
        )


def test_threshold_sweep_buy_rate_at_extremes():
    """τ=0.05 should capture almost all rows; τ=0.95 should capture almost none."""
    rng = np.random.default_rng(11)
    y = rng.integers(0, 2, size=500).astype(float)
    p = rng.uniform(size=500)
    rows = threshold_sweep(y, p)
    # At very low τ, nearly everything is predicted BUY
    assert rows[0]["buy_rate"] > 0.9, f"τ=0.05 buy_rate unexpectedly low: {rows[0]['buy_rate']}"
    # At very high τ, nearly nothing is predicted BUY
    assert rows[-1]["buy_rate"] < 0.1, f"τ=0.95 buy_rate unexpectedly high: {rows[-1]['buy_rate']}"


# ---------------------------------------------------------------------------
# threshold_sweep — input validation
# ---------------------------------------------------------------------------

def test_threshold_sweep_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        threshold_sweep(np.array([0.0, 1.0]), np.array([0.5]))


def test_threshold_sweep_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        threshold_sweep(np.array([]), np.array([]))


# ---------------------------------------------------------------------------
# pick_tau
# ---------------------------------------------------------------------------

def test_pick_tau_adjusts_upward_by_one_step():
    """pick_tau returns best tau + _TAU_STEP."""
    # Fabricate a sweep where tau=0.20 has the highest expected_cents.
    sweep = [
        {"tau": 0.10, "expected_cents_per_row": 0.01},
        {"tau": 0.20, "expected_cents_per_row": 0.05},
        {"tau": 0.30, "expected_cents_per_row": 0.03},
    ]
    result = pick_tau(sweep, tau_adjustment=_TAU_STEP)
    assert result == pytest.approx(0.20 + _TAU_STEP, abs=1e-9)


def test_pick_tau_clamps_at_upper_bound():
    """Adjustment never pushes τ above 1.0 − _TAU_STEP."""
    sweep = [{"tau": 0.95, "expected_cents_per_row": 1.0}]
    result = pick_tau(sweep, tau_adjustment=_TAU_STEP)
    assert result <= 1.0 - _TAU_STEP + 1e-9


def test_pick_tau_clamps_at_lower_bound():
    """Even with a zero or negative adjustment the result is ≥ _TAU_STEP."""
    sweep = [{"tau": 0.05, "expected_cents_per_row": 1.0}]
    result = pick_tau(sweep, tau_adjustment=0.0)
    assert result >= _TAU_STEP - 1e-9


def test_pick_tau_on_random_sweep():
    """pick_tau on a real sweep returns a float in (_TAU_STEP, 1-_TAU_STEP] range."""
    tau = pick_tau(_random_sweep())
    assert _TAU_STEP <= tau <= 1.0 - _TAU_STEP + 1e-9


# ---------------------------------------------------------------------------
# score_test
# ---------------------------------------------------------------------------

def test_score_test_returns_expected_keys():
    df = _synthetic_df()
    result_train = train_and_evaluate(df)
    pipeline = result_train["pipeline"]
    test_result = score_test(pipeline, df, tau=0.30)
    required = {
        "test_size", "test_positive_rate", "test_logloss", "test_brier",
        "test_precision", "test_recall", "test_f1", "test_buy_rate",
        "y_test", "p_test",
    }
    assert required.issubset(set(test_result.keys()))


def test_score_test_logloss_is_positive():
    df = _synthetic_df()
    result_train = train_and_evaluate(df)
    test_result = score_test(result_train["pipeline"], df, tau=0.30)
    assert test_result["test_logloss"] > 0


def test_score_test_brier_in_unit_interval():
    df = _synthetic_df()
    result_train = train_and_evaluate(df)
    test_result = score_test(result_train["pipeline"], df, tau=0.30)
    assert 0.0 <= test_result["test_brier"] <= 1.0


def test_score_test_size_matches_test_split():
    """Reported test_size matches the canonical test split row count."""
    df = _synthetic_df()
    _, _, test_df = _ev.split(df)
    result_train = train_and_evaluate(df)
    test_result = score_test(result_train["pipeline"], df, tau=0.30)
    assert test_result["test_size"] == len(test_df)


def test_score_test_predictions_in_unit_interval():
    df = _synthetic_df()
    result_train = train_and_evaluate(df)
    test_result = score_test(result_train["pipeline"], df, tau=0.30)
    p = test_result["p_test"]
    assert (p >= 0).all() and (p <= 1).all()


def test_score_test_does_not_use_train_or_val_rows():
    """y_test length equals test split length (no train/val leakage)."""
    df = _synthetic_df()
    _, _, test_df = _ev.split(df)
    result_train = train_and_evaluate(df)
    test_result = score_test(result_train["pipeline"], df, tau=0.30)
    assert len(test_result["y_test"]) == len(test_df)


def test_score_test_empty_test_raises():
    """If the split produces an empty test set, score_test raises ValueError."""
    df = _synthetic_df()
    # Keep only train rows — no test rows.
    train_only = df[pd.to_datetime(df["price_date"]) <= _ev.TRAIN_END].copy()
    result_train = train_and_evaluate(df)
    with pytest.raises(ValueError, match="test split is empty"):
        score_test(result_train["pipeline"], train_only, tau=0.30)


# ---------------------------------------------------------------------------
# log_experiment row format (via evaluate.log_experiment)
# ---------------------------------------------------------------------------

def test_log_experiment_row_has_all_phase2_fields(tmp_path, monkeypatch):
    """A Phase 2 notes string contains the required fields."""
    import fuel_signal.evaluate as ev

    results_path = tmp_path / "results.csv"
    monkeypatch.setattr(ev, "_RESULTS_CSV", results_path)

    notes = (
        "tau=0.35; criterion=max_expected_cents_val_adj+0.05; "
        "cost_model=TP+3.0c_FP-1.5c; "
        "val_logloss=0.4321; test_logloss=0.4567; "
        "val_BUY_rate=0.361; test_BUY_rate=0.269; "
        "val_P=0.500/R=0.600/F1=0.545; "
        "test_P=0.480/R=0.590/F1=0.529"
    )
    ev.log_experiment(
        name="logreg_cycle_features",
        features=FEATURE_COLUMNS,
        holdout_logloss=0.4567,
        holdout_brier=0.1234,
        notes=notes,
    )

    lines = results_path.read_text().splitlines()
    assert len(lines) == 2  # header + one row
    row_text = lines[1]
    # Required fields in the row
    assert "logreg_cycle_features" in row_text
    assert "tau=0.35" in row_text
    assert "val_logloss=" in row_text
    assert "test_logloss=" in row_text
    assert "val_BUY_rate=" in row_text
    assert "test_BUY_rate=" in row_text
    # Numeric scores are formatted correctly
    assert "0.456700" in row_text  # holdout_logloss
    assert "0.123400" in row_text  # holdout_brier


def test_log_experiment_holdout_logloss_formatted_to_6dp(tmp_path, monkeypatch):
    """holdout_logloss is written with 6 decimal places."""
    import fuel_signal.evaluate as ev

    results_path = tmp_path / "results.csv"
    monkeypatch.setattr(ev, "_RESULTS_CSV", results_path)

    ev.log_experiment("m", [], holdout_logloss=0.5, holdout_brier=0.2)
    content = results_path.read_text()
    # 0.5 formatted to 6dp → "0.500000"
    assert "0.500000" in content
    assert "0.200000" in content
