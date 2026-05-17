"""Tests for fuel_signal.train_lgbm — LightGBM pipeline, val scoring, CLI smoke.

Mirrors tests/test_train_logreg.py. Same synthetic-data strategy: a deterministic
linear relationship between two features and the label, producing a well-separated
problem that LightGBM reliably beats the marginal-rate baseline on.
"""

from __future__ import annotations

import datetime

import joblib
import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from fuel_signal import evaluate as _ev
from fuel_signal.features import FEATURE_COLUMNS
from fuel_signal.train_lgbm import (
    main,
    train_and_evaluate,
)


def _date_range(start: str, n_days: int) -> list[str]:
    d0 = datetime.date.fromisoformat(start)
    return [(d0 + datetime.timedelta(days=i)).isoformat() for i in range(n_days)]


def _synthetic_features_df(seed: int = 0) -> pd.DataFrame:
    """Build a feature frame with rows in train + val + test windows."""
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


# ---------------------------------------------------------------------------
# train_and_evaluate
# ---------------------------------------------------------------------------

def test_train_and_evaluate_returns_expected_keys():
    df = _synthetic_features_df()
    result = train_and_evaluate(df)
    expected_keys = {
        "pipeline", "feature_columns",
        "train_size", "val_size",
        "train_positive_rate", "val_positive_rate",
        "val_logloss", "val_brier",
        "baseline_prior", "baseline_val_logloss", "baseline_val_brier",
        "y_val", "p_val",
    }
    assert expected_keys.issubset(set(result.keys()))


def test_train_and_evaluate_uses_train_and_val_only():
    """train_size + val_size must equal train+val rows; test is untouched."""
    df = _synthetic_features_df()
    train, val, test = _ev.split(df)
    result = train_and_evaluate(df)
    assert result["train_size"] == len(train)
    assert result["val_size"] == len(val)
    assert len(test) > 0


def test_train_and_evaluate_beats_baseline():
    """LightGBM val log-loss is strictly below the marginal-rate baseline."""
    df = _synthetic_features_df()
    result = train_and_evaluate(df)
    assert result["val_logloss"] < result["baseline_val_logloss"]


def test_train_and_evaluate_predict_proba_shape():
    """p_val matches val row count and is in [0, 1]."""
    df = _synthetic_features_df()
    result = train_and_evaluate(df)
    assert result["p_val"].shape == (result["val_size"],)
    assert (result["p_val"] >= 0).all()
    assert (result["p_val"] <= 1).all()


def test_train_and_evaluate_empty_train_raises():
    df = _synthetic_features_df()
    df = df[pd.to_datetime(df["price_date"]) >= _ev.VAL_START]
    with pytest.raises(ValueError, match="train split is empty"):
        train_and_evaluate(df)


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

def test_cli_runs_end_to_end(tmp_path):
    """Click smoke test: features CSV → trained model + reliability plot on disk."""
    df = _synthetic_features_df()
    features_path = tmp_path / "features.csv"
    df.to_csv(features_path, index=False)
    model_path = tmp_path / "models" / "lgbm.joblib"
    reliability_path = tmp_path / "reliability_lgbm.png"

    runner = CliRunner()
    res = runner.invoke(
        main,
        [
            "--features-csv", str(features_path),
            "--model-out", str(model_path),
            "--reliability-out", str(reliability_path),
        ],
    )
    assert res.exit_code == 0, res.output
    assert "val log-loss" in res.output
    assert model_path.exists()
    assert reliability_path.exists()

    saved = joblib.load(model_path)
    assert "pipeline" in saved
    assert saved["feature_columns"] == FEATURE_COLUMNS

    X = df[FEATURE_COLUMNS].head(5).to_numpy(dtype=float)
    proba = saved["pipeline"].predict_proba(X)
    assert proba.shape == (5, 2)


def test_cli_missing_features_csv_errors(tmp_path):
    runner = CliRunner()
    res = runner.invoke(main, ["--features-csv", str(tmp_path / "nope.csv")])
    assert res.exit_code != 0
    assert "not found" in res.output.lower()


def test_cli_missing_columns_errors(tmp_path):
    bad = pd.DataFrame({"price_date": ["2020-01-01"], "label": [0]})
    bad_path = tmp_path / "bad.csv"
    bad.to_csv(bad_path, index=False)
    runner = CliRunner()
    res = runner.invoke(main, ["--features-csv", str(bad_path)])
    assert res.exit_code != 0
    assert "missing required columns" in res.output.lower()
