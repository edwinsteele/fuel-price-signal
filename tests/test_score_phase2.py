"""Tests for model-aware τ adjustment in score_phase2 (issue #123)
and multi-seed raw test-logloss banking (issue #145).

Covers pick_tau calibration_method routing, load_model_artifact returning
calibration_method as the third element of its 3-tuple, multi_seed_raw_logloss
vector/stats, and --seeds CLI validation.
"""

from __future__ import annotations

import datetime

import joblib
import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from fuel_signal.features import FEATURE_COLUMNS
from fuel_signal.score_phase2 import (
    _TAU_STEP,
    load_model_artifact,
    main,
    multi_seed_raw_logloss,
    pick_tau,
)

# ---------------------------------------------------------------------------
# Minimal sweep fixture — one row per τ, isotonic argmax at τ=0.60.
# ---------------------------------------------------------------------------

_SWEEP = [
    {"tau": 0.40, "expected_cents_per_row": 0.01},
    {"tau": 0.60, "expected_cents_per_row": 0.05},
    {"tau": 0.70, "expected_cents_per_row": 0.03},
]


# ---------------------------------------------------------------------------
# pick_tau — model-aware default
# ---------------------------------------------------------------------------

def test_pick_tau_isotonic_returns_val_argmax_no_nudge():
    """Isotonic model: pick_tau returns the val argmax τ with zero adjustment."""
    result = pick_tau(_SWEEP, calibration_method="isotonic")
    assert result == pytest.approx(0.60, abs=1e-9)


def test_pick_tau_sigmoid_applies_step_adjustment():
    """Sigmoid model: pick_tau applies the +_TAU_STEP nudge (backward-compat)."""
    result = pick_tau(_SWEEP, calibration_method="sigmoid")
    assert result == pytest.approx(0.60 + _TAU_STEP, abs=1e-9)


def test_pick_tau_raw_none_applies_step_adjustment():
    """Raw / no calibration_method: pick_tau applies the +_TAU_STEP nudge."""
    result = pick_tau(_SWEEP, calibration_method=None)
    assert result == pytest.approx(0.60 + _TAU_STEP, abs=1e-9)


# ---------------------------------------------------------------------------
# pick_tau — explicit override always wins
# ---------------------------------------------------------------------------

def test_pick_tau_explicit_override_beats_isotonic_default():
    """An explicit tau_adjustment overrides the isotonic default of 0.0."""
    result = pick_tau(_SWEEP, calibration_method="isotonic", tau_adjustment=0.05)
    assert result == pytest.approx(0.60 + 0.05, abs=1e-9)


def test_pick_tau_explicit_zero_override_beats_sigmoid_default():
    """An explicit tau_adjustment=0.0 overrides the sigmoid default of _TAU_STEP."""
    result = pick_tau(_SWEEP, calibration_method="sigmoid", tau_adjustment=0.0)
    assert result == pytest.approx(0.60, abs=1e-9)


# ---------------------------------------------------------------------------
# load_model_artifact — calibration_method in 3-tuple
# ---------------------------------------------------------------------------

def _minimal_logreg():
    """Return a tiny fitted LogisticRegression usable as base_pipeline."""
    X = np.array([[0.0], [1.0], [0.0], [1.0]])
    y = np.array([0, 1, 0, 1])
    clf = LogisticRegression()
    clf.fit(X, y)
    return clf


def test_load_model_artifact_isotonic_returns_calibration_method(tmp_path):
    """Isotonic-calibrated artifact: load_model_artifact returns calibration_method='isotonic'."""
    clf = _minimal_logreg()
    raw_p = clf.predict_proba(np.array([[0.0], [1.0]]))[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_p, np.array([0, 1]))

    artifact = {
        "base_pipeline": clf,
        "calibrator": calibrator,
        "calibration_method": "isotonic",
        "feature_columns": FEATURE_COLUMNS,
        "calibrated": True,
    }
    path = tmp_path / "model_iso.joblib"
    joblib.dump(artifact, path)

    _, _, cal_method = load_model_artifact(path)
    assert cal_method == "isotonic"


def test_load_model_artifact_raw_dict_returns_none_calibration(tmp_path):
    """Raw pipeline dict artifact: load_model_artifact returns calibration_method=None."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    clf = _minimal_logreg()
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    artifact = {"pipeline": pipe, "feature_columns": FEATURE_COLUMNS}
    path = tmp_path / "model_raw.joblib"
    joblib.dump(artifact, path)

    _, _, cal_method = load_model_artifact(path)
    assert cal_method is None


# ---------------------------------------------------------------------------
# multi_seed_raw_logloss — vector length and stats (issue #145)
# ---------------------------------------------------------------------------

def _date_range(start: str, n_days: int) -> list[str]:
    d0 = datetime.date.fromisoformat(start)
    return [(d0 + datetime.timedelta(days=i)).isoformat() for i in range(n_days)]


def _synthetic_df_for_seed_test(seed: int = 0) -> pd.DataFrame:
    """Minimal DataFrame with train + val + test rows for multi_seed_raw_logloss tests."""
    rng = np.random.default_rng(seed)
    train_dates = _date_range("2018-01-01", 400)
    val_dates = _date_range("2025-04-01", 60)
    test_dates = _date_range("2025-08-01", 60)
    all_dates = train_dates + val_dates + test_dates
    n = len(all_dates)
    X = rng.normal(size=(n, len(FEATURE_COLUMNS)))
    logits = 3.0 * X[:, 0] - 2.0 * X[:, 1] - 0.5
    probs = 1.0 / (1.0 + np.exp(-logits))
    labels = (rng.uniform(size=n) < probs).astype(int)
    rows = {col: X[:, i] for i, col in enumerate(FEATURE_COLUMNS)}
    rows["price_date"] = all_dates
    rows["label"] = labels
    rows["station_code"] = np.arange(n) % 10
    rows["today_price_cents"] = 160.0
    rows["future_min_cents"] = 159.0
    return pd.DataFrame(rows)


def test_multi_seed_raw_logloss_vector_length():
    """Output vector has one entry per seed."""
    df = _synthetic_df_for_seed_test()
    result = multi_seed_raw_logloss(df, FEATURE_COLUMNS, seeds=[42, 99])
    assert len(result["logloss_vector"]) == 2


def test_multi_seed_raw_logloss_mean_and_std():
    """Mean and std match numpy computations on the returned vector."""
    df = _synthetic_df_for_seed_test()
    result = multi_seed_raw_logloss(df, FEATURE_COLUMNS, seeds=[1, 7, 42])
    vec = np.array(result["logloss_vector"])
    assert result["logloss_mean"] == pytest.approx(vec.mean(), abs=1e-9)
    assert result["logloss_std"] == pytest.approx(vec.std(), abs=1e-9)


def test_multi_seed_raw_logloss_values_are_positive():
    """Each per-seed logloss is a positive finite float."""
    df = _synthetic_df_for_seed_test()
    result = multi_seed_raw_logloss(df, FEATURE_COLUMNS, seeds=[42])
    assert result["logloss_vector"][0] > 0
    assert np.isfinite(result["logloss_vector"][0])


# ---------------------------------------------------------------------------
# --seeds CLI validation (issue #145)
# ---------------------------------------------------------------------------

def test_seeds_without_model_path_errors(tmp_path):
    """--seeds without --model-path is rejected before any heavy work."""
    df = _synthetic_df_for_seed_test()
    features_path = tmp_path / "features.csv"
    df.to_csv(features_path, index=False)
    runner = CliRunner()
    res = runner.invoke(main, [
        "--features-csv", str(features_path),
        "--seeds", "1,7,42",
    ])
    assert res.exit_code != 0
    assert "--seeds requires --model-path" in res.output


def test_seeds_invalid_format_errors(tmp_path):
    """Non-integer --seeds value is rejected with a clear message."""
    import joblib
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    df = _synthetic_df_for_seed_test()
    features_path = tmp_path / "features.csv"
    df.to_csv(features_path, index=False)

    clf = _minimal_logreg()
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    artifact = {"pipeline": pipe, "feature_columns": FEATURE_COLUMNS}
    model_path = tmp_path / "model.joblib"
    joblib.dump(artifact, model_path)

    runner = CliRunner()
    res = runner.invoke(main, [
        "--features-csv", str(features_path),
        "--model-path", str(model_path),
        "--seeds", "1,abc,42",
    ])
    assert res.exit_code != 0
    assert "comma-separated list of integers" in res.output
