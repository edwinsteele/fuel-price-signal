"""Tests for model-aware τ adjustment in score_phase2 (issue #123).

Covers pick_tau calibration_method routing and load_model_artifact
returning calibration_method as the third element of its 3-tuple.
"""

from __future__ import annotations

import joblib
import numpy as np
import pytest
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from fuel_signal.features import FEATURE_COLUMNS
from fuel_signal.score_phase2 import (
    _TAU_STEP,
    load_model_artifact,
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
