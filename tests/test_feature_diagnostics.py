"""Tests for fuel_signal.feature_diagnostics."""

from __future__ import annotations

import datetime

import joblib
import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression

from fuel_signal import evaluate as _ev
from fuel_signal.feature_diagnostics import (
    error_summary_section,
    feature_importance_section,
    fn_fp_delta_section,
    main,
    run_diagnostics,
)
from fuel_signal.features import FEATURE_COLUMNS

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _date_range(start: str, n_days: int) -> list[str]:
    d0 = datetime.date.fromisoformat(start)
    return [(d0 + datetime.timedelta(days=i)).isoformat() for i in range(n_days)]


def _synthetic_df(seed: int = 0) -> pd.DataFrame:
    """Feature frame covering train + val windows."""
    rng = np.random.default_rng(seed)
    train_dates = _date_range("2018-01-01", 600)
    val_dates = _date_range("2025-04-01", 60)
    all_dates = train_dates + val_dates
    n = len(all_dates)
    X = rng.normal(size=(n, len(FEATURE_COLUMNS)))
    logits = 1.5 * X[:, 0] - 1.0 * X[:, 1] - 0.5
    probs = 1.0 / (1.0 + np.exp(-logits))
    labels = (rng.uniform(size=n) < probs).astype(int)
    rows = {col: X[:, i] for i, col in enumerate(FEATURE_COLUMNS)}
    rows["price_date"] = all_dates
    rows["label"] = labels
    rows["station_code"] = 1
    rows["today_price_cents"] = 160.0
    rows["future_min_cents"] = 159.0
    return pd.DataFrame(rows)


def _build_artifact(df: pd.DataFrame) -> dict:
    """Train a minimal LGBMClassifier and isotonic calibrator; return artifact dict."""
    train, _val, _test = _ev.split(df)
    X_train = train[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_train = train["label"].to_numpy(dtype=int)

    model = LGBMClassifier(random_state=42, verbose=-1, n_estimators=20)
    model.fit(X_train, y_train)

    raw_p = model.predict_proba(X_train)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_p, y_train)

    return {
        "base_pipeline": model,
        "calibrator": calibrator,
        "calibration_method": "isotonic",
        "feature_columns": FEATURE_COLUMNS,
        "calibrated": True,
    }


@pytest.fixture()
def artifact_path(tmp_path):
    df = _synthetic_df()
    artifact = _build_artifact(df)
    path = tmp_path / "lgbm_calibrated.joblib"
    joblib.dump(artifact, path)
    return path


@pytest.fixture()
def features_csv(tmp_path):
    df = _synthetic_df()
    path = tmp_path / "features.csv"
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Unit tests for section helpers
# ---------------------------------------------------------------------------

def test_feature_importance_section_has_all_features():
    df = _synthetic_df()
    artifact = _build_artifact(df)
    output = feature_importance_section(artifact)
    for col in FEATURE_COLUMNS:
        assert col in output


def test_feature_importance_section_gain_sums_to_100():
    df = _synthetic_df()
    artifact = _build_artifact(df)
    output = feature_importance_section(artifact)
    # Extract numbers from output lines (skip header lines)
    gain_vals = []
    for line in output.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0] not in ("feature", "-" * 10):
            try:
                gain_vals.append(float(parts[-2]))
            except ValueError:
                pass
    # Tolerance scales with feature count: each feature's gain% is displayed to 1 decimal,
    # so max per-feature rounding error is 0.05; allow 0.1 per feature to be safe.
    tol = len(FEATURE_COLUMNS) * 0.1
    assert abs(sum(gain_vals) - 100.0) < tol


def test_fn_fp_delta_section_has_all_features():
    df = _synthetic_df()
    artifact = _build_artifact(df)
    _train, val, _test = _ev.split(df)
    pred = np.zeros(len(val), dtype=int)
    pred[: len(val) // 2] = 1
    output = fn_fp_delta_section(val, artifact["feature_columns"], pred)
    for col in FEATURE_COLUMNS:
        assert col in output


def test_fn_fp_delta_section_sorted_by_abs_fn_tp():
    df = _synthetic_df()
    artifact = _build_artifact(df)
    _train, val, _test = _ev.split(df)
    rng = np.random.default_rng(7)
    pred = rng.integers(0, 2, size=len(val))
    output = fn_fp_delta_section(val, artifact["feature_columns"], pred)
    fn_tp_vals = []
    for line in output.splitlines():
        parts = line.strip().split()
        # data lines: feature  +/-x.xxx  +/-x.xxx
        if len(parts) == 3 and parts[1][0] in ("+", "-"):
            try:
                fn_tp_vals.append(abs(float(parts[1])))
            except ValueError:
                pass
    assert fn_tp_vals == sorted(fn_tp_vals, reverse=True)


def test_fn_fp_delta_section_excludes_id_columns():
    """station_code is excluded from delta rows; an exclusion note appears in the output."""
    df = _synthetic_df()
    _train, val, _test = _ev.split(df)
    pred = np.zeros(len(val), dtype=int)
    pred[: len(val) // 2] = 1

    output = fn_fp_delta_section(val, list(FEATURE_COLUMNS) + ["station_code"], pred)

    # station_code must not appear as a delta data row (lines with numeric deltas)
    for line in output.splitlines():
        if line.strip().startswith("station_code"):
            pytest.fail(f"station_code appeared as a delta data row: {line!r}")

    # Exclusion note mentions station_code and ID
    assert "station_code" in output
    assert "ID" in output

    # Regular numeric features are still present
    for col in FEATURE_COLUMNS:
        assert col in output


def test_fn_fp_delta_section_excludes_heuristic_id_columns():
    """High-cardinality integer columns are excluded from delta rows via the heuristic path."""
    df = _synthetic_df().copy()
    df["synthetic_id"] = np.arange(len(df), dtype=int)

    _train, val, _test = _ev.split(df)
    pred = np.zeros(len(val), dtype=int)
    pred[: len(val) // 2] = 1

    feature_cols = list(FEATURE_COLUMNS) + ["synthetic_id"]
    output = fn_fp_delta_section(val, feature_cols, pred)
    lines = output.splitlines()

    for line in lines:
        if line.strip().startswith("synthetic_id"):
            pytest.fail(f"synthetic_id appeared as a delta data row: {line!r}")

    assert any(
        "Excluded" in line and "ID" in line and "synthetic_id" in line for line in lines
    ), "Expected synthetic_id in the excluded ID columns note."

    numeric_features = [
        col for col in FEATURE_COLUMNS if pd.api.types.is_numeric_dtype(val[col].dtype)
    ]
    assert numeric_features
    non_id_col = numeric_features[0]
    assert any(
        line.strip().startswith(non_id_col) and any(ch.isdigit() for ch in line)
        for line in lines
    ), f"Expected {non_id_col} to appear as a delta row."


def test_error_summary_counts_sum_to_n():
    df = _synthetic_df()
    _train, val, _test = _ev.split(df)
    rng = np.random.default_rng(3)
    pred = rng.integers(0, 2, size=len(val))
    output = error_summary_section(val, pred)
    counts = []
    for group in ("TP", "FP", "TN", "FN"):
        for line in output.splitlines():
            if line.strip().startswith(group):
                counts.append(int(line.split()[1]))
    assert sum(counts) == len(val)


def test_error_summary_buy_rate_correct():
    """TP and FN groups always have buy_rate 100%; FP and TN always 0%."""
    df = _synthetic_df()
    _train, val, _test = _ev.split(df)
    rng = np.random.default_rng(5)
    pred = rng.integers(0, 2, size=len(val))
    output = error_summary_section(val, pred)
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("TP") or stripped.startswith("FN"):
            assert "100%" in line
        elif stripped.startswith("FP") or stripped.startswith("TN"):
            assert "0%" in line


# ---------------------------------------------------------------------------
# Integration: run_diagnostics
# ---------------------------------------------------------------------------

def test_run_diagnostics_returns_three_sections(artifact_path, features_csv):
    output = run_diagnostics(artifact_path, features_csv, threshold=0.40)
    assert "Feature importance" in output
    assert "FN" in output and "TP" in output
    assert "Error summary" in output


def test_run_diagnostics_threshold_affects_counts(artifact_path, features_csv):
    low_threshold = run_diagnostics(artifact_path, features_csv, threshold=0.10)
    high_threshold = run_diagnostics(artifact_path, features_csv, threshold=0.90)

    def _extract_rate(text: str) -> float:
        for line in text.splitlines():
            if "predicted-BUY rate:" in line:
                return float(line.rsplit(":", 1)[1].strip().rstrip("%"))
        raise AssertionError("predicted-BUY rate not found in output")

    assert _extract_rate(low_threshold) > _extract_rate(high_threshold)


# ---------------------------------------------------------------------------
# CLI smoke tests
# ---------------------------------------------------------------------------

def test_cli_runs_end_to_end(artifact_path, features_csv):
    runner = CliRunner()
    res = runner.invoke(
        main,
        [
            "--model-path", str(artifact_path),
            "--features-csv", str(features_csv),
        ],
    )
    assert res.exit_code == 0, res.output
    assert "Feature importance" in res.output
    assert "FN−TP" in res.output
    assert "Error summary" in res.output


def test_cli_missing_model_errors(tmp_path, features_csv):
    runner = CliRunner()
    res = runner.invoke(
        main,
        ["--model-path", str(tmp_path / "nope.joblib"), "--features-csv", str(features_csv)],
    )
    assert res.exit_code != 0
    assert "not found" in res.output.lower()


def test_cli_missing_features_csv_errors(tmp_path, artifact_path):
    runner = CliRunner()
    res = runner.invoke(
        main,
        ["--model-path", str(artifact_path), "--features-csv", str(tmp_path / "nope.csv")],
    )
    assert res.exit_code != 0
    assert "not found" in res.output.lower()


def test_cli_missing_columns_errors(tmp_path, artifact_path):
    bad = pd.DataFrame({"price_date": ["2025-04-01"], "label": [0]})
    bad_path = tmp_path / "bad.csv"
    bad.to_csv(bad_path, index=False)
    runner = CliRunner()
    res = runner.invoke(
        main,
        ["--model-path", str(artifact_path), "--features-csv", str(bad_path)],
    )
    assert res.exit_code != 0
    assert "missing columns" in res.output.lower()


def test_cli_bad_artifact_errors(tmp_path, features_csv):
    bad_artifact = {"pipeline": "wrong_key"}
    bad_path = tmp_path / "bad.joblib"
    joblib.dump(bad_artifact, bad_path)
    runner = CliRunner()
    res = runner.invoke(
        main,
        ["--model-path", str(bad_path), "--features-csv", str(features_csv)],
    )
    assert res.exit_code != 0
    assert "missing key" in res.output.lower()
