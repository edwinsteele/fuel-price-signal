"""Tests for fuel_signal.cv_report — paired walk-forward CV and logreg library CV."""

from __future__ import annotations

import datetime
import pathlib

import joblib
import numpy as np
import pandas as pd
from click.testing import CliRunner
from lightgbm import LGBMClassifier

from fuel_signal.cv_report import main, run_cv, run_paired_cv
from fuel_signal.features import FEATURE_COLUMNS

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _date_range(start: str, n_days: int) -> list[str]:
    d0 = datetime.date.fromisoformat(start)
    return [(d0 + datetime.timedelta(days=i)).isoformat() for i in range(n_days)]


# Feature names used by the paired-CV synthetic fixtures (independent of FEATURE_COLUMNS).
_FEATS_A = ["fa0", "fa1", "fa2"]  # 3-feature "model"
_FEATS_B = ["fa0", "fa1"]          # 2-feature "baseline"


def _synthetic_cv_df(seed: int = 0) -> pd.DataFrame:
    """Pre-test window frame with FEATURE_COLUMNS — used by run_cv tests."""
    rng = np.random.default_rng(seed)
    dates = _date_range("2020-01-01", 1096)
    n = len(dates)
    X = rng.normal(size=(n, len(FEATURE_COLUMNS)))
    logits = 1.5 * X[:, 0] - 1.0 * X[:, 1] - 0.5
    probs = 1.0 / (1.0 + np.exp(-logits))
    labels = (rng.uniform(size=n) < probs).astype(int)
    rows = {col: X[:, i] for i, col in enumerate(FEATURE_COLUMNS)}
    rows["price_date"] = dates
    rows["label"] = labels
    rows["station_code"] = np.arange(n) % 10
    rows["today_price_cents"] = 160.0
    rows["future_min_cents"] = 159.0
    return pd.DataFrame(rows)


def _synthetic_paired_df(seed: int = 0) -> pd.DataFrame:
    """Pre-test window frame with _FEATS_A columns — used by paired-CV tests."""
    rng = np.random.default_rng(seed)
    dates = _date_range("2020-01-01", 1096)
    n = len(dates)
    X = rng.normal(size=(n, 3))
    logits = 1.5 * X[:, 0] - 1.0 * X[:, 1] - 0.5
    probs = 1.0 / (1.0 + np.exp(-logits))
    labels = (rng.uniform(size=n) < probs).astype(int)
    return pd.DataFrame({
        "price_date": dates,
        "label": labels,
        "fa0": X[:, 0],
        "fa1": X[:, 1],
        "fa2": X[:, 2],
        "station_code": np.arange(n) % 10,
        "today_price_cents": 160.0,
        "future_min_cents": 159.0,
    })


def _make_lgbm_joblib(
    tmp_path: pathlib.Path,
    name: str,
    feature_cols: list[str],
) -> pathlib.Path:
    """Train a tiny LGBMClassifier on random data and save as a joblib artifact."""
    rng = np.random.default_rng(0)
    n = 300
    X = rng.normal(size=(n, len(feature_cols)))
    y = (rng.uniform(size=n) < 0.3).astype(int)
    m = LGBMClassifier(random_state=42, verbose=-1, n_estimators=5)
    m.fit(X, y)
    path = tmp_path / f"{name}.joblib"
    joblib.dump({"pipeline": m, "feature_columns": feature_cols}, path)
    return path


# ---------------------------------------------------------------------------
# run_cv — single-model logreg (library function)
# ---------------------------------------------------------------------------

def test_run_cv_returns_list_of_dicts():
    df = _synthetic_cv_df()
    results = run_cv(df, train_min_days=200, val_days=30, step_days=90)
    assert isinstance(results, list)
    assert len(results) >= 1
    expected_keys = {
        "fold", "train_start", "train_end", "val_start", "val_end",
        "train_rows", "val_rows", "val_buy_rate", "val_logloss", "baseline_logloss",
    }
    for r in results:
        assert expected_keys == set(r.keys())


def test_run_cv_fold_numbers_are_positive():
    df = _synthetic_cv_df()
    results = run_cv(df, train_min_days=200, val_days=30, step_days=90)
    for r in results:
        assert r["fold"] >= 1
        assert r["train_rows"] > 0
        assert r["val_rows"] > 0


def test_run_cv_dates_are_iso_strings_and_ordered():
    df = _synthetic_cv_df()
    results = run_cv(df, train_min_days=200, val_days=30, step_days=90)
    for r in results:
        for key in ("train_start", "train_end", "val_start", "val_end"):
            assert len(r[key]) == 10 and r[key][4] == "-" and r[key][7] == "-"
        assert r["train_end"] < r["val_start"]
        assert r["train_start"] <= r["train_end"]
        assert r["val_start"] <= r["val_end"]


def test_run_cv_buy_rate_is_proportion():
    df = _synthetic_cv_df()
    results = run_cv(df, train_min_days=200, val_days=30, step_days=90)
    for r in results:
        assert 0.0 <= r["val_buy_rate"] <= 1.0


def test_run_cv_logloss_is_positive():
    df = _synthetic_cv_df()
    results = run_cv(df, train_min_days=200, val_days=30, step_days=90)
    for r in results:
        assert r["val_logloss"] > 0.0
        assert r["baseline_logloss"] > 0.0


def test_run_cv_train_grows_with_folds():
    df = _synthetic_cv_df()
    results = run_cv(df, train_min_days=200, val_days=30, step_days=90)
    assert len(results) >= 2
    for prev, curr in zip(results, results[1:]):
        assert curr["train_rows"] >= prev["train_rows"]


def test_run_cv_empty_df_returns_empty_list():
    empty = pd.DataFrame(columns=["price_date", "label"] + FEATURE_COLUMNS)
    results = run_cv(empty, train_min_days=100, val_days=30, step_days=30)
    assert results == []


# ---------------------------------------------------------------------------
# run_paired_cv
# ---------------------------------------------------------------------------

def test_run_paired_cv_returns_correct_schema(tmp_path):
    df = _synthetic_paired_df()
    model_path = _make_lgbm_joblib(tmp_path, "model", _FEATS_A)
    baseline_path = _make_lgbm_joblib(tmp_path, "baseline", _FEATS_B)
    results = run_paired_cv(
        df, model_path, baseline_path,
        seed=42, train_min_days=200, val_days=30, step_days=90,
    )
    assert len(results) >= 1
    expected_keys = {
        "fold_idx", "train_start", "train_end", "val_start", "val_end",
        "n_val", "baseline_logloss", "model_logloss", "delta",
    }
    for r in results:
        assert set(r.keys()) == expected_keys


def test_run_paired_cv_delta_equals_difference(tmp_path):
    df = _synthetic_paired_df()
    model_path = _make_lgbm_joblib(tmp_path, "model", _FEATS_A)
    baseline_path = _make_lgbm_joblib(tmp_path, "baseline", _FEATS_B)
    results = run_paired_cv(
        df, model_path, baseline_path,
        seed=42, train_min_days=200, val_days=30, step_days=90,
    )
    for r in results:
        assert abs(r["delta"] - (r["model_logloss"] - r["baseline_logloss"])) < 1e-10


def test_run_paired_cv_logloss_is_positive(tmp_path):
    df = _synthetic_paired_df()
    model_path = _make_lgbm_joblib(tmp_path, "model", _FEATS_A)
    baseline_path = _make_lgbm_joblib(tmp_path, "baseline", _FEATS_B)
    results = run_paired_cv(
        df, model_path, baseline_path,
        seed=42, train_min_days=200, val_days=30, step_days=90,
    )
    for r in results:
        assert r["model_logloss"] > 0.0
        assert r["baseline_logloss"] > 0.0


def test_run_paired_cv_fold_idx_positive_and_n_val_nonzero(tmp_path):
    df = _synthetic_paired_df()
    model_path = _make_lgbm_joblib(tmp_path, "model", _FEATS_A)
    baseline_path = _make_lgbm_joblib(tmp_path, "baseline", _FEATS_B)
    results = run_paired_cv(
        df, model_path, baseline_path,
        seed=42, train_min_days=200, val_days=30, step_days=90,
    )
    for r in results:
        assert r["fold_idx"] >= 1
        assert r["n_val"] > 0


def test_run_paired_cv_dates_ordered(tmp_path):
    df = _synthetic_paired_df()
    model_path = _make_lgbm_joblib(tmp_path, "model", _FEATS_A)
    baseline_path = _make_lgbm_joblib(tmp_path, "baseline", _FEATS_B)
    results = run_paired_cv(
        df, model_path, baseline_path,
        seed=42, train_min_days=200, val_days=30, step_days=90,
    )
    for r in results:
        assert r["train_end"] < r["val_start"]
        assert r["train_start"] <= r["train_end"]
        assert r["val_start"] <= r["val_end"]


def test_run_paired_cv_empty_df_returns_empty(tmp_path):
    empty = pd.DataFrame(columns=["price_date", "label"] + _FEATS_A)
    model_path = _make_lgbm_joblib(tmp_path, "model", _FEATS_A)
    baseline_path = _make_lgbm_joblib(tmp_path, "baseline", _FEATS_B)
    results = run_paired_cv(
        empty, model_path, baseline_path,
        seed=42, train_min_days=100, val_days=30, step_days=30,
    )
    assert results == []


# ---------------------------------------------------------------------------
# CLI (paired)
# ---------------------------------------------------------------------------

def test_cli_paired_runs_end_to_end(tmp_path):
    df = _synthetic_paired_df()
    csv_path = tmp_path / "features.csv"
    df.to_csv(csv_path, index=False)
    model_path = _make_lgbm_joblib(tmp_path, "model", _FEATS_A)
    baseline_path = _make_lgbm_joblib(tmp_path, "baseline", _FEATS_B)

    runner = CliRunner()
    res = runner.invoke(main, [
        "--model", str(model_path),
        "--baseline", str(baseline_path),
        "--features", str(csv_path),
        "--train-min-days", "200",
        "--val-days", "30",
        "--step-days", "90",
    ])
    assert res.exit_code == 0, res.output
    assert "fold" in res.output
    assert "folds:" in res.output
    assert "→" in res.output
    assert "wins:" in res.output


def test_cli_paired_output_csv_schema(tmp_path):
    df = _synthetic_paired_df()
    csv_path = tmp_path / "features.csv"
    df.to_csv(csv_path, index=False)
    model_path = _make_lgbm_joblib(tmp_path, "model", _FEATS_A)
    baseline_path = _make_lgbm_joblib(tmp_path, "baseline", _FEATS_B)
    out_csv = tmp_path / "results.csv"

    runner = CliRunner()
    res = runner.invoke(main, [
        "--model", str(model_path),
        "--baseline", str(baseline_path),
        "--features", str(csv_path),
        "--output", str(out_csv),
        "--train-min-days", "200",
        "--val-days", "30",
        "--step-days", "90",
    ])
    assert res.exit_code == 0, res.output
    assert out_csv.exists()
    result_df = pd.read_csv(out_csv)
    assert set(result_df.columns) == {
        "fold_idx", "train_start", "train_end", "val_start", "val_end",
        "n_val", "baseline_logloss", "model_logloss", "delta",
    }


def test_cli_paired_output_dir_created(tmp_path):
    """--output creates parent directories that don't yet exist."""
    df = _synthetic_paired_df()
    csv_path = tmp_path / "features.csv"
    df.to_csv(csv_path, index=False)
    model_path = _make_lgbm_joblib(tmp_path, "model", _FEATS_A)
    baseline_path = _make_lgbm_joblib(tmp_path, "baseline", _FEATS_B)
    out_csv = tmp_path / "new_dir" / "results.csv"

    runner = CliRunner()
    res = runner.invoke(main, [
        "--model", str(model_path),
        "--baseline", str(baseline_path),
        "--features", str(csv_path),
        "--output", str(out_csv),
        "--train-min-days", "200",
        "--val-days", "30",
        "--step-days", "90",
    ])
    assert res.exit_code == 0, res.output
    assert out_csv.exists()


def test_cli_paired_missing_features_csv_errors(tmp_path):
    model_path = _make_lgbm_joblib(tmp_path, "model", _FEATS_A)
    baseline_path = _make_lgbm_joblib(tmp_path, "baseline", _FEATS_B)
    runner = CliRunner()
    res = runner.invoke(main, [
        "--model", str(model_path),
        "--baseline", str(baseline_path),
        "--features", str(tmp_path / "nope.csv"),
    ])
    assert res.exit_code != 0
    assert "not found" in res.output.lower()


def test_cli_paired_missing_model_errors(tmp_path):
    df = _synthetic_paired_df()
    csv_path = tmp_path / "features.csv"
    df.to_csv(csv_path, index=False)
    baseline_path = _make_lgbm_joblib(tmp_path, "baseline", _FEATS_B)
    runner = CliRunner()
    res = runner.invoke(main, [
        "--model", str(tmp_path / "missing.joblib"),
        "--baseline", str(baseline_path),
        "--features", str(csv_path),
    ])
    assert res.exit_code != 0


def test_cli_paired_missing_baseline_errors(tmp_path):
    df = _synthetic_paired_df()
    csv_path = tmp_path / "features.csv"
    df.to_csv(csv_path, index=False)
    model_path = _make_lgbm_joblib(tmp_path, "model", _FEATS_A)
    runner = CliRunner()
    res = runner.invoke(main, [
        "--model", str(model_path),
        "--baseline", str(tmp_path / "missing.joblib"),
        "--features", str(csv_path),
    ])
    assert res.exit_code != 0


def test_cli_paired_missing_columns_errors(tmp_path):
    bad = pd.DataFrame({"price_date": ["2020-01-01"], "fa0": [1.0]})
    csv_path = tmp_path / "bad.csv"
    bad.to_csv(csv_path, index=False)
    model_path = _make_lgbm_joblib(tmp_path, "model", _FEATS_A)
    baseline_path = _make_lgbm_joblib(tmp_path, "baseline", _FEATS_B)
    runner = CliRunner()
    res = runner.invoke(main, [
        "--model", str(model_path),
        "--baseline", str(baseline_path),
        "--features", str(csv_path),
    ])
    assert res.exit_code != 0
    assert "missing required columns" in res.output.lower()


# ---------------------------------------------------------------------------
# CLI — drop-feature mode
# ---------------------------------------------------------------------------

def test_cli_drop_feature_runs_end_to_end(tmp_path):
    df = _synthetic_paired_df()
    csv_path = tmp_path / "features.csv"
    df.to_csv(csv_path, index=False)
    model_path = _make_lgbm_joblib(tmp_path, "model", _FEATS_A)

    runner = CliRunner()
    res = runner.invoke(main, [
        "--model", str(model_path),
        "--drop-feature", "fa2",
        "--features", str(csv_path),
        "--train-min-days", "200",
        "--val-days", "30",
        "--step-days", "90",
    ])
    assert res.exit_code == 0, res.output
    assert "fold" in res.output
    assert "folds:" in res.output
    assert "wins:" in res.output


def test_cli_drop_feature_two_columns(tmp_path):
    df = _synthetic_paired_df()
    csv_path = tmp_path / "features.csv"
    df.to_csv(csv_path, index=False)
    model_path = _make_lgbm_joblib(tmp_path, "model", _FEATS_A)

    runner = CliRunner()
    res = runner.invoke(main, [
        "--model", str(model_path),
        "--drop-feature", "fa1",
        "--drop-feature", "fa2",
        "--features", str(csv_path),
        "--train-min-days", "200",
        "--val-days", "30",
        "--step-days", "90",
    ])
    assert res.exit_code == 0, res.output
    assert "folds:" in res.output


def test_cli_drop_feature_output_csv_schema(tmp_path):
    df = _synthetic_paired_df()
    csv_path = tmp_path / "features.csv"
    df.to_csv(csv_path, index=False)
    model_path = _make_lgbm_joblib(tmp_path, "model", _FEATS_A)
    out_csv = tmp_path / "results.csv"

    runner = CliRunner()
    res = runner.invoke(main, [
        "--model", str(model_path),
        "--drop-feature", "fa2",
        "--features", str(csv_path),
        "--output", str(out_csv),
        "--train-min-days", "200",
        "--val-days", "30",
        "--step-days", "90",
    ])
    assert res.exit_code == 0, res.output
    assert out_csv.exists()
    result_df = pd.read_csv(out_csv)
    assert set(result_df.columns) == {
        "fold_idx", "train_start", "train_end", "val_start", "val_end",
        "n_val", "baseline_logloss", "model_logloss", "delta",
    }


def test_cli_conflicting_flags_rejected(tmp_path):
    df = _synthetic_paired_df()
    csv_path = tmp_path / "features.csv"
    df.to_csv(csv_path, index=False)
    model_path = _make_lgbm_joblib(tmp_path, "model", _FEATS_A)
    baseline_path = _make_lgbm_joblib(tmp_path, "baseline", _FEATS_B)

    runner = CliRunner()
    res = runner.invoke(main, [
        "--model", str(model_path),
        "--baseline", str(baseline_path),
        "--drop-feature", "fa2",
        "--features", str(csv_path),
    ])
    assert res.exit_code != 0
    assert "mutually exclusive" in res.output.lower()


def test_cli_neither_flag_rejected(tmp_path):
    df = _synthetic_paired_df()
    csv_path = tmp_path / "features.csv"
    df.to_csv(csv_path, index=False)
    model_path = _make_lgbm_joblib(tmp_path, "model", _FEATS_A)

    runner = CliRunner()
    res = runner.invoke(main, [
        "--model", str(model_path),
        "--features", str(csv_path),
    ])
    assert res.exit_code != 0
    assert "--baseline" in res.output and "--drop-feature" in res.output
    assert "--single-window" in res.output


def test_cli_single_window_runs_end_to_end(tmp_path):
    df = _synthetic_cv_df()
    csv_path = tmp_path / "features.csv"
    df.to_csv(csv_path, index=False)

    runner = CliRunner()
    res = runner.invoke(main, [
        "--single-window",
        "--features", str(csv_path),
        "--train-min-days", "200",
        "--val-days", "30",
        "--step-days", "90",
    ])
    assert res.exit_code == 0, res.output
    assert "fold" in res.output
    assert "folds:" in res.output
    assert "wins:" in res.output
    assert "val_logloss=" in res.output


def test_cli_single_window_output_csv_schema(tmp_path):
    df = _synthetic_cv_df()
    csv_path = tmp_path / "features.csv"
    df.to_csv(csv_path, index=False)
    out_csv = tmp_path / "results.csv"

    runner = CliRunner()
    res = runner.invoke(main, [
        "--single-window",
        "--features", str(csv_path),
        "--output", str(out_csv),
        "--train-min-days", "200",
        "--val-days", "30",
        "--step-days", "90",
    ])
    assert res.exit_code == 0, res.output
    assert out_csv.exists()
    result_df = pd.read_csv(out_csv)
    assert set(result_df.columns) == {
        "fold", "train_start", "train_end", "val_start", "val_end",
        "train_rows", "val_rows", "val_buy_rate", "val_logloss", "baseline_logloss",
    }


def test_cli_single_window_conflicting_with_model_rejected(tmp_path):
    df = _synthetic_paired_df()
    csv_path = tmp_path / "features.csv"
    df.to_csv(csv_path, index=False)
    model_path = _make_lgbm_joblib(tmp_path, "model", _FEATS_A)

    runner = CliRunner()
    res = runner.invoke(main, [
        "--single-window",
        "--model", str(model_path),
        "--features", str(csv_path),
    ])
    assert res.exit_code != 0
    assert "mutually exclusive" in res.output.lower()


def test_cli_single_window_no_model_required(tmp_path):
    """--single-window works with only --features (no --model needed)."""
    df = _synthetic_cv_df()
    csv_path = tmp_path / "features.csv"
    df.to_csv(csv_path, index=False)

    runner = CliRunner()
    res = runner.invoke(main, [
        "--single-window",
        "--features", str(csv_path),
        "--train-min-days", "200",
        "--val-days", "30",
        "--step-days", "90",
    ])
    assert res.exit_code == 0, res.output


def test_cli_unknown_drop_feature_rejected(tmp_path):
    df = _synthetic_paired_df()
    csv_path = tmp_path / "features.csv"
    df.to_csv(csv_path, index=False)
    model_path = _make_lgbm_joblib(tmp_path, "model", _FEATS_A)

    runner = CliRunner()
    res = runner.invoke(main, [
        "--model", str(model_path),
        "--drop-feature", "not_a_real_column",
        "--features", str(csv_path),
    ])
    assert res.exit_code != 0
    assert "not_a_real_column" in res.output
    # Should list valid column names
    assert "fa0" in res.output
