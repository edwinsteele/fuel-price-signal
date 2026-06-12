"""Tests for fuel_signal.calibrate and the reliability_table helper in evaluate.

Design notes
------------
All tests are synthetic — no real DB or real model artifact required.  The
synthetic dataset uses the same pattern as test_train_logreg: a noisy linear
relationship so logreg can fit it, spanning the canonical train/val/test date
windows so evaluate.split() produces non-empty subsets.
"""

from __future__ import annotations

import datetime
import pathlib

import joblib
import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner
from sklearn.pipeline import Pipeline

from fuel_signal import evaluate as _ev
from fuel_signal.calibrate import (
    class_balance,
    compare_calibrations,
    main,
    pick_best,
)
from fuel_signal.features import FEATURE_COLUMNS
from fuel_signal.train_logreg import build_pipeline

# ---------------------------------------------------------------------------
# Shared synthetic dataset helpers
# ---------------------------------------------------------------------------

def _date_range(start: str, n_days: int) -> list[str]:
    d0 = datetime.date.fromisoformat(start)
    return [(d0 + datetime.timedelta(days=i)).isoformat() for i in range(n_days)]


def _synthetic_df(seed: int = 0) -> pd.DataFrame:
    """Feature frame spanning train/val/test windows with a separable label.

    2000 train days keeps walk_forward_folds (train_min_days=1825) working
    without injecting fold_params in CLI-level tests.
    """
    rng = np.random.default_rng(seed)
    train_dates = _date_range("2018-01-01", 2000)
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


def _save_logreg(df: pd.DataFrame, path: pathlib.Path) -> Pipeline:
    """Train a logreg on the train split and save to joblib; returns the pipeline."""
    train, _, _ = _ev.split(df)
    pipe = build_pipeline()
    pipe.fit(train[FEATURE_COLUMNS], train["label"].to_numpy(dtype=int))
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipe, "feature_columns": FEATURE_COLUMNS}, path)
    return pipe


# ---------------------------------------------------------------------------
# reliability_table (evaluate.py helper)
# ---------------------------------------------------------------------------

class TestReliabilityTable:
    def test_returns_expected_columns(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, size=200)
        p = rng.uniform(size=200)
        tbl = _ev.reliability_table(y, p)
        assert list(tbl.columns) == ["bin_mean_pred", "actual_rate", "count", "gap"]

    def test_bin_count_leq_n_bins(self):
        rng = np.random.default_rng(1)
        y = rng.integers(0, 2, size=300)
        p = rng.uniform(size=300)
        tbl = _ev.reliability_table(y, p, n_bins=10)
        assert len(tbl) <= 10
        assert tbl["count"].sum() == 300

    def test_gap_is_actual_minus_pred(self):
        rng = np.random.default_rng(2)
        y = rng.integers(0, 2, size=200)
        p = rng.uniform(size=200)
        tbl = _ev.reliability_table(y, p)
        np.testing.assert_allclose(tbl["gap"], tbl["actual_rate"] - tbl["bin_mean_pred"])

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            _ev.reliability_table(np.array([0, 1]), np.array([0.5]))

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            _ev.reliability_table(np.array([]), np.array([]))

    def test_constant_predictions_single_bin(self):
        y = np.array([0, 1, 0, 1])
        p = np.full(4, 0.5)
        tbl = _ev.reliability_table(y, p)
        assert len(tbl) == 1
        assert tbl["bin_mean_pred"].iloc[0] == pytest.approx(0.5)

    def test_invalid_n_bins_raises(self):
        y = np.array([0, 1])
        p = np.array([0.3, 0.7])
        with pytest.raises(ValueError, match="n_bins must be a positive integer"):
            _ev.reliability_table(y, p, n_bins=0)


# ---------------------------------------------------------------------------
# class_balance
# ---------------------------------------------------------------------------

class TestClassBalance:
    def test_returns_three_splits(self):
        df = _synthetic_df()
        cb = class_balance(df)
        assert set(cb["split"]) == {"train", "val", "test"}

    def test_buy_rate_in_unit_interval(self):
        df = _synthetic_df()
        cb = class_balance(df)
        assert (cb["buy_rate"].between(0.0, 1.0)).all()

    def test_row_counts_sum_to_less_than_total(self):
        # Buffer rows are dropped by evaluate.split, so sum < len(df)
        df = _synthetic_df()
        cb = class_balance(df)
        assert cb["n_rows"].sum() <= len(df)


# ---------------------------------------------------------------------------
# compare_calibrations
# ---------------------------------------------------------------------------

_SMALL_FOLDS = {"train_min_days": 200, "val_days": 30, "step_days": 30}


class TestCompareCalibrations:
    def test_returns_expected_keys(self, tmp_path):
        df = _synthetic_df()
        model_path = tmp_path / "models" / "logreg.joblib"
        _save_logreg(df, model_path)
        result = compare_calibrations(df, model_path, fold_params=_SMALL_FOLDS)
        assert set(result.keys()) == {
            "raw", "sigmoid", "isotonic", "y_val", "y_oof", "oof_buy_rate"
        }

    def test_all_variants_have_oof_and_val_metrics(self, tmp_path):
        df = _synthetic_df()
        model_path = tmp_path / "models" / "logreg.joblib"
        _save_logreg(df, model_path)
        result = compare_calibrations(df, model_path, fold_params=_SMALL_FOLDS)
        for name in ("raw", "sigmoid", "isotonic"):
            assert "oof_logloss" in result[name]
            assert "oof_brier" in result[name]
            assert "val_logloss" in result[name]
            assert "val_brier" in result[name]
            assert "p_val" in result[name]
            assert "p_oof" in result[name]

    def test_oof_buy_rate_is_in_unit_interval(self, tmp_path):
        df = _synthetic_df()
        model_path = tmp_path / "models" / "logreg.joblib"
        _save_logreg(df, model_path)
        result = compare_calibrations(df, model_path, fold_params=_SMALL_FOLDS)
        assert 0.0 < result["oof_buy_rate"] < 1.0

    def test_shipped_model_uses_100pct_trained_base(self, tmp_path):
        """Calibrated model's base_pipeline makes the same predictions as the 100%-trained raw model."""
        df = _synthetic_df()
        model_path = tmp_path / "models" / "logreg.joblib"
        raw_pipe = _save_logreg(df, model_path)
        result = compare_calibrations(df, model_path, fold_params=_SMALL_FOLDS)
        X_probe = df[FEATURE_COLUMNS].iloc[:20]
        p_raw = raw_pipe.predict_proba(X_probe)[:, 1]
        for name in ("sigmoid", "isotonic"):
            p_base = result[name]["model"].base_pipeline.predict_proba(X_probe)[:, 1]
            np.testing.assert_allclose(
                p_base, p_raw, rtol=1e-6,
                err_msg=f"{name} base_pipeline differs from 100%-trained raw pipeline",
            )

    def test_p_val_probabilities_in_unit_interval(self, tmp_path):
        df = _synthetic_df()
        model_path = tmp_path / "models" / "logreg.joblib"
        _save_logreg(df, model_path)
        result = compare_calibrations(df, model_path, fold_params=_SMALL_FOLDS)
        for name in ("raw", "sigmoid", "isotonic"):
            p = result[name]["p_val"]
            assert np.all((p >= 0.0) & (p <= 1.0)), f"{name} p_val probs out of range"
            p_oof = result[name]["p_oof"]
            assert np.all((p_oof >= 0.0) & (p_oof <= 1.0)), f"{name} p_oof probs out of range"

    def test_no_folds_raises(self, tmp_path):
        """Raises ValueError when train is too small for any fold."""
        df = _synthetic_df()
        model_path = tmp_path / "models" / "logreg.joblib"
        _save_logreg(df, model_path)
        # train_min_days > len(train) → no folds
        with pytest.raises(ValueError, match="no CV folds"):
            compare_calibrations(df, model_path, fold_params={"train_min_days": 99999})


# ---------------------------------------------------------------------------
# pick_best
# ---------------------------------------------------------------------------

class TestPickBest:
    def _make_compare(self, raw_ll, sig_ll, sig_br, iso_ll, iso_br, raw_br=0.2):
        n = 100
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, size=n)
        p = np.full(n, 0.3)
        dummy_model = object()
        return {
            "raw": {"oof_logloss": raw_ll, "oof_brier": raw_br, "p_val": p, "p_oof": p},
            "sigmoid": {"oof_logloss": sig_ll, "oof_brier": sig_br, "p_val": p, "p_oof": p, "model": dummy_model},
            "isotonic": {"oof_logloss": iso_ll, "oof_brier": iso_br, "p_val": p, "p_oof": p, "model": dummy_model},
            "y_val": y,
            "y_oof": y,
            "oof_buy_rate": float(y.mean()),
        }

    _FLAGGED = 0.10   # > _MISCAL_THRESHOLD (0.05) → proceed to calibration check
    _OK = 0.02        # ≤ _MISCAL_THRESHOLD → short-circuit to raw immediately

    def test_returns_raw_when_well_calibrated_regardless_of_logloss(self):
        # gate: max_gap ≤ threshold → raw, even if calibration would improve logloss
        compare = self._make_compare(raw_ll=0.50, sig_ll=0.45, sig_br=0.19, iso_ll=0.44, iso_br=0.19)
        name, model = pick_best(compare, self._OK)
        assert name == "raw"
        assert model is None

    def test_returns_raw_when_calibration_does_not_improve(self):
        compare = self._make_compare(raw_ll=0.50, sig_ll=0.55, sig_br=0.20, iso_ll=0.52, iso_br=0.20)
        name, model = pick_best(compare, self._FLAGGED)
        assert name == "raw"
        assert model is None

    def test_picks_sigmoid_when_it_wins(self):
        compare = self._make_compare(raw_ll=0.50, sig_ll=0.45, sig_br=0.19, iso_ll=0.55, iso_br=0.20)
        name, model = pick_best(compare, self._FLAGGED)
        assert name == "sigmoid"
        assert model is not None

    def test_picks_isotonic_when_it_wins(self):
        compare = self._make_compare(raw_ll=0.50, sig_ll=0.55, sig_br=0.20, iso_ll=0.45, iso_br=0.19)
        name, model = pick_best(compare, self._FLAGGED)
        assert name == "isotonic"

    def test_brier_regression_blocks_win(self):
        # sigmoid has lower logloss but blows up Brier beyond the limit
        compare = self._make_compare(raw_ll=0.50, raw_br=0.20, sig_ll=0.45, sig_br=0.21, iso_ll=0.55, iso_br=0.20)
        name, _ = pick_best(compare, self._FLAGGED)
        assert name == "raw"

    def test_picks_lower_logloss_when_both_candidates(self):
        compare = self._make_compare(raw_ll=0.50, sig_ll=0.44, sig_br=0.19, iso_ll=0.46, iso_br=0.19)
        name, _ = pick_best(compare, self._FLAGGED)
        assert name == "sigmoid"


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

class TestCalibrateCLI:
    def test_runs_end_to_end(self, tmp_path):
        df = _synthetic_df()
        features_path = tmp_path / "features.csv"
        df.to_csv(features_path, index=False)
        model_path = tmp_path / "models" / "logreg.joblib"
        _save_logreg(df, model_path)
        model_out = tmp_path / "models" / "logreg_calibrated.joblib"

        runner = CliRunner()
        result = runner.invoke(main, [
            "--features-csv", str(features_path),
            "--model-in", str(model_path),
            "--model-out", str(model_out),
            "--skip-results-csv",
        ])
        assert result.exit_code == 0, result.output
        assert "Class balance" in result.output
        assert "Reliability table" in result.output
        assert "Calibration comparison" in result.output
        assert "OOF" in result.output
        assert model_out.exists()

    def test_calibrated_artifact_is_loadable(self, tmp_path):
        df = _synthetic_df()
        features_path = tmp_path / "features.csv"
        df.to_csv(features_path, index=False)
        model_path = tmp_path / "models" / "logreg.joblib"
        _save_logreg(df, model_path)
        model_out = tmp_path / "models" / "logreg_calibrated.joblib"

        runner = CliRunner()
        result = runner.invoke(main, [
            "--features-csv", str(features_path),
            "--model-in", str(model_path),
            "--model-out", str(model_out),
            "--skip-results-csv",
        ])
        assert result.exit_code == 0, result.output

        saved = joblib.load(model_out)
        assert "feature_columns" in saved
        assert "calibrated" in saved

        # Reconstruct the pipeline and verify predict_proba works.
        # Calibrated artifacts store sklearn primitives; raw artifacts store "pipeline" directly.
        X = df[FEATURE_COLUMNS].iloc[:5]
        if saved.get("calibrated"):
            from fuel_signal.calibrate import _CalibratedPipeline
            assert "base_pipeline" in saved
            assert "calibrator" in saved
            assert "calibration_method" in saved
            pipeline = _CalibratedPipeline(
                saved["base_pipeline"], saved["calibrator"], saved["calibration_method"]
            )
        else:
            assert "pipeline" in saved
            pipeline = saved["pipeline"]
        proba = pipeline.predict_proba(X)
        assert proba.shape == (5, 2)
        assert np.all((proba >= 0.0) & (proba <= 1.0))

    def test_missing_features_csv_errors(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(main, [
            "--features-csv", str(tmp_path / "nope.csv"),
        ])
        assert result.exit_code != 0
        assert "not found" in result.output.lower()

    def test_missing_model_errors(self, tmp_path):
        df = _synthetic_df()
        features_path = tmp_path / "features.csv"
        df.to_csv(features_path, index=False)
        runner = CliRunner()
        result = runner.invoke(main, [
            "--features-csv", str(features_path),
            "--model-in", str(tmp_path / "no_model.joblib"),
        ])
        assert result.exit_code != 0
        assert "not found" in result.output.lower()
