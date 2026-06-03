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
from fuel_signal.features import FEATURE_COLUMNS, LGA_FEATURE_COLUMNS
from fuel_signal.train_lgbm import (
    main,
    train_and_evaluate,
)


def _date_range(start: str, n_days: int) -> list[str]:
    d0 = datetime.date.fromisoformat(start)
    return [(d0 + datetime.timedelta(days=i)).isoformat() for i in range(n_days)]


_ALL_FEATURE_COLUMNS = FEATURE_COLUMNS + LGA_FEATURE_COLUMNS
_SYNTHETIC_BRAND_COLUMNS = [
    "days_since_trough_entry_7_eleven",
    "days_since_trough_entry_bp",
    "days_since_trough_entry_shell",
]


def _synthetic_features_df(
    seed: int = 0,
    include_lga: bool = True,
    brand_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Build a feature frame with rows in train + val + test windows."""
    rng = np.random.default_rng(seed)

    train_dates = _date_range("2018-01-01", 800)
    val_dates = _date_range("2025-04-01", 60)
    test_dates = _date_range("2025-08-01", 60)
    all_dates = train_dates + val_dates + test_dates

    feature_cols = _ALL_FEATURE_COLUMNS if include_lga else list(FEATURE_COLUMNS)
    if brand_columns:
        feature_cols = feature_cols + list(brand_columns)
    n = len(all_dates)
    X = rng.normal(size=(n, len(feature_cols)))

    # Use strong coefficients so the model beats the baseline regardless of
    # how many noise features are in FEATURE_COLUMNS.
    logits = 3.0 * X[:, 0] - 2.0 * X[:, 1] - 0.5
    probs = 1.0 / (1.0 + np.exp(-logits))
    labels = (rng.uniform(size=n) < probs).astype(int)

    rows = {col: X[:, i] for i, col in enumerate(feature_cols)}
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
    """Click smoke test: Phase 4 features CSV → trained model + reliability plot on disk."""
    df = _synthetic_features_df(include_lga=True)
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
    assert "Phase 4" in res.output
    assert model_path.exists()
    assert reliability_path.exists()

    saved = joblib.load(model_path)
    assert "pipeline" in saved
    assert saved["feature_columns"] == _ALL_FEATURE_COLUMNS

    X = df[_ALL_FEATURE_COLUMNS].head(5).to_numpy(dtype=float)
    proba = saved["pipeline"].predict_proba(X)
    assert proba.shape == (5, 2)


def test_cli_no_lga_features_flag_trains_phase3c(tmp_path):
    """--no-lga-features trains the 15-feat Phase 3c schema on a CSV without LGA cols."""
    df = _synthetic_features_df(include_lga=False)
    features_path = tmp_path / "features.csv"
    df.to_csv(features_path, index=False)
    model_path = tmp_path / "models" / "lgbm_phase3c.joblib"
    reliability_path = tmp_path / "reliability_lgbm_phase3c.png"

    runner = CliRunner()
    res = runner.invoke(
        main,
        [
            "--features-csv", str(features_path),
            "--model-out", str(model_path),
            "--reliability-out", str(reliability_path),
            "--no-lga-features",
        ],
    )
    assert res.exit_code == 0, res.output
    assert "Phase 3c" in res.output

    saved = joblib.load(model_path)
    assert saved["feature_columns"] == FEATURE_COLUMNS


def test_cli_no_lga_features_with_lga_csv_errors(tmp_path):
    """--no-lga-features against a Phase 4 CSV (LGA cols present) is a hard error."""
    df = _synthetic_features_df(include_lga=True)
    features_path = tmp_path / "features.csv"
    df.to_csv(features_path, index=False)

    runner = CliRunner()
    res = runner.invoke(
        main,
        [
            "--features-csv", str(features_path),
            "--no-lga-features",
        ],
    )
    assert res.exit_code != 0
    assert "lga" in res.output.lower()


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


# ---------------------------------------------------------------------------
# Phase 4b — brand trough column wiring
# ---------------------------------------------------------------------------

def test_cli_default_picks_up_brand_columns(tmp_path):
    """Default schema picks up brand trough columns when present (Phase 4b)."""
    df = _synthetic_features_df(include_lga=True, brand_columns=_SYNTHETIC_BRAND_COLUMNS)
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
    assert "Phase 4b" in res.output

    saved = joblib.load(model_path)
    expected = FEATURE_COLUMNS + LGA_FEATURE_COLUMNS + sorted(_SYNTHETIC_BRAND_COLUMNS)
    assert saved["feature_columns"] == expected


def test_cli_no_brand_features_flag_trains_phase4(tmp_path):
    """--no-brand-features ignores brand cols even when present (reproduces Phase 4)."""
    df = _synthetic_features_df(include_lga=True, brand_columns=_SYNTHETIC_BRAND_COLUMNS)
    features_path = tmp_path / "features.csv"
    df.to_csv(features_path, index=False)
    model_path = tmp_path / "models" / "lgbm_phase4.joblib"
    reliability_path = tmp_path / "reliability_lgbm_phase4.png"

    runner = CliRunner()
    res = runner.invoke(
        main,
        [
            "--features-csv", str(features_path),
            "--model-out", str(model_path),
            "--reliability-out", str(reliability_path),
            "--no-brand-features",
        ],
    )
    assert res.exit_code == 0, res.output
    assert "Phase 4" in res.output
    assert "Phase 4b" not in res.output

    saved = joblib.load(model_path)
    assert saved["feature_columns"] == _ALL_FEATURE_COLUMNS


def test_cli_drop_feature_single(tmp_path):
    """--drop-feature COL removes that column from the resolved feature set."""
    df = _synthetic_features_df(include_lga=True)
    features_path = tmp_path / "features.csv"
    df.to_csv(features_path, index=False)
    model_path = tmp_path / "models" / "lgbm_drop.joblib"
    reliability_path = tmp_path / "reliability_drop.png"

    drop = FEATURE_COLUMNS[0]
    runner = CliRunner()
    res = runner.invoke(
        main,
        [
            "--features-csv", str(features_path),
            "--model-out", str(model_path),
            "--reliability-out", str(reliability_path),
            "--drop-feature", drop,
        ],
    )
    assert res.exit_code == 0, res.output
    assert drop in res.output

    saved = joblib.load(model_path)
    assert drop not in saved["feature_columns"]
    assert len(saved["feature_columns"]) == len(_ALL_FEATURE_COLUMNS) - 1


def test_cli_drop_feature_multiple(tmp_path):
    """--drop-feature is repeatable; multiple columns are removed."""
    df = _synthetic_features_df(include_lga=True)
    features_path = tmp_path / "features.csv"
    df.to_csv(features_path, index=False)
    model_path = tmp_path / "models" / "lgbm_drop_multi.joblib"
    reliability_path = tmp_path / "reliability_drop_multi.png"

    drop_a = FEATURE_COLUMNS[0]
    drop_b = LGA_FEATURE_COLUMNS[0]
    runner = CliRunner()
    res = runner.invoke(
        main,
        [
            "--features-csv", str(features_path),
            "--model-out", str(model_path),
            "--reliability-out", str(reliability_path),
            "--drop-feature", drop_a,
            "--drop-feature", drop_b,
        ],
    )
    assert res.exit_code == 0, res.output

    saved = joblib.load(model_path)
    assert drop_a not in saved["feature_columns"]
    assert drop_b not in saved["feature_columns"]
    assert len(saved["feature_columns"]) == len(_ALL_FEATURE_COLUMNS) - 2


def test_cli_drop_feature_unknown_errors(tmp_path):
    """--drop-feature with a column not in the resolved set is a hard error."""
    df = _synthetic_features_df(include_lga=True)
    features_path = tmp_path / "features.csv"
    df.to_csv(features_path, index=False)

    runner = CliRunner()
    res = runner.invoke(
        main,
        [
            "--features-csv", str(features_path),
            "--drop-feature", "not_a_real_feature_xyz",
        ],
    )
    assert res.exit_code != 0
    assert "not_a_real_feature_xyz" in res.output


def test_cli_seed_reproducible(tmp_path):
    """Same --seed → byte-identical model artifact; different --seed → different model."""
    df = _synthetic_features_df(include_lga=True)
    features_path = tmp_path / "features.csv"
    df.to_csv(features_path, index=False)

    def _run(seed: int, tag: str) -> np.ndarray:
        model_path = tmp_path / f"lgbm_{tag}.joblib"
        reliability_path = tmp_path / f"reliability_{tag}.png"
        runner = CliRunner()
        res = runner.invoke(
            main,
            [
                "--features-csv", str(features_path),
                "--model-out", str(model_path),
                "--reliability-out", str(reliability_path),
                "--seed", str(seed),
            ],
        )
        assert res.exit_code == 0, res.output
        saved = joblib.load(model_path)
        X = df[saved["feature_columns"]].to_numpy(dtype=float)
        return saved["pipeline"].predict_proba(X)[:, 1]

    proba_a = _run(7, "seed7a")
    proba_b = _run(7, "seed7b")
    proba_c = _run(13, "seed13")
    # Same seed → bit-exact predictions.
    np.testing.assert_array_equal(proba_a, proba_b)
    # Different seeds → bagging sees different rows; predictions must differ
    # somewhere. Compare full arrays, not rounded display strings.
    assert not np.array_equal(proba_a, proba_c), (
        "Different seeds produced identical predictions — --seed is not plumbed through."
    )


def test_cli_no_lga_features_with_brand_csv_errors(tmp_path):
    """--no-lga-features against a CSV with brand cols is also a hard error."""
    df = _synthetic_features_df(include_lga=True, brand_columns=_SYNTHETIC_BRAND_COLUMNS)
    features_path = tmp_path / "features.csv"
    df.to_csv(features_path, index=False)

    runner = CliRunner()
    res = runner.invoke(
        main,
        [
            "--features-csv", str(features_path),
            "--no-lga-features",
        ],
    )
    assert res.exit_code != 0
    assert "brand" in res.output.lower()
