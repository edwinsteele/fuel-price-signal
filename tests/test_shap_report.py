"""Tests for fuel_signal.shap_report — smoke tests and artifact validation."""

from __future__ import annotations

import datetime

import joblib
import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner
from lightgbm import LGBMClassifier

from fuel_signal.shap_report import (
    approx_interaction_scores,
    build_summary,
    compute_partner_scores,
    compute_shap,
    main,
    run_shap_report,
    save_dependence_plots,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FEATURES = ["feat_a", "feat_b", "feat_c"]


def _date_range(start: str, n: int) -> list[str]:
    d0 = datetime.date.fromisoformat(start)
    return [(d0 + datetime.timedelta(days=i)).isoformat() for i in range(n)]


def _synthetic_df(seed: int = 0, nan_feature: bool = False) -> pd.DataFrame:
    """Minimal frame spanning train + val + test canonical windows."""
    rng = np.random.default_rng(seed)
    train_dates = _date_range("2018-01-01", 400)
    val_dates = _date_range("2025-04-01", 60)
    test_dates = _date_range("2025-08-01", 60)
    all_dates = train_dates + val_dates + test_dates
    n = len(all_dates)

    X = rng.normal(size=(n, len(_FEATURES)))
    logits = 2.0 * X[:, 0] - 1.5 * X[:, 1]
    probs = 1.0 / (1.0 + np.exp(-logits))
    labels = (rng.uniform(size=n) < probs).astype(int)

    rows = {col: X[:, i] for i, col in enumerate(_FEATURES)}
    rows["price_date"] = all_dates
    rows["label"] = labels
    rows["station_code"] = 0
    rows["today_price_cents"] = 160.0
    rows["future_min_cents"] = 159.0

    if nan_feature:
        # Simulate a LGA feature that is all-NaN for half the rows.
        nan_col = np.where(rng.uniform(size=n) < 0.5, np.nan, rng.normal(size=n))
        rows["feat_nan"] = nan_col

    return pd.DataFrame(rows)


def _fit_model(df: pd.DataFrame, feature_columns: list[str]) -> LGBMClassifier:
    from fuel_signal import evaluate as _ev

    train, _val, _test = _ev.split(df)
    X = train[feature_columns].to_numpy(dtype=float)
    y = train["label"].to_numpy(dtype=int)
    m = LGBMClassifier(random_state=42, verbose=-1, n_estimators=10)
    m.fit(X, y)
    return m


def _make_bundle(tmp_path, df: pd.DataFrame, feature_columns: list[str]) -> tuple:
    model = _fit_model(df, feature_columns)
    bundle = {"pipeline": model, "feature_columns": feature_columns}
    mp = tmp_path / "model.joblib"
    joblib.dump(bundle, mp)
    fp = tmp_path / "features.csv"
    df.to_csv(fp, index=False)
    return mp, fp, model, feature_columns


# ---------------------------------------------------------------------------
# approx_interaction_scores
# ---------------------------------------------------------------------------

def test_approx_interaction_scores_shape():
    rng = np.random.default_rng(0)
    n, k = 300, 4
    X = rng.normal(size=(n, k))
    sv = rng.normal(size=(n, k))
    scores = approx_interaction_scores(0, sv, X)
    assert scores.shape == (k,)
    assert scores[0] == 0.0  # self-score is zero


def test_approx_interaction_scores_self_zero():
    rng = np.random.default_rng(1)
    n, k = 200, 5
    X = rng.normal(size=(n, k))
    sv = rng.normal(size=(n, k))
    for i in range(k):
        scores = approx_interaction_scores(i, sv, X)
        assert scores[i] == 0.0


# ---------------------------------------------------------------------------
# compute_partner_scores
# ---------------------------------------------------------------------------

def test_compute_partner_scores_columns():
    rng = np.random.default_rng(0)
    n, k = 200, 3
    X = rng.normal(size=(n, k))
    sv = rng.normal(size=(n, k))
    df = compute_partner_scores(["a", "b", "c"], X, sv)
    assert set(df.columns) == {"feature", "partner", "score", "pct_of_top", "pct_of_total"}


def test_compute_partner_scores_no_self_pairs():
    rng = np.random.default_rng(1)
    n, k = 200, 3
    X = rng.normal(size=(n, k))
    sv = rng.normal(size=(n, k))
    df = compute_partner_scores(["a", "b", "c"], X, sv)
    assert (df["feature"] == df["partner"]).sum() == 0


def test_compute_partner_scores_pct_of_top_le_one():
    rng = np.random.default_rng(2)
    n, k = 200, 4
    X = rng.normal(size=(n, k))
    sv = rng.normal(size=(n, k))
    df = compute_partner_scores(["a", "b", "c", "d"], X, sv)
    assert (df["pct_of_top"] <= 1.0 + 1e-9).all()


# ---------------------------------------------------------------------------
# compute_shap
# ---------------------------------------------------------------------------

def test_compute_shap_shape(tmp_path):
    df = _synthetic_df()
    from fuel_signal import evaluate as _ev

    model = _fit_model(df, _FEATURES)
    _, val, _ = _ev.split(df)
    X = val[_FEATURES].to_numpy(dtype=float)
    sv = compute_shap(model, X)
    assert sv.shape == X.shape


def test_compute_shap_returns_ndarray(tmp_path):
    df = _synthetic_df()
    from fuel_signal import evaluate as _ev

    model = _fit_model(df, _FEATURES)
    _, val, _ = _ev.split(df)
    X = val[_FEATURES].to_numpy(dtype=float)
    sv = compute_shap(model, X)
    assert isinstance(sv, np.ndarray)


# ---------------------------------------------------------------------------
# build_summary
# ---------------------------------------------------------------------------

def test_build_summary_columns():
    rng = np.random.default_rng(0)
    n, k = 100, 3
    X = rng.normal(size=(n, k))
    sv = rng.normal(size=(n, k))
    summary = build_summary(["a", "b", "c"], X, sv)
    assert list(summary.columns) == ["feature", "mean_abs_shap", "rank", "r", "nan_fraction"]


def test_build_summary_rank_is_1_indexed_and_consecutive():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 4))
    sv = rng.normal(size=(200, 4))
    summary = build_summary(["a", "b", "c", "d"], X, sv)
    assert sorted(summary["rank"].tolist()) == [1, 2, 3, 4]


def test_build_summary_sorted_descending_by_mean_abs_shap():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(200, 3))
    sv = rng.normal(size=(200, 3))
    summary = build_summary(["a", "b", "c"], X, sv)
    vals = summary["mean_abs_shap"].tolist()
    assert vals == sorted(vals, reverse=True)


def test_build_summary_nan_fraction_for_all_nan_feature():
    rng = np.random.default_rng(3)
    n, k = 100, 2
    X = rng.normal(size=(n, k)).astype(float)
    X[:, 1] = np.nan  # feature 1 is all-NaN
    sv = rng.normal(size=(n, k))
    summary = build_summary(["good", "bad"], X, sv)
    bad_row = summary[summary["feature"] == "bad"].iloc[0]
    assert bad_row["nan_fraction"] == pytest.approx(1.0)
    assert np.isnan(bad_row["r"])


def test_build_summary_r_is_signed_float():
    rng = np.random.default_rng(4)
    n = 500
    X = rng.normal(size=(n, 1))
    # SHAP strongly positively correlated with feature
    sv = X * 2.0 + rng.normal(scale=0.01, size=(n, 1))
    summary = build_summary(["f"], X, sv)
    assert summary.iloc[0]["r"] > 0.99  # should be near +1


def test_build_summary_nan_fraction_correct():
    rng = np.random.default_rng(5)
    n = 100
    X = rng.normal(size=(n, 1))
    X[:25, 0] = np.nan  # 25% NaN
    sv = rng.normal(size=(n, 1))
    summary = build_summary(["f"], X, sv)
    assert summary.iloc[0]["nan_fraction"] == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# save_dependence_plots
# ---------------------------------------------------------------------------

def test_save_dependence_plots_creates_pngs(tmp_path):
    rng = np.random.default_rng(0)
    n, k = 100, 3
    X = rng.normal(size=(n, k))
    sv = rng.normal(size=(n, k))
    save_dependence_plots(["a", "b", "c"], X, sv, tmp_path / "dep")
    for name in ["a", "b", "c"]:
        assert (tmp_path / "dep" / f"{name}.png").exists()


def test_save_dependence_plots_handles_all_nan(tmp_path):
    rng = np.random.default_rng(0)
    n, k = 50, 2
    X = rng.normal(size=(n, k))
    X[:, 1] = np.nan  # all-NaN
    sv = rng.normal(size=(n, k))
    save_dependence_plots(["ok", "nan_feat"], X, sv, tmp_path / "dep")
    assert (tmp_path / "dep" / "nan_feat.png").exists()


# ---------------------------------------------------------------------------
# run_shap_report — integration
# ---------------------------------------------------------------------------

def test_run_shap_report_writes_artifacts(tmp_path):
    df = _synthetic_df()
    mp, fp, _, feat_cols = _make_bundle(tmp_path, df, _FEATURES)
    out = tmp_path / "out"
    run_shap_report(mp, fp, "val", out)

    assert (out / "shap_values.npy").exists()
    assert (out / "summary.csv").exists()
    dep_dir = out / "dependence"
    assert dep_dir.is_dir()
    for feat in feat_cols:
        assert (dep_dir / f"{feat}.png").exists()


def test_run_shap_report_shap_values_shape(tmp_path):
    df = _synthetic_df()
    mp, fp, _, _ = _make_bundle(tmp_path, df, _FEATURES)
    result = run_shap_report(mp, fp, "val", tmp_path / "out")
    sv = result["shap_values"]
    assert sv.ndim == 2
    assert sv.shape[1] == len(_FEATURES)
    assert result["n_rows"] == sv.shape[0]


def test_run_shap_report_summary_csv_schema(tmp_path):
    df = _synthetic_df()
    mp, fp, _, _ = _make_bundle(tmp_path, df, _FEATURES)
    run_shap_report(mp, fp, "val", tmp_path / "out")
    summary = pd.read_csv(tmp_path / "out" / "summary.csv")
    assert set(summary.columns) == {"feature", "mean_abs_shap", "rank", "r", "nan_fraction"}
    assert len(summary) == len(_FEATURES)


def test_run_shap_report_writes_xval_and_feature_columns(tmp_path):
    import json as _json
    df = _synthetic_df()
    mp, fp, _, _ = _make_bundle(tmp_path, df, _FEATURES)
    out = tmp_path / "out"
    run_shap_report(mp, fp, "val", out)
    assert (out / "X_val.npy").exists()
    assert (out / "feature_columns.json").exists()
    fc = _json.loads((out / "feature_columns.json").read_text())
    assert fc == _FEATURES
    xv = np.load(out / "X_val.npy")
    assert xv.shape[1] == len(_FEATURES)


def test_run_shap_report_writes_partner_scores(tmp_path):
    df = _synthetic_df()
    mp, fp, _, _ = _make_bundle(tmp_path, df, _FEATURES)
    out = tmp_path / "out"
    run_shap_report(mp, fp, "val", out)
    assert (out / "partner_scores.csv").exists()
    ps = pd.read_csv(out / "partner_scores.csv")
    assert set(ps.columns) == {"feature", "partner", "score", "pct_of_top", "pct_of_total"}
    # Each (feature, partner) pair should have feature != partner
    assert (ps["feature"] == ps["partner"]).sum() == 0
    # pct_of_top should be <= 1 for all rows
    assert (ps["pct_of_top"] <= 1.0 + 1e-9).all()


def test_run_shap_report_handles_nan_feature(tmp_path):
    nan_feats = _FEATURES + ["feat_nan"]
    df = _synthetic_df(nan_feature=True)
    mp, fp, _, _ = _make_bundle(tmp_path, df, nan_feats)
    result = run_shap_report(mp, fp, "val", tmp_path / "out")
    summary = result["summary"]
    nan_row = summary[summary["feature"] == "feat_nan"].iloc[0]
    assert nan_row["nan_fraction"] > 0.0


def test_run_shap_report_test_split(tmp_path):
    df = _synthetic_df()
    mp, fp, _, _ = _make_bundle(tmp_path, df, _FEATURES)
    result = run_shap_report(mp, fp, "test", tmp_path / "out")
    assert result["n_rows"] > 0


def test_run_shap_report_npy_matches_returned_array(tmp_path):
    df = _synthetic_df()
    mp, fp, _, _ = _make_bundle(tmp_path, df, _FEATURES)
    result = run_shap_report(mp, fp, "val", tmp_path / "out")
    saved = np.load(tmp_path / "out" / "shap_values.npy")
    np.testing.assert_array_equal(saved, result["shap_values"])


# ---------------------------------------------------------------------------
# CLI smoke tests
# ---------------------------------------------------------------------------

def test_cli_runs_end_to_end(tmp_path):
    df = _synthetic_df()
    mp, fp, _, _ = _make_bundle(tmp_path, df, _FEATURES)
    out = tmp_path / "out"
    runner = CliRunner()
    res = runner.invoke(main, [
        "--model", str(mp),
        "--features", str(fp),
        "--split", "val",
        "--output", str(out),
    ])
    assert res.exit_code == 0, res.output
    assert "summary" in res.output.lower()
    assert (out / "shap_values.npy").exists()
    assert (out / "summary.csv").exists()


def test_cli_missing_model_errors(tmp_path):
    df = _synthetic_df()
    fp = tmp_path / "features.csv"
    df.to_csv(fp, index=False)
    runner = CliRunner()
    res = runner.invoke(main, [
        "--model", str(tmp_path / "nope.joblib"),
        "--features", str(fp),
        "--output", str(tmp_path / "out"),
    ])
    assert res.exit_code != 0
    assert "not found" in res.output.lower()


def test_cli_missing_features_errors(tmp_path):
    df = _synthetic_df()
    mp, _, _, _ = _make_bundle(tmp_path, df, _FEATURES)
    runner = CliRunner()
    res = runner.invoke(main, [
        "--model", str(mp),
        "--features", str(tmp_path / "nope.csv"),
        "--output", str(tmp_path / "out"),
    ])
    assert res.exit_code != 0
    assert "not found" in res.output.lower()


def test_cli_default_split_is_val(tmp_path):
    df = _synthetic_df()
    mp, fp, _, _ = _make_bundle(tmp_path, df, _FEATURES)
    out = tmp_path / "out"
    runner = CliRunner()
    res = runner.invoke(main, [
        "--model", str(mp),
        "--features", str(fp),
        "--output", str(out),
    ])
    assert res.exit_code == 0, res.output
    assert "val" in res.output
