"""Tests for fuel_signal.feature_redundancy — clustering + decomposition scores."""

from __future__ import annotations

import datetime
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner
from lightgbm import LGBMClassifier

from fuel_signal.feature_redundancy import (
    build_cluster_table,
    cluster_features,
    compute_interaction_matrix,
    decomposition_scores,
    main,
    run_redundancy_report,
    shap_correlation_matrix,
)

_FEATURES = ["feat_a", "feat_b", "feat_c", "feat_d"]


def _date_range(start: str, n: int) -> list[str]:
    d0 = datetime.date.fromisoformat(start)
    return [(d0 + datetime.timedelta(days=i)).isoformat() for i in range(n)]


def _synthetic_df(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    train_dates = _date_range("2018-01-01", 400)
    val_dates = _date_range("2025-04-01", 60)
    test_dates = _date_range("2025-08-01", 60)
    all_dates = train_dates + val_dates + test_dates
    n = len(all_dates)

    X = rng.normal(size=(n, len(_FEATURES)))
    logits = 2.0 * X[:, 0] - 1.5 * X[:, 1] + 0.5 * X[:, 2]
    probs = 1.0 / (1.0 + np.exp(-logits))
    labels = (rng.uniform(size=n) < probs).astype(int)

    rows = {col: X[:, i] for i, col in enumerate(_FEATURES)}
    rows["price_date"] = all_dates
    rows["label"] = labels
    rows["station_code"] = 0
    rows["today_price_cents"] = 160.0
    rows["future_min_cents"] = 159.0
    return pd.DataFrame(rows)


def _fit_model(df: pd.DataFrame, feature_columns: list[str]) -> LGBMClassifier:
    from fuel_signal import evaluate as _ev

    train, _val, _test = _ev.split(df)
    X = train[feature_columns].to_numpy(dtype=float)
    y = train["label"].to_numpy(dtype=int)
    m = LGBMClassifier(random_state=42, verbose=-1, n_estimators=15)
    m.fit(X, y)
    return m


def _make_bundle(tmp_path, df: pd.DataFrame, feature_columns: list[str]):
    model = _fit_model(df, feature_columns)
    bundle = {"pipeline": model, "feature_columns": feature_columns}
    mp = tmp_path / "model.joblib"
    joblib.dump(bundle, mp)
    fp = tmp_path / "features.csv"
    df.to_csv(fp, index=False)
    return mp, fp, model


# ---------------------------------------------------------------------------
# shap_correlation_matrix
# ---------------------------------------------------------------------------

def test_shap_corr_identical_columns_are_perfectly_correlated():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(500, 1))
    sv = np.hstack([base, base, rng.normal(size=(500, 1))])
    corr = shap_correlation_matrix(sv)
    assert corr[0, 1] == pytest.approx(1.0, abs=1e-12)
    assert corr[0, 0] == pytest.approx(1.0)
    # independent column is near-zero correlated with base
    assert abs(corr[0, 2]) < 0.2


def test_shap_corr_zero_variance_column_is_nan():
    n = 100
    sv = np.column_stack([np.linspace(0, 1, n), np.zeros(n)])
    corr = shap_correlation_matrix(sv)
    assert np.isnan(corr[0, 1])
    assert np.isnan(corr[1, 0])


# ---------------------------------------------------------------------------
# cluster_features
# ---------------------------------------------------------------------------

def test_cluster_groups_duplicate_features():
    rng = np.random.default_rng(1)
    base_a = rng.normal(size=500)
    base_b = rng.normal(size=500)
    # Two pairs of near-duplicates + one independent
    sv = np.column_stack([
        base_a,
        base_a + 0.01 * rng.normal(size=500),
        base_b,
        base_b + 0.01 * rng.normal(size=500),
        rng.normal(size=500),
    ])
    corr = shap_correlation_matrix(sv)
    labels, _Z = cluster_features(corr, threshold=0.3)
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]
    assert labels[4] not in {labels[0], labels[2]}


def test_cluster_table_siblings_listed():
    labels = np.array([1, 1, 2, 2, 3])
    mean_abs = np.array([0.5, 0.3, 0.2, 0.1, 0.4])
    table = build_cluster_table(["a", "b", "c", "d", "e"], mean_abs, labels)
    a_row = table[table["feature"] == "a"].iloc[0]
    assert a_row["siblings"] == "b"
    e_row = table[table["feature"] == "e"].iloc[0]
    assert e_row["siblings"] == ""


# ---------------------------------------------------------------------------
# decomposition_scores
# ---------------------------------------------------------------------------

def test_decomposition_concentrated_feature_has_low_entropy():
    F = 4
    M = np.zeros((F, F))
    np.fill_diagonal(M, 0.1)
    # feature 0: all partner mass on feature 1 (concentrated)
    M[0, 1] = M[1, 0] = 1.0
    # feature 2: partner mass split evenly across 0, 1, 3 (diffuse)
    M[2, 0] = M[0, 2] = 0.3
    M[2, 1] = M[1, 2] = 0.3
    M[2, 3] = M[3, 2] = 0.3
    df = decomposition_scores(["f0", "f1", "f2", "f3"], M)

    row_0 = df[df["feature"] == "f0"].iloc[0]
    row_2 = df[df["feature"] == "f2"].iloc[0]
    assert row_2["entropy_norm"] > row_0["entropy_norm"]
    assert row_0["top1_partner"] == "f1"
    # f0 also picks up 0.3 mass from f2 (symmetric interaction); f1 still dominates
    assert row_0["top1_share"] > 0.7
    assert row_2["n_partners_ge_5pct"] == 3


def test_decomposition_isolated_feature_handled():
    M = np.diag([0.2, 0.0, 0.0])  # f0 has main effect only, f1/f2 zero
    df = decomposition_scores(["f0", "f1", "f2"], M)
    row = df[df["feature"] == "f0"].iloc[0]
    assert row["total_partner_mass"] == 0.0
    assert row["entropy_norm"] == 0.0
    assert row["main_effect_share"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# compute_interaction_matrix
# ---------------------------------------------------------------------------

def test_compute_interaction_matrix_shape_and_symmetry(tmp_path):
    df = _synthetic_df()
    _mp, _fp, model = _make_bundle(tmp_path, df, _FEATURES)
    from fuel_signal import evaluate as _ev

    _, val, _ = _ev.split(df)
    X = val[_FEATURES].to_numpy(dtype=float)
    rng = np.random.default_rng(0)
    M, n_used = compute_interaction_matrix(model, X, sample_size=1_000, rng=rng)
    assert M.shape == (len(_FEATURES), len(_FEATURES))
    assert np.allclose(M, M.T, atol=1e-10)
    assert n_used == X.shape[0]  # smaller than cap


# ---------------------------------------------------------------------------
# run_redundancy_report — integration
# ---------------------------------------------------------------------------

_PAIRED_CV_COLS = {
    "paired_cv_median_delta",
    "paired_cv_worst_fold_delta",
    "paired_cv_fold_wins",
    "paired_cv_csv",
}


def test_run_redundancy_report_writes_all_artifacts(tmp_path):
    df = _synthetic_df()
    mp, fp, _ = _make_bundle(tmp_path, df, _FEATURES)
    out = tmp_path / "out"
    result = run_redundancy_report(
        mp, fp, "val", out,
        cluster_threshold=0.5,
        interaction_sample=1_000,
        seed=0,
        skip_paired_cv=True,
    )

    for name in [
        "shap_corr.csv",
        "clusters.csv",
        "dendrogram.png",
        "interaction_matrix.csv",
        "decomposition_candidates.csv",
        "feature_columns.json",
        "params.json",
    ]:
        assert (out / name).exists(), f"missing artifact: {name}"

    fc = json.loads((out / "feature_columns.json").read_text())
    assert fc == _FEATURES

    decomp = pd.read_csv(out / "decomposition_candidates.csv")
    assert len(decomp) == len(_FEATURES)
    assert {
        "feature", "main_effect", "total_partner_mass", "main_effect_share",
        "entropy_norm", "n_partners_ge_5pct", "top1_partner", "top1_share",
    }.issubset(decomp.columns)
    assert _PAIRED_CV_COLS.issubset(decomp.columns)
    # entropy_norm is sorted descending
    assert decomp["entropy_norm"].is_monotonic_decreasing

    clusters = pd.read_csv(out / "clusters.csv")
    assert result["n_clusters"] == clusters["cluster_id"].nunique()
    assert _PAIRED_CV_COLS.issubset(clusters.columns)
    # skip_paired_cv=True → paired_cv_median_delta is NaN throughout
    assert clusters["paired_cv_median_delta"].isna().all()
    assert decomp["paired_cv_median_delta"].isna().all()


def test_run_redundancy_report_paired_cv_populates_columns(tmp_path):
    """Paired CV path: columns are non-NaN and per-fold CSVs are written.

    Uses reduced cv_train_min_days/val_days/step_days so the synthetic data
    (400 continuous days in 2018) produces non-empty folds without needing a
    full ~5-year training window.
    """
    df = _synthetic_df()
    mp, fp, _ = _make_bundle(tmp_path, df, _FEATURES)
    out = tmp_path / "cv_out"
    run_redundancy_report(
        mp, fp, "val", out,
        cluster_threshold=0.5,
        interaction_sample=500,
        seed=0,
        skip_paired_cv=False,
        cv_seed=0,
        cv_train_min_days=50,
        cv_val_days=20,
        cv_step_days=20,
    )

    clusters = pd.read_csv(out / "clusters.csv")
    decomp = pd.read_csv(out / "decomposition_candidates.csv")

    assert _PAIRED_CV_COLS.issubset(clusters.columns)
    assert _PAIRED_CV_COLS.issubset(decomp.columns)

    # At least some clusters/features have CV results (non-NaN median delta)
    assert clusters["paired_cv_median_delta"].notna().any()
    assert decomp["paired_cv_median_delta"].notna().any()

    # fold_wins is N/M format for rows that ran
    wins_nonempty = clusters["paired_cv_fold_wins"].dropna()
    wins_nonempty = wins_nonempty[wins_nonempty != ""]
    assert wins_nonempty.str.match(r"^\d+/\d+$").all()

    # per-fold CSVs exist for non-empty entries
    first_csv = clusters["paired_cv_csv"].dropna().iloc[0]
    assert first_csv != ""
    assert (out / first_csv).exists()


def test_cli_smoke(tmp_path):
    df = _synthetic_df()
    mp, fp, _ = _make_bundle(tmp_path, df, _FEATURES)
    out = tmp_path / "cli_out"
    runner = CliRunner()
    res = runner.invoke(
        main,
        [
            "--model", str(mp),
            "--features", str(fp),
            "--split", "val",
            "--output", str(out),
            "--interaction-sample", "500",
            "--skip-paired-cv",
        ],
    )
    assert res.exit_code == 0, res.output
    assert (out / "clusters.csv").exists()
    assert (out / "decomposition_candidates.csv").exists()
