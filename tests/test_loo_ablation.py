"""Tests for fuel_signal.loo_ablation — fast, synthetic-data only."""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from fuel_signal.features import FEATURE_COLUMNS
from fuel_signal.loo_ablation import main, run_loo


def _date_range(start: str, n_days: int) -> list[str]:
    d0 = datetime.date.fromisoformat(start)
    return [(d0 + datetime.timedelta(days=i)).isoformat() for i in range(n_days)]


def _synthetic_df(seed: int = 0) -> pd.DataFrame:
    """Tiny synthetic frame covering train + val windows (2 seeds, fast)."""
    rng = np.random.default_rng(seed)
    train_dates = _date_range("2018-01-01", 500)
    val_dates = _date_range("2025-04-01", 60)
    all_dates = train_dates + val_dates
    n = len(all_dates)
    X = rng.normal(size=(n, len(FEATURE_COLUMNS)))
    logits = 3.0 * X[:, 0] - 2.0 * X[:, 1] - 0.5
    probs = 1.0 / (1.0 + np.exp(-logits))
    labels = (rng.uniform(size=n) < probs).astype(int)
    rows = {col: X[:, i] for i, col in enumerate(FEATURE_COLUMNS)}
    rows["price_date"] = all_dates
    rows["label"] = labels
    rows["station_code"] = np.arange(n) % 5
    rows["today_price_cents"] = 160.0
    rows["future_min_cents"] = 159.0
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# run_loo
# ---------------------------------------------------------------------------


def test_run_loo_returns_expected_keys():
    df = _synthetic_df()
    report = run_loo(df, [FEATURE_COLUMNS[-1]], seeds=[1, 7])
    expected = {
        "drop_columns",
        "seeds",
        "baseline_scores",
        "baseline_mean",
        "baseline_std",
        "loo_scores",
        "loo_mean",
        "loo_std",
        "delta",
        "verdict",
    }
    assert expected.issubset(set(report.keys()))


def test_run_loo_delta_math():
    """Δ must equal loo_mean − baseline_mean exactly."""
    df = _synthetic_df()
    report = run_loo(df, [FEATURE_COLUMNS[-1]], seeds=[1, 7])
    assert pytest.approx(report["delta"]) == report["loo_mean"] - report["baseline_mean"]


def test_run_loo_std_matches_numpy():
    """baseline_std / loo_std must match numpy std(ddof=1) of per-seed scores."""
    df = _synthetic_df()
    report = run_loo(df, [FEATURE_COLUMNS[-1]], seeds=[1, 7])
    assert pytest.approx(report["baseline_std"]) == float(
        np.std(report["baseline_scores"], ddof=1)
    )
    assert pytest.approx(report["loo_std"]) == float(
        np.std(report["loo_scores"], ddof=1)
    )


def test_run_loo_verdict_values():
    """Verdict must be one of the three documented strings."""
    df = _synthetic_df()
    report = run_loo(df, [FEATURE_COLUMNS[-1]], seeds=[1, 7])
    valid_verdicts = {
        "within noise / redundant",
        "feature contributes (starved)",
        "feature harmful (unexpected)",
    }
    assert report["verdict"] in valid_verdicts


def test_run_loo_verdict_logic():
    """Within-noise verdict fires when |Δ| < baseline_std; outside-band verdicts
    reflect sign of Δ."""
    df = _synthetic_df()
    # Use a non-informative feature (last column, which has near-zero coefficient)
    # to get a within-noise result at 2 seeds.
    report = run_loo(df, [FEATURE_COLUMNS[-1]], seeds=[1, 7])
    delta = report["delta"]
    bstd = report["baseline_std"]
    if abs(delta) < bstd:
        assert report["verdict"] == "within noise / redundant"
    elif delta > 0:
        assert report["verdict"] == "feature contributes (starved)"
    else:
        assert report["verdict"] == "feature harmful (unexpected)"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_no_drop_errors(tmp_path):
    df = _synthetic_df()
    csv_path = tmp_path / "features.csv"
    df.to_csv(csv_path, index=False)
    runner = CliRunner()
    res = runner.invoke(main, ["--features-csv", str(csv_path), "--seeds", "1,7"])
    assert res.exit_code != 0
    assert "nothing to ablate" in res.output.lower()


def test_cli_unknown_drop_errors(tmp_path):
    df = _synthetic_df()
    csv_path = tmp_path / "features.csv"
    df.to_csv(csv_path, index=False)
    runner = CliRunner()
    res = runner.invoke(
        main,
        ["--features-csv", str(csv_path), "--drop", "not_a_real_column", "--seeds", "1"],
    )
    assert res.exit_code != 0
    assert "unknown feature" in res.output.lower()


def test_cli_missing_csv_errors(tmp_path):
    runner = CliRunner()
    res = runner.invoke(
        main,
        ["--features-csv", str(tmp_path / "nope.csv"), "--drop", FEATURE_COLUMNS[-1], "--seeds", "1"],
    )
    assert res.exit_code != 0
    assert "not found" in res.output.lower()


def test_cli_end_to_end(tmp_path):
    """Smoke test: runs to completion and prints the expected output sections."""
    df = _synthetic_df()
    csv_path = tmp_path / "features.csv"
    df.to_csv(csv_path, index=False)
    runner = CliRunner()
    res = runner.invoke(
        main,
        [
            "--features-csv", str(csv_path),
            "--drop", FEATURE_COLUMNS[-1],
            "--seeds", "1,7",
        ],
    )
    assert res.exit_code == 0, res.output
    assert "LOO ablation" in res.output
    assert "baseline" in res.output
    assert "Verdict" in res.output
