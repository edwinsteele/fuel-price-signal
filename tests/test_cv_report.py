"""Tests for fuel_signal.cv_report — walk-forward CV runner and CLI smoke."""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
from click.testing import CliRunner

from fuel_signal import evaluate as _ev
from fuel_signal.cv_report import main, run_cv
from fuel_signal.features import FEATURE_COLUMNS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _date_range(start: str, n_days: int) -> list[str]:
    d0 = datetime.date.fromisoformat(start)
    return [(d0 + datetime.timedelta(days=i)).isoformat() for i in range(n_days)]


def _synthetic_cv_df(seed: int = 0) -> pd.DataFrame:
    """Daily-resolution frame spanning the pre-test window only.

    Covers 2020-01-01 to 2022-12-31 (~1096 days). With train_min_days=200,
    val_days=30, step_days=90 this produces ~9 folds, each fast to fit.
    Label is a noisy linear function of two features so logreg gets real signal.
    """
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


# ---------------------------------------------------------------------------
# run_cv — structure
# ---------------------------------------------------------------------------

def test_run_cv_returns_list_of_dicts():
    df = _synthetic_cv_df()
    results = run_cv(df, train_min_days=200, val_days=30, step_days=90)
    assert isinstance(results, list)
    assert len(results) >= 1
    expected_keys = {"fold", "train_rows", "val_rows", "val_buy_rate", "val_logloss", "baseline_logloss"}
    for r in results:
        assert expected_keys == set(r.keys())


def test_run_cv_fold_numbers_are_positive():
    df = _synthetic_cv_df()
    results = run_cv(df, train_min_days=200, val_days=30, step_days=90)
    for r in results:
        assert r["fold"] >= 1
        assert r["train_rows"] > 0
        assert r["val_rows"] > 0


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
    """Expanding window: each fold's train_rows >= the previous fold's."""
    df = _synthetic_cv_df()
    results = run_cv(df, train_min_days=200, val_days=30, step_days=90)
    assert len(results) >= 2
    for prev, curr in zip(results, results[1:]):
        assert curr["train_rows"] >= prev["train_rows"]


def test_run_cv_skips_empty_val_folds():
    """Folds whose val window falls in a data gap are skipped (no crash, no spurious rows)."""
    # Only keep dates that deliberately avoid 2020-04-01..2020-10-31 so
    # some val windows land in the gap.
    df = _synthetic_cv_df()
    gap_start = pd.Timestamp("2020-04-01")
    gap_end = pd.Timestamp("2020-10-31")
    dates_ts = pd.to_datetime(df["price_date"])
    df_with_gap = df[~((dates_ts >= gap_start) & (dates_ts <= gap_end))].copy()

    # Should not raise; may return fewer folds than the gapless version.
    results = run_cv(df_with_gap, train_min_days=200, val_days=30, step_days=90)
    gapless_results = run_cv(df, train_min_days=200, val_days=30, step_days=90)
    assert len(results) <= len(gapless_results)


def test_run_cv_empty_df_returns_empty_list():
    empty = pd.DataFrame(columns=["price_date", "label"] + FEATURE_COLUMNS)
    results = run_cv(empty, train_min_days=100, val_days=30, step_days=30)
    assert results == []


def test_run_cv_excludes_test_window():
    """Rows at or after TEST_START never appear in any fold (walk_forward_folds contract)."""
    df = _synthetic_cv_df()
    # Add rows inside the test window — they must be ignored.
    extra_dates = _date_range(_ev.TEST_START, 30)
    rng = np.random.default_rng(99)
    extra = {col: rng.normal(size=30) for col in FEATURE_COLUMNS}
    extra["price_date"] = extra_dates
    extra["label"] = rng.integers(0, 2, size=30)
    extra["station_code"] = 0
    extra["today_price_cents"] = 160.0
    extra["future_min_cents"] = 159.0
    df_extended = pd.concat([df, pd.DataFrame(extra)], ignore_index=True)

    # Total train_rows across all folds must equal what we'd get without the extra rows.
    results_clean = run_cv(df, train_min_days=200, val_days=30, step_days=90)
    results_ext = run_cv(df_extended, train_min_days=200, val_days=30, step_days=90)
    assert len(results_clean) == len(results_ext)
    for rc, re in zip(results_clean, results_ext):
        assert rc["train_rows"] == re["train_rows"]
        assert rc["val_rows"] == re["val_rows"]


# ---------------------------------------------------------------------------
# CLI smoke tests
# ---------------------------------------------------------------------------

def test_cli_runs_end_to_end(tmp_path):
    df = _synthetic_cv_df()
    csv_path = tmp_path / "features.csv"
    df.to_csv(csv_path, index=False)

    runner = CliRunner()
    res = runner.invoke(
        main,
        [
            "--features-csv", str(csv_path),
            "--train-min-days", "200",
            "--val-days", "30",
            "--step-days", "90",
        ],
    )
    assert res.exit_code == 0, res.output
    assert "fold" in res.output
    assert "logloss" in res.output
    assert "folds:" in res.output


def test_cli_output_contains_summary_line(tmp_path):
    df = _synthetic_cv_df()
    csv_path = tmp_path / "features.csv"
    df.to_csv(csv_path, index=False)

    runner = CliRunner()
    res = runner.invoke(
        main,
        ["--features-csv", str(csv_path), "--train-min-days", "200",
         "--val-days", "30", "--step-days", "90"],
    )
    assert res.exit_code == 0, res.output
    lines = [ln for ln in res.output.splitlines() if ln.startswith("folds:")]
    assert len(lines) == 1
    assert "±" in lines[0]


def test_cli_missing_csv_errors(tmp_path):
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
