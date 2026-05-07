"""Tests for fuel_signal.evaluate — canonical split, scoring, and experiment logging."""

import numpy as np
import pandas as pd
import pytest

import fuel_signal.evaluate as ev
from fuel_signal.evaluate import (
    TEST_END,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
    VAL_END,
    VAL_START,
    baseline_prior,
    brier,
    log_experiment,
    log_loss,
    split,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(dates_labels: list[tuple[str, int]]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "price_date": d,
            "label": lbl,
            "station_code": 1,
            "today_price_cents": 160.0,
            "future_min_cents": 162.0,
        }
        for d, lbl in dates_labels
    ])


# ---------------------------------------------------------------------------
# split() — boundary assignment
# ---------------------------------------------------------------------------

def test_split_assigns_boundary_dates():
    """Last day of train and first/last days of val/test land in the right bucket."""
    df = _make_df([
        ("2020-06-15", 0),  # interior train
        (TRAIN_END, 0),     # last day of train
        (VAL_START, 0),     # first day of val
        (VAL_END, 0),       # last day of val
        (TEST_START, 0),    # first day of test
        (TEST_END, 0),      # last day of test
    ])
    train, val, test = split(df)
    assert set(train["price_date"]) == {"2020-06-15", TRAIN_END}
    assert set(val["price_date"]) == {VAL_START, VAL_END}
    assert set(test["price_date"]) == {TEST_START, TEST_END}


def test_split_excludes_pre_train_start():
    """Rows before TRAIN_START are excluded from train (canonical window is enforced)."""
    df = _make_df([
        ("2015-01-01", 0),  # before TRAIN_START — must be excluded
        ("2016-07-31", 0),  # day before TRAIN_START — must be excluded
        (TRAIN_START, 0),   # first day of train — must be included
        ("2020-06-15", 0),  # interior train
    ])
    train, _val, _test = split(df)
    all_train_dates = set(train["price_date"])
    assert "2015-01-01" not in all_train_dates
    assert "2016-07-31" not in all_train_dates
    assert TRAIN_START in all_train_dates
    assert "2020-06-15" in all_train_dates


# ---------------------------------------------------------------------------
# split() — buffer rows are absent from all three splits
# ---------------------------------------------------------------------------

def test_split_drops_buffer_rows():
    """Rows in either 7-day buffer window are excluded from all three subsets."""
    buf1 = "2025-03-18"   # TRAIN_END + 1 day (inside buffer 1)
    buf2 = "2025-06-24"   # VAL_END + 1 day (inside buffer 2)
    df = _make_df([
        ("2024-01-01", 0),  # train
        (buf1, 0),          # buffer 1 — must be dropped
        ("2025-04-01", 0),  # val
        (buf2, 0),          # buffer 2 — must be dropped
        ("2025-08-01", 0),  # test
    ])
    train, val, test = split(df)
    all_dates = set(train["price_date"]) | set(val["price_date"]) | set(test["price_date"])
    assert buf1 not in all_dates, f"buffer row {buf1} appeared in a split"
    assert buf2 not in all_dates, f"buffer row {buf2} appeared in a split"


def test_split_drops_all_buffer_dates():
    """Every date in each buffer window is absent from all splits."""
    # Buffer 1: 2025-03-18 through 2025-03-24 (7 days)
    buffer1_dates = [f"2025-03-{d:02d}" for d in range(18, 25)]
    # Buffer 2: 2025-06-24 through 2025-06-30 (7 days)
    buffer2_dates = [f"2025-06-{d:02d}" for d in range(24, 31)]
    all_buffer = buffer1_dates + buffer2_dates

    df = _make_df([(d, 0) for d in all_buffer] + [("2024-01-01", 0)])
    train, val, test = split(df)
    all_dates = set(train["price_date"]) | set(val["price_date"]) | set(test["price_date"])
    for d in all_buffer:
        assert d not in all_dates, f"buffer date {d} appeared in a split"


# ---------------------------------------------------------------------------
# split() — no date appears in more than one split
# ---------------------------------------------------------------------------

def test_split_no_date_in_multiple_splits():
    """No price_date appears in more than one of train, val, test."""
    dates = [
        "2018-03-10",
        TRAIN_END,
        VAL_START,
        "2025-05-01",
        VAL_END,
        TEST_START,
        "2025-10-15",
        TEST_END,
    ]
    df = _make_df([(d, 0) for d in dates])
    train, val, test = split(df)
    train_d = set(train["price_date"])
    val_d = set(val["price_date"])
    test_d = set(test["price_date"])
    assert train_d & val_d == set(), f"overlap train∩val: {train_d & val_d}"
    assert train_d & test_d == set(), f"overlap train∩test: {train_d & test_d}"
    assert val_d & test_d == set(), f"overlap val∩test: {val_d & test_d}"


# ---------------------------------------------------------------------------
# Scoring — log_loss and brier
# ---------------------------------------------------------------------------

def test_baseline_prior_empty_raises():
    """baseline_prior() raises ValueError on an empty training DataFrame."""
    empty = pd.DataFrame(columns=["label", "price_date"])
    with pytest.raises(ValueError, match="baseline_prior"):
        baseline_prior(empty)


def test_log_loss_perfect_predictor():
    y = np.array([1.0, 0.0, 1.0])
    # Perfect predictions → loss → 0 (clipped, not exactly 0)
    assert log_loss(y, np.array([1.0, 0.0, 1.0])) < 1e-10


def test_log_loss_worst_predictor():
    y = np.array([1.0, 0.0])
    # Predicting exactly wrong → high loss
    assert log_loss(y, np.array([0.0, 1.0])) > 30


def test_brier_perfect():
    y = np.array([1.0, 0.0, 1.0])
    assert brier(y, np.array([1.0, 0.0, 1.0])) == pytest.approx(0.0)


def test_brier_worst():
    y = np.array([1.0, 0.0])
    assert brier(y, np.array([0.0, 1.0])) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Baseline entropy sanity test (from issue #26)
# ---------------------------------------------------------------------------

def test_baseline_logloss_matches_entropy():
    """Scoring the marginal-rate baseline on a test set with the same class balance
    as train produces log loss ≈ H(p) = -(p log p + (1-p) log(1-p)).

    Tolerance of 0.01 reflects the difference between train and test class balance.
    When both are equal (as here), the equality is exact — the assertion is strict.
    """
    p = 0.25    # easy fraction: 25 positives in 100 rows
    n = 100
    n_pos = int(n * p)

    labels = [1] * n_pos + [0] * (n - n_pos)

    # Build a single DataFrame: 100 train rows + 100 test rows, all with label rate p
    rows = (
        [{"price_date": "2020-01-01", "label": lbl, "station_code": i,
          "today_price_cents": 160.0, "future_min_cents": 162.0}
         for i, lbl in enumerate(labels)]
        + [{"price_date": "2025-08-01", "label": lbl, "station_code": i + n,
            "today_price_cents": 160.0, "future_min_cents": 162.0}
           for i, lbl in enumerate(labels)]
    )
    df = pd.DataFrame(rows)

    train, _val, test = split(df)

    p_hat = baseline_prior(train)
    assert p_hat == pytest.approx(p, abs=1e-9)

    pred = np.full(len(test), p_hat)
    expected = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    assert abs(log_loss(test["label"].values, pred) - expected) < 0.01


# ---------------------------------------------------------------------------
# log_experiment — writes to CSV
# ---------------------------------------------------------------------------

def test_log_experiment_creates_file_with_header(tmp_path, monkeypatch):
    """log_experiment creates results.csv with header if it doesn't exist."""
    results_path = tmp_path / "results.csv"
    monkeypatch.setattr(ev, "_RESULTS_CSV", results_path)

    log_experiment("baseline", [], holdout_logloss=0.573, brier=0.192)

    assert results_path.exists()
    lines = results_path.read_text().splitlines()
    assert lines[0].startswith("timestamp,git_sha,name")
    assert len(lines) == 2  # header + one data row


def test_log_experiment_appends_row(tmp_path, monkeypatch):
    """Second call appends without re-writing the header."""
    results_path = tmp_path / "results.csv"
    monkeypatch.setattr(ev, "_RESULTS_CSV", results_path)

    log_experiment("run_1", ["feat_a"], holdout_logloss=0.55, brier=0.18)
    log_experiment("run_2", ["feat_a", "feat_b"], holdout_logloss=0.50, brier=0.17)

    lines = results_path.read_text().splitlines()
    assert len(lines) == 3  # header + 2 data rows
    assert "run_1" in lines[1]
    assert "run_2" in lines[2]


def test_log_experiment_features_pipe_separated(tmp_path, monkeypatch):
    """Feature list is written as pipe-separated values."""
    results_path = tmp_path / "results.csv"
    monkeypatch.setattr(ev, "_RESULTS_CSV", results_path)

    log_experiment("m", ["cycle_pct_through", "station_price_cents"], holdout_logloss=0.5, brier=0.2)

    content = results_path.read_text()
    assert "cycle_pct_through|station_price_cents" in content
