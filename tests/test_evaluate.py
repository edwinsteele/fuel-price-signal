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
    walk_forward_folds,
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


def test_log_loss_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        log_loss(np.array([1.0, 0.0, 1.0]), np.array([0.5, 0.5]))


def test_log_loss_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        log_loss(np.array([]), np.array([]))


def test_brier_perfect():
    y = np.array([1.0, 0.0, 1.0])
    assert brier(y, np.array([1.0, 0.0, 1.0])) == pytest.approx(0.0)


def test_brier_worst():
    y = np.array([1.0, 0.0])
    assert brier(y, np.array([0.0, 1.0])) == pytest.approx(1.0)


def test_brier_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        brier(np.array([1.0, 0.0, 1.0]), np.array([0.5, 0.5]))


def test_brier_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        brier(np.array([]), np.array([]))


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

    log_experiment("baseline", [], holdout_logloss=0.573, holdout_brier=0.192)

    assert results_path.exists()
    lines = results_path.read_text().splitlines()
    assert lines[0].startswith("timestamp,git_sha,name,features,train_start,train_end")
    assert len(lines) == 2  # header + one data row


def test_log_experiment_appends_row(tmp_path, monkeypatch):
    """Second call appends without re-writing the header."""
    results_path = tmp_path / "results.csv"
    monkeypatch.setattr(ev, "_RESULTS_CSV", results_path)

    log_experiment("run_1", ["feat_a"], holdout_logloss=0.55, holdout_brier=0.18)
    log_experiment("run_2", ["feat_a", "feat_b"], holdout_logloss=0.50, holdout_brier=0.17)

    lines = results_path.read_text().splitlines()
    assert len(lines) == 3  # header + 2 data rows
    assert "run_1" in lines[1]
    assert "run_2" in lines[2]


def test_log_experiment_features_pipe_separated(tmp_path, monkeypatch):
    """Feature list is written as pipe-separated values."""
    results_path = tmp_path / "results.csv"
    monkeypatch.setattr(ev, "_RESULTS_CSV", results_path)

    log_experiment("m", ["cycle_pct_through", "station_price_cents"], holdout_logloss=0.5, holdout_brier=0.2)

    content = results_path.read_text()
    assert "cycle_pct_through|station_price_cents" in content


def test_log_experiment_raises_on_schema_drift(tmp_path, monkeypatch):
    """log_experiment raises ValueError if existing file has a different header."""
    results_path = tmp_path / "results.csv"
    monkeypatch.setattr(ev, "_RESULTS_CSV", results_path)
    # Write a file with an old/mismatched header
    results_path.write_text("timestamp,git_sha,name,holdout_logloss\n")

    import pytest
    with pytest.raises(ValueError, match="header does not match current schema"):
        log_experiment("m", [], holdout_logloss=0.5, holdout_brier=0.2)


# ---------------------------------------------------------------------------
# walk_forward_folds — helpers
# ---------------------------------------------------------------------------

def _make_daily_df(start: str, end: str) -> pd.DataFrame:
    """Daily-resolution DataFrame from start to end inclusive."""
    dates = pd.date_range(start, end, freq="D")
    return pd.DataFrame({
        "price_date": dates.strftime("%Y-%m-%d"),
        "label": 0,
        "station_code": 1,
        "today_price_cents": 160.0,
        "future_min_cents": 162.0,
    })


# ---------------------------------------------------------------------------
# walk_forward_folds — structural properties
# ---------------------------------------------------------------------------

def test_wff_no_overlap():
    """No date appears in both train and val within any fold."""
    df = _make_daily_df("2020-01-01", "2023-12-31")
    for train_df, val_df in walk_forward_folds(
        df, train_min_days=100, val_days=30, step_days=30, buffer_days=7
    ):
        train_dates = set(train_df["price_date"])
        val_dates = set(val_df["price_date"])
        assert train_dates.isdisjoint(val_dates)


def test_wff_buffer_gap_respected():
    """No date from the buffer window appears in either train or val."""
    buffer_days = 7
    df = _make_daily_df("2020-01-01", "2023-12-31")
    for train_df, val_df in walk_forward_folds(
        df, train_min_days=100, val_days=30, step_days=30, buffer_days=buffer_days
    ):
        train_end = pd.to_datetime(train_df["price_date"]).max()
        val_start = pd.to_datetime(val_df["price_date"]).min()
        buffer_dates = set(
            pd.date_range(
                train_end + pd.Timedelta(days=1),
                val_start - pd.Timedelta(days=1),
                freq="D",
            ).strftime("%Y-%m-%d")
        )
        all_fold_dates = set(train_df["price_date"]) | set(val_df["price_date"])
        overlap = buffer_dates & all_fold_dates
        assert not overlap, f"Buffer dates found in fold: {overlap}"


def test_wff_train_always_before_val():
    """In every fold the max train date is strictly before the min val date."""
    df = _make_daily_df("2020-01-01", "2023-12-31")
    for train_df, val_df in walk_forward_folds(
        df, train_min_days=100, val_days=30, step_days=30, buffer_days=7
    ):
        train_max = pd.to_datetime(train_df["price_date"]).max()
        val_min = pd.to_datetime(val_df["price_date"]).min()
        assert train_max < val_min


def test_wff_monotonic_val_windows():
    """Each successive fold's val window starts strictly later than the previous."""
    df = _make_daily_df("2020-01-01", "2023-12-31")
    folds = list(walk_forward_folds(
        df, train_min_days=100, val_days=30, step_days=30, buffer_days=7
    ))
    assert len(folds) >= 2, "Need at least 2 folds to check monotonicity"
    prev_val_start = None
    for train_df, val_df in folds:
        val_start = pd.to_datetime(val_df["price_date"]).min()
        if prev_val_start is not None:
            assert val_start > prev_val_start
        prev_val_start = val_start


# ---------------------------------------------------------------------------
# walk_forward_folds — fold count and boundary conditions
# ---------------------------------------------------------------------------

def test_wff_fold_count():
    """Generator yields exactly the expected number of folds.

    With min_date=2020-01-01, max_date=2020-12-31 (365-day span in a leap year),
    train_min_days=30, val_days=10, step_days=10, buffer_days=3:

      val_end(i) = min_date + (30−1) + i×10 + (3+1) + (10−1) = min_date + 42 + 10i

    Valid while 42 + 10i ≤ 365  →  i ≤ 32.3  →  folds 0…32 = 33 folds.
    """
    df = _make_daily_df("2020-01-01", "2020-12-31")
    folds = list(walk_forward_folds(
        df, train_min_days=30, val_days=10, step_days=10, buffer_days=3
    ))
    assert len(folds) == 33


def test_wff_excludes_test_window():
    """Rows from TEST_START onwards never appear in any fold even if present in df."""
    df = _make_daily_df("2024-01-01", "2025-12-31")
    for train_df, val_df in walk_forward_folds(
        df, train_min_days=30, val_days=10, step_days=10, buffer_days=3
    ):
        for part in [train_df, val_df]:
            late = pd.to_datetime(part["price_date"]) >= pd.Timestamp(TEST_START)
            assert not late.any(), (
                f"Post-test date in fold: {part.loc[late, 'price_date'].tolist()}"
            )


def test_wff_empty_df_yields_nothing():
    """Empty DataFrame produces no folds."""
    df = pd.DataFrame(
        columns=["price_date", "label", "station_code", "today_price_cents", "future_min_cents"]
    )
    assert list(walk_forward_folds(df)) == []


@pytest.mark.parametrize("kwargs,match", [
    ({"step_days": 0},        "step_days"),
    ({"step_days": -1},       "step_days"),
    ({"val_days": 0},         "val_days"),
    ({"train_min_days": 0},   "train_min_days"),
    ({"buffer_days": -1},     "buffer_days"),
])
def test_wff_invalid_params_raise(kwargs, match):
    """Non-positive step/val/train or negative buffer raises ValueError immediately."""
    df = _make_daily_df("2020-01-01", "2023-12-31")
    with pytest.raises(ValueError, match=match):
        list(walk_forward_folds(df, **kwargs))


def test_wff_train_grows_each_fold():
    """Training set grows by step_days rows between consecutive folds."""
    df = _make_daily_df("2020-01-01", "2023-12-31")
    step = 30
    folds = list(walk_forward_folds(
        df, train_min_days=100, val_days=30, step_days=step, buffer_days=7
    ))
    assert len(folds) >= 2
    for (t1, _), (t2, _) in zip(folds, folds[1:]):
        assert len(t2) - len(t1) == step
