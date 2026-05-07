"""Evaluation harness for the ML price-movement model.

Defines the canonical train/val/test date ranges and provides splitting, scoring,
and experiment logging functions. Every experiment must be scored against the same
fixed holdout window so results are directly comparable across runs.

## Canonical split

| Split    | Start      | End        |
|----------|------------|------------|
| Train    | 2016-08-01 | 2025-03-17 |
| (buffer) | 2025-03-18 | 2025-03-24 | ← dropped; prevents label leakage
| Val      | 2025-03-25 | 2025-06-23 |
| (buffer) | 2025-06-24 | 2025-06-30 | ← dropped; prevents label leakage
| Test     | 2025-07-01 | 2025-12-31 |

## Rationale

Test (last 6 months of 2025): clean normal-cycle period that avoids the Jan–Feb 2026
irregular compressed cycles and the Mar–Apr 2026 Middle East supply shock. Class balance
in test (≈26% BUY / 74% WAIT) matches the overall dataset — no split-induced distribution
shift to account for. CSV data ends 2026-04-26; test ends well clear of that boundary so
the 7-day forward label horizon is not a concern.

7-day buffers prevent label leakage: a train row with price_date = 2025-03-17 has its BUY
label computed from prices through 2025-03-24. If val started on 2025-03-18, that train
label would incorporate val-period data. The buffer gap must be ≥ horizon_days (7 days).

Val (3 months): sufficient for LightGBM hyperparameter search in phase 2.

## Cardinal rule

Choose the window once, write it as constants, never adjust based on results.
Nudging dates after seeing scores is the leaderboard-overfitting cardinal sin.
"""

from __future__ import annotations

import csv
import datetime
import pathlib
import subprocess

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Canonical split boundaries
# ---------------------------------------------------------------------------

TRAIN_START = "2016-08-01"
TRAIN_END = "2025-03-17"
VAL_START = "2025-03-25"   # TRAIN_END + 8 days (≥ 7-day label horizon buffer)
VAL_END = "2025-06-23"
TEST_START = "2025-07-01"  # VAL_END + 8 days
TEST_END = "2025-12-31"

_RESULTS_CSV = pathlib.Path(__file__).parent.parent / "experiments" / "results.csv"
_CSV_HEADER = [
    "timestamp", "git_sha", "name", "features",
    "train_end", "val_start", "val_end", "test_start", "test_end",
    "holdout_logloss", "holdout_brier", "notes",
]


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, val, test) split by price_date; buffer rows are dropped.

    Rows whose price_date falls in either 7-day buffer window are excluded from
    all three subsets: (TRAIN_END, VAL_START) exclusive and (VAL_END, TEST_START)
    exclusive. Rows after TEST_END are also excluded.
    """
    dates = pd.to_datetime(df["price_date"])
    train = df[dates <= TRAIN_END].copy()
    val = df[(dates >= VAL_START) & (dates <= VAL_END)].copy()
    test = df[(dates >= TEST_START) & (dates <= TEST_END)].copy()
    return train, val, test


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Binary cross-entropy log loss (lower is better).

    Clips predictions to [1e-15, 1-1e-15] to avoid log(0).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 1e-15, 1 - 1e-15)
    return float(-np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))


def brier(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Brier score — mean squared error of probability predictions (lower is better)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_pred - y_true) ** 2))


def baseline_prior(df_train: pd.DataFrame) -> float:
    """Return the marginal positive rate from the training set.

    A constant predictor at this value is the 'do nothing' baseline: among all
    constant predictors, it minimises log loss. All models must beat this floor.
    """
    return float(df_train["label"].mean())


# ---------------------------------------------------------------------------
# Experiment logging
# ---------------------------------------------------------------------------

def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def log_experiment(
    name: str,
    features: list[str],
    holdout_logloss: float,
    brier: float,
    notes: str = "",
) -> None:
    """Append one row to experiments/results.csv with a UTC timestamp and git sha.

    Creates the file with a header row if it does not exist yet. The `brier`
    parameter is the holdout Brier score (not the module-level brier() function).
    """
    _RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not _RESULTS_CSV.exists() or _RESULTS_CSV.stat().st_size == 0
    with _RESULTS_CSV.open("a", newline="") as fh:
        writer = csv.writer(fh)
        if write_header:
            writer.writerow(_CSV_HEADER)
        writer.writerow([
            datetime.datetime.now(datetime.timezone.utc).isoformat(),
            _git_sha(),
            name,
            "|".join(features),
            TRAIN_END,
            VAL_START,
            VAL_END,
            TEST_START,
            TEST_END,
            f"{holdout_logloss:.6f}",
            f"{brier:.6f}",
            notes,
        ])
