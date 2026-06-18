"""Tests for experiments/lib/realised — the paired realised-backtest harness (#255).

The pure helpers (τ selection, fold planning, saving math) are unit-tested here on
synthetic data. The full DB-backed walk-forward run is exercised by a manual smoke
against the real DB (read-only) — it needs the price series and is not reproduced
in CI.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from experiments.lib.realised import (
    ArmSpec,
    _plan_folds,
    _saving_pct,
    _train_calibrate_select_tau,
    run_paired_realised_backtest,
)


def _synth_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Synthetic daily rows with two features carrying signal into a binary label."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2017-01-01", periods=n, freq="D").strftime("%Y-%m-%d")
    f1 = rng.normal(size=n)
    f2 = rng.normal(size=n)
    p = 1.0 / (1.0 + np.exp(-(0.9 * f1 - 0.5 * f2)))
    label = (rng.uniform(size=n) < p).astype(int)
    return pd.DataFrame({"price_date": dates, "label": label, "f1": f1, "f2": f2})


def test_saving_pct_basic_and_guards():
    assert _saving_pct(200.0, 190.0) == pytest.approx(5.0)
    assert _saving_pct(200.0, 200.0) == pytest.approx(0.0)
    assert math.isnan(_saving_pct(0.0, 190.0))      # no always-buy baseline
    assert math.isnan(_saving_pct(200.0, float("nan")))  # model CPL undefined


def test_plan_folds_indexes_sequentially_and_subsets():
    df = _synth_df(n=400)
    outer = {"train_min_days": 60, "val_days": 30, "step_days": 30, "buffer_days": 5}

    plans = _plan_folds(df, outer, None)
    assert len(plans) >= 2
    assert [p.fold for p in plans] == list(range(1, len(plans) + 1))
    # train_index actually selects rows from the frame
    assert df.loc[plans[0].train_index].shape[0] > 0
    assert plans[0].val_start <= plans[0].val_end

    subset = _plan_folds(df, outer, {2})
    assert [p.fold for p in subset] == [2]


def test_train_calibrate_select_tau_returns_pipeline_and_valid_tau():
    df = _synth_df(n=240)
    inner = {"train_min_days": 50, "val_days": 20, "step_days": 20, "buffer_days": 3}

    pipe, tau = _train_calibrate_select_tau(df, ["f1", "f2"], seed=42, inner_fold_params=inner)

    proba = pipe.predict_proba(df[["f1", "f2"]].head(5))
    assert proba.shape == (5, 2)
    assert np.all((proba >= 0.0) & (proba <= 1.0))
    assert 0.05 <= tau <= 0.95  # pick_tau clamps to [_TAU_STEP, 1 - _TAU_STEP]


def test_paired_backtest_rejects_duplicate_arm_names():
    """Duplicate ArmSpec names are rejected before any DB access (they key histories)."""
    df = _synth_df(n=60)
    with pytest.raises(ValueError, match="names must be unique"):
        run_paired_realised_backtest(
            [ArmSpec("dup", df), ArmSpec("dup", df.copy())], ["f1", "f2"]
        )


def test_paired_backtest_rejects_reserved_always_buy_arm_when_collecting_fills():
    """'always_buy' is reserved for the baseline ledger when collect_fills=True."""
    df = _synth_df(n=60)
    with pytest.raises(ValueError, match="reserved arm name"):
        run_paired_realised_backtest(
            [ArmSpec("always_buy", df)], ["f1", "f2"], collect_fills=True
        )


def test_paired_backtest_rejects_mismatched_index():
    """Arms must share an index so a fold's train rows select identically."""
    df = _synth_df(n=60)
    other = df.copy()
    other.index = other.index + 1000
    with pytest.raises(ValueError, match="index differs"):
        run_paired_realised_backtest(
            [ArmSpec("baseline", df), ArmSpec("candidate", other)], ["f1", "f2"]
        )


def test_train_calibrate_select_tau_raises_without_oof_folds():
    df = _synth_df(n=60)
    # Inner window longer than the data → pool_oof_predictions yields no folds.
    inner = {"train_min_days": 5000, "val_days": 30, "step_days": 30}
    with pytest.raises(ValueError, match="no OOF folds"):
        _train_calibrate_select_tau(df, ["f1", "f2"], seed=42, inner_fold_params=inner)
