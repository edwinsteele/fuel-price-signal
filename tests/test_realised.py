"""Tests for experiments/lib/realised — the paired realised-backtest harness (#255).

The pure helpers (τ selection, fold planning, saving math) are unit-tested here on
synthetic data. The full DB-backed walk-forward run is exercised by a manual smoke
against the real DB (read-only) — it needs the price series and is not reproduced
in CI.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import experiments.lib.realised as realised_module
from experiments.lib.realised import (
    ArmSpec,
    BaselineCache,
    BaselineCacheMismatch,
    _arm_cols,
    _baseline_cache_fingerprint,
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


def test_arm_cols_resolves_explicit_none_not_falsy():
    """None → shared list; an explicit list is used as-is; an empty list raises."""
    df = _synth_df(n=10)
    shared = ["f1", "f2"]
    assert _arm_cols(ArmSpec("a", df), shared) == shared            # None → shared
    assert _arm_cols(ArmSpec("a", df, feature_columns=["f1"]), shared) == ["f1"]
    with pytest.raises(ValueError, match="empty feature_columns"):  # [] is loud, not a fallback
        _arm_cols(ArmSpec("a", df, feature_columns=[]), shared)


def test_train_calibrate_select_tau_raises_without_oof_folds():
    df = _synth_df(n=60)
    # Inner window longer than the data → pool_oof_predictions yields no folds.
    inner = {"train_min_days": 5000, "val_days": 30, "step_days": 30}
    with pytest.raises(ValueError, match="no OOF folds"):
        _train_calibrate_select_tau(df, ["f1", "f2"], seed=42, inner_fold_params=inner)


# ── baseline caching (fps-e2l) ───────────────────────────────────────────────
#
# The full DB-backed run is a manual smoke (this module's docstring). The
# caching logic added here is intricate enough (which arm gets fit, which
# gets skipped, what gets reused vs recomputed per fold) that it needs direct
# coverage, so these tests mock the DB (_db.open_db/load_history) and the
# expensive-fit/replay calls (_train_calibrate_select_tau, aggregate_backtest)
# to run a full run_paired_realised_backtest() end to end without either a
# real database or real LightGBM fits.

class _FakeConn:
    def close(self):
        pass


class _FakePipe:
    """Just enough of a fitted-pipeline interface for ModelStrategy's
    __post_init__ hasattr(pipeline, "predict_proba") check."""

    def __init__(self, tag):
        self.tag = tag

    def predict_proba(self, X):  # pragma: no cover - never actually called
        raise NotImplementedError


def _fill_record(date, station_code, price, litres, spend_cents, emergency=False):
    return SimpleNamespace(
        date=date, station_code=station_code, price=price, litres=litres,
        spend_cents=spend_cents, emergency=emergency,
    )


def _install_fakes(monkeypatch, *, fit_calls: list, load_history_calls: list, agg_calls: list):
    """Patch out the DB + the expensive fit/replay calls with deterministic fakes.

    aggregate_backtest's economics are keyed on (arm-shaped, val_start) only —
    deterministic, so two runs over the same fold plan reproduce byte-identical
    numbers regardless of which run actually did the fitting.
    """
    monkeypatch.setattr(realised_module._db, "open_db", lambda *a, **k: _FakeConn())
    monkeypatch.setattr(realised_module, "load_history", lambda *a, **k: (
        load_history_calls.append(1), object()
    )[1])

    def fake_train_calibrate(train_df, feature_columns, seed, inner_fold_params):
        fit_calls.append(tuple(feature_columns))
        tau = 0.30 if "cand" not in feature_columns else 0.35
        return _FakePipe(tuple(feature_columns)), tau

    monkeypatch.setattr(realised_module, "_train_calibrate_select_tau", fake_train_calibrate)

    def fake_aggregate_backtest(history, strategy, station_codes, val_start, val_end, tank, collect_fills=False):
        is_model = hasattr(strategy, "threshold")
        agg_calls.append((getattr(strategy, "name", "always_buy"), val_start))
        cpl = (150.0 + strategy.threshold) if is_model else 200.0
        litres = 10.0
        spend = cpl * litres
        out = {"total_spend_cents": spend, "total_litres": litres, "fill_events": 1, "cpl": cpl}
        if collect_fills:
            out["fills"] = [_fill_record(val_start, station_codes[0], cpl, litres, spend)]
        return out

    monkeypatch.setattr(realised_module, "aggregate_backtest", fake_aggregate_backtest)


def _cache_test_setup(seed: int = 7):
    df = _synth_df(n=280, seed=seed)
    df["cand"] = np.random.default_rng(seed + 1).normal(size=len(df))
    baseline_cols = ["f1", "f2"]
    candidate_cols = ["f1", "f2", "cand"]
    outer = {"train_min_days": 60, "val_days": 30, "step_days": 30, "buffer_days": 5}
    inner = {"train_min_days": 15, "val_days": 10, "step_days": 10, "buffer_days": 2}
    arms = [
        ArmSpec("R0", df, feature_columns=baseline_cols),
        ArmSpec("candidate", df, feature_columns=candidate_cols),
    ]
    return arms, baseline_cols, outer, inner


def test_baseline_cache_captured_when_held_tau_is_none(monkeypatch):
    fit_calls, load_history_calls, agg_calls = [], [], []
    _install_fakes(monkeypatch, fit_calls=fit_calls, load_history_calls=load_history_calls, agg_calls=agg_calls)
    arms, baseline_cols, outer, inner = _cache_test_setup()

    result = run_paired_realised_backtest(
        arms, baseline_cols, station_codes=[1], seed=42,
        outer_fold_params=outer, inner_fold_params=inner, collect_fills=True, verbose=False,
    )

    n_folds = result.per_window["fold"].nunique()
    assert n_folds >= 2, "fixture must produce at least 2 folds for this test to mean anything"
    assert isinstance(result.baseline_cache, BaselineCache)
    assert set(result.baseline_cache.per_fold) == set(result.per_window["fold"].unique())
    assert result.meta["baseline_cache_used"] is False
    assert result.meta["baseline_cache_hit_folds"] == []
    # Both arms loaded (no cache yet) and fit every fold.
    assert len(load_history_calls) == 2
    assert len(fit_calls) == n_folds * 2


def test_baseline_cache_reuse_skips_baseline_refit_and_history_load(monkeypatch):
    """Second call, passing the first call's cache back in, must reuse arms[0]'s
    fit and economics — not refit it or reload its PriceHistory — while still
    fitting the candidate fresh, and produce numerically identical results."""
    fit_calls, load_history_calls, agg_calls = [], [], []
    _install_fakes(monkeypatch, fit_calls=fit_calls, load_history_calls=load_history_calls, agg_calls=agg_calls)
    arms1, baseline_cols, outer, inner = _cache_test_setup()

    result1 = run_paired_realised_backtest(
        arms1, baseline_cols, station_codes=[1], seed=42,
        outer_fold_params=outer, inner_fold_params=inner, collect_fills=True, verbose=False,
    )
    n_folds = result1.per_window["fold"].nunique()

    fit_calls.clear()
    load_history_calls.clear()
    agg_calls.clear()
    arms2, _, _, _ = _cache_test_setup()  # fresh ArmSpecs, same underlying data/config
    result2 = run_paired_realised_backtest(
        arms2, baseline_cols, station_codes=[1], seed=42,
        outer_fold_params=outer, inner_fold_params=inner, collect_fills=True, verbose=False,
        baseline_cache=result1.baseline_cache,
    )

    assert len(load_history_calls) == 1, "baseline history must not be reloaded when fully cached"
    assert len(fit_calls) == n_folds, "only the candidate should be refit, not the baseline"
    assert result2.meta["baseline_cache_used"] is True
    assert result2.meta["baseline_cache_hit_folds"] == sorted(result1.per_window["fold"].unique().tolist())

    pd.testing.assert_frame_equal(
        result1.per_window.reset_index(drop=True), result2.per_window.reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        result1.aggregate.reset_index(drop=True), result2.aggregate.reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        result1.deltas.reset_index(drop=True), result2.deltas.reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        result1.fills.sort_values(["fold", "arm", "date"]).reset_index(drop=True),
        result2.fills.sort_values(["fold", "arm", "date"]).reset_index(drop=True),
    )


def test_baseline_cache_rejects_explicit_held_tau():
    """Caching is only valid when held_tau is None (baseline's own τ IS the
    held τ in that mode) — an explicit override needs the baseline scored at
    a tau the cache never captured."""
    df = _synth_df(n=60)
    cache = BaselineCache(fingerprint={}, per_fold={})
    with pytest.raises(BaselineCacheMismatch, match="held_tau is None"):
        run_paired_realised_backtest(
            [ArmSpec("baseline", df)], ["f1", "f2"], held_tau=0.4, baseline_cache=cache,
        )


def test_baseline_cache_rejects_fingerprint_mismatch(monkeypatch):
    fit_calls, load_history_calls, agg_calls = [], [], []
    _install_fakes(monkeypatch, fit_calls=fit_calls, load_history_calls=load_history_calls, agg_calls=agg_calls)
    arms1, baseline_cols, outer, inner = _cache_test_setup()

    result1 = run_paired_realised_backtest(
        arms1, baseline_cols, station_codes=[1], seed=42,
        outer_fold_params=outer, inner_fold_params=inner, collect_fills=True, verbose=False,
    )

    arms2, _, _, _ = _cache_test_setup()
    with pytest.raises(BaselineCacheMismatch, match="fingerprint mismatch"):
        run_paired_realised_backtest(
            arms2, baseline_cols, station_codes=[1], seed=99,  # different seed than the cache
            outer_fold_params=outer, inner_fold_params=inner, collect_fills=True, verbose=False,
            baseline_cache=result1.baseline_cache,
        )


def test_baseline_cache_rejects_fold_window_drift_despite_matching_fingerprint(monkeypatch):
    """Belt-and-braces: even with a matching fingerprint, a per-fold val window
    that doesn't match the cache's is refused rather than silently reused."""
    fit_calls, load_history_calls, agg_calls = [], [], []
    _install_fakes(monkeypatch, fit_calls=fit_calls, load_history_calls=load_history_calls, agg_calls=agg_calls)
    arms, baseline_cols, outer, inner = _cache_test_setup()

    result = run_paired_realised_backtest(
        arms, baseline_cols, station_codes=[1], seed=42,
        outer_fold_params=outer, inner_fold_params=inner, collect_fills=True, verbose=False,
    )
    fold = next(iter(result.baseline_cache.per_fold))
    tampered = dict(result.baseline_cache.per_fold)
    tampered[fold] = {**tampered[fold], "val_start": "1900-01-01"}
    bad_cache = BaselineCache(fingerprint=result.baseline_cache.fingerprint, per_fold=tampered)

    arms2, _, _, _ = _cache_test_setup()
    with pytest.raises(BaselineCacheMismatch, match="despite a matching fingerprint"):
        run_paired_realised_backtest(
            arms2, baseline_cols, station_codes=[1], seed=42,
            outer_fold_params=outer, inner_fold_params=inner, collect_fills=True, verbose=False,
            baseline_cache=bad_cache,
        )


def test_baseline_cache_fingerprint_covers_the_documented_keys():
    df = _synth_df(n=60)
    arm = ArmSpec("baseline", df)
    fp = _baseline_cache_fingerprint(
        arm, ["f1", "f2"], [1, 2], 42, {"train_min_days": 10}, {"val_days": 5}, {3, 1}, True,
    )
    assert fp == {
        "baseline_arm": "baseline",
        "baseline_feature_columns": ["f1", "f2"],
        "station_codes": [1, 2],
        "seed": 42,
        "outer_fold_params": {"train_min_days": 10},
        "inner_fold_params": {"val_days": 5},
        "fold_subset": [1, 3],
        "collect_fills": True,
    }
