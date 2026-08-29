"""Tests for experiments/pipeline/shock_folds.py — the batch-level shock-fold cache
(fps-3tu follow-up).

Mirrors test_noise_floor.py's fixture style: a minimal, non-DB-backed batch dir, since
computing the shock-fold set only ever fits the baseline (no realised backtest, no
placebo arms).
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from experiments.pipeline.batch_freeze import resolve_baseline_columns
from experiments.pipeline.shock_folds import (
    SHOCK_FOLDS_FILENAME,
    compute_shock_folds_for_batch,
    load_cached_shock_folds,
)
from fuel_signal.features import FEATURE_COLUMNS, LGA_FEATURE_COLUMNS, NETWORK_FEATURE_COLUMNS, baseline_fingerprint

# Small toy fold geometry (mirrors test_runner.py's SMALL_OUTER_FOLDS) so a handful of
# synthetic days produce several non-empty WFCV folds.
SMALL_OUTER = {"train_min_days": 30, "val_days": 15, "step_days": 15}


def _baseline_df(n_days: int = 120, n_stations: int = 2, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Well before evaluate.TEST_START (2025-07-01) — walk_forward_folds only ever iterates
    # the pre-test portion of the frame, so a fixture dated after it yields zero folds.
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D").strftime("%Y-%m-%d")
    rows = []
    for station in range(n_stations):
        for i, d in enumerate(dates):
            row = {"price_date": d, "station_code": station, "label": int(rng.uniform() < 0.5)}
            for c in FEATURE_COLUMNS + LGA_FEATURE_COLUMNS + NETWORK_FEATURE_COLUMNS:
                row[c] = float(rng.normal())
            rows.append(row)
    return pd.DataFrame(rows)


def _write_batch_dir(tmp_path: pathlib.Path, df: pd.DataFrame, name: str = "batch1") -> pathlib.Path:
    batch_dir = tmp_path / name
    batch_dir.mkdir()
    df.to_parquet(batch_dir / "features.parquet")
    (batch_dir / "baseline_columns.json").write_text(json.dumps(resolve_baseline_columns(df)))
    return batch_dir


def test_compute_shock_folds_for_batch_writes_expected_payload(tmp_path):
    df = _baseline_df()
    batch_dir = _write_batch_dir(tmp_path, df)

    payload = compute_shock_folds_for_batch(batch_dir, **SMALL_OUTER, verbose=False)

    out_path = batch_dir / SHOCK_FOLDS_FILENAME
    assert out_path.exists()
    on_disk = json.loads(out_path.read_text())
    assert on_disk == payload
    assert isinstance(payload["shock_folds"], list)
    assert payload["n_folds"] > 0
    assert len(payload["shock_folds"]) > 0
    assert set(payload["fold_baseline_ll"].keys()) == {str(i) for i in range(1, payload["n_folds"] + 1)}
    expected_fp = baseline_fingerprint(resolve_baseline_columns(df))
    assert payload["baseline_fingerprint"] == expected_fp


def test_compute_shock_folds_for_batch_refuses_to_overwrite_without_force(tmp_path):
    df = _baseline_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    compute_shock_folds_for_batch(batch_dir, **SMALL_OUTER, verbose=False)

    with pytest.raises(FileExistsError, match="already exists"):
        compute_shock_folds_for_batch(batch_dir, **SMALL_OUTER, verbose=False)


def test_compute_shock_folds_for_batch_force_overwrites(tmp_path):
    df = _baseline_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    compute_shock_folds_for_batch(batch_dir, **SMALL_OUTER, verbose=False)

    second = compute_shock_folds_for_batch(batch_dir, force=True, **SMALL_OUTER, verbose=False)
    on_disk = json.loads((batch_dir / SHOCK_FOLDS_FILENAME).read_text())
    assert on_disk == second


def test_load_cached_shock_folds_returns_none_for_missing_batch_dir():
    assert load_cached_shock_folds(None, expected_baseline_fingerprint="54:abc") is None


def test_load_cached_shock_folds_returns_none_when_file_absent(tmp_path):
    assert load_cached_shock_folds(tmp_path, expected_baseline_fingerprint="54:abc") is None


def test_load_cached_shock_folds_returns_none_on_fingerprint_mismatch(tmp_path):
    df = _baseline_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    compute_shock_folds_for_batch(batch_dir, **SMALL_OUTER, verbose=False)

    assert load_cached_shock_folds(batch_dir, expected_baseline_fingerprint="54:not-a-real-fingerprint") is None


def test_load_cached_shock_folds_returns_the_cached_set_on_a_match(tmp_path):
    df = _baseline_df()
    batch_dir = _write_batch_dir(tmp_path, df)
    payload = compute_shock_folds_for_batch(batch_dir, **SMALL_OUTER, verbose=False)

    fp = baseline_fingerprint(resolve_baseline_columns(df))
    loaded = load_cached_shock_folds(batch_dir, expected_baseline_fingerprint=fp)
    assert loaded == frozenset(payload["shock_folds"])
