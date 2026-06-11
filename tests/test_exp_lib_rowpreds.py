"""Tests for experiments/lib/rowpreds — RowPredCollector."""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

from experiments.lib.rowpreds import PROBA_DTYPE, SEED_DTYPE, RowPredCollector


def _ident(n: int = 4, fold: int = 0) -> pd.DataFrame:
    return pd.DataFrame({
        "fold": np.int8(fold),
        "station_code": [f"S{i}" for i in range(n)],
        "price_date": pd.date_range("2024-01-01", periods=n, freq="D"),
        "label": np.zeros(n, dtype=np.int8),
        "is_hard25": np.ones(n, dtype=np.int8),
    })


def test_add_appends_run_seed_proba_columns():
    collector = RowPredCollector(_ident())
    proba = np.array([0.1, 0.2, 0.3, 0.4])
    collector.add("R0", 42, proba)

    assert len(collector._blocks) == 1
    block = collector._blocks[0]
    assert set(["run", "seed", "proba"]).issubset(block.columns)


def test_seed_dtype_is_int16():
    collector = RowPredCollector(_ident())
    collector.add("R0", 42, np.zeros(4))
    assert collector._blocks[0]["seed"].dtype == SEED_DTYPE


def test_proba_dtype_is_float32():
    collector = RowPredCollector(_ident())
    collector.add("R0", 42, np.zeros(4, dtype=np.float64))
    assert collector._blocks[0]["proba"].dtype == PROBA_DTYPE


def test_seed_dtype_does_not_overflow_at_int8_boundary():
    # int8 max is 127; 200 would overflow silently to -56
    collector = RowPredCollector(_ident())
    collector.add("R0", 200, np.zeros(4))
    assert int(collector._blocks[0]["seed"].iloc[0]) == 200


def test_accumulates_across_runs_and_seeds():
    collector = RowPredCollector(_ident(n=3))
    for run in ["R0", "R1"]:
        for seed in [42, 43]:
            collector.add(run, seed, np.zeros(3))
    assert len(collector._blocks) == 4


def test_ident_base_update_between_folds():
    collector = RowPredCollector(_ident(n=4, fold=0))
    collector.add("R0", 42, np.zeros(4))

    collector.ident_base = _ident(n=5, fold=1)
    collector.add("R0", 42, np.zeros(5))

    df = pd.concat(collector._blocks, ignore_index=True)
    assert len(df) == 9
    assert set(df["fold"].unique()) == {0, 1}


def test_to_parquet_shape_and_dtypes(tmp_path: pathlib.Path):
    n = 6
    collector = RowPredCollector(_ident(n=n))
    for run in ["R0", "R1"]:
        collector.add(run, 42, np.random.default_rng(0).random(n).astype(np.float32))

    out = tmp_path / "rowpreds.parquet"
    collector.to_parquet(out)

    assert out.exists()
    roundtrip = pd.read_parquet(out)
    assert len(roundtrip) == 2 * n
    assert roundtrip["seed"].dtype == SEED_DTYPE
    assert roundtrip["proba"].dtype == PROBA_DTYPE


def test_to_parquet_column_set(tmp_path: pathlib.Path):
    collector = RowPredCollector(_ident())
    collector.add("R0", 42, np.zeros(4))

    df = collector.to_parquet(tmp_path / "out.parquet")
    expected_cols = {"fold", "station_code", "price_date", "label", "is_hard25",
                     "run", "seed", "proba"}
    assert expected_cols.issubset(set(df.columns))
