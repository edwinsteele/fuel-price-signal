from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

SEED_DTYPE = np.int16
PROBA_DTYPE = np.float32


class RowPredCollector:
    """Accumulate per-(run, seed) row-level prediction blocks and write to parquet.

    Typical usage::

        collector = RowPredCollector(ident_base)  # ident_base built for first fold
        for fold_idx, ...:
            ident = pd.DataFrame({...})           # experiment-specific, per fold
            collector.ident_base = ident
            for run_name in RUNS:
                for seed in SEEDS:
                    _, proba, _ = fit_score(train_df, val_df, cols, seed)
                    collector.add(run_name, seed, proba)
        collector.to_parquet(out_dir / "rowpreds.parquet")

    The caller builds ``ident_base`` (fold / station_code / price_date / label /
    cohort masks) once per fold and assigns it to ``collector.ident_base``.
    ``add`` copies it and appends ``run``, ``seed`` (int16), and ``proba``
    (float32) columns.  ``to_parquet`` concatenates all blocks and writes with
    zstd compression.
    """

    def __init__(self, ident_base: pd.DataFrame) -> None:
        self.ident_base = ident_base
        self._blocks: list[pd.DataFrame] = []

    def add(self, run: str, seed: int, proba: np.ndarray) -> None:
        """Append one (run, seed) block using the current ``ident_base``."""
        _info = np.iinfo(SEED_DTYPE)
        if not (_info.min <= seed <= _info.max):
            raise ValueError(f"seed {seed} out of range for {SEED_DTYPE} [{_info.min}, {_info.max}]")
        proba = np.asarray(proba)
        if proba.ndim != 1:
            raise ValueError(f"'proba' must be 1D, got shape {proba.shape!r}")
        if proba.shape[0] != len(self.ident_base):
            raise ValueError(
                f"Length mismatch: proba ({proba.shape[0]}) vs ident_base ({len(self.ident_base)})"
            )
        block = self.ident_base.copy()
        block["run"] = run
        block["seed"] = SEED_DTYPE(seed)
        block["proba"] = proba.astype(PROBA_DTYPE)
        self._blocks.append(block)

    def to_parquet(self, path: pathlib.Path, compression: str = "zstd") -> pd.DataFrame:
        """Concatenate all blocks and write to *path*. Returns the combined DataFrame."""
        if not self._blocks:
            raise ValueError("No blocks added — call add() at least once before writing.")
        df = pd.concat(self._blocks, ignore_index=True)
        df.to_parquet(path, index=False, compression=compression)
        return df
