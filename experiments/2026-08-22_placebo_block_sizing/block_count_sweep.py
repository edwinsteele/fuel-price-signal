"""Regenerates the two measured constants in experiments/pipeline/placebo.py (fps-d7m).

    PYTHONPATH=. uv run python experiments/2026-08-22_placebo_block_sizing/block_count_sweep.py

Both tables are SYNTHETIC and deterministic — no batch data, no DB, ~30s. That is the point:
`MIN_BLOCKS` and `PLACEBO_BLOCK_DAYS` are the two numbers in placebo.py that came out of a
measurement rather than a principle, and until this file existed the numbers lived only in a
docstring with no way to re-derive them. Anyone moving either constant should re-run this
FIRST and paste the new tables into placebo.py alongside the change.

The probe series is deliberately adversarial: a monotone drift plus a 31-day cycle, i.e.
drift-dominated. That is the hardest shape for any time-axis reordering to scramble, and it
is the shape that broke the previous circular-shift construction (see the bead and this
dir's README). A gentler series would flatter the numbers.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

from experiments.pipeline.placebo import (
    MIN_BLOCKS,
    PLACEBO_BLOCK_DAYS,
    _block_derangement,
)

OUT_DIR = pathlib.Path(__file__).parent
N_SEEDS = 500
# placebo.py screens on abs(self_correlation) <= noise_floor.MAX_SELF_CORRELATION.
# Imported rather than hardcoded so this file cannot drift from the gate it is measuring.
from experiments.pipeline.noise_floor import MAX_SELF_CORRELATION  # noqa: E402


def adversarial_series(n: int) -> np.ndarray:
    """Drift-dominated level-like series: the hardest case for a time-axis reorder."""
    return np.linspace(120.0, 190.0, n) + 8.0 * np.sin(np.arange(n) * 2 * np.pi / 31.0)


def permute(values: np.ndarray, block: int, seed: int) -> np.ndarray:
    blocks = [values[i:i + block] for i in range(0, len(values), block)]
    order = _block_derangement(len(blocks), np.random.default_rng(seed))
    return np.concatenate([blocks[j] for j in order])


def self_correlations(values: np.ndarray, block: int, n_seeds: int) -> np.ndarray:
    src = pd.Series(values)
    return np.array([abs(pd.Series(permute(values, block, s)).corr(src)) for s in range(n_seeds)])


def sweep_block_count() -> pd.DataFrame:
    """Table 1 — sets MIN_BLOCKS. How well does the permutation decorrelate, by BLOCK COUNT?

    Block count K is what the permutation has to work with; the spread of |self-correlation|
    grows as K shrinks. MIN_BLOCKS is the smallest K whose WORST case over `N_SEEDS` still
    clears MAX_SELF_CORRELATION — below that, a level-like column starts failing the screen
    and being substituted away, which is the systematic exclusion fps-d7m exists to remove.
    """
    rows = []
    for k in (8, 12, 16, 20, 24, 30, 40, 59):
        cs = self_correlations(adversarial_series(k * PLACEBO_BLOCK_DAYS), PLACEBO_BLOCK_DAYS, N_SEEDS)
        rows.append({
            "n_blocks": k, "median": np.median(cs), "p95": np.quantile(cs, 0.95),
            "p99": np.quantile(cs, 0.99), "max": cs.max(),
            "clears_gate": bool(cs.max() <= MAX_SELF_CORRELATION),
        })
    return pd.DataFrame(rows)


def sweep_block_length() -> pd.DataFrame:
    """Table 2 — sets PLACEBO_BLOCK_DAYS. What does block LENGTH cost in fidelity?

    A placebo has to resemble a real feature, so the permutation should preserve the column's
    local texture. A circular shift introduced exactly one seam (the wrap); block permutation
    introduces one per boundary, so shorter blocks mean more seams and lower retained lag-1
    autocorrelation. Held at a batch0-sized 3572 positions so the comparison is like-for-like.
    """
    n = 3572
    v = adversarial_series(n)
    lag1 = lambda a: pd.Series(a).autocorr(1)  # noqa: E731
    rows = [{"block_days": "original", "n_blocks": 0, "lag1_autocorr": lag1(v),
             "median_self_corr": 0.0, "seams": 0}]
    for block in (14, 29, 61, 127, 251):
        cs = self_correlations(v, block, 60)
        ac = np.median([lag1(permute(v, block, s)) for s in range(60)])
        rows.append({"block_days": block, "n_blocks": -(-n // block), "lag1_autocorr": ac,
                     "median_self_corr": np.median(cs), "seams": -(-n // block) - 1})
    return pd.DataFrame(rows)


def main() -> None:
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)  # noqa: E731

    t1 = sweep_block_count()
    print(f"\n=== Table 1: decorrelation vs BLOCK COUNT ({N_SEEDS} seeds) — sets MIN_BLOCKS ===")
    print(f"gate: abs(self_correlation) <= {MAX_SELF_CORRELATION}   shipped MIN_BLOCKS = {MIN_BLOCKS}")
    print(t1.to_string(index=False, float_format=fmt))
    chosen = t1[t1.clears_gate].n_blocks.min()
    print(f"-> smallest block count whose WORST case clears the gate: {chosen}")
    assert chosen == MIN_BLOCKS, (
        f"this sweep now picks MIN_BLOCKS={chosen}, but placebo.py ships {MIN_BLOCKS} — "
        "reconcile before trusting either."
    )

    t2 = sweep_block_length()
    print("\n=== Table 2: fidelity vs BLOCK LENGTH — sets PLACEBO_BLOCK_DAYS ===")
    print(f"shipped PLACEBO_BLOCK_DAYS = {PLACEBO_BLOCK_DAYS}  (batch0 cycle_mean_length: mean 30.5, max 35.3)")
    print(t2.to_string(index=False, float_format=fmt))

    t1.to_csv(OUT_DIR / "block_count_sweep.csv", index=False)
    t2.to_csv(OUT_DIR / "block_length_sweep.csv", index=False)
    print(f"\nwrote {OUT_DIR}/block_count_sweep.csv and block_length_sweep.csv")


if __name__ == "__main__":
    main()
