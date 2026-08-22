"""Full screen of batch0's 54 locked columns under the shipped placebo construction (fps-d7m).

    PYTHONPATH=. uv run python experiments/2026-08-22_placebo_block_sizing/batch0_screen.py

This is fps-d7m's acceptance-criterion-2 evidence, made re-runnable. Unlike
`block_count_sweep.py` it needs the frozen batch (features.parquet + baseline_columns.json)
and takes a few minutes — it builds a placebo for every column at every seed.

It reports BOTH reasons a candidate can be skipped, because conflating them is exactly the
error the PR review caught in this work's own prose:
  * correlation — the placebo stayed too close to its source. Under block permutation this
    fires on ZERO batch0 columns; under the previous circular shift it fired on eight.
  * all-NaN — the source column has no values to correlate (5 of batch0's 35
    `days_since_trough_entry_<lga>` columns). Unverifiable, so skipped and substituted. This
    DOES fire, on every run, and it is why the production bank contains two fallback draws.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

from experiments.pipeline.noise_floor import MAX_SELF_CORRELATION
from experiments.pipeline.placebo import (
    PLACEBO_BLOCK_SEED_POOL,
    axis_is_market_wide,
    make_placebo_series,
    screen_draws,
    select_draws,
)
from fuel_signal.features import load_features

BATCH_DIR = pathlib.Path("experiments/batches/batch0")
OUT_DIR = pathlib.Path(__file__).parent
N_SEEDS = 5
N_DRAWS = 20  # noise_floor.DEFAULT_N_DRAWS — the production default


def main() -> None:
    frame = load_features(BATCH_DIR / "features.csv")  # .parquet sibling, batch_freeze convention
    columns = json.loads((BATCH_DIR / "baseline_columns.json").read_text())
    print(f"batch0: {len(frame):,} rows, {len(columns)} locked columns, "
          f"{frame['price_date'].nunique()} dates, {frame['station_code'].nunique()} stations")
    print(f"gate: abs(self_correlation) <= {MAX_SELF_CORRELATION}, {N_SEEDS} seeds per column\n")

    rows = []
    for column in columns:
        source = frame[column]
        corrs = [make_placebo_series(frame, column, seed).corr(source)
                 for seed in PLACEBO_BLOCK_SEED_POOL[:N_SEEDS]]
        finite = [abs(c) for c in corrs if pd.notna(c)]
        rows.append({
            "column": column,
            "market_wide": axis_is_market_wide(frame, column),
            "all_nan": bool(source.isna().all()),
            "median_self_corr": np.median(finite) if finite else np.nan,
            "max_self_corr": max(finite) if finite else np.nan,
        })
    screen = pd.DataFrame(rows)
    usable = screen[~screen.all_nan]
    failures = usable[usable.max_self_corr > MAX_SELF_CORRELATION]

    print(f"all-NaN columns (unverifiable, skipped): {int(screen.all_nan.sum())}")
    print(f"usable columns: {len(usable)}   checks: {len(usable) * N_SEEDS}")
    print(f"CORRELATION failures: {len(failures)}")
    print(f"median of per-column max: {usable.max_self_corr.median():.3f}   "
          f"worst: {usable.max_self_corr.max():.3f}\n")
    if len(failures):
        print(failures.to_string(index=False))

    print("Weakest draws (the NAMED LIMITATION — mostly-cross-sectional columns):")
    print(usable.nlargest(4, "max_self_corr")[["column", "market_wide", "median_self_corr",
                                               "max_self_corr"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print(f"\n=== end-to-end at the production default (n_draws={N_DRAWS}) ===")
    primaries = [c for c, _ in select_draws(columns, N_DRAWS)]
    drawn = screen_draws(frame, columns, N_DRAWS, max_self_correlation=MAX_SELF_CORRELATION)
    drawn_columns = [c for c, _, _ in drawn]
    dropped = [c for c in primaries if c not in drawn_columns]
    print(f"primaries substituted ({len(dropped)}): {dropped}")
    print(f"  reason all-NaN: {[c for c in dropped if bool(frame[c].isna().all())]}")
    print(f"  reason correlation: {[c for c in dropped if not bool(frame[c].isna().all())]}")
    print(f"substitutes drawn: {[c for c in drawn_columns if c not in primaries]}")
    print(f"max abs(self_correlation) among the {len(drawn)} drawn: "
          f"{max(abs(c) for _, _, c in drawn):.3f}")

    screen.to_csv(OUT_DIR / "batch0_screen.csv", index=False)
    print(f"\nwrote {OUT_DIR}/batch0_screen.csv")


if __name__ == "__main__":
    main()
