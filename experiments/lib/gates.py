from __future__ import annotations

import numpy as np
import pandas as pd


def seed_variance_gate(
    df_rows: pd.DataFrame,
    cohort_ll_map: dict[str, str],
) -> tuple[dict, list]:
    """Check seed variance per (fold, run) cell for each cohort.

    Returns (seed_var_summary, seed_var_flags). Prints flagged cells.
    Raises ValueError if any cohort median seed_std is NaN or <= 0 — a zero
    denominator would silently broadcast NaN and mask real outliers.
    """
    seed_var_flags: list[dict] = []
    seed_var_summary: dict[str, dict] = {}
    for cohort, col in cohort_ll_map.items():
        agg = df_rows.groupby(["fold", "run"], as_index=False).agg(
            seed_std=(col, lambda s: float(np.nanstd(s, ddof=1)))
        )
        cohort_med = float(np.nanmedian(agg["seed_std"])) if len(agg) else float("nan")
        if not np.isfinite(cohort_med) or cohort_med <= 0:
            raise ValueError(
                f"Seed-variance gate: cohort {cohort!r} median seed_std is "
                f"{cohort_med!r} (n_cells={len(agg)}). Investigate before trusting aggregates."
            )
        agg["seed_std_ratio"] = agg["seed_std"] / cohort_med
        flagged = agg[agg["seed_std_ratio"] > 5.0]
        seed_var_summary[cohort] = {
            "cohort_median_seed_std": cohort_med,
            "n_cells": int(len(agg)),
            "n_flagged_gt_5x": int(len(flagged)),
        }
        for _, r in flagged.iterrows():
            seed_var_flags.append({
                "cohort": cohort,
                "fold": int(r["fold"]),
                "run": r["run"],
                "seed_std": float(r["seed_std"]),
                "ratio_vs_cohort_median": float(r["seed_std_ratio"]),
            })

    if seed_var_flags:
        print("\n!! SEED-VARIANCE FLAGS (seed_std > 5× cohort median) !!", flush=True)
        for f in seed_var_flags:
            print(
                f"   [{f['cohort']:<6}] fold={f['fold']:>2}  run={f['run']:<12}  "
                f"seed_std={f['seed_std']:.4f}  ratio={f['ratio_vs_cohort_median']:.1f}×",
                flush=True,
            )
    else:
        print(
            "\nSeed-variance gate: no flagged cells (all seed_std ≤ 5× cohort median).",
            flush=True,
        )

    return seed_var_summary, seed_var_flags
