from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Sign convention: Δ = run − R0; negative is better (lower log-loss).
# All threshold comparisons are "value <= threshold", so a negative target_max
# demands improvement and a small positive worst_fold_max caps regression.


@dataclass(frozen=True)
class GateSpec:
    """Thresholds for the three standard paired-WFCV decision gates.

    cohort_col: delta column for gates 1 and 2 (e.g. ``delta_ll_hard25_median``).
    pop_col:    delta column for gate 3        (e.g. ``delta_ll_all_median``).
    target_fold: fold index that must show improvement (gate 1).
    target_max: Δ must be <= this on target_fold (typically negative, e.g. -0.04).
    worst_fold_max: no fold's Δ may exceed this (typically small positive, e.g. +0.01).
    net_pop_max: mean Δ across folds on pop_col must be <= this (typically 0.0).
    """

    cohort_col: str
    pop_col: str
    target_fold: int
    target_max: float
    worst_fold_max: float
    net_pop_max: float


def evaluate_gates(
    fold_run: pd.DataFrame,
    spec: GateSpec,
    run: str,
) -> list[dict]:
    """Evaluate the three standard gates for one run against the GateSpec.

    Returns a list of dicts ``{name, threshold, value, passed}`` — one per gate.
    The caller can write this directly into ``meta.json`` and print a verdict table.

    Sign convention (single-sourced here): Δ = run − R0; negative is better.
    A gate passes when ``value <= threshold``.
    """
    sub = fold_run[fold_run["run"] == run]

    # Gate 1: target-fold cohort Δ
    target_rows = sub.loc[sub["fold"] == spec.target_fold, spec.cohort_col]
    if len(target_rows) == 0:
        target_value = float("nan")
    elif len(target_rows) == 1:
        target_value = float(target_rows.iloc[0])
    else:
        raise ValueError(
            f"Expected at most one row for (run={run!r}, fold={spec.target_fold}) "
            f"but found {len(target_rows)} rows."
        )

    # Gate 2: worst-fold (maximum) cohort Δ across all folds
    worst_value = float(sub[spec.cohort_col].max()) if len(sub) else float("nan")

    # Gate 3: mean net-population Δ across all folds
    net_pop_value = float(sub[spec.pop_col].mean()) if len(sub) else float("nan")

    return [
        {
            "name": f"target_fold_{spec.target_fold}_{spec.cohort_col}",
            "threshold": spec.target_max,
            "value": target_value,
            "passed": target_value <= spec.target_max,
        },
        {
            "name": f"worst_fold_{spec.cohort_col}",
            "threshold": spec.worst_fold_max,
            "value": worst_value,
            "passed": worst_value <= spec.worst_fold_max,
        },
        {
            "name": f"net_pop_{spec.pop_col}",
            "threshold": spec.net_pop_max,
            "value": net_pop_value,
            "passed": net_pop_value <= spec.net_pop_max,
        },
    ]


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
