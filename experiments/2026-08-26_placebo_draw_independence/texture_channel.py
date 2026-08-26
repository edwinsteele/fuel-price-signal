"""Could the DELTAS tell a different story from the placebo COLUMNS? (bd fps-3jj.21)

`measure_independence.py` shows that reusing a source column across draws barely moves the
correlation between the placebo COLUMNS. But a placebo reaches its delta by two channels and
column correlation only sees one:

  alignment — does this shuffle happen to line up with the target? (correlation sees this)
  texture   — how much does adding a column OF THIS SHAPE perturb the fit at all, regardless
              of alignment? Distribution, cardinality, NaN rate, autocorrelation. (blind)

Two placebos from the same source have IDENTICAL texture, so if texture drives delta variance
then reuse correlates the deltas, the sample std shrinks, and the bar gets too EASY — the bias
direction fps-3jj.14 exists to remove.

Testable on committed data, no fits: batch1's arity-1 floor is 20 draws over 20 DISTINCT source
columns spanning a handful of texture families. If texture drove the delta, draws sharing a
family would cluster. One-way ANOVA of delta by family answers it, and the design's own
resolution is reported alongside (this repo does not read a quiet F as "no effect").

Second half: how much reuse would the proposed construction actually do? Exposure bounds the
harm regardless of what the ANOVA says.
"""
from __future__ import annotations

import itertools
import json
import pathlib

import numpy as np
import pandas as pd
from scipy import stats

BATCH = pathlib.Path("/Users/esteele/Code/fuel-price-signal/experiments/batches/batch1")


def texture_family(column: str) -> str:
    if column.startswith("days_since_trough_entry_"):
        return "lga_trough_counter"
    if column.startswith("network_"):
        return "network"
    if column in {"cycle_pct_through", "cycle_days_since_peak"}:
        return "cycle_phase"
    if column.startswith("cycle_"):
        return "cycle_magnitude"
    return "price_level_other"


def main() -> None:
    floor = json.loads((BATCH / "noise_floor_k1.json").read_text())
    draws = pd.DataFrame(floor["placebo_draws"])
    draws["delta"] = floor["deltas_cpl_held"]
    draws["family"] = draws.source_column.map(texture_family)
    print(f"batch1 arity-1 floor: {len(draws)} draws, "
          f"band mean {draws.delta.mean():+.4f}, std {draws.delta.std(ddof=1):.4f} c/L\n")

    print("=== delta by texture family ===")
    print(draws.groupby("family")["delta"].agg(
        n="size", mean="mean", std=lambda s: s.std(ddof=1)).to_string())

    groups = [g["delta"].to_numpy() for _, g in draws.groupby("family") if len(g) > 1]
    n, k_groups = len(draws), len(draws.family.unique())
    f_stat, p = stats.f_oneway(*groups)
    k_bar = n / k_groups

    # ICC(1) from the ANOVA F: the share of delta variance attributable to texture, i.e. the
    # correlation two draws would inherit purely from sharing a source column.
    icc = (f_stat - 1.0) / (f_stat + k_bar - 1.0)
    df1, df2 = len(groups) - 1, n - len(groups)
    f_crit = stats.f.ppf(0.95, df1, df2)
    icc_detectable = (f_crit - 1.0) / (f_crit + k_bar - 1.0)
    # Upper confidence bound on ICC — the pessimistic case the decision must survive.
    f_upper = f_stat / stats.f.ppf(0.05, df1, df2)
    icc_upper = (f_upper - 1.0) / (f_upper + k_bar - 1.0)

    print(f"\n  one-way ANOVA: F({df1},{df2}) = {f_stat:.3f}, p = {p:.3f}")
    print(f"  ICC (texture's share of delta variance): {icc:+.3f}")
    print(f"  95% upper bound on ICC:                  {icc_upper:+.3f}   <- the pessimistic case")
    print(f"  smallest ICC this design could resolve:  {icc_detectable:.3f}")

    lga = draws[draws.family == "lga_trough_counter"]["delta"]
    print(f"\n  tightest texture group available ({len(lga)} near-identical LGA counters):")
    print(f"    std {lga.std(ddof=1):.4f} vs whole-bank {draws.delta.std(ddof=1):.4f} c/L "
          f"(ratio {lga.std(ddof=1)/draws.delta.std(ddof=1):.2f}x)")

    print("\n=== how much reuse does the proposal actually do? ===")
    n_cols = 49  # usable (non-all-NaN) batch1 lock columns
    for n_draws, arity in ((20, 3), (20, 4), (20, 10), (20, 35)):
        picks = n_draws * arity
        # Deck scheme: every column used floor(picks/n_cols) or ceil(...) times.
        base, extra = divmod(picks, n_cols)
        counts = np.array([base + 1] * extra + [base] * (n_cols - extra))
        # Draw PAIRS that share at least one source column, upper-bounded by counting
        # co-occurrences: a column used c times contributes C(c,2) sharing pairs.
        sharing = int(sum(c * (c - 1) // 2 for c in counts))
        total_pairs = n_draws * (n_draws - 1) // 2
        frac = min(1.0, sharing / total_pairs)
        for label, rho in (("measured ICC", max(icc, 0.0)), ("95% upper ICC", max(icc_upper, 0.0))):
            rho_bar = frac * rho
            n_eff = n_draws / (1.0 + (n_draws - 1) * rho_bar)
            if label == "measured ICC":
                print(f"  {n_draws} draws x arity {arity}: {picks} picks from {n_cols} columns, "
                      f"<={sharing}/{total_pairs} draw pairs share a column ({frac:.0%})")
            print(f"      {label:>14} {rho:.3f} -> mean pairwise rho {rho_bar:.4f}, "
                  f"effective draws {n_eff:.1f}")

    print("\n=== today's construction, for comparison ===")
    print("  10 draws x arity 3, all columns distinct, but a SEED COLLISION puts one draw pair "
          "at 0.965\n  placebo-column correlation (fps-3jj.20). Treating that one pair as fully "
          "shared:")
    rho_bar = (1 / 45) * 1.0
    print(f"      1/45 pairs at rho 1.0 -> mean pairwise rho {rho_bar:.4f}, "
          f"effective draws {10 / (1 + 9 * rho_bar):.1f}")


if __name__ == "__main__":
    main()
