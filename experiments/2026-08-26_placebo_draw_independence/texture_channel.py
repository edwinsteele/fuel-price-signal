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

import json
import pathlib

import numpy as np
import pandas as pd
from scipy import stats

from experiments.pipeline.placebo import MAX_RULER_ARITY

# Repo-relative, like every sibling experiment (`2026-08-22_placebo_block_sizing`,
# `2026-08-23_placebo_arity`) — run from the repo root, as the README's commands do.
# Its only input, `noise_floor_k1.json`, is tracked in git.
BATCH = pathlib.Path("experiments/batches/batch1")


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

    def exposure(n_draws: int, arity: int) -> float:
        """Upper bound on the fraction of DRAW PAIRS sharing at least one source column.

        Deck scheme: over m picks from n columns every column is used floor(m/n) or
        ceil(m/n) times, so a column used c times contributes C(c,2) sharing pairs. That
        counts column-sharings rather than pairs, so it over-counts when one pair shares
        several columns — deliberately, since the conservative direction for a safety
        argument is to overstate the exposure. At high arity it is exact anyway by
        pigeonhole (two 35-column draws from 49 columns must share >= 21).
        """
        picks = n_draws * arity
        base, extra = divmod(picks, n_cols)
        counts = [base + 1] * extra + [base] * (n_cols - extra)
        sharing = sum(c * (c - 1) // 2 for c in counts)
        return min(1.0, sharing / (n_draws * (n_draws - 1) // 2))

    def n_eff(n_draws: int, frac_pairs: float, rho: float) -> float:
        rho_bar = frac_pairs * rho
        return n_draws / (1.0 + (n_draws - 1) * rho_bar)

    rho_hi = max(icc_upper, 0.0)
    print(f"  20-draw bank, effective draws at ICC 0 and at the {rho_hi:.3f} upper bound:")
    print(f"  {'arity':>5} {'picks':>6} {'pairs sharing':>14} {'n_eff@0':>9} {'n_eff@hi':>9}")
    rows = {}
    for arity in range(1, 7):
        frac = exposure(20, arity)
        rows[arity] = n_eff(20, frac, rho_hi)
        print(f"  {arity:>5} {20*arity:>6} {frac:>13.1%} {20.0:>9.1f} {rows[arity]:>9.2f}")

    # The cap is a COMPARISON, so what it compares against decides it. Print every defensible
    # scoring of "the ruler we already accept" rather than the one that flatters the answer —
    # the first version of this analysis charged the seed-collision pair rho = 1.0 (a DELTA
    # correlation guessed from a placebo-COLUMN correlation of 0.965) while charging reuse the
    # measured delta-space bound, and the cap turned entirely on that asymmetry.
    print("\n=== what is the cap compared AGAINST? ===")
    baselines = {
        "collision pair charged rho=1.0 (flattering)": n_eff(10, 1 / 45, 1.0),
        "collision pair charged rho=upper (symmetrised)": n_eff(10, 1 / 45, rho_hi),
        "post-fix 10x3 bank (no reuse, no collision)": n_eff(10, exposure(10, 3), rho_hi),
    }
    for label, value in baselines.items():
        clears = [a for a in sorted(rows) if rows[a] >= value]
        print(f"  {label:<48} n_eff {value:5.2f}  -> arities clearing it: "
              f"{max(clears) if clears else 'none'}")
    cap = min(
        max((a for a in sorted(rows) if rows[a] >= v), default=0) for v in baselines.values()
    )
    print(f"\n  Cap that clears EVERY baseline: {cap}")
    print(f"  placebo.MAX_RULER_ARITY = {MAX_RULER_ARITY}")
    if cap != MAX_RULER_ARITY:
        print("  !! MISMATCH — re-derive MAX_RULER_ARITY or record why it differs.")


if __name__ == "__main__":
    main()
