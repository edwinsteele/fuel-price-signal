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
import math
import pathlib

import pandas as pd
from scipy import stats

from experiments.pipeline.placebo import TEXTURE_ICC_BOUND

# Repo-relative, like every sibling experiment (`2026-08-22_placebo_block_sizing`,
# `2026-08-23_placebo_arity`) — run from the repo root, as the README's commands do.
# Its only input, `noise_floor_k1.json`, is tracked in git.
BATCH = pathlib.Path("experiments/batches/batch1")


def _shared_fraction_n_eff(n_draws: int, arity: int, icc: float, n_cols: int = 49) -> float:
    """Effective draws charging each pair `icc * shared / arity`, mirroring
    `placebo.effective_n_draws` — reproduced here rather than imported so this script keeps
    measuring the model independently of the code that ships it."""
    picks, lap = list(range(n_cols)), 1
    while len(picks) < n_draws * arity + n_cols:
        picks += [(i + lap) % n_cols for i in range(n_cols)]
        lap += 1
    bank, cur, used = [], [], set()
    for c in picks:
        if c in used:
            continue
        cur.append(c)
        used.add(c)
        if len(cur) == arity:
            bank.append(set(cur))
            cur, used = [], set()
            if len(bank) == n_draws:
                break
    pairs = [icc * len(a & b) / arity for a, b in itertools.combinations(bank, 2)]
    rho_bar = sum(pairs) / len(pairs) if pairs else 0.0
    return n_draws / (1 + (n_draws - 1) * rho_bar)


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

    # EVERY group, including one with a single draw. A singleton contributes nothing to the
    # WITHIN-group sum of squares (correctly — it has no within-group variation) but it does
    # contribute to the BETWEEN-group sum of squares, and dropping it discards that. The old
    # `if len(g) > 1` filter silently deleted such groups while `n`/`k_groups` below stayed
    # computed on the unfiltered `draws`, so a dropped family desynced the F stat from its own
    # df/k_bar and nothing in the output revealed it (bd fps-8o0; see
    # `2026-08-27_texture_icc/measure_icc.py::_anova_icc`, corrected the same way in PR #337).
    groups = [g["delta"].to_numpy() for _, g in draws.groupby("family")]
    n, k_groups = len(draws), len(groups)
    if sum(len(g) for g in groups) <= len(groups):
        raise SystemExit(
            f"every one of the {k_groups} texture families has exactly one draw — there is no "
            "WITHIN-family variation to separate alignment from texture."
        )
    f_stat, p = stats.f_oneway(*groups)
    # Standard ANOVA k_bar for unbalanced designs (reduces to the common group size when
    # balanced) — the naive n/k_groups this used to be is only correct when every family has
    # the same count, which batch1's families do not (bd fps-8o0; same formula as
    # `2026-08-27_texture_icc/measure_icc.py::_anova_icc`).
    sizes = [len(g) for g in groups]
    k_bar = (n - sum(s * s for s in sizes) / n) / (k_groups - 1)

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

    # fps-3jj.25 removed the arity CAP this section used to derive. Reuse is priced into the
    # threshold instead, so what matters is no longer "which arity is still allowed" but "what
    # does a bank at this width cost in effective draws, and therefore in c/L". Both charges are
    # printed because the difference between them is why a cap ever looked necessary: charging
    # any overlap the FULL icc put a 20-draw arity-10 bank at 2.4 effective draws, while
    # charging the shared FRACTION — what `placebo.effective_n_draws` does — puts it at 9.04.
    # Bars are a GRADING quantity, so they are quoted on the batch's live ruler
    # (`noise_floor.json`), not on the k1 floor the ANOVA above needs for its 20 distinct-column
    # draws. Quoting them on different bands would make two different numbers look like one.
    grading = json.loads((BATCH / "noise_floor.json").read_text())
    grade_deltas = pd.Series(grading["deltas_cpl_held"], dtype=float)
    grade_mean, grade_std = float(grade_deltas.mean()), float(grade_deltas.std(ddof=1))
    print("\n=== binary charge vs FRACTION charge (why the cap was an artefact) ===")
    print(f"  bars quoted on the LIVE ruler: {grading.get('n_placebo_columns', 1)}-column band, "
          f"mean {grade_mean:+.4f}, std {grade_std:.4f} c/L over {len(grade_deltas)} draws")
    print(f"  {'arity':>5} {'n_eff binary':>13} {'n_eff fraction':>15} {'bar(1 cand)':>12}")
    for arity in (1, 2, 3, 4, 5, 6, 10, 20, 35):
        frac = exposure(20, arity)
        binary = n_eff(20, frac, rho_hi)
        shared = _shared_fraction_n_eff(20, arity, rho_hi)
        bar = (
            grade_mean - stats.t.ppf(0.95, df=shared - 1)
            * math.sqrt(1 + 1 / shared) * grade_std
            if shared >= 2 else float("nan")
        )
        print(f"  {arity:>5} {binary:>13.2f} {shared:>15.2f} {bar:>12.3f}")
    print(f"\n  placebo.TEXTURE_ICC_BOUND = {TEXTURE_ICC_BOUND} (the value shipped)")
    if abs(rho_hi - TEXTURE_ICC_BOUND) > 5e-4:
        print(f"  !! MISMATCH — this run measures {rho_hi:.3f}; update TEXTURE_ICC_BOUND "
              "or record why it differs.")


if __name__ == "__main__":
    main()
