"""What can a same-column noise floor actually SEE? (bd fps-3jj.23, run this FIRST)

`fps-3jj.23` proposes replacing `placebo.TEXTURE_ICC_BOUND` — a 0.391 upper bound from an
underpowered family-level ANOVA — with a measured number, by computing a noise floor whose
draws all share one source column and comparing its variance to the committed 20-draw floor's.

Before spending ~2h of fits on that, this asks the question the repo's own convention requires
of any null result ("a quiet result reads 'could not see it', never 'it is not there'"): what
is the smallest ICC each candidate design can resolve, and what upper bound does each leave
behind if the point estimate lands at zero? A design whose best case is a LOOSER bound than the
0.391 already shipped cannot discharge the bead no matter how the fits come out.

Second half: where is the constant load-bearing at all? Its whole justification is that it
"sets every bar", so the bar is computed across the constant's entire possible range at every
arity, on batch1's live ruler and shipped code.

    PYTHONPATH=. uv run python experiments/2026-08-27_texture_icc/power.py

Pure arithmetic over committed artifacts — no fits, no DB, seconds to run. Prints and exits.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
from bank_model import model_bank  # sibling module; this dir is sys.path[0] when run as a script
from scipy.stats import f as f_dist

from experiments.pipeline.dossier_tables import family_wise_z_threshold
from experiments.pipeline.placebo import TEXTURE_ICC_BOUND, effective_n_draws

BATCH = pathlib.Path("experiments/batches/batch1")

#: Seconds per draw, measured: the committed 20-draw k1 floor's own `wall_seconds` / 20
#: (13717 / 20 = 686). Used only to price a design, so a rough figure is fine — but it is a
#: MEASURED rough figure off the artifact, not a guess.
SECONDS_PER_DRAW = 686.0

#: The committed multi-column floor every design below is compared against. 20 draws over 20
#: DISTINCT source columns at arity 1, so its variance is alignment + texture; a pinned bank's
#: within-column variance is alignment alone, and the gap is the texture component.
MULTI_FLOOR = "noise_floor_k1.json"

#: The batch's live grading ruler — where the BARS come from. Deliberately a different file
#: from MULTI_FLOOR: bars are a grading quantity and must be quoted on the band that actually
#: grades, while the ICC arithmetic needs k1's 20 distinct-column draws.
GRADING_FLOOR = "noise_floor.json"

#: Usable (non-all-NaN) batch1 lock columns — the ceiling on how many DISTINCT columns a
#: design can pin. batch1 declares 54; `bank_model.BATCH1_UNSCREENABLE` names the five that
#: are entirely NaN in the frozen frame and are skipped by the screen on every lap.
N_USABLE_COLUMNS = 49


def icc_from_anova_design(k_columns: int, m_seeds: int) -> tuple[float, float]:
    """(smallest resolvable ICC, 95% upper bound if the point estimate lands at exactly 0) for
    a balanced one-way ANOVA of delta grouped BY SOURCE COLUMN.

    This is the DIRECT estimator of the quantity `effective_n_draws` charges: between-column
    variance is texture (fixed by the column), within-column is alignment (varies with the
    block seed), and ICC(1) = (F - 1) / (F + m - 1). It needs no second floor, so nothing
    depends on the committed floor's own sampling noise — and unlike the family ANOVA it
    measures the by-COLUMN quantity, closing the family-vs-column gap the bead names as the
    reason 0.391 is not established as an upper bound at all.
    """
    df1, df2 = k_columns - 1, k_columns * (m_seeds - 1)
    f_crit = f_dist.ppf(0.95, df1, df2)
    resolvable = (f_crit - 1.0) / (f_crit + m_seeds - 1.0)
    # Upper confidence limit when the observed F is exactly 1 (point estimate ICC = 0) — the
    # best case for the design, and therefore the honest way to price it before running.
    # ONE-SIDED 5% tail, the same one `texture_channel.py` used to produce the 0.391 this would
    # replace; `icc_from_ratio_design` is held to the identical tail so the two are comparable.
    f_upper = 1.0 / f_dist.ppf(0.05, df1, df2)
    upper = (f_upper - 1.0) / (f_upper + m_seeds - 1.0)
    return resolvable, upper


def icc_from_ratio_design(n_pinned: int, n_multi: int = 20) -> tuple[float, float]:
    """(smallest resolvable ICC, 95% upper bound at a point estimate of 0) for the bead's own
    construction: one pinned column, `n_pinned` seeds, variance compared to the committed
    `n_multi`-draw multi-column floor. ICC = 1 - var_pinned / var_multi.

    An UNPAIRED variance ratio across two separately-computed banks, which is where its power
    goes: the F has df (n_pinned - 1, n_multi - 1), and a variance ratio needs a lot of both.
    """
    df1, df2 = n_pinned - 1, n_multi - 1
    # ONE-SIDED 5% tail, matching how the 0.391 being replaced was itself derived
    # (`texture_channel.py`: `f_stat / f.ppf(0.05, df1, df2)`). An earlier version used the
    # 0.975/0.025 two-sided pair here while `icc_from_anova_design` used the one-sided 5% tail,
    # so the two designs were printed in the same column under the same header at DIFFERENT
    # alphas — Design A's numbers were 97.5% bounds masquerading as 95% ones, which inflated
    # both its own figures and the gap between the designs (PR #337 review).
    resolvable = 1.0 - f_dist.ppf(0.05, df1, df2)
    upper = 1.0 - 1.0 / f_dist.ppf(0.95, df1, df2)
    return resolvable, upper


def main() -> None:
    baseline_columns = json.loads((BATCH / "baseline_columns.json").read_text())
    grading = json.loads((BATCH / GRADING_FLOOR).read_text())
    deltas = np.asarray(grading["deltas_cpl_held"], dtype=float)
    mean, std = float(deltas.mean()), float(deltas.std(ddof=1))

    print("=" * 78)
    print("PART 1 — what each design can see, BEFORE spending the compute")
    print("=" * 78)
    print(f"The value shipped today: TEXTURE_ICC_BOUND = {TEXTURE_ICC_BOUND} (a 95% upper bound")
    print("on the ICC by texture FAMILY, resolvable only down to 0.359). A design is worth")
    print("running only if it can leave a TIGHTER bound than that, or resolve the by-column")
    print("quantity the family bound is not established to cover.\n")

    print("Design A — the bead as written: 1 pinned column x n seeds, variance ratio against")
    print(f"           the committed {MULTI_FLOOR} (20 draws).")
    print("  (both designs at the ONE-SIDED 5% tail — the tail 0.391 itself was derived at)")
    print(f"  {'draws':>6} {'hours':>6} {'resolvable ICC':>15} {'upper bnd @ICC=0':>17}")
    for n in (10, 15, 20, 30):
        res, upper = icc_from_ratio_design(n)
        print(f"  {n:>6} {n * SECONDS_PER_DRAW / 3600:>6.1f} {res:>15.3f} {upper:>17.3f}")

    print("\nDesign B — k pinned columns x m seeds, one-way ANOVA of delta BY COLUMN. Measures")
    print("           the by-column quantity directly; needs no second floor.")
    print(f"  {'k':>3} {'m':>3} {'draws':>6} {'hours':>6} {'resolvable ICC':>15} {'upper bnd @ICC=0':>17}")
    for k, m in ((4, 5), (5, 4), (5, 5), (6, 5), (8, 4), (8, 5), (10, 5), (12, 5)):
        res, upper = icc_from_anova_design(k, m)
        n = k * m
        print(f"  {k:>3} {m:>3} {n:>6} {n * SECONDS_PER_DRAW / 3600:>6.1f} {res:>15.3f} {upper:>17.3f}")

    print("\nSmallest Design B that leaves an upper bound below the 0.391 already shipped:")
    winner = None
    for n_total in range(9, 81):
        for m in range(3, 9):
            if n_total % m:
                continue
            k = n_total // m
            if k < 3 or k > N_USABLE_COLUMNS:
                continue
            _res, upper = icc_from_anova_design(k, m)
            if upper < TEXTURE_ICC_BOUND and (winner is None or n_total < winner[2]):
                winner = (k, m, n_total, upper)
    if winner is None:
        print("  none within 80 draws.")
    else:
        k, m, n_total, upper = winner
        print(f"  {k} columns x {m} seeds = {n_total} draws "
              f"({n_total * SECONDS_PER_DRAW / 3600:.1f}h) -> upper bound {upper:.3f}")

    print()
    print("=" * 78)
    print("PART 2 — where the constant is load-bearing at all")
    print("=" * 78)
    print("Single-candidate bar (c/L) over the constant's ENTIRE possible range, 20-draw bank,")
    print(f"quoted on batch1's live ruler ({GRADING_FLOOR}: arity "
          f"{grading.get('n_placebo_columns', 1)}, {deltas.size} draws, mean {mean:+.4f}, "
          f"std {std:.4f}).")
    print("A row's SPREAD is the most the bar can move no matter what this measurement finds.\n")

    iccs = [0.0, 0.2, TEXTURE_ICC_BOUND, 0.6, 0.8, 1.0]
    print(f"  {'arity':>5} " + " ".join(f"{i:>8.3f}" for i in iccs) + f" {'spread':>8}")
    for arity in (1, 2, 3, 4, 6, 10, 20, 35):
        bank = model_bank(baseline_columns, 20, arity)
        row = []
        for icc in iccs:
            n_eff = effective_n_draws(bank, icc=icc)
            row.append(
                mean - family_wise_z_threshold(n_candidates=1, n_draws=n_eff) * std
                if n_eff >= 2 else float("nan")
            )
        finite = [v for v in row if v == v]
        cells = " ".join(f"{v:>8.3f}" if v == v else f"{'no band':>8}" for v in row)
        print(f"  {arity:>5} {cells} {max(finite) - min(finite):>8.3f}")

    print("\n  Read the spread against the realised arbiter's own decision quantum (~0.05 c/L")
    print("  per buy/wait flip at 7d, smaller at batch1's 1d cadence) and against results.csv's")
    print("  0.03-0.26 c/L range for single-column features. docs/routines/generator.md invites")
    print("  candidates of 2-4 columns; batch1's five were 1- and 3-column.")


if __name__ == "__main__":
    main()
