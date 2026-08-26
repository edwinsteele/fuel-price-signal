"""The texture ICC by SOURCE COLUMN, measured. (bd fps-3jj.23)

Reads a pinned-source noise floor — every draw built from one of a few deliberate source
columns, cycled round-robin, each at its own block seed (`noise_floor --same-source-column`) —
and reports the intraclass correlation of `delta_cpl_held` grouped BY SOURCE COLUMN.

That grouping is the whole point. A placebo reaches its delta through two channels:

  alignment — did THIS rearrangement happen to line up with the target? Varies with the seed.
  texture   — how much does a column OF THIS SHAPE perturb the fit at all, regardless of
              alignment? Distribution, cardinality, NaN rate, autocorrelation. Fixed by the
              source column.

Two draws from the same column share texture exactly, so the correlation they inherit from
sharing a column IS the quantity `placebo.effective_n_draws` charges. Between-column variance
is texture; within-column is alignment.

Two estimators are reported, deliberately:

  ANOVA  (primary) — ICC(1) from a one-way ANOVA over the pinned bank alone. Directly the
                     by-COLUMN quantity, so it closes the family-vs-column gap that makes the
                     shipped 0.391 not established as an upper bound at all. Depends on no
                     other artifact, hence on no other artifact's sampling noise.
  RATIO  (the bead's own construction) — 1 - var_within / var_multi, against the committed
                     20-draw multi-column floor. Reported for continuity with how the bead
                     framed the measurement, and because a large disagreement between the two
                     would say something real (the pinned columns are not representative of
                     the bank's texture mix).

Run `power.py` in this directory FIRST — it prices what each design can resolve, and the
answer decides whether a given bank is worth the fits at all.

    PYTHONPATH=. uv run python experiments/2026-08-27_texture_icc/measure_icc.py \\
        --pinned experiments/batches/batch1/noise_floor_icc.json

Prints and exits. The decision it feeds is whether `placebo.TEXTURE_ICC_BOUND` is replaced,
raised, or explicitly held.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pandas as pd
from bank_model import model_bank  # sibling module; this dir is sys.path[0] when run as a script
from scipy.stats import f as f_dist

from experiments.pipeline.dossier_tables import (
    NULL_METHOD_PLACEBO_COLUMN,
    NULL_METHOD_PLACEBO_PINNED_COLUMN,
    family_wise_z_threshold,
)
from experiments.pipeline.placebo import TEXTURE_ICC_BOUND, effective_n_draws

BATCH = pathlib.Path("experiments/batches/batch1")
DEFAULT_PINNED = BATCH / "noise_floor_icc.json"
MULTI_FLOOR = BATCH / "noise_floor_k1.json"
GRADING_FLOOR = BATCH / "noise_floor.json"

#: Everything that must be IDENTICAL between the pinned bank and the multi-column floor for
#: the ratio estimator to mean anything — the same discipline `compare_arity.py` enforces, and
#: the same one `baseline_fingerprint` enforces between runs. The pinned source columns are
#: supposed to be the ONLY thing that changed.
HELD_FIXED = ("baseline_fingerprint", "tank_params", "seed", "n_windows")

#: The multi-column floor must ALSO be arity 1 and the expected draw count. Checked separately
#: from HELD_FIXED because arity is stored as `n_placebo_columns` and a pre-fps-3jj.14 floor
#: omits it entirely, which reads as 1 — the same absent-as-1 rule `_noise_band` applies.
#: Without this guard batch1's `noise_floor.json` (arity 3, 10 draws) clears every other check
#: — it shares fingerprint, tank_params, seed AND n_windows with `noise_floor_k1.json` — so
#: `--multi .../noise_floor.json` would report an "ICC by variance ratio" against a 3-column
#: band, which is not the alignment+texture quantity the estimator subtracts (PR #337 review).
MULTI_FLOOR_ARITY = 1


def _load(path: pathlib.Path) -> dict:
    if not path.exists():
        raise SystemExit(
            f"missing {path}. If that is the pinned bank, build it first — see this "
            "directory's README for the command and for why the draw count is not a free "
            "choice. If it is the comparison floor or the grading ruler, it is a committed "
            "artifact of the batch: check --multi, or that the batch dir is the real one."
        )
    payload = json.loads(path.read_text())
    payload["_deltas"] = np.asarray(payload["deltas_cpl_held"], dtype=float)
    return payload


def _anova_icc(frame: pd.DataFrame) -> dict:
    """ICC(1) of `delta` grouped by `source_column`, with its 95% CI and — required by this
    repo's convention, not optional — the smallest value the design could have resolved.

    Uses the harmonic-style average group size `k_bar` so an unbalanced bank (a screen retry
    can never unbalance it, but a non-multiple `n_draws` can) is priced correctly rather than
    silently treated as balanced.
    """
    # EVERY group, including one with a single draw. A singleton contributes nothing to the
    # WITHIN-column sum of squares (correctly — it has no within-column variation) but it does
    # contribute to the BETWEEN-column sum of squares, and dropping it discards that. The
    # earlier `if len(g) > 1` filter silently deleted such columns AND reported the post-drop
    # `n`/`k`, so nothing in the output revealed the loss: a 5-draw, 3-column bank printed
    # `n=4, k=2`. `icc_upper` is what the verdict at the bottom of this script turns on, so a
    # quietly dropped draw could flip the printed recommendation (PR #337 review). Inherited
    # from `2026-08-26_placebo_draw_independence/texture_channel.py`, which still has it —
    # tracked separately, that file is outside this change.
    groups = [g["delta"].to_numpy() for _, g in frame.groupby("source_column")]
    if len(groups) < 2:
        raise SystemExit(
            "the pinned bank has fewer than two source columns — an ICC BY COLUMN needs "
            "between-column variation. Rebuild with several --same-source-column flags."
        )
    if sum(len(g) for g in groups) <= len(groups):
        raise SystemExit(
            f"every one of the {len(groups)} pinned columns has exactly one draw — there is no "
            "WITHIN-column variation to separate alignment from texture. Raise --n-draws to at "
            "least twice the pinned column count."
        )
    n = int(sum(len(g) for g in groups))
    k = len(groups)
    sizes = np.array([len(g) for g in groups], dtype=float)
    # Standard ANOVA k_bar for unbalanced one-way designs; reduces to the common group size
    # when the design is balanced, which is what a well-chosen n_draws makes it.
    k_bar = (n - (sizes**2).sum() / n) / (k - 1)

    grand = np.concatenate(groups).mean()
    ss_between = float(sum(len(g) * (g.mean() - grand) ** 2 for g in groups))
    ss_within = float(sum(((g - g.mean()) ** 2).sum() for g in groups))
    df1, df2 = k - 1, n - k
    ms_between, ms_within = ss_between / df1, ss_within / df2
    f_stat = ms_between / ms_within
    p_value = float(1.0 - f_dist.cdf(f_stat, df1, df2))

    def _icc(f: float) -> float:
        return (f - 1.0) / (f + k_bar - 1.0)

    f_crit = f_dist.ppf(0.95, df1, df2)
    return {
        "n": n, "k": k, "k_bar": k_bar, "df1": df1, "df2": df2,
        "f_stat": f_stat, "p_value": p_value,
        "icc": _icc(f_stat),
        # A SYMMETRIC two-sided 95% interval. The earlier version paired a 2.5%-tail lower with
        # a 5%-tail upper and called the pair a "95% CI" (PR #337 review).
        "icc_ci_lower": _icc(f_stat / f_dist.ppf(0.975, df1, df2)),
        "icc_ci_upper": _icc(f_stat / f_dist.ppf(0.025, df1, df2)),
        # The decision quantity, reported separately and labelled: a ONE-SIDED 95% upper bound
        # at the 5% tail — the identical construction `texture_channel.py` used to produce the
        # 0.391 this run would replace, so the two are directly comparable.
        "icc_upper": _icc(f_stat / f_dist.ppf(0.05, df1, df2)),
        "icc_resolvable": _icc(f_crit),
        "ms_within": ms_within,
    }


def _ratio_icc(var_within: float, df_within: int, multi: dict) -> dict:
    """ICC = 1 - var_within / var_multi, with the F-test on the variance ratio, its CI, and the
    smallest ratio this design can call significant — the shape `compare_arity.py` established.

    `var_within` is the pinned bank's WITHIN-column variance (alignment alone), not its total:
    a multi-pinned-column bank's total variance contains between-column texture, which is
    exactly what is being isolated. At one pinned column the two coincide.

    **STATED ASSUMPTION, not a demonstrated fact: Var(alignment) is the same in both banks.**
    The two variances come from separately-computed banks, so `1 - var_within / var_multi` is
    the ICC only if a draw's alignment variance does not itself depend on which source column
    it was built from. Plausible — same batch, same seed, same folds, same fit path, and the
    same block-permutation construction — but assumed. The ANOVA estimator above does not need
    it, which is why that one is primary and this one is reported for continuity (PR #337
    review).
    """
    var_multi = float(multi["_deltas"].var(ddof=1))
    df_multi = multi["_deltas"].size - 1
    ratio = var_within / var_multi
    p_two_sided = 2 * min(
        float(f_dist.cdf(ratio, df_within, df_multi)),
        1.0 - float(f_dist.cdf(ratio, df_within, df_multi)),
    )
    lo = ratio / f_dist.ppf(0.975, df_within, df_multi)
    hi = ratio / f_dist.ppf(0.025, df_within, df_multi)
    return {
        "var_within": var_within, "var_multi": var_multi, "ratio": ratio,
        "df_within": df_within, "df_multi": df_multi, "p_value": p_two_sided,
        "icc": 1.0 - ratio, "icc_lower": 1.0 - hi, "icc_upper": 1.0 - lo,
        # Significant only if the ratio falls below the lower critical value — i.e. the pinned
        # bank is DETECTABLY narrower. Anything above it is "could not see it", never "absent".
        "icc_resolvable": 1.0 - float(f_dist.ppf(0.025, df_within, df_multi)),
    }


def _bars(icc: float, mean: float, std: float, arities=(1, 3, 4, 10, 35)) -> dict[int, float]:
    """Single-candidate bar at each arity for a 20-draw bank, using the shipped
    `effective_n_draws` over `bank_model.model_bank`'s reconstruction of the overlap."""
    columns = json.loads((BATCH / "baseline_columns.json").read_text())
    out: dict[int, float] = {}
    for arity in arities:
        n_eff = effective_n_draws(model_bank(columns, 20, arity), icc=icc)
        out[arity] = (
            mean - family_wise_z_threshold(n_candidates=1, n_draws=n_eff) * std
            if n_eff >= 2 else float("nan")
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pinned", type=pathlib.Path, default=DEFAULT_PINNED)
    parser.add_argument("--multi", type=pathlib.Path, default=MULTI_FLOOR)
    args = parser.parse_args()

    pinned, multi = _load(args.pinned), _load(args.multi)

    # Comparability first, as a hard stop — a mismatch makes the ratio estimator meaningless,
    # and a pinned bank stamped as an ordinary floor means the wrong file was passed.
    if pinned.get("null_method") != NULL_METHOD_PLACEBO_PINNED_COLUMN:
        raise SystemExit(
            f"{args.pinned} declares null_method {pinned.get('null_method')!r}, not "
            f"{NULL_METHOD_PLACEBO_PINNED_COLUMN!r} — this is an ordinary floor, whose draws "
            "use DISTINCT source columns and therefore carry no within-column variation to "
            "measure. Build the pinned bank with --same-source-column."
        )
    if multi.get("null_method") != NULL_METHOD_PLACEBO_COLUMN:
        raise SystemExit(f"{args.multi} is not a {NULL_METHOD_PLACEBO_COLUMN!r} floor.")
    if pinned.get("partial") or multi.get("partial"):
        raise SystemExit("a partial-fold floor is not comparable — recompute without --fold-subset.")
    for key in HELD_FIXED:
        if pinned.get(key) != multi.get(key):
            raise SystemExit(
                f"floors differ on {key} ({pinned.get(key)!r} vs {multi.get(key)!r}) — not "
                "comparable; the pinned source columns are supposed to be the only difference."
            )
    # Absent `n_placebo_columns` reads as 1 (what a pre-fps-3jj.14 floor in fact was), the same
    # rule `_noise_band` applies. batch1's live ruler shares every HELD_FIXED key with the k1
    # floor, so without this it would sail through as a comparison band at the wrong arity.
    multi_arity = multi.get("n_placebo_columns") or 1
    if multi_arity != MULTI_FLOOR_ARITY:
        raise SystemExit(
            f"{args.multi} is an arity-{multi_arity} floor. The ratio estimator subtracts a "
            "1-column bank's alignment variance from a 1-column bank's total; a "
            f"{multi_arity}-column band measures a different, wider quantity and would "
            "understate the ICC. Pass the arity-1 floor (batch1: noise_floor_k1.json)."
        )
    pinned_arity = pinned.get("n_placebo_columns") or 1
    if pinned_arity != 1:
        raise SystemExit(f"the pinned bank declares arity {pinned_arity}; it must be 1.")

    frame = pd.DataFrame(pinned["placebo_draws"])
    # One record per draw MEMBER, so this positional join is only correct at arity 1 (checked
    # above, and enforced by compute_noise_floor). Asserted rather than assumed: at arity > 1
    # it would silently mis-pair deltas with columns.
    if len(frame) != len(pinned["deltas_cpl_held"]):
        raise SystemExit(
            f"{len(frame)} draw records against {len(pinned['deltas_cpl_held'])} deltas — "
            "expected one record per draw (arity 1)."
        )
    frame["delta"] = np.asarray(pinned["deltas_cpl_held"], dtype=float)[frame["draw"] - 1]

    print(f"pinned bank: {args.pinned}")
    print(f"  columns {pinned.get('same_source_columns')}")
    print(f"  {len(frame)} draws, mean {frame.delta.mean():+.4f}, "
          f"std {frame.delta.std(ddof=1):.4f} c/L")
    print(f"  baseline_fingerprint {pinned.get('baseline_fingerprint')}  "
          f"tank_params {pinned.get('tank_params')}  seed {pinned.get('seed')}  "
          f"wall {pinned.get('wall_seconds', 0) / 3600:.1f}h")
    print()
    print("=== delta by source column ===")
    print(frame.groupby("source_column")["delta"].agg(
        n="size", mean="mean", std=lambda s: s.std(ddof=1)).to_string())

    anova = _anova_icc(frame)
    print("\n=== ICC by COLUMN — one-way ANOVA (PRIMARY) ===")
    print(f"  F({anova['df1']},{anova['df2']}) = {anova['f_stat']:.3f}, p = {anova['p_value']:.3f}"
          f"   (k={anova['k']} columns, k_bar={anova['k_bar']:.2f} draws each)")
    print(f"  ICC                                    {anova['icc']:+.3f}")
    print(f"  95% CI (two-sided)                     "
          f"[{anova['icc_ci_lower']:+.3f}, {anova['icc_ci_upper']:+.3f}]")
    print(f"  95% upper bound (ONE-SIDED, 5% tail)   {anova['icc_upper']:+.3f}"
          "   <- compare to 0.391, derived the same way")
    print(f"  smallest ICC this design could resolve {anova['icc_resolvable']:.3f}"
          "   <- below this reads 'could not see it', NEVER 'it is not there'")

    ratio = _ratio_icc(anova["ms_within"], anova["df2"], multi)
    print(f"\n=== ICC by variance ratio against {args.multi.name} (the bead's construction) ===")
    print(f"  within-column var {ratio['var_within']:.6f} vs multi-column var "
          f"{ratio['var_multi']:.6f}  (ratio {ratio['ratio']:.3f})")
    print(f"  F({ratio['df_within']},{ratio['df_multi']}) two-sided p = {ratio['p_value']:.3f}")
    print(f"  ICC                                   {ratio['icc']:+.3f}")
    print(f"  95% CI                                [{ratio['icc_lower']:+.3f}, {ratio['icc_upper']:+.3f}]")
    print(f"  smallest ICC this design could resolve {ratio['icc_resolvable']:.3f}")

    grading = _load(GRADING_FLOOR)
    mean = float(grading["_deltas"].mean())
    std = float(grading["_deltas"].std(ddof=1))
    print(f"\n=== what it does to the bar (live ruler: arity "
          f"{grading.get('n_placebo_columns', 1)}, {grading['_deltas'].size} draws) ===")
    labels = {
        "shipped bound": TEXTURE_ICC_BOUND,
        "ANOVA point": max(anova["icc"], 0.0),
        "ANOVA upper": max(anova["icc_upper"], 0.0),
    }
    arities = (1, 3, 4, 10, 35)
    print(f"  {'ICC':>14} {'value':>7} " + " ".join(f"{'k=%d' % a:>9}" for a in arities))
    for label, value in labels.items():
        bars = _bars(value, mean, std, arities)
        cells = " ".join(
            f"{bars[a]:>9.3f}" if bars[a] == bars[a] else f"{'no band':>9}" for a in arities
        )
        print(f"  {label:>14} {value:>7.3f} {cells}")

    print(f"\n  placebo.TEXTURE_ICC_BOUND = {TEXTURE_ICC_BOUND} (shipped)")
    if anova["icc_upper"] > TEXTURE_ICC_BOUND:
        print(f"  !! the measured 95% upper bound ({anova['icc_upper']:.3f}) EXCEEDS it — the "
              "shipped value is\n     not established as conservative by this run. Raise it, or "
              "state explicitly that it\n     is held and why (bd fps-3jj.23's acceptance "
              "criteria allow the latter, with a reason).")
    else:
        print(f"  the measured 95% upper bound ({anova['icc_upper']:.3f}) is TIGHTER — "
              "replace the constant with it.")


if __name__ == "__main__":
    main()
