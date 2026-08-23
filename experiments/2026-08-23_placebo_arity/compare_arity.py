"""Does the placebo band widen with arity? (bd fps-3jj.14)

Reads two floors computed on the SAME frozen batch, same seed, same folds, same cadence —
differing only in how many placebo columns each draw adds — and answers the question the
bead actually asks: "is the band materially wider at k=3 than at k=1", not "what is the
k=3 band to three significant figures".

    PYTHONPATH=. uv run python experiments/2026-08-23_placebo_arity/compare_arity.py

Deliberately prints and exits. The decision it feeds is which floor grades batch1.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
from scipy.stats import f as f_dist

from experiments.pipeline.dossier_tables import family_wise_z_threshold

BATCH_DIR = pathlib.Path("experiments/batches/batch1")
FLOORS = {1: "noise_floor.json", 3: "noise_floor_k3.json"}

#: batch1's own size (docs/routines/generator.md § Batch sizing) — the n the family-wise
#: correction is made over. Not a constant: a later batch is 10-15.
N_CANDIDATES = 5


def _load(path: pathlib.Path) -> dict:
    payload = json.loads(path.read_text())
    deltas = np.asarray(payload["deltas_cpl_held"], dtype=float)
    return {
        "arity": int(payload.get("n_placebo_columns", 1)),
        "n": deltas.size,
        "mean": float(deltas.mean()),
        "std": float(deltas.std(ddof=1)),
        "deltas": deltas,
        "fingerprint": payload.get("baseline_fingerprint"),
        "tank_params": payload.get("tank_params"),
        "seed": payload.get("seed"),
        "partial": payload.get("partial"),
        "wall_seconds": payload.get("wall_seconds"),
    }


def _bar(band: dict, n_candidates: int) -> float:
    """The delta_cpl_held a candidate must reach to clear, in c/L (a cost: more negative
    is better, so the bar is below the band mean)."""
    z = family_wise_z_threshold(n_candidates=n_candidates, n_draws=band["n"])
    return band["mean"] - z * band["std"]


def main() -> None:
    bands = {}
    for arity, name in FLOORS.items():
        path = BATCH_DIR / name
        if not path.exists():
            raise SystemExit(
                f"missing {path} — run:\n  PYTHONPATH=. uv run python -m "
                f"experiments.pipeline.noise_floor batch1 --arity {arity} --n-draws 10 "
                f"--out-name {name}"
            )
        bands[arity] = _load(path)

    # Comparability first. Two floors are only comparable if everything EXCEPT arity is
    # held — the same discipline baseline_fingerprint enforces between runs. A mismatch
    # here would make the whole comparison meaningless, so it is a hard stop, not a note.
    a, b = bands[1], bands[3]
    for key in ("fingerprint", "tank_params", "seed"):
        if a[key] != b[key]:
            raise SystemExit(
                f"floors differ on {key} ({a[key]!r} vs {b[key]!r}) — not comparable; "
                "arity is supposed to be the only thing that changed."
            )
    if a["partial"] or b["partial"]:
        raise SystemExit("a partial-fold floor is not a valid ruler — recompute without --fold-subset.")
    for arity, band in bands.items():
        if band["arity"] != arity:
            raise SystemExit(f"{FLOORS[arity]} declares arity {band['arity']}, expected {arity}")

    print(f"baseline_fingerprint {a['fingerprint']}  tank_params {a['tank_params']}  seed {a['seed']}")
    print()
    print(f"{'arity':>6} {'n':>4} {'mean c/L':>10} {'std c/L':>10} {'min':>8} {'max':>8}")
    for arity, band in bands.items():
        print(
            f"{arity:>6} {band['n']:>4} {band['mean']:>+10.4f} {band['std']:>10.4f} "
            f"{band['deltas'].min():>+8.4f} {band['deltas'].max():>+8.4f}"
        )

    ratio = b["std"] / a["std"]
    print(f"\nstd ratio (k=3 / k=1): {ratio:.3f}")

    # An F-test on the variance ratio. Reported WITH its power, because the honest reading
    # of "not significant" at n=10 vs n=20 is "this design cannot see a small widening",
    # not "there is no widening".
    f_stat = (b["std"] ** 2) / (a["std"] ** 2)
    df_b, df_a = b["n"] - 1, a["n"] - 1
    p_two_sided = 2 * min(
        f_dist.cdf(f_stat, df_b, df_a), 1.0 - f_dist.cdf(f_stat, df_b, df_a)
    )
    lo = f_stat / f_dist.ppf(0.975, df_b, df_a)
    hi = f_stat / f_dist.ppf(0.025, df_b, df_a)
    print(f"F({df_b},{df_a}) = {f_stat:.3f}, two-sided p = {p_two_sided:.3f}")
    print(f"95% CI on the VARIANCE ratio: [{lo:.3f}, {hi:.3f}]  -> std ratio [{lo**0.5:.3f}, {hi**0.5:.3f}]")
    detectable = f_dist.ppf(0.975, df_b, df_a) ** 0.5
    print(
        f"smallest std ratio this design could call significant: {detectable:.2f}x "
        "— anything under that is UNRESOLVED, not absent."
    )

    # The decision-relevant quantity: how far does the bar actually move, in c/L?
    print(f"\n{'arity':>6} {'single-cand bar':>17} {'batch bar (n=%d)' % N_CANDIDATES:>18}")
    for arity, band in bands.items():
        print(f"{arity:>6} {_bar(band, 1):>+17.4f} {_bar(band, N_CANDIDATES):>+18.4f}")
    move_single = _bar(b, 1) - _bar(a, 1)
    move_batch = _bar(b, N_CANDIDATES) - _bar(a, N_CANDIDATES)
    print(f"\nbar moves by {move_single:+.4f} c/L (single) and {move_batch:+.4f} c/L (batch of {N_CANDIDATES})")
    print(
        "\nRead against the realised arbiter's own quantum (~0.05 c/L per buy/wait flip at "
        "7d, SMALLER at batch1's 1d cadence — see project_cadence_ceiling) and against "
        "results.csv's 0.03-0.26 c/L range for single-column features."
    )


if __name__ == "__main__":
    main()
