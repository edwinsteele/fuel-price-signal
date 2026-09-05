"""fps-nas criteria 3-4: does a broad-Sydney CPL delta track the five-station one?

Runs ONE batch1 candidate through the realised arbiter twice — at the incumbent
five commute stations, and at a `experiments.lib.universe` sample of the width
`timing.py` shows is affordable — and reads the two out side by side.

**Run `timing.py` first.** `--n-stations` is required and deliberately has no
default: the universe width is a costing decision, and this script must not be
the place it gets made by accident.

Two things this script refuses to do, both for the same reason (see
`feedback_disjoint_basket_comparison`):

* It never differences the two populations' CPLs. Five stations and 200 stations
  fill different tanks on different days at different points in the cycle; the
  difference has no shared denominator. Each population's delta is reported
  against ITS OWN fold-clustered 2*SE, and the homogeneity read is whether both
  land on the same side of zero with overlapping intervals — plus the paired
  per-fold correlation, which IS legitimate because the 14 folds are shared.
* It does not call `experiments.pipeline.runner.run_candidate`. That would be
  less code, but `run_candidate` unconditionally writes `<batch_dir>/
  r0_cache.joblib` from whatever universe it just ran (runner.py `_save_baseline
  _cache`), so a broad-universe run would silently replace the five-station R0
  cache that every subsequent batch1 candidate depends on. Filed as a follow-up;
  until it is fixed, a wide run must go through `run_paired_realised_backtest`
  directly, as here.

Run: PYTHONPATH=. uv run python \
        experiments/2026-09-05_arbiter_universe_width/homogeneity.py --n-stations 200 \
        2>&1 | tee experiments/2026-09-05_arbiter_universe_width/homogeneity.log
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sqlite3
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402

from experiments.lib.constants import SEEDS  # noqa: E402
from experiments.lib.flips import cascade_window_days, summarise_flips  # noqa: E402
from experiments.lib.io import write_meta  # noqa: E402
from experiments.lib.realised import (  # noqa: E402
    ArmSpec,
    _plan_folds,
    run_paired_realised_backtest,
)
from experiments.lib.universe import (  # noqa: E402
    UniverseSpec,
    describe_universe,
    eligible_pool_digest,
    sample_station_universe,
)
from experiments.pipeline.runner import (  # noqa: E402
    BASELINE_ARM,
    CANDIDATE_ARM,
    DEFAULT_INNER_FOLD_PARAMS,
    _make_lookup_provider,
    load_candidate_module,
)
from fuel_signal.backtest import TankParams  # noqa: E402
from fuel_signal.config import PREFERRED_STATIONS  # noqa: E402
from fuel_signal.features import load_features  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
UNIVERSE_SEED = 20260905  # same value timing.py draws with — keep them in step.


def _fold_clustered(deltas: pd.Series) -> dict:
    """Mean per-fold delta and its fold-clustered 2*SE.

    The fold is the independent replicate, not the station-day: stations inside one
    90-day window share a cycle and a shock, so sizing an interval on station-days
    would understate it by roughly the square root of the universe width — which
    would manufacture exactly the resolution improvement this bead is trying to
    MEASURE.
    """
    values = deltas.dropna()
    n = len(values)
    sd = float(values.std(ddof=1)) if n > 1 else float("nan")
    se = sd / math.sqrt(n) if n > 1 else float("nan")
    mean = float(values.mean()) if n else float("nan")
    return {
        "n_folds": n,
        "mean_per_fold_delta": mean,
        "sd": sd,
        "se": se,
        "two_se": 2 * se,
        "interval": [mean - 2 * se, mean + 2 * se],
        "inside_own_2se": bool(abs(mean) < 2 * se) if n > 1 else None,
    }


def _run(arms, baseline_columns, codes, *, seed, db_path, tank, label) -> dict:
    print(f"\n=== {label}: {len(codes)} stations ===", flush=True)
    result = run_paired_realised_backtest(
        arms, baseline_columns,
        station_codes=list(codes),
        seed=seed,
        inner_fold_params=dict(DEFAULT_INNER_FOLD_PARAMS),
        db_path=db_path,
        tank=tank,
        collect_fills=True,
        verbose=True,
    )
    agg = result.aggregate.set_index("arm")
    deltas = result.deltas[result.deltas["arm"] == CANDIDATE_ARM]
    flips = summarise_flips(
        result.fills, BASELINE_ARM, CANDIDATE_ARM, shock_folds=set(),
        window_days=cascade_window_days(
            result.meta["tank_params"], exact_fields=result.meta["tank_params_fields"]
        ),
    )
    return {
        "label": label,
        "n_stations": len(codes),
        "station_codes": sorted(codes),
        "always_cpl": float(agg.loc[BASELINE_ARM, "always_cpl"]),
        "cpl_baseline_held": float(agg.loc[BASELINE_ARM, "cpl_held"]),
        "cpl_candidate_held": float(agg.loc[CANDIDATE_ARM, "cpl_held"]),
        # The headline: pooled over spend and litres, exactly what the dossier reads.
        "pooled_delta_cpl_held": float(
            agg.loc[CANDIDATE_ARM, "cpl_held"] - agg.loc[BASELINE_ARM, "cpl_held"]
        ),
        "per_fold_delta_cpl_held": {
            int(r.fold): float(r.delta_cpl_held) for r in deltas.itertuples()
        },
        "fold_clustered": _fold_clustered(deltas["delta_cpl_held"]),
        # The resolution statistic the bead is actually buying: 92 independent
        # decisions at five stations — what does it become here?
        "n_flips": flips["n_flips"],
        "n_decisions": flips["n_decisions"],
        "wall_seconds": result.meta["total_wall_seconds"],
        "tank_params": result.meta["tank_params"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch-dir", type=pathlib.Path,
                    default=pathlib.Path("experiments/batches/batch1"))
    ap.add_argument("--candidate", type=pathlib.Path,
                    default=pathlib.Path("experiments/candidates/batch1/tgp_cycle_displacement.py"),
                    help="batch1's cleanest signal (-0.2015 c/L) — the best available "
                         "signal-to-noise for a homogeneity read.")
    ap.add_argument("--n-stations", type=int, required=True,
                    help="Broad-universe width. Required: run timing.py first.")
    ap.add_argument("--seed", type=int, default=SEEDS[0])
    args = ap.parse_args()

    batch_dir = args.batch_dir
    db_path = batch_dir / "fuel_signal.db"
    frame = load_features(batch_dir / "features.csv")
    baseline_columns = json.loads((batch_dir / "baseline_columns.json").read_text())
    tank = TankParams()

    candidate = load_candidate_module(args.candidate)
    candidate_frame = candidate.add_columns(frame.copy())
    candidate_cols = list(baseline_columns) + list(candidate.COLUMNS)
    provider = _make_lookup_provider(candidate_frame, list(candidate.COLUMNS))
    arms = [
        ArmSpec(BASELINE_ARM, candidate_frame, feature_columns=baseline_columns),
        ArmSpec(CANDIDATE_ARM, candidate_frame, feature_columns=candidate_cols,
                extra_feature_provider=provider),
    ]

    # The universe span is the whole evaluation window, not one fold: a station has to
    # be replayable in EVERY fold or it silently contributes to some and not others,
    # which reweights the pooled CPL along the time axis.
    # Gate on the ACTUAL val windows, not just their envelope. A span-level gate lets
    # through a station that is entirely dark for one whole fold — measured at 189 of
    # 599 on batch1, station 414 among them at worst-window coverage 0.0 — and
    # aggregate_backtest skips such a cell silently (PR #361 review finding 1).
    windows = tuple((p.val_start, p.val_end) for p in _plan_folds(candidate_frame, {}, None))
    spec = UniverseSpec(
        n=args.n_stations, seed=UNIVERSE_SEED,
        start_date=windows[0][0], end_date=windows[-1][1], windows=windows,
    )
    conn = sqlite3.connect(db_path)
    try:
        broad_codes = sample_station_universe(conn, spec)
        # The spec records the knobs; the digest records what those knobs selected out
        # of THIS db. daily_prices is rebuilt by fill.py, so the spec alone does not
        # pin a universe (review finding 2).
        pool_digest = eligible_pool_digest(conn, spec)
        populations = {
            "five": describe_universe(conn, sorted(PREFERRED_STATIONS), spec),
            "broad": describe_universe(conn, broad_codes, spec),
        }
    finally:
        conn.close()
    print(f"[universe] broad n={len(broad_codes)} across "
          f"{populations['broad']['n_councils']} councils; five across "
          f"{populations['five']['n_councils']}", flush=True)

    runs = [
        _run(arms, baseline_columns, sorted(PREFERRED_STATIONS),
             seed=args.seed, db_path=db_path, tank=tank, label="five"),
        _run(arms, baseline_columns, broad_codes,
             seed=args.seed, db_path=db_path, tank=tank, label="broad"),
    ]
    five, broad = runs

    # Paired by FOLD — the one comparison the two populations can legitimately share,
    # because the fold grid is identical. Everything else is read per-population.
    common = sorted(set(five["per_fold_delta_cpl_held"]) & set(broad["per_fold_delta_cpl_held"]))
    a = pd.Series([five["per_fold_delta_cpl_held"][f] for f in common], index=common)
    b = pd.Series([broad["per_fold_delta_cpl_held"][f] for f in common], index=common)
    homogeneity = {
        "n_common_folds": len(common),
        "per_fold_sign_agreement": int(((a * b) > 0).sum()),
        "per_fold_pearson_r": float(a.corr(b)) if len(common) > 2 else None,
        "same_sign_headline": bool(
            five["pooled_delta_cpl_held"] * broad["pooled_delta_cpl_held"] > 0
        ),
        "intervals_overlap": bool(
            five["fold_clustered"]["interval"][0] <= broad["fold_clustered"]["interval"][1]
            and broad["fold_clustered"]["interval"][0] <= five["fold_clustered"]["interval"][1]
        ),
        "decisions_ratio": (
            broad["n_decisions"] / five["n_decisions"] if five["n_decisions"] else None
        ),
        "not_computed": (
            "A difference of the two pooled CPLs. Disjoint baskets, no shared "
            "denominator — read each delta against its own 2*SE instead."
        ),
    }

    meta = {
        "issue": "fps-nas",
        "question": "criteria 3-4 — broad-Sydney vs five-station realised CPL for one candidate",
        "batch_dir": str(batch_dir),
        "candidate": candidate.NAME,
        "seed": args.seed,
        "universe_seed": UNIVERSE_SEED,
        "universe_spec": spec.as_dict(),
        "eligible_pool_digest": pool_digest,
        "inner_fold_params": dict(DEFAULT_INNER_FOLD_PARAMS),
        "populations": populations,
        "runs": runs,
        "homogeneity": homogeneity,
    }
    # From write_meta's RETURN, not `meta` — see fps-6rm / timing.py. This artifact
    # happened to satisfy the cadence-stamp scanner anyway, because `_run` copies
    # `tank_params` into each run row, but that is an accident of this script's shape
    # and not something the next edit to it should have to rely on.
    stamped = write_meta(HERE, meta, baseline_columns=baseline_columns, tank=tank)
    (HERE / "homogeneity.json").write_text(json.dumps(stamped, indent=2, default=str))

    print("\n" + "=" * 72)
    for r in runs:
        fc = r["fold_clustered"]
        print(
            f"{r['label']:>6}  n={r['n_stations']:>3}  pooled Δ={r['pooled_delta_cpl_held']:+.4f} c/L"
            f"  per-fold mean {fc['mean_per_fold_delta']:+.4f} ± {fc['two_se']:.4f}"
            f"  decisions={r['n_decisions']}"
        )
    print(f"\nsame sign: {homogeneity['same_sign_headline']}  "
          f"intervals overlap: {homogeneity['intervals_overlap']}  "
          f"per-fold sign agreement: {homogeneity['per_fold_sign_agreement']}/{len(common)}  "
          f"r={homogeneity['per_fold_pearson_r']}")
    print(f"decisions {five['n_decisions']} -> {broad['n_decisions']} "
          f"({homogeneity['decisions_ratio']}x)")
    print(f"\nwrote {HERE / 'homogeneity.json'}")


if __name__ == "__main__":
    main()
