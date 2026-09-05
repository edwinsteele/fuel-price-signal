"""fps-nas acceptance criterion 1: is the realised arbiter's cost fit-dominated?

The bead's costing argument is STRUCTURAL, not measured: per fold per arm, the
expensive work (`_train_calibrate_select_tau` — one full LGBM on all 714 stations'
fold-train, plus a walk-forward OOF loop of several more fits to fit the isotonic
calibrator) does not scale with the replay universe, while only `load_history` and
the `aggregate_backtest` tank walk do. If that holds, going 5 -> 50 -> 200 stations
multiplies the cheap part only, and a ~40x wider arbiter is nearly free.

It may well NOT hold. `ModelStrategy.decide` calls `predict_proba` on ONE row per
station-day, and at the locked 1d cadence a 90-day fold is 90 decisions per station
— so the replay is 200x90 = 18,000 single-row sklearn calls per arm per fold at
n=200, each carrying full per-call overhead. This script measures the three pieces
SEPARATELY rather than timing the black box, because "the total got 3x slower" does
not tell you which knob to turn:

    fit      _train_calibrate_select_tau   — expected flat in n
    history  load_history                  — one query per station + shared caches
    replay   aggregate_backtest            — expected linear in n

Run: PYTHONPATH=. uv run python experiments/2026-09-05_arbiter_universe_width/timing.py \
        2>&1 | tee experiments/2026-09-05_arbiter_universe_width/timing.log

Writes timing.json beside this script. One fold, one arm (the frozen baseline) —
enough for the scaling law, and a paired 2-arm run is just this twice.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sqlite3
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from experiments.lib.constants import SEEDS  # noqa: E402
from experiments.lib.io import write_meta  # noqa: E402
from experiments.lib.realised import _plan_folds, _train_calibrate_select_tau  # noqa: E402
from experiments.lib.universe import (  # noqa: E402
    UniverseSpec,
    describe_universe,
    sample_station_universe,
)
from experiments.pipeline.runner import DEFAULT_INNER_FOLD_PARAMS  # noqa: E402
from fuel_signal.backtest import (  # noqa: E402
    ModelStrategy,
    TankParams,
    _evaluation_dates,
    load_history,
)
from fuel_signal.backtest_phase2 import aggregate_backtest  # noqa: E402
from fuel_signal.config import PREFERRED_STATIONS  # noqa: E402
from fuel_signal.features import load_features  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
UNIVERSE_SEED = 20260905


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch-dir", type=pathlib.Path,
                    default=pathlib.Path("experiments/batches/batch1"))
    ap.add_argument("--fold", type=int, default=11,
                    help="1-indexed outer fold to time (default: 11). NOT the mid-span "
                         "fold 7: station 414 sits at 0.10 coverage there and 0.00 in "
                         "folds 5-6, so the five-station arm would really be timing four "
                         "stations. Folds 8-14 are clean for all five.")
    ap.add_argument("--sizes", type=int, nargs="+", default=[5, 50, 200])
    ap.add_argument("--seed", type=int, default=SEEDS[0])
    args = ap.parse_args()

    batch_dir = args.batch_dir
    db_path = batch_dir / "fuel_signal.db"
    # load_features() prefers the parquet sibling — batch dirs carry only the parquet.
    frame = load_features(batch_dir / "features.csv")
    baseline_columns = json.loads((batch_dir / "baseline_columns.json").read_text())
    tank = TankParams()
    inner_fold_params = dict(DEFAULT_INNER_FOLD_PARAMS)

    (plan,) = _plan_folds(frame, {}, {args.fold})
    print(f"[fold {plan.fold}] val {plan.val_start} -> {plan.val_end}", flush=True)

    # The universe spec's span is the FOLD's val window, not the whole eval span:
    # coverage is being asked "can this station be replayed here", and here is this
    # fold. A station dark for all of 2022 is fine for a 2024 fold.
    def spec_for(n: int) -> UniverseSpec:
        # The span IS this fold's val window, and it is also passed as the single
        # gating window: coverage has to hold where the replay happens, and for a
        # one-fold timing run those are the same interval.
        return UniverseSpec(
            n=n, seed=UNIVERSE_SEED, start_date=plan.val_start, end_date=plan.val_end,
            windows=((plan.val_start, plan.val_end),),
        )

    # Fit once: it is station-independent by construction (train_df is the whole
    # fold-train over all 714 stations regardless of the replay universe), so timing
    # it inside the size loop would measure the same work three times and inflate
    # every total by an identical constant.
    t0 = time.perf_counter()
    cal_pipe, own_tau = _train_calibrate_select_tau(
        frame.loc[plan.train_index], baseline_columns, args.seed, inner_fold_params
    )
    fit_seconds = time.perf_counter() - t0
    print(f"[fit] {fit_seconds:.1f}s  tau={own_tau:.4f}  "
          f"(train rows={len(plan.train_index):,})", flush=True)

    eval_dates = _evaluation_dates(plan.val_start, plan.val_end, tank.evaluation_interval_days)
    rows = []
    for n in args.sizes:
        conn = sqlite3.connect(db_path)
        try:
            if n == len(PREFERRED_STATIONS):
                # Size 5 IS the incumbent population, not a 5-station sample of the
                # network — the point of the row is to time what the arbiter does today.
                codes = sorted(PREFERRED_STATIONS)
                source = "PREFERRED_STATIONS"
            else:
                codes = sample_station_universe(conn, spec_for(n))
                source = f"sample_station_universe(seed={UNIVERSE_SEED})"
            described = describe_universe(conn, codes, spec_for(len(codes)))

            t0 = time.perf_counter()
            history = load_history(conn, codes, eval_dates=eval_dates)
            history_seconds = time.perf_counter() - t0
        finally:
            conn.close()

        t0 = time.perf_counter()
        agg = aggregate_backtest(
            history,
            ModelStrategy(pipeline=cal_pipe, feature_columns=baseline_columns, threshold=own_tau),
            codes, plan.val_start, plan.val_end, tank,
        )
        replay_seconds = time.perf_counter() - t0

        row = {
            "n_stations": len(codes),
            "source": source,
            "n_councils": described["n_councils"],
            "coverage_min": described["coverage_min"],
            "history_seconds": history_seconds,
            "replay_seconds": replay_seconds,
            "scaling_seconds": history_seconds + replay_seconds,
            "fill_events": agg["fill_events"],
            "cpl": agg["cpl"],
            "seconds_per_station": (history_seconds + replay_seconds) / len(codes),
        }
        rows.append(row)
        print(
            f"[n={len(codes):>3}] history={history_seconds:6.1f}s  replay={replay_seconds:7.1f}s"
            f"  fills={agg['fill_events']:>6}  cpl={agg['cpl']:.2f}"
            f"  per-station={row['seconds_per_station']:.2f}s",
            flush=True,
        )

    base = rows[0]
    for row in rows:
        # The question is not "did it get slower" (it must) but "did the SCALING part
        # overtake the fixed fit". Both framings are printed so neither can be quoted
        # without the other.
        row["scaling_vs_fit"] = row["scaling_seconds"] / fit_seconds
        row["scaling_x_vs_smallest"] = row["scaling_seconds"] / base["scaling_seconds"]
        row["stations_x_vs_smallest"] = row["n_stations"] / base["n_stations"]
        row["one_arm_fold_seconds"] = fit_seconds + row["scaling_seconds"]
        row["fit_share"] = fit_seconds / row["one_arm_fold_seconds"]

    meta = {
        "issue": "fps-nas",
        "question": "criterion 1 — is the realised arbiter's per-fold cost fit-dominated?",
        "batch_dir": str(batch_dir),
        "fold": plan.fold,
        "val_start": plan.val_start,
        "val_end": plan.val_end,
        "seed": args.seed,
        "universe_seed": UNIVERSE_SEED,
        "inner_fold_params": inner_fold_params,
        "n_baseline_columns": len(baseline_columns),
        "fit_seconds": fit_seconds,
        "own_tau": own_tau,
        "by_size": rows,
        # A full paired run is 14 folds x 2 arms, with the baseline arm servable from
        # r0_cache.joblib on a repeat run — so this is a floor on a fresh batch, not a
        # forecast for one with a warm cache.
        "projected_full_run_hours": {
            str(r["n_stations"]): 14 * 2 * r["one_arm_fold_seconds"] / 3600 for r in rows
        },
    }
    write_meta(HERE, meta, baseline_columns=baseline_columns, tank=tank)
    (HERE / "timing.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"\nwrote {HERE / 'timing.json'}")
    print("\nfit is dominant where fit_share > 0.5:")
    for r in rows:
        print(f"  n={r['n_stations']:>3}  fit_share={r['fit_share']:.2f}  "
              f"scaling/fit={r['scaling_vs_fit']:.2f}  "
              f"{r['stations_x_vs_smallest']:.0f}x stations -> "
              f"{r['scaling_x_vs_smallest']:.1f}x scaling cost")


if __name__ == "__main__":
    main()
