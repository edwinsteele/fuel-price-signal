"""Stage 1 (fit-free): how far does the perfect-foresight ceiling move when the
decision cadence gets finer?  bd fps-fii.

The whole "we're near the floor, stop doing feature work" judgement rests on
#262's window-level number:

    always-buy      193.41 c/L
    model           189.79   <- captured 3.62
    perfect oracle  188.14   <- 1.66 still available

but that 1.66 is measured on a **7-day decision grid**, which is a TankParams
default, not a fact about the world. With a 50 L tank burning 25 L per weekly
interval the car must fill at least every second interval, so the strategy's
entire freedom is *which of two adjacent weeks to buy in*. A finer grid can
exploit within-week dips the weekly grid cannot see.

This stage sweeps cadence on the ORACLE and ALWAYS-BUY arms only. Both are
fit-free, so the whole sweep is seconds — it answers "how much bigger could the
pot be?" on its own. The model arm (which also gets more chances at a finer
cadence, so headroom does NOT automatically grow) is stage 2, model_cadence.py.

WINDOW LEVEL ONLY. Do not slice any of this by cycle zone — per-zone headroom is
not identified (fps-1785999730023-4-264564ac / experiments/2026-08-20_headroom_attribution).

  PYTHONPATH=. uv run python experiments/2026-08-20_cadence_ceiling/oracle_cadence.py
"""
from __future__ import annotations

import pathlib
import time

import pandas as pd

from experiments.lib.constants import BASELINE_COLUMNS, BASELINE_FINGERPRINT
from experiments.lib.io import current_git_sha, write_meta

# _plan_folds is imported rather than re-derived on purpose: the fold windows the
# oracle runs over MUST be the same objects the model harness plans, and this repo
# has already paid twice for re-deriving something the lock owns (fps-sa1, fps-zci).
from experiments.lib.realised import _plan_folds
from fuel_signal import db as _db
from fuel_signal.backtest import AlwaysBuyStrategy, TankParams, load_history, run_oracle_backtest
from fuel_signal.backtest_phase2 import aggregate_backtest
from fuel_signal.config import PREFERRED_STATIONS
from fuel_signal.features import load_features

HERE = pathlib.Path(__file__).resolve().parent
# 7/3/1 per fps-fii, plus 2 — see model_cadence.py's CADENCES note: 3d is
# structurally unusable as a ceiling and 2d is the nearest provably-safe cadence.
CADENCES = (7, 3, 2, 1)


def main() -> None:
    t0 = time.perf_counter()
    df = load_features()
    df["price_date"] = pd.to_datetime(df["price_date"])
    station_codes = list(PREFERRED_STATIONS)

    plans = _plan_folds(df, {}, None)
    print(f"[folds] {len(plans)} walk-forward windows, {len(station_codes)} stations", flush=True)

    conn = _db.open_db()
    try:
        # Oracle and always-buy read station prices only — no PIT eval_dates grid
        # needed (that is a ModelStrategy concern, and it belongs to stage 2).
        history = load_history(conn, station_codes)
    finally:
        conn.close()

    summary_rows: list[dict] = []
    fold_rows: list[dict] = []
    all_fills: list[dict] = []

    for cadence in CADENCES:
        ct0 = time.perf_counter()
        tank = TankParams(evaluation_interval_days=cadence)
        o_spend = o_litres = a_spend = a_litres = 0.0
        for p in plans:
            fo_spend = fo_litres = 0.0
            for sc in station_codes:
                r = run_oracle_backtest(
                    history, sc, p.val_start, p.val_end, tank, collect_fills=True
                )
                if pd.isna(r.realised_cpl):
                    print(f"[warn] oracle NaN cadence={cadence} fold={p.fold} station={sc}", flush=True)
                    continue
                fo_spend += r.total_spend_cents
                fo_litres += r.total_litres
                for fr in r.fills:
                    all_fills.append({
                        "cadence": cadence, "fold": p.fold, "arm": "oracle",
                        "date": fr.date, "station_code": fr.station_code, "price": fr.price,
                        "litres": fr.litres, "spend_cents": fr.spend_cents,
                        "emergency": fr.emergency,
                    })
            always = aggregate_backtest(
                history, AlwaysBuyStrategy(), station_codes, p.val_start, p.val_end, tank,
                collect_fills=True,
            )
            for fr in always["fills"]:
                all_fills.append({
                    "cadence": cadence, "fold": p.fold, "arm": "always_buy",
                    "date": fr.date, "station_code": fr.station_code, "price": fr.price,
                    "litres": fr.litres, "spend_cents": fr.spend_cents,
                    "emergency": fr.emergency,
                })
            o_spend += fo_spend
            o_litres += fo_litres
            a_spend += always["total_spend_cents"]
            a_litres += always["total_litres"]
            fold_rows.append({
                "cadence": cadence, "fold": p.fold,
                "val_start": p.val_start, "val_end": p.val_end,
                "oracle_cpl": fo_spend / fo_litres if fo_litres > 0 else float("nan"),
                "always_cpl": always["cpl"],
                "oracle_litres": fo_litres, "always_litres": always["total_litres"],
            })

        fills = pd.DataFrame([r for r in all_fills if r["cadence"] == cadence])
        oracle_fills = fills[fills["arm"] == "oracle"]
        summary_rows.append({
            "cadence_days": cadence,
            "always_cpl": a_spend / a_litres if a_litres > 0 else float("nan"),
            "oracle_cpl": o_spend / o_litres if o_litres > 0 else float("nan"),
            "always_minus_oracle": (a_spend / a_litres) - (o_spend / o_litres),
            "oracle_fills": len(oracle_fills),
            "oracle_emergency_pct": 100 * oracle_fills["emergency"].mean(),
            "oracle_mean_fill_litres": oracle_fills["litres"].mean(),
            "oracle_litres": o_litres,
            "always_litres": a_litres,
            "wall_s": time.perf_counter() - ct0,
        })
        s = summary_rows[-1]
        print(
            f"[cadence {cadence:>2}d] oracle {s['oracle_cpl']:.2f}  always {s['always_cpl']:.2f}  "
            f"spread {s['always_minus_oracle']:.2f}  fills {s['oracle_fills']} "
            f"({s['oracle_emergency_pct']:.0f}% emerg, mean {s['oracle_mean_fill_litres']:.1f} L)  "
            f"litres o/a {o_litres:.0f}/{a_litres:.0f}  [{s['wall_s']:.1f}s]",
            flush=True,
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(HERE / "oracle_cadence.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(HERE / "oracle_cadence_per_fold.csv", index=False)
    pd.DataFrame(all_fills).to_parquet(HERE / "cadence_fills.parquet")

    base = summary.loc[summary["cadence_days"] == 7, "oracle_cpl"].iloc[0]
    summary["ceiling_move_vs_7d"] = base - summary["oracle_cpl"]
    print("\n=== oracle ceiling vs decision cadence (window level, pooled) ===", flush=True)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.2f}"), flush=True)
    print(
        "\nReference (#262, 7d): always 193.41 / model 189.79 / oracle 188.14 "
        f"→ 1.66 c/L headroom. This run's 7d oracle: {base:.2f}.",
        flush=True,
    )

    meta_dir = HERE / "stage1_oracle"  # own dir so the two stages' meta.json do not collide
    meta_dir.mkdir(exist_ok=True)
    write_meta(meta_dir, {
        "stage": "1_oracle_only",
        "git_sha": current_git_sha(),
        "cadences": list(CADENCES),
        "n_folds": len(plans),
        "station_codes": station_codes,
        "tank": "TankParams() with evaluation_interval_days swept",
        "baseline_fingerprint": BASELINE_FINGERPRINT,
        "n_baseline_columns": len(BASELINE_COLUMNS),
        "summary": summary.to_dict("records"),
        "wall_seconds": time.perf_counter() - t0,
    })
    print(f"[done] {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
