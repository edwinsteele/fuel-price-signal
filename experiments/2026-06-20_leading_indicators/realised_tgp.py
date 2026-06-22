"""Realised/CpL arbiter for the TGP velocity candidate (#268, #215).

The WFCV log-loss screen (paired_wfcv_velocity.py) liked `tgp_delta_7d`: strong
independent signal that RESCUES the shock regime, unlike the raw gap
`station_minus_tgp_cents` (helps calm, hurts shock). Log-loss is a SCREEN, not the
verdict — this runs the realised/CpL backtest, the arbiter, on three arms:

  baseline   54 production features
  vel7       + tgp_delta_7d            (does the velocity convert to spend?)
  gap_vel7   + tgp_delta_7d + gap      (does the raw gap earn its place, or is it
                                        dead weight in low-headroom calm?)

TGP has no production PriceHistory source yet (graduation = separate PR), so it's
injected in-process via the #268 ArmSpec.extra_feature_provider seam: the training
frame carries the candidate columns (computed in-script, identical to
paired_wfcv_velocity.add_candidate_columns) and the live replay gets the value from
a closure over the same PIT series. PIT = TGP ffilled across weekends + lagged 1
day; accessed via .asof(as_of) (latest value <= decision date) to match
PriceHistory.avg_price_at's on-or-before semantics — no leakage.

Each arm is scored at its OWN per-fold τ and at the baseline's held τ (clean
attribution — the #254 lesson). Pooled CpL is the headline.

Run (heavy — 3 arms x 14 outer folds x inner OOF, single seed):
  PYTHONPATH=. uv run python experiments/2026-06-20_leading_indicators/realised_tgp.py \\
    2>&1 | tee experiments/2026-06-20_leading_indicators/run_realised.log

A first read can subset folds via FOLD_SUBSET below.
"""
from __future__ import annotations

import pathlib
import time

import pandas as pd

from experiments.lib.realised import ArmSpec, run_paired_realised_backtest
from fuel_signal.features import (
    FEATURE_COLUMNS,
    LGA_FEATURE_COLUMNS,
    NETWORK_FEATURE_COLUMNS,
    load_features,
)

HERE = pathlib.Path(__file__).resolve().parent
TGP_XLSX = HERE / "data" / "AIP_TGP_2026-06-19.xlsx"

GAP = "station_minus_tgp_cents"
VEL7 = "tgp_delta_7d"
BASE = FEATURE_COLUMNS + LGA_FEATURE_COLUMNS + NETWORK_FEATURE_COLUMNS  # 54, production

# Inner OOF folds for the per-fold isotonic calibrator + τ. Must fit inside outer
# fold 1's train (~1825d), so a 3y inner min-train (same rationale as
# realised_by_regime.py). The choice only moves the operating-point τ.
INNER_FOLDS = {"train_min_days": 1095, "val_days": 90, "step_days": 90}

# Set to e.g. {1, 7, 9, 13} for a fast first read; None = all folds (the verdict).
FOLD_SUBSET: set[int] | None = None


def _load_tgp_pit() -> pd.Series:
    """Daily PIT Sydney ULP TGP (c/L): weekday series, ffill weekends, lag 1 day."""
    tgp = pd.read_excel(TGP_XLSX, sheet_name="Petrol TGP")
    dcol = tgp.columns[0]
    tgp = tgp[[dcol, "Sydney"]].rename(columns={dcol: "date", "Sydney": "tgp"})
    tgp["date"] = pd.to_datetime(tgp["date"], errors="coerce")
    s = tgp.dropna(subset=["date"]).set_index("date")["tgp"].sort_index()
    return s.asfreq("D").ffill().shift(1)


def main() -> None:
    t0 = time.perf_counter()

    print("Loading features ...", flush=True)
    df = load_features()
    df["price_date"] = pd.to_datetime(df["price_date"])
    print(f"  rows={len(df):,}", flush=True)

    pit = _load_tgp_pit()
    vel7 = pit - pit.shift(7)

    # Training frames (must share index with the baseline). Candidate values are
    # computed exactly as paired_wfcv_velocity.add_candidate_columns.
    df_vel7 = df.copy()
    df_vel7[VEL7] = df_vel7["price_date"].map(vel7)
    df_gap_vel7 = df_vel7.copy()
    df_gap_vel7[GAP] = df_gap_vel7["station_price_cents"] - df_gap_vel7["price_date"].map(pit)

    for name, col in ((VEL7, df_vel7[VEL7]), (GAP, df_gap_vel7[GAP])):
        print(f"  {name}: null={col.isna().mean():.4%}  mean={col.mean():.2f}  sd={col.std():.2f}",
              flush=True)

    # Live providers: same PIT series via .asof (latest value <= decision date).
    def vel7_provider(as_of, station_code, station_price):
        return {VEL7: float(vel7.asof(pd.Timestamp(as_of)))}

    def gap_vel7_provider(as_of, station_code, station_price):
        ts = pd.Timestamp(as_of)
        return {VEL7: float(vel7.asof(ts)), GAP: station_price - float(pit.asof(ts))}

    arms = [
        ArmSpec("baseline", df, feature_columns=BASE),
        ArmSpec("vel7", df_vel7, feature_columns=BASE + [VEL7],
                extra_feature_provider=vel7_provider),
        ArmSpec("gap_vel7", df_gap_vel7, feature_columns=BASE + [VEL7, GAP],
                extra_feature_provider=gap_vel7_provider),
    ]

    res = run_paired_realised_backtest(
        arms, BASE, seed=42, held_tau=None,
        inner_fold_params=INNER_FOLDS, fold_subset=FOLD_SUBSET,
    )

    res.per_window.to_csv(HERE / "runs_realised.csv", index=False)
    res.aggregate.to_csv(HERE / "aggregate_realised.csv", index=False)
    res.deltas.to_csv(HERE / "deltas_realised.csv", index=False)

    print("\n=== AGGREGATE — pooled CpL per arm (own + held τ) ===", flush=True)
    print(res.aggregate.to_string(index=False), flush=True)

    print("\n=== DELTAS vs baseline (candidate − baseline; negative = cheaper) ===", flush=True)
    print("    pooled CpL deltas (own + held τ):", flush=True)
    base_agg = res.aggregate.set_index("arm")
    for arm in ("vel7", "gap_vel7"):
        d_own = base_agg.loc[arm, "cpl_own"] - base_agg.loc["baseline", "cpl_own"]
        d_held = base_agg.loc[arm, "cpl_held"] - base_agg.loc["baseline", "cpl_held"]
        print(f"    {arm:<10}  Δcpl_own={d_own:+.4f}  Δcpl_held={d_held:+.4f}", flush=True)

    print("\n=== per-fold deltas (held τ) ===", flush=True)
    piv = res.deltas.pivot_table(index="fold", columns="arm", values="delta_cpl_held")
    print(piv.to_string(float_format=lambda x: f"{x:+.4f}"), flush=True)

    print(f"\n[wall] harness {res.meta['total_wall_seconds']:.1f}s  "
          f"total {time.perf_counter()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
