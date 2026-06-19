"""Gate-1 cheap pre-read (#259): proxy economic regret stratified by cycle regime.

NOT the gate — a directional screen (two-exams caveat: proxy expected_cents_per_row
!= realised CPL). Reads the already-computed #254 rowpreds (R0 = honest live
post-#250 baseline arm), joins the production `cycle_pct_through` regime tag, and
asks: is the model leaving more economic value on the table in the late-descent /
overdue zone than in normal rows?

Metric — per (regime, seed):
  peak_cents  = max over tau of expected_cents_per_row   (value the model extracts)
  oracle_cents = base_rate * TP_REWARD                   (perfect classifier ceiling)
  regret      = oracle_cents - peak_cents                (value left on the table)
Regret normalises out the differing base rates, so the three buckets are
comparable. Gate-1 pre-read fires (directional) iff regret is materially higher
in late-descent/overdue than in normal.

Run:
  PYTHONPATH=. uv run python experiments/2026-06-18_late_descent_gate1/proxy_regret_by_regime.py
"""
from __future__ import annotations

import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.lib.zones import CYCLE_REGIME_BANDS, assign_regime
from fuel_signal.features import load_features
from fuel_signal.score_phase2 import _TP_REWARD_CENTS, threshold_sweep

HERE = pathlib.Path(__file__).resolve().parent
ROWPREDS = HERE.parent / "2026-06-16_regime_cycle_length" / "rowpreds.parquet"


def main() -> None:
    t0 = time.perf_counter()

    rp = pd.read_parquet(ROWPREDS)
    rp = rp[rp["run"] == "R0"].copy()
    print(f"[load] rowpreds R0 {rp.shape}  ({time.perf_counter()-t0:.1f}s)", flush=True)

    feats = load_features()
    # cycle_pct_through is a per-date quantity; join on (station_code, price_date)
    # to be exact even though it is broadcast across stations.
    rp["price_date"] = pd.to_datetime(rp["price_date"])
    feats["price_date"] = pd.to_datetime(feats["price_date"])
    tag = feats[["station_code", "price_date", "cycle_pct_through"]]
    # many_to_one: features has one row per (station, date); fail loudly if not,
    # else a dup key would silently duplicate rows and inflate the regret metrics.
    rp = rp.merge(tag, on=["station_code", "price_date"], how="left", validate="many_to_one")
    missing = rp["cycle_pct_through"].isna().sum()
    if missing:
        print(f"[warn] {missing} rows lost the pct join — dropping", flush=True)
        rp = rp.dropna(subset=["cycle_pct_through"])
    rp["regime"] = rp["cycle_pct_through"].map(assign_regime)
    print(f"[join] tagged {rp.shape}  ({time.perf_counter()-t0:.1f}s)", flush=True)
    print(rp.groupby("regime").size().to_string(), flush=True)

    # Per (regime, seed): sweep tau, take peak expected_cents and the oracle ceiling.
    recs = []
    for regime, lo, hi in CYCLE_REGIME_BANDS:
        sub_r = rp[rp["regime"] == regime]
        for seed, sub in sub_r.groupby("seed"):
            y = sub["label"].to_numpy(dtype=int)
            p = sub["proba"].to_numpy(dtype=float)
            sweep = threshold_sweep(y, p)
            best = max(sweep, key=lambda r: r["expected_cents_per_row"])
            base_rate = float(y.mean())
            oracle = base_rate * _TP_REWARD_CENTS
            recs.append({
                "regime": regime,
                "seed": int(seed),
                "n": len(sub),
                "base_rate": base_rate,
                "peak_cents": best["expected_cents_per_row"],
                "tau": best["tau"],
                "buy_rate": best["buy_rate"],
                "oracle_cents": oracle,
                "regret": oracle - best["expected_cents_per_row"],
            })
    per_seed = pd.DataFrame(recs)

    summ = (
        per_seed.groupby("regime")
        .agg(
            n=("n", "first"),
            base_rate=("base_rate", "mean"),
            peak_cents_mean=("peak_cents", "mean"),
            peak_cents_std=("peak_cents", "std"),
            tau_med=("tau", "median"),
            buy_rate_mean=("buy_rate", "mean"),
            oracle_cents=("oracle_cents", "mean"),
            regret_mean=("regret", "mean"),
            regret_std=("regret", "std"),
        )
        .reindex([b[0] for b in CYCLE_REGIME_BANDS])
    )
    summ.to_csv(HERE / "proxy_regret_by_regime.csv")
    print("\n=== proxy regret by regime (mean over 5 seeds) ===", flush=True)
    print(summ.to_string(float_format=lambda x: f"{x:.4f}"), flush=True)

    # Plot: expected_cents_per_row vs tau, one line per regime (seed-pooled),
    # with oracle ceilings as dashed h-lines + peak markers.
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"normal": "#2c7fb8", "late_descent": "#d95f0e", "overdue": "#cb181d"}
    for regime, _, _ in CYCLE_REGIME_BANDS:
        sub = rp[rp["regime"] == regime]
        y = sub["label"].to_numpy(dtype=int)
        p = sub["proba"].to_numpy(dtype=float)
        sweep = threshold_sweep(y, p)
        taus = [r["tau"] for r in sweep]
        cents = [r["expected_cents_per_row"] for r in sweep]
        best = max(sweep, key=lambda r: r["expected_cents_per_row"])
        oracle = float(y.mean()) * _TP_REWARD_CENTS
        c = colors[regime]
        ax.plot(taus, cents, label=f"{regime} (n={len(sub):,})", color=c)
        ax.plot(best["tau"], best["expected_cents_per_row"], "o", color=c)
        ax.axhline(oracle, ls="--", lw=0.8, color=c, alpha=0.6)
    ax.set_xlabel("tau")
    ax.set_ylabel("expected_cents_per_row (proxy)")
    ax.set_title("Gate-1 pre-read: proxy economics by cycle regime\n(dashed = oracle ceiling; gap = regret)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "proxy_regret_by_regime.png", dpi=120)
    print(f"\n[plot] wrote proxy_regret_by_regime.png  ({time.perf_counter()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
