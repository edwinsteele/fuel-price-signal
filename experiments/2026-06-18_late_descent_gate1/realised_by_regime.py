"""Gate-1 realised gate (#259): per-regime realised CPL via the #255 fill ledger.

The arbiter for the Layer-1 proxy pre-read (proxy_regret_by_regime.py). Runs the
production baseline through the paired realised-backtest harness with
collect_fills=True, tags every fill by the cycle regime AT ITS FILL DATE, and
compares the model's realised CPL to a regime-matched always-buy baseline.

Gate fires (realised) iff the model's saving% vs always-buy is materially LOWER
in late_descent than in overdue — i.e. the proxy skill-gap shows up in spend.

Path-dependency caveat: CPL is a tank-path metric, so a model fill tagged to
regime X can be cheap because of a wait decision made in regime Y. This is
"realised CPL conditional on filling during regime X" — a valid existence check,
not a clean causal attribution.

Heavy run (14 folds × inner OOF, one arm). Run:
  PYTHONPATH=. uv run python experiments/2026-06-18_late_descent_gate1/realised_by_regime.py
"""
from __future__ import annotations

import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.lib.realised import ArmSpec, run_paired_realised_backtest
from fuel_signal.features import (
    FEATURE_COLUMNS,
    LGA_FEATURE_COLUMNS,
    NETWORK_FEATURE_COLUMNS,
    load_features,
)

HERE = pathlib.Path(__file__).resolve().parent
FEATURES = FEATURE_COLUMNS + LGA_FEATURE_COLUMNS + NETWORK_FEATURE_COLUMNS  # 54, production

BANDS = [
    ("normal", 0.0, 0.6),
    ("late_descent", 0.6, 1.0),
    ("overdue", 1.0, np.inf),
]

# Inner OOF folds for the per-fold isotonic calibrator + τ pick. MUST be smaller
# than the outer train window: the harness runs this inside each outer fold's
# train, and outer fold 1's train is only ~1825d (the outer train_min_days), so
# the inner default (also 1825d) yields zero folds. A 3y inner min-train keeps the
# production 90d val/step granularity while fitting inside fold 1. The choice only
# moves the operating-point τ (applied uniformly across regimes) — the per-regime
# comparison is a post-hoc tag on the SAME fitted models, so it's unaffected.
INNER_FOLDS = {"train_min_days": 1095, "val_days": 90, "step_days": 90}


def _band(pct: float) -> str:
    for name, lo, hi in BANDS:
        if lo <= pct < hi:
            return name
    return "normal"


def _cpl(frame: pd.DataFrame) -> float:
    litres = frame["litres"].sum()
    return frame["spend_cents"].sum() / litres if litres > 0 else float("nan")


def main() -> None:
    t0 = time.perf_counter()
    df = load_features()
    df["price_date"] = pd.to_datetime(df["price_date"])

    res = run_paired_realised_backtest(
        [ArmSpec("baseline", df)], FEATURES, collect_fills=True, seed=42,
        inner_fold_params=INNER_FOLDS,
    )
    fills = res.fills.copy()
    print(f"\n[ledger] {len(fills)} fills "
          f"({(fills['arm'] == 'baseline').sum()} model / "
          f"{(fills['arm'] == 'always_buy').sum()} always-buy)", flush=True)

    # Tag each fill by the cycle regime at its fill date (per station_code, date).
    fills["date"] = pd.to_datetime(fills["date"])
    tag = df[["station_code", "price_date", "cycle_pct_through"]].rename(
        columns={"price_date": "date"}
    )
    # many_to_one: features has one row per (station, date) — guard against a dup
    # key silently duplicating fills (which would skew the per-regime CPL).
    fills = fills.merge(tag, on=["station_code", "date"], how="left", validate="many_to_one")
    # Persist the FULL merged ledger (pre-drop, NaN pct rows kept) so post-hoc
    # cleanup checks (chosen-only saving%, drop regime-correlation) run with no
    # re-fit. See cleanup_checks.py.
    fills.to_parquet(HERE / "realised_fills.parquet")
    print(f"[ledger] wrote realised_fills.parquet ({len(fills)} rows)", flush=True)
    # A NaN pct (fill date absent from the feature frame) would silently bucket
    # into 'normal' via _band — biasing the gate's deciding regime. Drop loudly.
    missing = fills["cycle_pct_through"].isna().sum()
    if missing:
        print(f"[warn] {missing}/{len(fills)} fills lost the pct join — dropping", flush=True)
        fills = fills.dropna(subset=["cycle_pct_through"])
    fills["regime"] = fills["cycle_pct_through"].map(_band)

    # Per regime: model CPL vs regime-matched always-buy CPL → saving%.
    recs = []
    for regime, _, _ in BANDS:
        sub = fills[fills["regime"] == regime]
        model = sub[sub["arm"] == "baseline"]
        always = sub[sub["arm"] == "always_buy"]
        model_cpl = _cpl(model)
        always_cpl = _cpl(always)
        saving = (
            (always_cpl - model_cpl) / always_cpl * 100
            if always_cpl > 0 and not np.isnan(model_cpl)
            else float("nan")
        )
        recs.append({
            "regime": regime,
            "model_fills": len(model),
            "model_litres": model["litres"].sum(),
            "model_cpl": model_cpl,
            "always_cpl": always_cpl,
            "saving_pct": saving,
            "model_emergency_frac": model["emergency"].mean() if len(model) else float("nan"),
        })
    summ = pd.DataFrame(recs).set_index("regime")
    summ.to_csv(HERE / "realised_by_regime.csv")
    print("\n=== realised saving% vs regime-matched always-buy ===", flush=True)
    print(summ.to_string(float_format=lambda x: f"{x:.3f}"), flush=True)

    # Aggregate (sanity vs the harness's own pooled CPL).
    print("\n=== harness aggregate (all regimes pooled) ===", flush=True)
    print(res.aggregate.to_string(index=False), flush=True)

    # Plot: saving% by regime.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"normal": "#2c7fb8", "late_descent": "#d95f0e", "overdue": "#cb181d"}
    bars = summ["saving_pct"]
    ax.bar(bars.index, bars.values, color=[colors[r] for r in bars.index])
    for r, v in bars.items():
        ax.text(r, v, f"{v:.2f}%", ha="center", va="bottom")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("realised saving% vs always-buy")
    ax.set_title(
        "Gate-1 realised: model advantage by cycle regime\n"
        "(low in late_descent = proxy skill-gap confirmed on spend)"
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "realised_by_regime.png", dpi=120)
    print(f"\n[plot] wrote realised_by_regime.png  ({time.perf_counter()-t0:.1f}s total)", flush=True)
    print(f"[wall] harness {res.meta['total_wall_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
