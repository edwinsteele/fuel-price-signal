"""Economic headroom map (#262): model CPL vs perfect-foresight oracle CPL by zone.

#259 disproved the "late descent is a decision-skill soft spot" story: on chosen
(~emergency) fills the model pays a flat ~175 c/L in every cycle regime, and the
headline regime gradient was an artifact of emergency dilution + a regime-varying
always-buy denominator. always-buy is the wrong yardstick.

The right one is a perfect-foresight oracle ceiling. headroom = model_cpl −
oracle_cpl is recoverable economic opportunity; this map shows WHERE (if anywhere)
it concentrates, agnostic to the cycle-regime axis (the story just disproved).

LEAKY-CEILING, NECESSARY-NOT-SUFFICIENT CAVEAT: the oracle sees the future, so a
gap proves money EXISTS in a zone (necessary) — not that a PIT-safe feature can
CAPTURE it (sufficient). Flat-bottom troughs decouple log-loss from CPL, so any
feature chasing a hot zone must still clear the realised arbiter (WFCV log-loss is
only a screen). A tie in a zone is a hard "stop digging" signal; a gap is only
permission to keep looking.

Two passes, same 14-fold walk-forward windows / stations / seed / tank:
  - MODEL: the production 54-feat baseline through the #255 realised harness with
    collect_fills=True (identical to the #259 gate-1 run) → model + always-buy fills.
  - ORACLE: run_oracle_backtest over each fold's val window (fit-free, fast).
Both fill ledgers are tagged by zone at each fill's own date and pooled within
zone. CPL is a tank-path metric → this is "CPL conditional on filling in zone X",
a valid existence check, not clean causal attribution (the #259 caveat).

Heavy: the model pass retrains per fold (~520s). Oracle adds little. Run:
  PYTHONPATH=. uv run python experiments/2026-06-19_headroom_map/headroom_map.py
"""
from __future__ import annotations

import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.lib.realised import ArmSpec, run_paired_realised_backtest
from fuel_signal import db as _db
from fuel_signal.backtest import TankParams, load_history, run_oracle_backtest
from fuel_signal.config import PREFERRED_STATIONS
from fuel_signal.features import (
    FEATURE_COLUMNS,
    LGA_FEATURE_COLUMNS,
    NETWORK_FEATURE_COLUMNS,
    load_features,
)

HERE = pathlib.Path(__file__).resolve().parent
FEATURES = FEATURE_COLUMNS + LGA_FEATURE_COLUMNS + NETWORK_FEATURE_COLUMNS  # 54, production

# Same inner-OOF gotcha as the gate-1 run: fold 1's outer train is ~1825d, so the
# inner default (also 1825d) yields 0 folds. A 3y inner min-train fits inside it;
# only moves τ uniformly (the per-zone tag is post-hoc on the same fitted models).
INNER_FOLDS = {"train_min_days": 1095, "val_days": 90, "step_days": 90}

REGIME_BANDS = [("normal", 0.0, 0.6), ("late_descent", 0.6, 1.0), ("overdue", 1.0, np.inf)]


def _regime(pct: float) -> str:
    for name, lo, hi in REGIME_BANDS:
        if lo <= pct < hi:
            return name
    return "normal"


def _cpl(frame: pd.DataFrame) -> float:
    litres = frame["litres"].sum()
    return frame["spend_cents"].sum() / litres if litres > 0 else float("nan")


def _headroom_by_axis(model: pd.DataFrame, oracle: pd.DataFrame, axis: str) -> pd.DataFrame:
    """model_cpl − oracle_cpl per zone of `axis`, with fill/litre counts per arm."""
    recs = []
    zones = sorted(set(model[axis].dropna().unique()) | set(oracle[axis].dropna().unique()),
                   key=str)
    for z in zones:
        m = model[model[axis] == z]
        o = oracle[oracle[axis] == z]
        m_cpl, o_cpl = _cpl(m), _cpl(o)
        recs.append({
            "axis": axis, "zone": z,
            "model_fills": len(m), "model_litres": m["litres"].sum(), "model_cpl": m_cpl,
            "oracle_fills": len(o), "oracle_litres": o["litres"].sum(), "oracle_cpl": o_cpl,
            "headroom_cpl": m_cpl - o_cpl,
        })
    return pd.DataFrame(recs)


def _date_zone_lookup(feat: pd.DataFrame) -> pd.DataFrame:
    """Per-DATE cycle_pct_through + network_px_std (both network-wide → identical
    across stations on a date), so fills tag by date alone. Tagging on (station,
    date) instead would spuriously miss ~20% of fills: a fill lands on the eval
    grid and the engine carries the last price forward, but features.csv has no
    row for a station on a date it didn't report — even though the network-wide
    cycle/volatility values exist. Date-only keying drops only dates absent for
    the whole network (label-horizon trim at the window tail)."""
    return (
        feat.groupby("price_date")[["cycle_pct_through", "network_px_std"]]
        .first()
        .reset_index()
        .rename(columns={"price_date": "date"})
    )


def _tag(fills: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    """Attach zone columns (regime / quarter / volatility tercile) at each fill date."""
    fills = fills.copy()
    fills["date"] = pd.to_datetime(fills["date"])
    fills = fills.merge(lookup, on="date", how="left", validate="many_to_one")
    missing = fills["cycle_pct_through"].isna().sum()
    if missing:
        print(f"[warn] {missing}/{len(fills)} fills lost the date join — dropping", flush=True)
        fills = fills.dropna(subset=["cycle_pct_through"])
    fills["regime"] = fills["cycle_pct_through"].map(_regime)
    fills["quarter"] = "Q" + fills["date"].dt.quarter.astype(str)
    return fills


def main() -> None:
    t0 = time.perf_counter()
    df = load_features()
    df["price_date"] = pd.to_datetime(df["price_date"])
    station_codes = list(PREFERRED_STATIONS)
    tank = TankParams()

    # --- Pass 1: model + always-buy fills via the #255 realised harness ---------
    res = run_paired_realised_backtest(
        [ArmSpec("baseline", df)], FEATURES, collect_fills=True, seed=42,
        inner_fold_params=INNER_FOLDS, tank=tank,
    )
    model_fills = res.fills[res.fills["arm"] == "baseline"].copy()
    windows = res.per_window[["fold", "val_start", "val_end"]].drop_duplicates("fold")
    print(f"\n[model] {len(model_fills)} model fills over {len(windows)} folds", flush=True)

    # --- Pass 2: oracle over the SAME fold val windows (fit-free) ---------------
    conn = _db.open_db()
    try:
        history = load_history(conn, station_codes)  # oracle only needs station prices
    finally:
        conn.close()
    oracle_rows = []
    for _, w in windows.iterrows():
        for sc in station_codes:
            r = run_oracle_backtest(
                history, sc, w["val_start"], w["val_end"], tank, collect_fills=True
            )
            for fr in r.fills:
                oracle_rows.append({
                    "fold": w["fold"], "date": fr.date, "station_code": fr.station_code,
                    "price": fr.price, "litres": fr.litres, "spend_cents": fr.spend_cents,
                    "emergency": fr.emergency,
                })
    oracle_fills = pd.DataFrame(oracle_rows)
    print(f"[oracle] {len(oracle_fills)} oracle fills "
          f"({oracle_fills['emergency'].mean():.2%} emergency)", flush=True)

    # --- Tag both ledgers by zone and persist (post-hoc, no re-fit) -------------
    lookup = _date_zone_lookup(df)
    model_fills = _tag(model_fills, lookup)
    oracle_fills = _tag(oracle_fills, lookup)
    # Shared volatility terciles over the union of fill network_px_std values.
    vol = pd.concat([model_fills["network_px_std"], oracle_fills["network_px_std"]])
    edges = np.array(vol.quantile([0.0, 1 / 3, 2 / 3, 1.0]))
    edges[0], edges[-1] = -np.inf, np.inf
    labels = ["vol_low", "vol_mid", "vol_high"]
    for f in (model_fills, oracle_fills):
        f["volatility"] = pd.cut(f["network_px_std"], bins=edges, labels=labels)
    model_fills.to_parquet(HERE / "model_fills.parquet")
    oracle_fills.to_parquet(HERE / "oracle_fills.parquet")

    # --- Headroom per axis ------------------------------------------------------
    print(f"\n[overall] model_cpl={_cpl(model_fills):.2f}  oracle_cpl={_cpl(oracle_fills):.2f}  "
          f"headroom={_cpl(model_fills) - _cpl(oracle_fills):.2f}", flush=True)
    maps = {ax: _headroom_by_axis(model_fills, oracle_fills, ax)
            for ax in ("regime", "quarter", "volatility", "fold")}
    out = pd.concat(maps.values(), ignore_index=True)
    out.to_csv(HERE / "headroom_map.csv", index=False)
    for ax, m in maps.items():
        print(f"\n=== headroom by {ax} (model_cpl − oracle_cpl) ===", flush=True)
        print(m.to_string(index=False, float_format=lambda x: f"{x:.2f}"), flush=True)

    # --- Plot: headroom by each axis -------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax_obj, ax_name in zip(axes, ("regime", "quarter", "volatility"), strict=True):
        m = maps[ax_name].set_index("zone")["headroom_cpl"]
        ax_obj.bar(m.index.astype(str), m.values, color="#d95f0e")
        for x, v in zip(m.index.astype(str), m.values, strict=True):
            ax_obj.text(x, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        ax_obj.set_title(f"by {ax_name}")
        ax_obj.set_ylabel("headroom c/L (model − oracle)")
        ax_obj.grid(axis="y", alpha=0.3)
        ax_obj.tick_params(axis="x", labelrotation=30)
    fig.suptitle("Economic headroom = recoverable c/L (leaky upper bound; necessary not sufficient)")
    fig.tight_layout()
    fig.savefig(HERE / "headroom_map.png", dpi=120)
    print(f"\n[done] {time.perf_counter() - t0:.1f}s total "
          f"(harness {res.meta['total_wall_seconds']:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
