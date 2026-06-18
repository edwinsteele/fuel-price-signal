"""Gate-1 realised gate (#259): two post-hoc cleanup checks on the saved ledger.

Reads realised_fills.parquet (written by realised_by_regime.py) — NO re-fit.

Check 1 — chosen-only (~emergency) saving%: the headline saving% mixes the
model's volitional buys with forced tank-floor (emergency) fills. normal is ~88%
forced = an abstention zone, not skill. Restricting the model side to ~emergency
fills isolates decision skill. always_buy never abstains, so its ledger has no
emergency fills → it stays the clean blind denominator.

Check 2 — drop regime-correlation: ~9.5% of fills lost the (station,date) pct
join and were dropped from the gate. If those drops cluster in one regime, that
band's saving% is built on a thinner/biased sample. The dropped rows have no pct
by construction, so we assign an APPROXIMATE regime via a backward as-of join to
the nearest prior feature row, then compare the drop regime mix to the retained.

  PYTHONPATH=. uv run python experiments/2026-06-18_late_descent_gate1/cleanup_checks.py
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

from fuel_signal.features import load_features

HERE = pathlib.Path(__file__).resolve().parent
BANDS = [("normal", 0.0, 0.6), ("late_descent", 0.6, 1.0), ("overdue", 1.0, np.inf)]


def _band(pct: float) -> str:
    if pd.isna(pct):
        return "unmatched"
    for name, lo, hi in BANDS:
        if lo <= pct < hi:
            return name
    return "normal"


def _cpl(frame: pd.DataFrame) -> float:
    litres = frame["litres"].sum()
    return frame["spend_cents"].sum() / litres if litres > 0 else float("nan")


def main() -> None:
    fills = pd.read_parquet(HERE / "realised_fills.parquet")
    fills["date"] = pd.to_datetime(fills["date"])
    retained = fills[fills["cycle_pct_through"].notna()].copy()
    retained["regime"] = retained["cycle_pct_through"].map(_band)

    # ---- Check 1: chosen-only saving% ----------------------------------------
    print("=== Check 1: chosen-only (~emergency) saving% vs always-buy ===")
    recs = []
    for regime, _, _ in BANDS:
        sub = retained[retained["regime"] == regime]
        model = sub[sub["arm"] == "baseline"]
        always = sub[sub["arm"] == "always_buy"]
        chosen = model[~model["emergency"]]
        always_cpl = _cpl(always)

        def sav(frame: pd.DataFrame) -> float:
            c = _cpl(frame)
            return (always_cpl - c) / always_cpl * 100 if always_cpl > 0 and not np.isnan(c) else float("nan")

        recs.append({
            "regime": regime,
            "model_fills": len(model),
            "chosen_fills": len(chosen),
            "emergency_frac": model["emergency"].mean() if len(model) else float("nan"),
            "always_emerg_frac": always["emergency"].mean() if len(always) else float("nan"),
            "model_cpl_all": _cpl(model),
            "model_cpl_chosen": _cpl(chosen),
            "always_cpl": always_cpl,
            "saving_all_%": sav(model),
            "saving_chosen_%": sav(chosen),
        })
    summ = pd.DataFrame(recs).set_index("regime")
    summ.to_csv(HERE / "cleanup_chosen_only.csv")
    print(summ.to_string(float_format=lambda x: f"{x:.3f}"))

    # ---- Check 2: drop regime-correlation ------------------------------------
    print("\n=== Check 2: dropped-fill regime mix (approx, as-of backward join) ===")
    dropped = fills[fills["cycle_pct_through"].isna()].copy()
    df = load_features()
    df["price_date"] = pd.to_datetime(df["price_date"])
    tag = (
        df[["station_code", "price_date", "cycle_pct_through"]]
        .rename(columns={"price_date": "date"})
        .sort_values("date")
    )
    dropped = dropped.sort_values("date")
    dropped["approx_pct"] = pd.merge_asof(
        dropped[["date", "station_code"]], tag,
        on="date", by="station_code", direction="backward",
    )["cycle_pct_through"].to_numpy()
    dropped["regime"] = dropped["approx_pct"].map(_band)

    # Model arm only (the gate-relevant ledger); compare drop mix vs retained mix.
    drop_model = dropped[dropped["arm"] == "baseline"]
    ret_model = retained[retained["arm"] == "baseline"]
    print(f"model fills: {len(ret_model)} retained, {len(drop_model)} dropped "
          f"({len(drop_model) / (len(ret_model) + len(drop_model)) * 100:.1f}%)")
    mix = pd.DataFrame({
        "retained_%": ret_model["regime"].value_counts(normalize=True) * 100,
        "dropped_%": drop_model["regime"].value_counts(normalize=True) * 100,
        "dropped_n": drop_model["regime"].value_counts(),
    }).fillna(0.0)
    mix.to_csv(HERE / "cleanup_drop_mix.csv")
    print(mix.to_string(float_format=lambda x: f"{x:.2f}"))
    unmatched = (drop_model["regime"] == "unmatched").sum()
    print(f"\ndrops with no prior feature row even via as-of: {unmatched} "
          f"({unmatched / len(drop_model) * 100:.1f}% of model drops)")
    print("\nyear distribution of model drops:")
    print(drop_model["date"].dt.year.value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
