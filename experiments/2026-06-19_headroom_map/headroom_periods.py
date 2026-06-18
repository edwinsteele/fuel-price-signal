"""Post-hoc headroom slices (#262), no re-fit — reads the saved fill ledgers.

`headroom_map.py` writes `model_fills.parquet` + `oracle_fills.parquet` (each fill
tagged by date with cycle_pct_through / network_px_std). This script re-derives the
two ad-hoc slices discussed in the README without re-running the 574s harness:

  1. headroom by FIXED network_px_std cent-bands (the hump, vs the production
     tercile cut which smeared it into a monotonic ramp);
  2. top headroom PERIODS (per calendar month), to localise episodes for feature
     work — writes headroom_periods.csv.

Reminder (README § noise floor): per-zone headroom can go negative — that is an
attribution artifact of slicing a path-coupled total by fill date (the arms fill on
different dates AND counts), NOT the oracle losing. Trust the contiguous / large
buckets; monthly noise floor ≈±3 c/L. See #265 for the tighter attribution.

Run:
  PYTHONPATH=. uv run python experiments/2026-06-19_headroom_map/headroom_periods.py
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
VOL_BANDS = [0, 8, 12, 16, np.inf]
VOL_LABELS = ["<8c", "8-12c", "12-16c", ">=16c"]


def _cpl(frame: pd.DataFrame) -> float:
    litres = frame["litres"].sum()
    return frame["spend_cents"].sum() / litres if litres > 0 else float("nan")


def main() -> None:
    m = pd.read_parquet(HERE / "model_fills.parquet")
    o = pd.read_parquet(HERE / "oracle_fills.parquet")

    print(f"\n[overall] model {_cpl(m):.2f}  oracle {_cpl(o):.2f}  "
          f"headroom {_cpl(m) - _cpl(o):.2f} c/L")

    # 1. Fixed cent-band volatility cut.
    rows = []
    for f in (m, o):
        f["band"] = pd.cut(f["network_px_std"], bins=VOL_BANDS, labels=VOL_LABELS)
    for lab in VOL_LABELS:
        mm, oo = m[m["band"] == lab], o[o["band"] == lab]
        rows.append({"band": lab, "model_fills": len(mm), "oracle_fills": len(oo),
                     "model_cpl": _cpl(mm), "oracle_cpl": _cpl(oo),
                     "headroom_cpl": _cpl(mm) - _cpl(oo)})
    print("\n=== headroom by fixed network_px_std band ===")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # 2. Top headroom periods (per month).
    for f in (m, o):
        f["month"] = pd.to_datetime(f["date"]).dt.to_period("M").astype(str)
    rows = []
    for ym in sorted(set(m["month"]) | set(o["month"])):
        mm, oo = m[m["month"] == ym], o[o["month"] == ym]
        rows.append({"month": ym, "model_fills": len(mm),
                     "mean_vol_c": mm["network_px_std"].mean(),
                     "model_cpl": _cpl(mm), "oracle_cpl": _cpl(oo),
                     "headroom_cpl": _cpl(mm) - _cpl(oo)})
    periods = pd.DataFrame(rows)
    periods.to_csv(HERE / "headroom_periods.csv", index=False)
    print("\n=== top 12 months by headroom (mean_vol_c = mean network_px_std) ===")
    print(periods.sort_values("headroom_cpl", ascending=False).head(12)
          .to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"\n[wrote] {HERE / 'headroom_periods.csv'} ({len(periods)} months)")


if __name__ == "__main__":
    main()
