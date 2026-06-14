"""EDA (#237 side-check): does the expanding-window cycle_mean_length lag the
true cycle-length regime, and how badly does the dsp-reset reconstruction (used
by experiments/lib/features/cycle_shape.py) over-segment vs find_peaks?

True per-cycle length via the heatmap recipe: scipy.signal.find_peaks
(distance>=7, prominence>=1c) on the metro-average daily E10 series. No fitting.
"""
from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks

from fuel_signal.features import load_features

OUT = pathlib.Path(__file__).parent

df = load_features()
df["price_date"] = pd.to_datetime(df["price_date"])
pd_ = (
    df.drop_duplicates("price_date")
    .assign(sydney_avg=lambda d: d["station_price_cents"] - d["station_minus_sydney_avg_cents"])
    .loc[:, ["price_date", "sydney_avg", "cycle_days_since_peak", "cycle_mean_length"]]
    .sort_values("price_date")
    .reset_index(drop=True)
)

# --- True cycles via find_peaks (heatmap recipe) ---
s = pd_.set_index("price_date")["sydney_avg"].asfreq("D").interpolate()
peaks, _ = find_peaks(s.values, distance=7, prominence=1.0)
peak_dates = s.index[peaks]
true_len = peak_dates.to_series().diff().dt.days.dropna()
true_cyc = pd.DataFrame({"start_date": peak_dates[1:], "length_days": true_len.values})
true_cyc = true_cyc[(true_cyc["length_days"] >= 7) & (true_cyc["length_days"] <= 90)]
true_cyc["trailing8_median"] = true_cyc["length_days"].rolling(8, min_periods=3).median()

# --- dsp-reset reconstruction (what cycle_shape.py does) ---
dsp = pd_["cycle_days_since_peak"]
start = (dsp < dsp.shift(1) - 1).fillna(False)
start.iloc[0] = True
pd_["cycle_id"] = start.cumsum()
dsp_cyc = pd_.groupby("cycle_id").agg(
    start_date=("price_date", "min"), length_days=("price_date", "size"),
).reset_index()
dsp_cyc = dsp_cyc[(dsp_cyc["length_days"] >= 7) & (dsp_cyc["length_days"] <= 90)]

fold1 = pd.Timestamp("2021-11-05")
folds = [fold1 + pd.Timedelta(days=90 * i) for i in range(16)]
folds = [f for f in folds if f <= pd_["price_date"].max()]

fig, ax = plt.subplots(figsize=(13, 6))
ax.scatter(true_cyc["start_date"], true_cyc["length_days"], s=30, color="#34495e",
           alpha=0.7, label="true cycle length (find_peaks, peak→peak)")
ax.plot(true_cyc["start_date"], true_cyc["trailing8_median"], color="#16a085", lw=2.0,
        label="trailing-8-cycle median (regime-local baseline)")
ax.plot(pd_["price_date"], pd_["cycle_mean_length"], color="#c0392b", lw=2.0,
        label="cycle_mean_length (feature — expanding all-history mean)")
ax.axvline(pd.Timestamp("2020-03-01"), color="grey", ls=":", lw=1.2, alpha=0.7)
ax.text(pd.Timestamp("2020-03-01"), 5, " COVID", color="grey", fontsize=9)
for f in folds:
    ax.axvline(f, color="#2980b9", ls="--", lw=0.5, alpha=0.3)
ax.text(folds[0], 72, " fold val windows →", color="#2980b9", fontsize=8)
ax.set_xlabel("date"); ax.set_ylabel("cycle length (days)")
ax.set_title("Cycle-length regime vs the stale feature — #237 phase-axis contamination check")
ax.grid(alpha=0.3); ax.legend(loc="upper left", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "regime_cycle_length.png", dpi=120)

# --- Reports ---
print("Cycles per year (find_peaks vs dsp-reset reconstruction):")
ty = true_cyc.set_index("start_date")["length_days"].groupby(lambda d: d.year).count()
dy = dsp_cyc.set_index("start_date")["length_days"].groupby(lambda d: d.year).count()
for yr in sorted(set(ty.index) | set(dy.index)):
    print(f"  {yr}: find_peaks={ty.get(yr, 0):>3}   dsp-reset={dy.get(yr, 0):>3}")

print("\ncycle_mean_length feature trajectory (expanding mean):")
for d in ["2018-06-30", "2021-11-05", "2023-06-01", "2026-05-30"]:
    v = pd_.loc[pd_["price_date"] <= d, "cycle_mean_length"].iloc[-1]
    print(f"  {d}: {v:.1f}d")

val_true = true_cyc[true_cyc["start_date"] >= fold1]["length_days"].median()
val_feat = pd_[pd_["price_date"] >= fold1]["cycle_mean_length"].median()
print(f"\nValidation era ({fold1.date()}+):")
print(f"  median TRUE length (find_peaks): {val_true:.1f}d")
print(f"  median cycle_mean_length       : {val_feat:.1f}d")
print(f"  feature under-reports by       : {val_true - val_feat:.1f}d "
      f"({100 * (val_true - val_feat) / val_true:.0f}%)")
print(f"  => pct_through inflated by ~    : {val_true / val_feat:.2f}x")
print(f"\nWrote {OUT / 'regime_cycle_length.png'}")
