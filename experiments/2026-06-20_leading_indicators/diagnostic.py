import sqlite3
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fuel_signal.config import SYDNEY_METRO_COUNCILS

HERE = "experiments/2026-06-20_leading_indicators"

# --- pump: Sydney-metro E10 daily median, continuous daily index ---
con = sqlite3.connect("fuel_signal.db")
ph = ",".join("?" * len(SYDNEY_METRO_COUNCILS))
raw = pd.read_sql_query(
    f"""SELECT d.price_date AS d, d.price_decicents AS p
        FROM daily_prices d JOIN stations s ON s.station_code = d.station_code
        WHERE d.fuel_type_id = 1 AND s.council IN ({ph})""",
    con, params=list(SYDNEY_METRO_COUNCILS))
con.close()
raw["date"] = pd.to_datetime(raw["d"], format="%Y%m%d")
pump = raw.groupby("date")["p"].median().div(10)
pump = pump.asfreq("D").interpolate(limit=5)

# --- tgp ---
tgp = pd.read_excel(f"{HERE}/data/AIP_TGP_2026-06-19.xlsx", sheet_name="Petrol TGP")
dcol = tgp.columns[0]
tgp = tgp[[dcol, "Sydney"]].rename(columns={dcol: "date", "Sydney": "tgp"})
tgp["date"] = pd.to_datetime(tgp["date"], errors="coerce")
tgp = tgp.dropna(subset=["date"]).set_index("date")["tgp"]

df = pd.DataFrame({"e10": pump}).join(tgp).loc["2016-01-01":]
df["tgp"] = df["tgp"].ffill()
df = df.dropna(subset=["e10", "tgp"])
e = df["e10"].to_numpy()

# --- cycle peaks & troughs on the aggregate series (oracle, look-ahead) ---
troughs, _ = find_peaks(-e, prominence=10, distance=21)
peaks, _ = find_peaks(e, prominence=10, distance=21)
df["is_trough"] = False
df.iloc[troughs, df.columns.get_loc("is_trough")] = True

# next-trough lookup for every day
n = len(df)
next_tr_idx = np.full(n, -1)
j = 0
for i in range(n):
    while j < len(troughs) and troughs[j] < i:
        j += 1
    if j < len(troughs):
        next_tr_idx[i] = troughs[j]
# descent phase = after a peak, before the next trough
in_descent = np.zeros(n, dtype=bool)
for p_i in peaks:
    nt = next_tr_idx[p_i]
    if nt > p_i:
        in_descent[p_i:nt + 1] = True

idx = np.arange(n)
valid = (next_tr_idx >= 0)
depth_remaining = np.where(valid, e[np.clip(next_tr_idx, 0, n - 1)] * -1 + e, np.nan)
days_to_trough = np.where(valid, next_tr_idx - idx, np.nan)
gap = (df["e10"] - df["tgp"]).to_numpy()
slope7 = df["e10"].diff(7).to_numpy() / 7.0  # trailing c/day (negative in descent)

m = valid & in_descent & np.isfinite(slope7) & (slope7 < -0.1)
g, dep, dtt, sl = gap[m], depth_remaining[m], days_to_trough[m], slope7[m]
est_days = g / (-sl)

def r2(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    return np.corrcoef(x[ok], y[ok])[0, 1] ** 2

print(f"descent rows: {m.sum()}  (of {n} days, {len(troughs)} troughs, {len(peaks)} peaks)")
print(f"gap vs depth_remaining   r2 = {r2(g, dep):.3f}")
print(f"gap vs days_to_trough    r2 = {r2(g, dtt):.3f}")
print(f"est_days(gap/slope) vs days_to_trough  r2 = {r2(est_days, dtt):.3f}")

# trough "kiss" gap distribution
kg = (df.loc[df["is_trough"], "e10"] - df.loc[df["is_trough"], "tgp"])
print(f"\ntrough kiss gap (pump-TGP at trough) c/L: "
      f"mean {kg.mean():.1f}  sd {kg.std():.1f}  "
      f"p10 {kg.quantile(.1):.1f}  median {kg.median():.1f}  p90 {kg.quantile(.9):.1f}")
print(f"descent slope c/day: mean {sl.mean():.2f}  sd {sl.std():.2f}  "
      f"p10 {np.percentile(sl,10):.2f}  p90 {np.percentile(sl,90):.2f}")

# --- plots ---
fig, ax = plt.subplots(2, 2, figsize=(14, 11))
ax[0, 0].hexbin(g, dep, gridsize=40, cmap="Blues", mincnt=1)
lim = max(g.max(), dep.max())
ax[0, 0].plot([0, lim], [0, lim], "r--", lw=1, label="y=x")
ax[0, 0].set(xlabel="gap to TGP (pump-TGP, c/L)", ylabel="actual depth remaining (c/L)",
             title=f"Depth remaining vs gap  (r2={r2(g,dep):.2f})")
ax[0, 0].legend(fontsize=8)

ax[0, 1].hexbin(g, dtt, gridsize=40, cmap="Greens", mincnt=1)
ax[0, 1].set(xlabel="gap to TGP (c/L)", ylabel="actual days to trough",
             title=f"Days to trough vs gap  (r2={r2(g,dtt):.2f})")

ax[1, 0].hexbin(np.clip(est_days, 0, 80), np.clip(dtt, 0, 80), gridsize=40,
                cmap="Purples", mincnt=1)
ax[1, 0].plot([0, 60], [0, 60], "r--", lw=1, label="y=x")
ax[1, 0].set(xlabel="estimated days = gap / -slope7", ylabel="actual days to trough",
             title=f"Time projection vs actual  (r2={r2(est_days,dtt):.2f})")
ax[1, 0].legend(fontsize=8)

ax[1, 1].hist(kg, bins=30, color="tab:orange", alpha=0.8)
ax[1, 1].axvline(0, color="k", lw=1)
ax[1, 1].axvline(kg.mean(), color="r", lw=1, ls="--", label=f"mean {kg.mean():.1f}")
ax[1, 1].set(xlabel="pump - TGP at trough (the 'kiss' gap, c/L)", ylabel="cycles",
             title="Trough kiss-gap distribution")
ax[1, 1].legend(fontsize=8)

fig.tight_layout()
out = f"{HERE}/gap_to_tgp_diagnostic.png"
fig.savefig(out, dpi=110)
print("\nwrote", out)
