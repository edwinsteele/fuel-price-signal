import sqlite3
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fuel_signal.config import SYDNEY_METRO_COUNCILS

HERE = "experiments/2026-06-20_leading_indicators"

# --- pump side: Sydney-metro E10 daily median ---
con = sqlite3.connect("fuel_signal.db")
ph = ",".join("?" * len(SYDNEY_METRO_COUNCILS))
q = f"""
SELECT d.price_date AS d, d.price_decicents AS p
FROM daily_prices d JOIN stations s ON s.station_code = d.station_code
WHERE d.fuel_type_id = 1 AND s.council IN ({ph})
"""
raw = pd.read_sql_query(q, con, params=list(SYDNEY_METRO_COUNCILS))
con.close()
raw["date"] = pd.to_datetime(raw["d"], format="%Y%m%d")
pump = raw.groupby("date")["p"].median().div(10).rename("e10")  # decicents -> c/L

# --- wholesale side: AIP Sydney ULP TGP ---
tgp = pd.read_excel(f"{HERE}/data/AIP_TGP_2026-06-19.xlsx", sheet_name="Petrol TGP")
dcol = tgp.columns[0]
tgp = tgp[[dcol, "Sydney"]].rename(columns={dcol: "date", "Sydney": "tgp"})
tgp["date"] = pd.to_datetime(tgp["date"], errors="coerce")
tgp = tgp.dropna(subset=["date"]).set_index("date")["tgp"]

# --- align on common daily window from 2016 ---
df = pd.concat([pump, tgp], axis=1).loc["2016-01-01":]
df["tgp"] = df["tgp"].ffill()  # TGP is weekday-only; carry over weekends
df = df.dropna(subset=["e10"])
print("aligned rows:", len(df), "|", df.index.min().date(), "->", df.index.max().date())
print("E10  c/L:", round(df.e10.min(), 1), "->", round(df.e10.max(), 1))
print("TGP  c/L:", round(df.tgp.min(), 1), "->", round(df.tgp.max(), 1))
print("retail margin (E10-TGP) c/L: mean", round((df.e10 - df.tgp).mean(), 1),
      "min", round((df.e10 - df.tgp).min(), 1), "max", round((df.e10 - df.tgp).max(), 1))

# --- plot: full window + two shock zooms ---
fig, axes = plt.subplots(3, 1, figsize=(15, 13))
windows = [("2016-01-01", "2026-06-19", "Full window 2016-2026"),
           ("2020-01-01", "2020-12-31", "2020 COVID crash"),
           ("2022-01-01", "2022-12-31", "2022 Ukraine spike")]
for ax, (a, b, title) in zip(axes, windows):
    seg = df.loc[a:b]
    ax.plot(seg.index, seg.e10, lw=0.8, color="tab:blue", label="Sydney E10 daily median (pump)")
    ax.plot(seg.index, seg.tgp, lw=1.2, color="tab:red", label="Sydney ULP TGP (wholesale)")
    ax.fill_between(seg.index, seg.tgp, seg.e10, where=(seg.e10 >= seg.tgp),
                    color="tab:blue", alpha=0.08)
    ax.set_title(title)
    ax.set_ylabel("c/L (GST incl.)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
fig.tight_layout()
out = f"{HERE}/tgp_vs_e10_overlay.png"
fig.savefig(out, dpi=110)
print("wrote", out)

# --- calm (non-shock) period: Jan 2023 - Jan 2024 ---
seg = df.loc["2023-01-01":"2024-01-31"]
fig2, ax = plt.subplots(figsize=(15, 5))
ax.plot(seg.index, seg.e10, lw=1.0, color="tab:blue", label="Sydney E10 daily median (pump)")
ax.plot(seg.index, seg.tgp, lw=1.4, color="tab:red", label="Sydney ULP TGP (wholesale)")
ax.fill_between(seg.index, seg.tgp, seg.e10, where=(seg.e10 >= seg.tgp),
                color="tab:blue", alpha=0.08)
ax.set_title("Calm cycling: Jan 2023 - Jan 2024")
ax.set_ylabel("c/L (GST incl.)")
ax.legend(loc="upper left", fontsize=8)
ax.grid(alpha=0.3)
fig2.tight_layout()
out2 = f"{HERE}/tgp_vs_e10_calm_2023.png"
fig2.savefig(out2, dpi=110)
print("wrote", out2)
margin = (seg.e10 - seg.tgp)
print("2023 calm: margin c/L mean", round(margin.mean(), 1),
      "min", round(margin.min(), 1), "max", round(margin.max(), 1),
      "| TGP range", round(seg.tgp.min(), 1), "->", round(seg.tgp.max(), 1))
