import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fuel_signal.score_phase2 import threshold_sweep

df = pd.read_parquet("experiments/2026-06-16_regime_cycle_length/rowpreds.parquet")
taus = np.round(np.arange(0.01, 0.601, 0.01), 4)

curves = {}
for run, label in [("R0", "stale (expanding mean)"), ("R1", "honest (regime median)")]:
    sub = df[df.run == run]
    g = sub.groupby(["fold", "station_code", "price_date"], observed=True).agg(
        proba=("proba", "mean"), label=("label", "first")
    )
    rows = threshold_sweep(g.label.to_numpy(), g.proba.to_numpy(), taus=taus)
    curves[run] = (label, pd.DataFrame(rows))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
for run, color in [("R0", "#888"), ("R1", "#c0392b")]:
    lab, c = curves[run]
    ax1.plot(c.tau, c.buy_rate, color=color, label=lab)
    ax2.plot(c.tau, c.expected_cents_per_row, color=color, label=lab)
    star = c.loc[c.expected_cents_per_row.idxmax()]
    ax2.scatter([star.tau], [star.expected_cents_per_row], color=color, zorder=5)
    ax2.annotate(f"τ*={star.tau:.2f}", (star.tau, star.expected_cents_per_row),
                 textcoords="offset points", xytext=(5, -10), color=color)

for ax, t in [(ax1, "Buy-rate vs τ"), (ax2, "Proxy economics (c/row) vs τ")]:
    ax.axvline(0.20, ls=":", c="#c0392b", alpha=0.5)
    ax.axvline(0.25, ls=":", c="#888", alpha=0.5)
    ax.set_xlabel("τ"); ax.set_title(t); ax.legend()
fig.tight_layout()
fig.savefig("experiments/2026-06-16_regime_cycle_length/tau_coordinate.png", dpi=110)

print("=== buy-rate at the two operating points ===")
for run in ["R0", "R1"]:
    lab, c = curves[run]
    br = {t: float(c.loc[c.tau == t, "buy_rate"].iloc[0]) for t in (0.20, 0.25)}
    star = c.loc[c.expected_cents_per_row.idxmax()]
    print(f"{run} {lab:26s} buy@0.20={br[0.20]:.3f} buy@0.25={br[0.25]:.3f} "
          f"| τ*={star.tau:.2f} buy@τ*={star.buy_rate:.3f} c/row*={star.expected_cents_per_row:.4f}")

# does honest@0.20 match stale@0.25 in buy-rate? (coordinate-shift test)
r0 = curves["R0"][1]; r1 = curves["R1"][1]
stale_25 = float(r0.loc[r0.tau == 0.25, "buy_rate"].iloc[0])
match = r1.iloc[(r1.buy_rate - stale_25).abs().idxmin()]
print(f"\nstale buys {stale_25:.3f} at τ=0.25; honest reaches that buy-rate at τ={match.tau:.2f}")
