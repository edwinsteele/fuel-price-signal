import numpy as np
import pandas as pd
from fuel_signal.score_phase2 import threshold_sweep

df = pd.read_parquet("experiments/2026-06-16_regime_cycle_length/rowpreds.parquet")
taus = np.round(np.arange(0.02, 0.601, 0.02), 3)
shock = {1, 4, 9, 13}

# seed-average proba per row, per run
g = (df.groupby(["run", "fold", "station_code", "price_date"], observed=True)
       .agg(proba=("proba", "mean"), label=("label", "first")).reset_index())

print(f"{'fold':>4} {'reg':>6} {'R0 τ*':>6} {'R1 τ*':>6} {'R0 pk':>7} {'R1 pk':>7} "
      f"{'Δcpr':>8} {'disagree@.21':>12}")
for fold in sorted(g.fold.unique()):
    reg = "shock" if fold in shock else ("ELONG" if fold == 7 else "norm")
    row = {}
    for run in ["R0", "R1"]:
        sub = g[(g.run == run) & (g.fold == fold)]
        sweep = pd.DataFrame(threshold_sweep(sub.label.to_numpy(), sub.proba.to_numpy(), taus=taus))
        star = sweep.loc[sweep.expected_cents_per_row.idxmax()]
        row[run] = (sub, star.tau, star.expected_cents_per_row)
    # decision disagreement at a common tau
    s0 = g[(g.run == "R0") & (g.fold == fold)].set_index(["station_code", "price_date"]).proba
    s1 = g[(g.run == "R1") & (g.fold == fold)].set_index(["station_code", "price_date"]).proba
    j = pd.concat([s0.rename("p0"), s1.rename("p1")], axis=1).dropna()
    disagree = float(((j.p0 >= 0.21) != (j.p1 >= 0.21)).mean())
    dcpr = row["R1"][2] - row["R0"][2]
    print(f"{fold:>4} {reg:>6} {row['R0'][1]:>6.2f} {row['R1'][1]:>6.2f} "
          f"{row['R0'][2]:>7.4f} {row['R1'][2]:>7.4f} {dcpr:>+8.4f} {disagree:>11.1%}")
