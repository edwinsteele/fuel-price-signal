"""Why the forced-fill constraint is cadence-conditional (fps-fii teaching figure).

Left:  the tank-level ladder each cadence affords. Depletion per decide point is
       D = daily x interval, so a coarser grid means fewer rungs between "full"
       and the emergency floor — i.e. fewer chances to say "wait".
Right: what that does to the ledger. The FORCED fills barely move (they are a
       function of distance driven, which is fixed); the chosen fills stack up
       on top of them.

  PYTHONPATH=. uv run python experiments/2026-08-20_cadence_ceiling/plot_decision_freedom.py
"""
from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import pandas as pd

from fuel_signal.backtest import TankParams

HERE = pathlib.Path(__file__).resolve().parent
CADENCES = (7, 2, 1)


def main() -> None:
    t = TankParams()
    size, floor = t.tank_size_litres, t.floor_fraction * t.tank_size_litres
    s = pd.read_csv(HERE / "model_cadence.csv").set_index("cadence_days")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    # --- left: the level ladder ---------------------------------------------
    for x, c in enumerate(CADENCES):
        D = t.daily_consumption_litres * c
        lv, rungs = size, []
        while lv > -1e-9:
            rungs.append(lv)
            lv -= D
        free = [r for r in rungs if r - D > floor]      # rungs from which "wait" is still a choice
        ax1.bar(x, size, width=0.5, color="#f0f0f0", edgecolor="#bbb", zorder=1)
        ax1.bar(x, floor, width=0.5, color="#f4c7c3", edgecolor="#d95f5f", zorder=2)
        for r in rungs:
            ax1.plot([x - 0.25, x + 0.25], [r, r],
                     color="#2b7bba" if r in free else "#d95f0e", lw=2.5, zorder=3)
        ax1.text(x, size + 2.5, f"{len(free)} wait{'s' if len(free) != 1 else ''}",
                 ha="center", fontsize=11, weight="bold",
                 color="#2b7bba" if len(free) > 1 else "#d95f0e")
    ax1.set_xticks(range(len(CADENCES)))
    ax1.set_xticklabels([f"{c}-day\nD={t.daily_consumption_litres * c:.2f} L" for c in CADENCES])
    ax1.set_ylabel("tank level (L)")
    ax1.set_ylim(0, size + 7)
    ax1.set_title("Rungs the tank affords between full and the floor\n"
                  "(blue = you may still choose to wait; orange = the tank decides)",
                  fontsize=10)
    ax1.axhline(floor, color="#d95f5f", ls="--", lw=1, zorder=4)
    ax1.text(-0.45, floor + 1, "emergency floor (10%)", fontsize=8, color="#d95f5f")

    # --- right: chosen vs forced fills --------------------------------------
    forced = [s.loc[c, "model_fills"] * s.loc[c, "model_emergency_pct"] / 100 for c in CADENCES]
    chosen = [s.loc[c, "model_fills"] - f for c, f in zip(CADENCES, forced, strict=True)]
    xs = range(len(CADENCES))
    ax2.bar(xs, forced, width=0.5, color="#d95f0e", label="forced (emergency)")
    ax2.bar(xs, chosen, width=0.5, bottom=forced, color="#2b7bba",
            label="chosen — what a feature can act on")
    for x, (f, ch) in enumerate(zip(forced, chosen, strict=True)):
        ax2.text(x, f / 2, f"{f:.0f}", ha="center", va="center", color="white",
                 fontsize=10, weight="bold")
        ax2.text(x, f + ch / 2, f"{ch:.0f}", ha="center", va="center", color="white",
                 fontsize=11, weight="bold")
    ax2.set_xticks(list(xs))
    ax2.set_xticklabels([f"{c}-day" for c in CADENCES])
    ax2.set_ylabel("model fills (14 folds x 5 stations)")
    ax2.set_title("Forced fills barely move — discretionary decisions stack on top",
                  fontsize=10)
    ax2.legend(fontsize=9, loc="upper left")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("The tank forces the same number of fills however often you check — "
                 "checking more often only adds decisions (bd fps-fii)", fontsize=12)
    fig.tight_layout()
    fig.savefig(HERE / "decision_freedom.png", dpi=130)
    print(f"wrote {HERE / 'decision_freedom.png'}", flush=True)


if __name__ == "__main__":
    main()
