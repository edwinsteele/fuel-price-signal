"""Per-zone headroom under all six attribution conventions.

The figure's job is to show that the per-zone decomposition is NOT IDENTIFIED:
each zone's headroom moves further when you change a free bookkeeping convention
(cost basis x interval label -- neither has physical content) than the zones move
apart from each other. The grey bar is the #262 purchase-event estimator for
reference; it is the one that produced impossible negatives.

Usage:
  PYTHONPATH=. uv run python experiments/2026-08-20_headroom_attribution/plot_regime.py
"""
from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
ORDER = {"regime": ["normal", "late_descent", "overdue"],
         "volatility": ["<8c", "8-12c", "12-16c", ">=16c"]}
MARK = {("fifo", "start"): "o", ("fifo", "mid"): "s", ("fifo", "end"): "^",
        ("average", "start"): "o", ("average", "mid"): "s", ("average", "end"): "^"}
COL = {"fifo": "#4C72B0", "average": "#DD8452"}


def panel(ax, grid, old, axis, order):
    for i, z in enumerate(order):
        ax.bar(i, old.get(z, np.nan), 0.55, color="#dddddd", edgecolor="#999999",
               zorder=1, label="purchase-event (#262)" if i == 0 else None)
    g = grid[grid.axis == axis]
    for (cost, label), sub in g.groupby(["cost", "label"], observed=True):
        xs = [order.index(z) for z in sub.zone if z in order]
        ys = [v for z, v in zip(sub.zone, sub.headroom_cpl) if z in order]
        ax.scatter(xs, ys, marker=MARK[(cost, label)], s=64, zorder=3,
                   color=COL[cost], edgecolor="black", linewidth=0.5,
                   label=f"{cost}/{label}")
    # the span each zone covers across the six conventions
    for i, z in enumerate(order):
        v = g[g.zone == z]["headroom_cpl"]
        if len(v):
            ax.plot([i, i], [v.min(), v.max()], color="black", lw=1.2, zorder=2)
    ax.axhline(0, color="crimson", lw=1.2, ls="--")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_ylabel("headroom  model_cpl - oracle_cpl  (c/L)")
    ax.set_title(axis)
    ax.grid(axis="y", alpha=0.3)


def main() -> None:
    grid = pd.read_csv(HERE / "convention_grid.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))
    for ax, axis in zip(axes, ("regime", "volatility")):
        old = pd.read_csv(HERE / f"headroom_{axis}.csv", index_col="zone")["old_purchase_event"]
        panel(ax, grid, old, axis, ORDER[axis])
    axes[0].legend(fontsize=7.5, loc="upper left", ncol=2)
    fig.suptitle(
        "Per-zone headroom is not identified: each zone moves MORE across six free bookkeeping\n"
        "conventions (vertical bars) than the zones differ from each other. Dashed red = 0, which "
        "a perfect-foresight oracle can never cross.", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(HERE / "headroom_attribution.png", dpi=140)
    print("wrote", HERE / "headroom_attribution.png")


if __name__ == "__main__":
    main()
