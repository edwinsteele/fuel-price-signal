"""Plot realised CPL vs tau at each cadence, against that cadence's oracle floor.

The point the plot has to make: the vertical distance between a cadence's curve and
its own oracle line is headroom, and moving ALONG a curve barely changes it, while
moving BETWEEN curves changes it a lot. tau is the flat axis; cadence is not.

  PYTHONPATH=. uv run python experiments/2026-08-21_tau_cadence/plot_tau_sweep.py
"""
from __future__ import annotations

import pathlib

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set before pyplot)

HERE = pathlib.Path(__file__).resolve().parent
PROD_TAU = 0.25

COLOURS = {7: "#1f77b4", 2: "#ff7f0e", 1: "#2ca02c"}


def main() -> None:
    s = pd.read_csv(HERE / "tau_sweep.csv")
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.6))

    for cad in sorted(s.cadence_days.unique(), reverse=True):
        c = s[s.cadence_days == cad].sort_values("tau")
        col = COLOURS[cad]
        ax.plot(c.tau, c.model_cpl, "o-", color=col, label=f"{cad}d model", zorder=3)
        ax.axhline(c.oracle_cpl.iloc[0], color=col, ls=":", lw=1.4, zorder=1)
        best = c.loc[c.model_cpl.idxmin()]
        ax.plot([best.tau], [best.model_cpl], "*", ms=17, color=col,
                mec="black", mew=0.7, zorder=4)
        ax.annotate(f"{cad}d oracle {c.oracle_cpl.iloc[0]:.2f}",
                    xy=(0.86, c.oracle_cpl.iloc[0]), fontsize=8, color=col,
                    va="bottom", ha="right")

    ax.axvline(PROD_TAU, color="0.35", ls="--", lw=1.2, zorder=2)
    ax.annotate("production τ=0.25", xy=(PROD_TAU, ax.get_ylim()[1]),
                xytext=(PROD_TAU + 0.012, ax.get_ylim()[1] - 0.4),
                fontsize=9, color="0.25")
    ax.set_xlabel("τ (decision threshold)")
    ax.set_ylabel("realised CPL (c/L, pooled over 14 folds × 5 stations)")
    ax.set_title("Realised CPL vs τ — dotted line is that cadence's oracle floor\n"
                 "★ = best τ in hindsight", fontsize=10)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.25)
    ax.set_xlim(0.0, 0.95)

    # Right panel: the same data as distance from each cadence's own best, which is
    # what "is the operating point binding?" actually asks. Cadences are stacked on
    # different CPL levels in the left panel, so the flatness is easier to see here.
    # Read the RIGHT-HAND side of this panel: the hypothesis under test was that a
    # finer grid wants a CHOOSIER model, i.e. a higher tau.
    for cad in sorted(s.cadence_days.unique(), reverse=True):
        c = s[s.cadence_days == cad].sort_values("tau")
        ax2.plot(c.tau, c.model_cpl - c.model_cpl.min(), "o-",
                 color=COLOURS[cad], label=f"{cad}d", zorder=3)
    ax2.axvline(PROD_TAU, color="0.35", ls="--", lw=1.2)
    ax2.axhline(0.05, color="crimson", ls=":", lw=1.4)
    ax2.annotate("~0.05 c/L = one decision flip at 7d", xy=(0.42, 0.06),
                 fontsize=8.5, color="crimson")
    ax2.set_ylim(-0.02, 1.2)
    ax2.set_xlim(0.0, 0.85)
    ax2.set_xlabel("τ (decision threshold)")
    ax2.set_ylabel("cost of the wrong τ (c/L above that cadence's best)")
    # NB the flatness is DIRECTIONAL and the title must not overclaim it. Above the
    # optimum -- the choosier side, exactly where "a finer grid should want a choosier
    # model" pointed -- the 1d curve is far flatter than 7d (0.21 vs 1.06 c/L at
    # tau=0.50). BELOW the optimum it is the other way round (1.01 vs 0.13 at 0.05):
    # at 7d an over-permissive model is forced to fill anyway, while at 1d it really
    # can buy constantly at bad prices. Only the upper side answers this experiment.
    ax2.set_title("How much does picking τ badly cost?\n"
                  "ABOVE the optimum — where the hypothesis pointed — finer is FLATTER",
                  fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.25)

    fig.suptitle("fps-929 — the daily-cadence plateau is not an operating-point limit",
                 fontsize=12)
    fig.tight_layout()
    out = HERE / "tau_sweep.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
