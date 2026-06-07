"""Per-fold bar chart of A's contribution, decomposed by bucket.

Two panels:
- Top: per-fold mean delta_ll on hard25 rows (the headline cohort). One bar
  per fold, red above zero = A hurts on hard25, blue below zero = A helps.
- Bottom: per-fold sum of delta_ll, stacked by bucket. Shows where each
  fold's net effect came from.

Reads step5_rowdelta.parquet.
"""
from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = pathlib.Path(__file__).parent

BUCKETS = ["ext_descent", "normal_descent", "elong_ascent", "normal_ascent"]
BUCKET_COLOURS = {
    "ext_descent":    "#c0392b",  # red — A's bad bucket
    "normal_descent": "#2980b9",  # blue — A's good bucket
    "elong_ascent":   "#e67e22",  # orange — neutral
    "normal_ascent":  "#7f8c8d",  # grey — neutral
}


def main() -> None:
    rd = pd.read_parquet(OUT / "step5_rowdelta.parquet")

    # Panel 1: per-fold mean delta_ll on hard25.
    hard = rd[rd["is_hard25"] == 1]
    pf_hard = (
        hard.groupby("fold")
        .agg(mean_delta=("delta_ll", "mean"), n=("delta_ll", "size"))
        .reset_index()
        .sort_values("fold")
    )

    # Panel 2: per-fold sum of delta_ll by bucket (all rows).
    pf_sum = (
        rd.groupby(["fold", "bucket"])["delta_ll"].sum().unstack(fill_value=0.0)
    )
    pf_sum = pf_sum.reindex(columns=BUCKETS, fill_value=0.0)

    folds = sorted(rd["fold"].unique())

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    # ---- Panel 1: per-fold mean delta_ll on hard25 ----
    ax = axes[0]
    colours = ["#c0392b" if v > 0 else "#2980b9" for v in pf_hard["mean_delta"]]
    ax.bar(pf_hard["fold"], pf_hard["mean_delta"], color=colours,
           edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("Per-fold mean delta_ll on hard25 rows\n(positive = A hurts)")
    ax.set_title("A's effect by fold — hard25 cohort (median across seeds)")
    for f, v in zip(pf_hard["fold"], pf_hard["mean_delta"]):
        ax.annotate(f"{v:+.3f}", (f, v),
                    textcoords="offset points",
                    xytext=(0, 4 if v >= 0 else -12),
                    ha="center", fontsize=8)

    # ---- Panel 2: stacked sum of delta_ll per bucket ----
    ax = axes[1]
    # Separate positive and negative contributions per bucket so the stack
    # shows above-zero vs below-zero contributions clearly.
    pos = pf_sum.clip(lower=0)
    neg = pf_sum.clip(upper=0)
    bot_pos = np.zeros(len(folds))
    bot_neg = np.zeros(len(folds))
    for b in BUCKETS:
        ax.bar(folds, pos[b].values, bottom=bot_pos,
               color=BUCKET_COLOURS[b], edgecolor="black", linewidth=0.3,
               label=b)
        ax.bar(folds, neg[b].values, bottom=bot_neg,
               color=BUCKET_COLOURS[b], edgecolor="black", linewidth=0.3)
        bot_pos = bot_pos + pos[b].values
        bot_neg = bot_neg + neg[b].values
    # Net line: total per fold
    net = pf_sum.sum(axis=1).values
    ax.plot(folds, net, "k.-", lw=1.2, label="net (sum)")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("Fold")
    ax.set_ylabel("Per-fold sum of delta_ll\n(positive = A hurt the fold overall)")
    ax.set_title("Decomposition: where did each fold's net effect come from?")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xticks(folds)

    fig.tight_layout()
    out_png = OUT / "step5c_perfold_bars.png"
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
