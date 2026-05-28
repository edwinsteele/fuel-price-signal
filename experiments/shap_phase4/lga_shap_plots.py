"""LGA-only SHAP plots for #136.

Produces two figures restricted to the 35 days_since_trough_entry_<lga>
features:

  lga_beeswarm.png       — beeswarm summary, all 29 non-trivial LGAs ranked
  lga_dependence_grid.png — 5×6 grid of dependence scatters, all non-trivial
                             LGAs, ordered by mean|SHAP|. Annotated with
                             Pearson r and mean|SHAP|.

The 6 all-NaN LGAs (bayside, botany_bay, camden, hunters_hill, lane_cove,
waverley) are excluded — they have no signal by data construction.
"""

from __future__ import annotations

import pathlib

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

from fuel_signal import evaluate as _ev  # noqa: E402

OUT = pathlib.Path(__file__).parent
FEATURES_CSV = pathlib.Path("data/features.csv")
MODEL_PATH = pathlib.Path("data/models/lgbm.joblib")
SHAP_PATH = OUT / "shap_values.npy"

LGA_PREFIX = "days_since_trough_entry_"


def main() -> None:
    bundle = joblib.load(MODEL_PATH)
    feature_columns: list[str] = bundle["feature_columns"]

    df = pd.read_csv(FEATURES_CSV)
    _train, val, _test = _ev.split(df)
    X_val = val[feature_columns].to_numpy(dtype=float)
    sv = np.load(SHAP_PATH)
    assert sv.shape == X_val.shape

    # Find LGA feature indices.
    lga_idxs = [i for i, c in enumerate(feature_columns) if c.startswith(LGA_PREFIX)]
    print(f"Found {len(lga_idxs)} LGA features")

    # Drop all-NaN LGAs.
    keep: list[int] = []
    for i in lga_idxs:
        if not np.all(np.isnan(X_val[:, i])):
            keep.append(i)
    dropped = len(lga_idxs) - len(keep)
    print(f"Dropping {dropped} all-NaN LGA features; keeping {len(keep)}")

    # Per-LGA stats.
    stats = []
    for i in keep:
        col = X_val[:, i]
        mask = ~np.isnan(col)
        v = col[mask]
        s = sv[mask, i]
        mean_abs = float(np.mean(np.abs(s)))
        if np.std(v) == 0 or np.std(s) == 0:
            r = float("nan")
        else:
            r = float(np.corrcoef(v, s)[0, 1])
        stats.append((i, feature_columns[i], mean_abs, r))
    # Sort by mean|SHAP| desc.
    stats.sort(key=lambda t: t[2], reverse=True)

    # ------------------------------------------------------------------
    # Plot 1 — LGA-only beeswarm.
    # ------------------------------------------------------------------
    ordered_idxs = [t[0] for t in stats]
    sv_lga = sv[:, ordered_idxs]
    X_lga = X_val[:, ordered_idxs]
    names = [feature_columns[i].replace(LGA_PREFIX, "") for i in ordered_idxs]

    print("\nRendering LGA-only beeswarm…")
    shap.summary_plot(sv_lga, X_lga, feature_names=names, show=False, max_display=len(names))
    fig = plt.gcf()
    fig.set_size_inches(10, max(6, 0.28 * len(names)))
    fig.suptitle(f"#136 LGA-only SHAP beeswarm ({len(names)} non-trivial LGAs, val n={len(val):,})", fontsize=11)
    fig.tight_layout()
    out_png = OUT / "lga_beeswarm.png"
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"  saved {out_png}")

    # ------------------------------------------------------------------
    # Plot 2 — dependence grid, all non-trivial LGAs, ordered by mean|SHAP|.
    # ------------------------------------------------------------------
    n = len(stats)
    ncols = 6
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.6, nrows * 2.1), sharey=True)
    axes = np.atleast_2d(axes)
    flat_axes = axes.ravel()

    rng = np.random.default_rng(0)
    n_plot_target = 4000

    print("\nPer-LGA dependence summary (ordered by mean|SHAP| desc):")
    print(f"{'rank':>4} {'lga':<22} {'|SHAP|':>8} {'r':>7}")
    print("-" * 45)
    for rank, (idx, _full, mean_abs, r) in enumerate(stats):
        ax = flat_axes[rank]
        col = X_val[:, idx]
        mask = ~np.isnan(col)
        v = col[mask]
        s = sv[mask, idx]
        n_plot = min(n_plot_target, len(v))
        sub = rng.choice(len(v), size=n_plot, replace=False)
        ax.scatter(v[sub], s[sub], s=1.5, alpha=0.25)
        ax.axhline(0, color="k", lw=0.5, alpha=0.4)
        slug = feature_columns[idx].replace(LGA_PREFIX, "")
        ax.set_title(f"{rank + 1}. {slug}\nr={r:+.2f}  |S|={mean_abs:.3f}", fontsize=7)
        ax.tick_params(labelsize=6)
        print(f"{rank + 1:>4} {slug:<22} {mean_abs:>8.4f} {r:>+7.3f}")

    # Hide unused axes.
    for j in range(n, len(flat_axes)):
        flat_axes[j].axis("off")

    for ax in axes[:, 0]:
        ax.set_ylabel("SHAP", fontsize=7)
    for ax in axes[-1, :]:
        ax.set_xlabel("days_since_trough", fontsize=7)

    fig.suptitle(f"#136 LGA dependence grid — {n} non-trivial LGAs, ordered by mean|SHAP|", fontsize=11)
    fig.tight_layout()
    out_png = OUT / "lga_dependence_grid.png"
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"\nSaved {out_png}")


if __name__ == "__main__":
    main()
