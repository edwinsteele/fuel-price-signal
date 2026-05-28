"""LGA dependence plots with cycle_pct_through interaction colouring.

Mirrors the shap.dependence_plot style used in experiments/shap/ — but as
a single grid covering all 29 non-trivial LGA features, ordered by
mean|SHAP| descending and sharing one colourbar.

Reveals how the model reads each LGA's days_since_trough at different
positions in the local cycle: early-cycle (blue) vs late-cycle (red).
"""

from __future__ import annotations

import pathlib

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fuel_signal import evaluate as _ev  # noqa: E402

OUT = pathlib.Path(__file__).parent
FEATURES_CSV = pathlib.Path("data/features.csv")
MODEL_PATH = pathlib.Path("data/models/lgbm.joblib")
SHAP_PATH = OUT / "shap_values.npy"

LGA_PREFIX = "days_since_trough_entry_"
COLOUR_FEATURE = "cycle_pct_through"
CMAP = "coolwarm"
N_PLOT_PER_PANEL = 4000


def main() -> None:
    bundle = joblib.load(MODEL_PATH)
    feature_columns: list[str] = bundle["feature_columns"]

    df = pd.read_csv(FEATURES_CSV)
    _train, val, _test = _ev.split(df)
    X_val = val[feature_columns].to_numpy(dtype=float)
    sv = np.load(SHAP_PATH)
    assert sv.shape == X_val.shape

    if COLOUR_FEATURE not in feature_columns:
        raise SystemExit(f"{COLOUR_FEATURE} not in model schema")
    c_idx = feature_columns.index(COLOUR_FEATURE)
    c_full = X_val[:, c_idx]

    # Non-trivial LGAs only.
    lga_idxs = [i for i, c in enumerate(feature_columns) if c.startswith(LGA_PREFIX)]
    keep = [i for i in lga_idxs if not np.all(np.isnan(X_val[:, i]))]
    print(f"Plotting {len(keep)} non-trivial LGA features, coloured by {COLOUR_FEATURE}")

    stats = []
    for i in keep:
        col = X_val[:, i]
        mask = ~np.isnan(col)
        s = sv[mask, i]
        mean_abs = float(np.mean(np.abs(s)))
        stats.append((i, mean_abs))
    stats.sort(key=lambda t: t[1], reverse=True)
    ordered = [t[0] for t in stats]

    # Robust colour range for cycle_pct_through (drop outliers / NaN).
    c_finite = c_full[np.isfinite(c_full)]
    if len(c_finite) == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin = float(np.percentile(c_finite, 1))
        vmax = float(np.percentile(c_finite, 99))

    n = len(ordered)
    ncols = 6
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.8, nrows * 2.3), sharey=True)
    axes = np.atleast_2d(axes)
    flat_axes = axes.ravel()

    rng = np.random.default_rng(0)
    scatter_handle = None

    for rank, idx in enumerate(ordered):
        ax = flat_axes[rank]
        col = X_val[:, idx]
        mask = ~np.isnan(col) & np.isfinite(c_full)
        v = col[mask]
        s = sv[mask, idx]
        c = c_full[mask]

        slug = feature_columns[idx].replace(LGA_PREFIX, "")
        if len(v) == 0:
            ax.text(0.5, 0.5, "no finite data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{rank + 1}. {slug}", fontsize=7)
            continue

        n_plot = min(N_PLOT_PER_PANEL, len(v))
        sub = rng.choice(len(v), size=n_plot, replace=False)
        sc = ax.scatter(
            v[sub], s[sub],
            c=c[sub], cmap=CMAP, vmin=vmin, vmax=vmax,
            s=2.0, alpha=0.55, edgecolors="none",
        )
        if scatter_handle is None:
            scatter_handle = sc

        ax.axhline(0, color="k", lw=0.5, alpha=0.4)
        mean_abs = stats[rank][1]
        if np.std(v) and np.std(s):
            r = float(np.corrcoef(v, s)[0, 1])
        else:
            r = float("nan")
        ax.set_title(f"{rank + 1}. {slug}\nr={r:+.2f}  |S|={mean_abs:.3f}", fontsize=7)
        ax.tick_params(labelsize=6)

    for j in range(n, len(flat_axes)):
        flat_axes[j].axis("off")

    for ax in axes[:, 0]:
        ax.set_ylabel("SHAP", fontsize=7)
    for ax in axes[-1, :]:
        ax.set_xlabel("days_since_trough", fontsize=7)

    fig.suptitle(
        f"#136 LGA dependence × {COLOUR_FEATURE} — {n} LGAs, val n={len(val):,}",
        fontsize=11,
    )

    # Shared colourbar on the right.
    fig.tight_layout(rect=[0, 0, 0.94, 0.97])
    cbar_ax = fig.add_axes([0.955, 0.05, 0.012, 0.88])
    cb = fig.colorbar(scatter_handle, cax=cbar_ax)
    cb.set_label(COLOUR_FEATURE, fontsize=8)
    cb.ax.tick_params(labelsize=7)

    out_png = OUT / f"lga_dependence_by_{COLOUR_FEATURE}.png"
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
