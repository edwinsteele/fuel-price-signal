"""Step 5 of #136 — multi-LGA dependence plot comparison.

Compare days_since_trough_entry_<lga> dependence on SHAP across the rank
spectrum to distinguish "no signal" (flat dependence) from "redundant with
woollahra" (similar shape, lower amplitude).

Six LGAs chosen:
  - Top   : woollahra, randwick, blue_mountains
  - Mid   : parramatta
  - Bottom: sutherland_shire, penrith     (memory-named "leaders", low SHAP)

Outputs:
  - printed Pearson r + per-decile mean SHAP for each LGA
  - dependence_grid.png         2×3 grid of dependence scatters, shared y-axis
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

LGAS = [
    ("woollahra", "Top (rank 1)"),
    ("randwick", "Top (rank 2)"),
    ("blue_mountains", "Top (rank 3)"),
    ("parramatta", "Mid (rank 5)"),
    ("sutherland_shire", "Memory leader, SHAP 23/35"),
    ("penrith", "Memory leader, SHAP 19/35"),
]


def main() -> None:
    bundle = joblib.load(MODEL_PATH)
    feature_columns: list[str] = bundle["feature_columns"]

    df = pd.read_csv(FEATURES_CSV)
    _train, val, _test = _ev.split(df)
    X_val = val[feature_columns].to_numpy(dtype=float)
    sv = np.load(SHAP_PATH)
    assert sv.shape == X_val.shape

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=True)
    axes = axes.ravel()

    print(f"{'LGA':<22} {'rows':>8} {'NaN':>6} {'r':>7} {'|SHAP|':>8} {'min_dec':>9} {'max_dec':>9}")
    print("-" * 75)

    rng = np.random.default_rng(0)

    for ax, (slug, label) in zip(axes, LGAS):
        col = f"days_since_trough_entry_{slug}"
        if col not in feature_columns:
            ax.text(0.5, 0.5, f"{col} missing", ha="center", va="center", transform=ax.transAxes)
            continue
        idx = feature_columns.index(col)
        vals = X_val[:, idx]
        shaps = sv[:, idx]

        nan_mask = np.isnan(vals)
        n_nan = int(nan_mask.sum())
        v = vals[~nan_mask]
        s = shaps[~nan_mask]
        if len(v) == 0:
            ax.text(0.5, 0.5, "all NaN", ha="center", va="center", transform=ax.transAxes)
            print(f"{slug:<22} {len(v):>8,} {n_nan:>6,} {'-':>7} {'-':>8} {'-':>9} {'-':>9}  ({label})")
            continue

        mean_abs = float(np.mean(np.abs(s)))
        # Pearson; guard against zero variance (all values identical)
        if np.std(v) == 0 or np.std(s) == 0:
            r = float("nan")
        else:
            r = float(np.corrcoef(v, s)[0, 1])

        try:
            qs = pd.qcut(v, 10, duplicates="drop")
            dec = (
                pd.DataFrame({"v": v, "s": s, "bin": qs})
                .groupby("bin", observed=True)["s"]
                .mean()
            )
            dec_min = float(dec.min())
            dec_max = float(dec.max())
        except ValueError:
            dec_min = float("nan")
            dec_max = float("nan")

        print(
            f"{slug:<22} {len(v):>8,} {n_nan:>6,} {r:>+7.3f} "
            f"{mean_abs:>8.4f} {dec_min:>+9.4f} {dec_max:>+9.4f}  ({label})"
        )

        n_plot = min(8000, len(v))
        sub = rng.choice(len(v), size=n_plot, replace=False)
        ax.scatter(v[sub], s[sub], s=2, alpha=0.25)
        ax.axhline(0, color="k", lw=0.5, alpha=0.5)
        ax.set_title(f"{slug}  r={r:+.2f}  mean|SHAP|={mean_abs:.3f}\n{label}", fontsize=9)
        ax.set_xlabel("days_since_trough_entry")

    axes[0].set_ylabel("SHAP value (log-odds toward BUY)")
    axes[3].set_ylabel("SHAP value (log-odds toward BUY)")
    fig.suptitle("#136 Step 5: dependence shape across SHAP-rank spectrum (val, n=62,152)", fontsize=11)
    fig.tight_layout()
    out_png = OUT / "dependence_grid.png"
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"\nSaved {out_png}")


if __name__ == "__main__":
    main()
