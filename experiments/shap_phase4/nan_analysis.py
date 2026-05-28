"""SHAP NaN analysis for #136 — does woollahra's high SHAP rank come from
the integer feature value (genuine leadership signal) or from LGBM routing
NaN-valued rows along a learnt "missing → era marker" path (artefact)?

Inputs:
  - data/models/lgbm.joblib           Phase 4 lock model (50-feat)
  - data/features.csv                 features table
  - experiments/shap_phase4/shap_values.npy   saved val SHAP from lock day

Outputs (this script): printed report covering steps 0-4 of #136.
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

VERIFY_N = 1000
TARGET_COL = "days_since_trough_entry_woollahra"


def main() -> None:
    print(f"Loading model {MODEL_PATH}…")
    bundle = joblib.load(MODEL_PATH)
    model = bundle["pipeline"]
    feature_columns: list[str] = bundle["feature_columns"]
    print(f"  features ({len(feature_columns)}): ends with {feature_columns[-1]}")
    assert TARGET_COL in feature_columns, f"{TARGET_COL} not in model schema"
    target_idx = feature_columns.index(TARGET_COL)
    print(f"  {TARGET_COL} index = {target_idx}")

    print(f"\nLoading {FEATURES_CSV}…")
    df = pd.read_csv(FEATURES_CSV)
    _train, val, _test = _ev.split(df)
    X_val = val[feature_columns].to_numpy(dtype=float)
    n_val = len(val)
    print(f"  val rows: {n_val:,}")

    print(f"\nLoading {SHAP_PATH}…")
    sv = np.load(SHAP_PATH)
    print(f"  shap_values shape: {sv.shape}")
    assert sv.shape == (n_val, len(feature_columns)), (
        f"shape mismatch: shap {sv.shape} vs val ({n_val}, {len(feature_columns)})"
    )

    # ------------------------------------------------------------------
    # Step 0 — verification gate: rerun SHAP on first VERIFY_N rows, compare.
    # ------------------------------------------------------------------
    print(f"\n[Step 0] verification gate — re-running TreeExplainer on first {VERIFY_N} val rows…")
    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_values(X_val[:VERIFY_N])
    sv_check = raw[1] if isinstance(raw, list) else raw
    diff = np.abs(sv_check - sv[:VERIFY_N])
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())
    print(f"  max abs diff   : {max_diff:.3e}")
    print(f"  mean abs diff  : {mean_diff:.3e}")
    if max_diff > 1e-6:
        print("  WARNING: max diff exceeds 1e-6 — model/SHAP may have drifted; results may not be trustworthy.")
    else:
        print("  OK — join is verified.")

    # ------------------------------------------------------------------
    # Step 1 — partition val rows on NaN-mask for TARGET_COL.
    # ------------------------------------------------------------------
    col_vals = X_val[:, target_idx]
    nan_mask = np.isnan(col_vals)
    n_nan = int(nan_mask.sum())
    n_nonnan = int((~nan_mask).sum())
    print(f"\n[Step 1] partition val on {TARGET_COL} NaN-mask:")
    print(f"  NaN rows     : {n_nan:>7,} ({100 * n_nan / n_val:.2f}%)")
    print(f"  non-NaN rows : {n_nonnan:>7,} ({100 * n_nonnan / n_val:.2f}%)")

    # ------------------------------------------------------------------
    # Step 2 — decompose mean|SHAP| by NaN group.
    # mean|SHAP| = (1/N) Σ|s_i|.  Per-group "contribution to the mean" is
    # (1/N_total) Σ_in_group |s_i|, so contributions sum to overall mean|SHAP|.
    # ------------------------------------------------------------------
    shap_col = sv[:, target_idx]
    total_mean_abs = float(np.mean(np.abs(shap_col)))
    contrib_nan = float(np.sum(np.abs(shap_col[nan_mask])) / n_val) if n_nan else 0.0
    contrib_nonnan = float(np.sum(np.abs(shap_col[~nan_mask])) / n_val) if n_nonnan else 0.0
    # Also report group-internal mean|SHAP| (per-row average within each group).
    grp_mean_nan = float(np.mean(np.abs(shap_col[nan_mask]))) if n_nan else 0.0
    grp_mean_nonnan = float(np.mean(np.abs(shap_col[~nan_mask]))) if n_nonnan else 0.0
    print(f"\n[Step 2] mean|SHAP| decomposition for {TARGET_COL}:")
    print(f"  overall mean|SHAP|        : {total_mean_abs:.4f}")
    print(f"  contribution from NaN     : {contrib_nan:.4f} ({100 * contrib_nan / total_mean_abs:.1f}% of total)")
    print(f"  contribution from non-NaN : {contrib_nonnan:.4f} ({100 * contrib_nonnan / total_mean_abs:.1f}% of total)")
    print(f"  within-group mean|SHAP| NaN     : {grp_mean_nan:.4f}")
    print(f"  within-group mean|SHAP| non-NaN : {grp_mean_nonnan:.4f}")
    print(f"  expected NaN share if neutral   : {100 * n_nan / n_val:.2f}%")

    # ------------------------------------------------------------------
    # Step 3 — signed SHAP on NaN rows. Consistent +ve = NaN routes toward BUY
    # (calendar/era marker hypothesis); mixed = no label-leakage via missingness.
    # ------------------------------------------------------------------
    if n_nan:
        nan_signed = shap_col[nan_mask]
        nonnan_signed = shap_col[~nan_mask]
        print("\n[Step 3] signed SHAP distribution:")
        for label, arr in (("NaN rows    ", nan_signed), ("non-NaN rows", nonnan_signed)):
            print(
                f"  {label} mean={float(arr.mean()):+.4f}  "
                f"median={float(np.median(arr)):+.4f}  std={float(arr.std()):.4f}"
            )
            print(
                f"               %positive={100 * float((arr > 0).mean()):.1f}%   "
                f"%negative={100 * float((arr < 0).mean()):.1f}%"
            )

    # ------------------------------------------------------------------
    # Step 4 — dependence plot on non-NaN subset: feature value vs SHAP.
    # Genuine leadership signal = monotone decreasing
    #   (low days_since = recent trough → +SHAP toward BUY).
    # ------------------------------------------------------------------
    if n_nonnan:
        vals = col_vals[~nan_mask]
        shaps = shap_col[~nan_mask]
        # Pearson on the raw (val, shap) pairs.
        corr = float(np.corrcoef(vals, shaps)[0, 1])
        # Bin into deciles and report mean SHAP per decile.
        try:
            qs = pd.qcut(vals, 10, duplicates="drop")
            bin_summary = (
                pd.DataFrame({"days": vals, "shap": shaps, "bin": qs})
                .groupby("bin", observed=True)["shap"]
                .agg(["count", "mean", "median"])
            )
        except ValueError:
            bin_summary = None

        print("\n[Step 4] non-NaN dependence (value vs SHAP):")
        print(f"  Pearson r(value, shap) = {corr:+.3f}   (expect strongly negative for genuine leadership)")
        if bin_summary is not None:
            print(f"  per-decile of {TARGET_COL}:")
            for idx, row in bin_summary.iterrows():
                print(
                    f"    {str(idx):<28}  n={int(row['count']):>5,}  "
                    f"mean_shap={row['mean']:+.4f}  median_shap={row['median']:+.4f}"
                )

        # Save scatter plot.
        fig, ax = plt.subplots(figsize=(8, 5))
        # subsample for plotting if huge
        n_plot = min(20000, len(vals))
        rng = np.random.default_rng(0)
        idx = rng.choice(len(vals), size=n_plot, replace=False)
        ax.scatter(vals[idx], shaps[idx], s=2, alpha=0.3)
        ax.axhline(0, color="k", lw=0.5, alpha=0.5)
        ax.set_xlabel(TARGET_COL)
        ax.set_ylabel(f"SHAP value for {TARGET_COL}")
        ax.set_title(f"#136 dependence: {TARGET_COL}  (non-NaN, n={n_nonnan:,}, plot subsample={n_plot:,})")
        fig.tight_layout()
        out_png = OUT / "nan_analysis_dependence_woollahra.png"
        fig.savefig(out_png, dpi=120)
        plt.close(fig)
        print(f"  saved {out_png}")

    print("\nDone.")


if __name__ == "__main__":
    main()
