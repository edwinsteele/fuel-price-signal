"""SHAP diagnostic for the 15-feature LGBM model (post-#127 stickiness_score).

Trains LGBM seed=42 on train, computes TreeExplainer SHAP on val, saves:
  - mean_abs_shap_ranking.csv
  - summary.png (beeswarm)
  - dep_station_minus_lga_mean_cents__by__station_minus_sydney_avg_cents.png
  - dep_stickiness_score__by__station_minus_lga_mean_cents.png
  - shap_values.npy

Mirrors the Phase 3b SHAP investigation (experiments/shap/) so plots are directly
comparable. The right-hump shrinkage test compares the 14-feat vs 15-feat
dep_station_minus_lga_mean_cents__by__station_minus_sydney_avg_cents.png plots.
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from lightgbm import LGBMClassifier  # noqa: E402

from fuel_signal import evaluate as _ev  # noqa: E402
from fuel_signal.features import FEATURE_COLUMNS  # noqa: E402

OUT = pathlib.Path(__file__).parent
FEATURES_CSV = pathlib.Path("data/features.csv")


def main() -> None:
    print(f"Loading {FEATURES_CSV}…")
    df = pd.read_csv(FEATURES_CSV)
    train, val, _test = _ev.split(df)
    print(f"  train rows: {len(train):,}  val rows: {len(val):,}")
    print(f"  features ({len(FEATURE_COLUMNS)}): {FEATURE_COLUMNS}")

    X_train = train[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_train = train["label"].to_numpy(dtype=int)
    X_val = val[FEATURE_COLUMNS].to_numpy(dtype=float)

    print("Fitting LGBM (seed=42)…")
    model = LGBMClassifier(random_state=42, verbose=-1)
    model.fit(X_train, y_train)

    print("Running TreeExplainer on val…")
    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_values(X_val)
    # newer SHAP: returns (n, k) for binary classifier (positive-class margin contributions)
    # older SHAP: returns list [neg, pos]; use pos
    if isinstance(raw, list):
        sv = raw[1]
    else:
        sv = raw
    print(f"  shap_values shape: {sv.shape}")

    np.save(OUT / "shap_values.npy", sv)

    mean_abs = np.mean(np.abs(sv), axis=0)
    ranking = (
        pd.DataFrame({"feature": FEATURE_COLUMNS, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    ranking.to_csv(OUT / "mean_abs_shap_ranking.csv", index=False)
    print("\nMean |SHAP| ranking:")
    for i, row in ranking.iterrows():
        print(f"  {i + 1:>2}. {row['feature']:<35} {row['mean_abs_shap']:.4f}")

    print("\nRendering beeswarm summary…")
    shap.summary_plot(sv, X_val, feature_names=FEATURE_COLUMNS, show=False)
    plt.tight_layout()
    plt.savefig(OUT / "summary.png", dpi=120)
    plt.close()

    print("Rendering dep: station_minus_lga_mean_cents coloured by station_minus_sydney_avg_cents…")
    lga_idx = FEATURE_COLUMNS.index("station_minus_lga_mean_cents")
    syd_idx = FEATURE_COLUMNS.index("station_minus_sydney_avg_cents")
    shap.dependence_plot(
        lga_idx,
        sv,
        X_val,
        feature_names=FEATURE_COLUMNS,
        interaction_index=syd_idx,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(OUT / "dep_station_minus_lga_mean_cents__by__station_minus_sydney_avg_cents.png", dpi=120)
    plt.close()

    print("Rendering dep: stickiness_score coloured by station_minus_lga_mean_cents…")
    sticky_idx = FEATURE_COLUMNS.index("stickiness_score")
    shap.dependence_plot(
        sticky_idx,
        sv,
        X_val,
        feature_names=FEATURE_COLUMNS,
        interaction_index=lga_idx,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(OUT / "dep_stickiness_score__by__station_minus_lga_mean_cents.png", dpi=120)
    plt.close()

    print(f"\nDone. Artifacts in {OUT}/")


if __name__ == "__main__":
    main()
