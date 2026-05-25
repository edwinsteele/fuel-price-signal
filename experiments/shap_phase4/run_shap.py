"""SHAP diagnostic for the 50-feature Phase 4 LGBM (15 base + 35 LGA trough).

Trains LGBM seed=42 on train, computes TreeExplainer SHAP on val, saves:
  - mean_abs_shap_ranking.csv        all 50 features ranked
  - mean_abs_shap_lga_ranking.csv    LGA features only, ranked
  - mean_abs_shap_by_cohort.csv      mean|SHAP| per LGA feature, sliced by
                                     signed-days-to-trough cohort (matches
                                     trough_weakness bins)
  - summary.png                      beeswarm of top-20 features
  - shap_values.npy                  raw (n_val, 50) matrix

Cross-references performed in script output (printed):
  - 6 LGAs expected to be all-NaN inputs (bayside, botany_bay, camden,
    hunters_hill, lane_cove, waverley) should have mean|SHAP| ≈ 0.
  - 4 leader LGAs from project_leadership_architecture (sutherland_shire,
    northern_beaches, penrith, ku_ring_gai) should rank in the upper half
    of LGA features; in the lead −7..−1 cohort especially.

Cohort method mirrors experiments/trough_weakness/run.py exactly so bins
are directly comparable. Trough detection uses smoothed local-min over the
full per-station history (post-hoc — fine for SHAP slicing, would be
forward-looking as a feature).
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
from scipy.signal import find_peaks  # noqa: E402

from fuel_signal import evaluate as _ev  # noqa: E402
from fuel_signal.features import FEATURE_COLUMNS, LGA_FEATURE_COLUMNS  # noqa: E402

OUT = pathlib.Path(__file__).parent
FEATURES_CSV = pathlib.Path("data/features.csv")

ALL_FEATURES = FEATURE_COLUMNS + LGA_FEATURE_COLUMNS

# From handover: 6 LGAs have all-NaN columns (no stations or < 3-station floor).
EXPECTED_ZERO_LGAS = [
    "days_since_trough_entry_bayside",
    "days_since_trough_entry_botany_bay",
    "days_since_trough_entry_camden",
    "days_since_trough_entry_hunters_hill",
    "days_since_trough_entry_lane_cove",
    "days_since_trough_entry_waverley",
]

# Top 4 by trough_lead_consistency per project_leadership_architecture.
LEADER_LGAS = [
    "days_since_trough_entry_sutherland_shire",
    "days_since_trough_entry_northern_beaches",
    "days_since_trough_entry_penrith",
    "days_since_trough_entry_ku_ring_gai",
]

SMOOTH_WINDOW = 7
MIN_TROUGH_SPACING = 10
SNAP_RADIUS = 5
BINS = [
    (-999, -8,  "lead −8+"),
    (-7,   -4,  "lead −7..−4"),
    (-3,   -1,  "lead −3..−1"),
    (0,    0,   "trough (0)"),
    (1,    3,   "post +1..+3"),
    (4,    7,   "post +4..+7"),
    (8,    999, "mid-cycle +8+"),
]


def _signed_days_to_trough(station_df: pd.DataFrame) -> pd.Series:
    sorted_df = station_df.sort_values("price_date")
    original_index = sorted_df.index
    prices = sorted_df["today_price_cents"].to_numpy()
    if len(prices) < SMOOTH_WINDOW * 2:
        return pd.Series(np.nan, index=original_index)

    smooth = pd.Series(prices).rolling(SMOOTH_WINDOW, center=True, min_periods=1).mean().to_numpy()
    trough_idx, _ = find_peaks(-smooth, distance=MIN_TROUGH_SPACING)
    if len(trough_idx) == 0:
        return pd.Series(np.nan, index=original_index)

    snapped = np.empty_like(trough_idx)
    for i, t in enumerate(trough_idx):
        lo = max(0, t - SNAP_RADIUS)
        hi = min(len(prices), t + SNAP_RADIUS + 1)
        snapped[i] = lo + int(np.argmin(prices[lo:hi]))
    trough_idx = np.unique(snapped)

    all_idx = np.arange(len(prices))
    nearest_pos = np.searchsorted(trough_idx, all_idx)
    nearest_pos = np.clip(nearest_pos, 1, len(trough_idx) - 1)
    left = trough_idx[nearest_pos - 1]
    right = trough_idx[nearest_pos]
    left_dist = all_idx - left
    right_dist = all_idx - right
    use_left = np.abs(left_dist) <= np.abs(right_dist)
    return pd.Series(np.where(use_left, left_dist, right_dist).astype(float), index=original_index)


def _bin_label(d: float) -> str:
    if pd.isna(d):
        return "nan"
    d = int(d)
    for lo, hi, name in BINS:
        if lo <= d <= hi:
            return name
    return "nan"


def main() -> None:
    print(f"Loading {FEATURES_CSV}…")
    df = pd.read_csv(FEATURES_CSV, parse_dates=["price_date"])

    print("Computing per-station trough distances for cohort slicing…")
    df["days_to_trough"] = (
        df.groupby("station_code", group_keys=False).apply(_signed_days_to_trough)
    )
    df["bin"] = df["days_to_trough"].apply(_bin_label)

    train, val, _test = _ev.split(df)
    print(f"  train rows: {len(train):,}  val rows: {len(val):,}")
    print(f"  features: {len(FEATURE_COLUMNS)} base + {len(LGA_FEATURE_COLUMNS)} LGA = {len(ALL_FEATURES)}")

    X_train = train[ALL_FEATURES].to_numpy(dtype=float)
    y_train = train["label"].to_numpy(dtype=int)
    X_val = val[ALL_FEATURES].to_numpy(dtype=float)

    print("\nFitting LGBM seed=42 on 50-feat…")
    model = LGBMClassifier(random_state=42, verbose=-1)
    model.fit(X_train, y_train)

    print("Running TreeExplainer on val…")
    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_values(X_val)
    if isinstance(raw, list):
        sv = raw[1]
    else:
        sv = raw
    print(f"  shap_values shape: {sv.shape}")
    np.save(OUT / "shap_values.npy", sv)

    # -------- overall ranking --------
    mean_abs = np.mean(np.abs(sv), axis=0)
    ranking = (
        pd.DataFrame({
            "feature": ALL_FEATURES,
            "mean_abs_shap": mean_abs,
            "kind": ["base"] * len(FEATURE_COLUMNS) + ["lga"] * len(LGA_FEATURE_COLUMNS),
        })
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    ranking.to_csv(OUT / "mean_abs_shap_ranking.csv", index=False)

    print("\nMean |SHAP| ranking (top 25):")
    for i, row in ranking.head(25).iterrows():
        tag = "" if row["kind"] == "base" else "  [LGA]"
        print(f"  {i + 1:>2}. {row['feature']:<45} {row['mean_abs_shap']:.4f}{tag}")

    # -------- LGA-only ranking --------
    lga_ranking = (
        ranking[ranking["kind"] == "lga"]
        .drop(columns=["kind"])
        .reset_index(drop=True)
    )
    lga_ranking.to_csv(OUT / "mean_abs_shap_lga_ranking.csv", index=False)

    print(f"\nLGA feature ranking — top 10 of {len(lga_ranking)}:")
    for i, row in lga_ranking.head(10).iterrows():
        print(f"  {i + 1:>2}. {row['feature']:<45} {row['mean_abs_shap']:.4f}")

    # -------- expected-zero sanity check --------
    print("\nExpected-zero LGAs (all-NaN inputs, should be SHAP ≈ 0):")
    feat_to_shap = dict(zip(ALL_FEATURES, mean_abs))
    for f in EXPECTED_ZERO_LGAS:
        v = feat_to_shap[f]
        flag = "✓" if v < 1e-6 else f"✗ ({v:.6f})"
        print(f"  {f:<45} {flag}")

    # -------- leader cross-reference --------
    print("\nLeader-LGA cross-reference (top 4 by trough_lead_consistency):")
    name_to_rank = {row["feature"]: i + 1 for i, row in lga_ranking.iterrows()}
    for f in LEADER_LGAS:
        rank = name_to_rank.get(f, "—")
        v = feat_to_shap[f]
        print(f"  {f:<45} LGA-rank {rank}/{len(lga_ranking)}   mean|SHAP|={v:.4f}")

    # -------- cohort slicing --------
    # val carries the "bin" column computed before split; row order is preserved
    # so SHAP rows align with val rows positionally.
    bin_arr = val["bin"].to_numpy()
    bin_order = [b[2] for b in BINS]
    cohort_rows = []
    for name in bin_order:
        mask = bin_arr == name
        n = int(mask.sum())
        if n == 0:
            continue
        sv_bin = sv[mask]
        mean_abs_bin = np.mean(np.abs(sv_bin), axis=0)
        row = {"bin": name, "n": n}
        for f, v in zip(ALL_FEATURES, mean_abs_bin):
            row[f] = v
        cohort_rows.append(row)
    cohort_df = pd.DataFrame(cohort_rows)
    cohort_df.to_csv(OUT / "mean_abs_shap_by_cohort.csv", index=False)

    # Pretty print: top LGA features per cohort
    print("\nTop 5 LGA features by mean|SHAP| within each cohort:")
    for _, r in cohort_df.iterrows():
        lga_vals = [(f, r[f]) for f in LGA_FEATURE_COLUMNS]
        lga_vals.sort(key=lambda kv: kv[1], reverse=True)
        print(f"\n  {r['bin']}  (n={int(r['n']):,})")
        for f, v in lga_vals[:5]:
            tag = "  ★ leader" if f in LEADER_LGAS else ""
            print(f"    {f:<45} {v:.4f}{tag}")

    # -------- beeswarm summary --------
    print("\nRendering beeswarm summary (top 20)…")
    shap.summary_plot(sv, X_val, feature_names=ALL_FEATURES, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(OUT / "summary.png", dpi=120)
    plt.close()

    print(f"\nDone. Artifacts in {OUT}/")


if __name__ == "__main__":
    main()
