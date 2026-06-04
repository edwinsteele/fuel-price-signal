"""Cheap test of the engineered phase-residual feature.

Hypothesis: cycle_pct_through × {station_minus_last_min_cents,
station_price_cents} showed a clean SHAP interaction saddle in step 1.
That suggests the model is reconstructing 'observed price minus expected
price at this cycle phase' from 4 inputs (station_price, last_cycle_min,
last_cycle_max, cycle_pct_through). If we hand it that residual directly,
the saddle should collapse into a single main-effect curve and val logloss
should not degrade (and may improve, particularly if the engineered formula
generalizes to regimes the tree hasn't trained on).

Engineered feature:
    expected_price = last_cycle_min + cycle_pct_through *
                     (last_cycle_max - last_cycle_min)
    station_minus_expected_phase_price = station_price - expected_price

A linear interpolation between trough and peak; the simplest possible
phase model. If a richer shape is needed (sinusoid, asymmetric ramp) we
iterate.

Protocol: single val window, 3 seeds for stability (per
[[feedback_seed_discipline]]), Phase 4 feature set (15 base + 35 LGA).
Reports baseline vs +engineered. SHAP attribution of the new feature
relative to its parents is the secondary outcome.
"""

from __future__ import annotations

import json
import pathlib
import time

import joblib
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier

from fuel_signal import evaluate as _ev
from fuel_signal.features import FEATURE_COLUMNS, LGA_FEATURE_COLUMNS, load_features

OUT = pathlib.Path(__file__).parent
SEEDS = (42, 43, 44)

NEW_COL = "station_minus_expected_phase_price"


def add_engineered(df: pd.DataFrame) -> pd.DataFrame:
    """Add station_minus_expected_phase_price column to df (in place copy)."""
    expected = df["cycle_last_min_cents"] + df["cycle_pct_through"] * (
        df["cycle_last_max_cents"] - df["cycle_last_min_cents"]
    )
    out = df.copy()
    out[NEW_COL] = df["station_price_cents"] - expected
    return out


def fit_and_score(
    train: pd.DataFrame, val: pd.DataFrame, cols: list[str], seed: int, label: str
) -> dict:
    X_train = train[cols].to_numpy(dtype=float)
    y_train = train["label"].to_numpy(dtype=int)
    X_val = val[cols].to_numpy(dtype=float)
    y_val = val["label"].to_numpy(dtype=int)

    t0 = time.perf_counter()
    model = LGBMClassifier(random_state=seed, verbose=-1, subsample=0.8, subsample_freq=1)
    model.fit(X_train, y_train)
    fit_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    p_val = model.predict_proba(X_val)[:, 1]
    pred_s = time.perf_counter() - t0

    ll = _ev.log_loss(y_val, p_val)
    br = _ev.brier(y_val, p_val)
    print(f"  [{label} fit] {fit_s:.1f}s  [predict_proba val] {pred_s:.2f}s")
    return {"model": model, "logloss": ll, "brier": br,
            "X_val": X_val, "p_val": p_val, "fit_s": fit_s, "pred_s": pred_s}


def mean_abs_shap(model: LGBMClassifier, X: np.ndarray, label: str) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    explainer = shap.TreeExplainer(model.booster_)
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        sv = sv[1] if len(sv) == 2 else sv[0]
    sv = np.asarray(sv)
    out = np.mean(np.abs(sv), axis=0)
    dt = time.perf_counter() - t0
    print(f"  [SHAP {label} n={X.shape[0]}] {dt:.1f}s")
    return out, dt


def main() -> None:
    overall_t0 = time.perf_counter()
    print("Loading features (parquet cache via load_features) …")
    t0 = time.perf_counter()
    df = load_features()
    df["price_date"] = pd.to_datetime(df["price_date"])
    print(f"  [load_features] {time.perf_counter() - t0:.1f}s  rows={len(df):,}")

    # Sanity: required parents must exist (LGA features may be NaN per-row, fine).
    needed = {"station_price_cents", "cycle_pct_through",
              "cycle_last_min_cents", "cycle_last_max_cents", "label"}
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f"missing columns: {missing}")

    df = add_engineered(df)
    print(f"  added {NEW_COL}.  describe:")
    print(df[NEW_COL].describe().to_string())

    train, val, _test = _ev.split(df)
    print(f"  train={len(train):,}  val={len(val):,}")

    baseline_cols = FEATURE_COLUMNS + LGA_FEATURE_COLUMNS
    engineered_cols = baseline_cols + [NEW_COL]
    print(f"  baseline features: {len(baseline_cols)}")
    print(f"  engineered features: {len(engineered_cols)}")

    rows = []
    shap_baseline_acc = []
    shap_engineered_acc = []

    for seed in SEEDS:
        print(f"\n=== seed {seed} ===")
        b = fit_and_score(train, val, baseline_cols, seed, "baseline")
        e = fit_and_score(train, val, engineered_cols, seed, "engineered")
        delta = e["logloss"] - b["logloss"]
        print(f"  baseline   logloss={b['logloss']:.6f} brier={b['brier']:.6f}")
        print(f"  engineered logloss={e['logloss']:.6f} brier={e['brier']:.6f}")
        print(f"  delta logloss (engineered - baseline) = {delta:+.6f}")

        # Subsample for SHAP to keep runtime down.
        rng = np.random.default_rng(seed)
        n_shap = min(8000, b["X_val"].shape[0])
        idx = rng.choice(b["X_val"].shape[0], size=n_shap, replace=False)
        b_shap, b_shap_s = mean_abs_shap(b["model"], b["X_val"][idx], "baseline")
        e_shap, e_shap_s = mean_abs_shap(e["model"], e["X_val"][idx], "engineered")
        shap_baseline_acc.append(b_shap)
        shap_engineered_acc.append(e_shap)

        rows.append({
            "seed": seed,
            "baseline_logloss": b["logloss"],
            "engineered_logloss": e["logloss"],
            "delta_logloss": delta,
            "baseline_brier": b["brier"],
            "engineered_brier": e["brier"],
            "baseline_fit_s": b["fit_s"],
            "engineered_fit_s": e["fit_s"],
            "baseline_shap_s": b_shap_s,
            "engineered_shap_s": e_shap_s,
        })

    results = pd.DataFrame(rows)
    results.to_csv(OUT / "step2_results.csv", index=False)
    print("\nPer-seed results:")
    print(results.to_string(index=False))

    mean_b = results["baseline_logloss"].mean()
    mean_e = results["engineered_logloss"].mean()
    std_b = results["baseline_logloss"].std(ddof=1)
    std_e = results["engineered_logloss"].std(ddof=1)
    mean_d = results["delta_logloss"].mean()
    std_d = results["delta_logloss"].std(ddof=1)
    print(f"\nbaseline   mean logloss = {mean_b:.6f}  seed std = {std_b:.6f}")
    print(f"engineered mean logloss = {mean_e:.6f}  seed std = {std_e:.6f}")
    print(f"delta      mean         = {mean_d:+.6f}  seed std = {std_d:.6f}")
    print(f"|delta| / seed std       = {abs(mean_d) / max(std_d, 1e-9):.2f}")

    # SHAP ranking: focus on the engineered feature and its parents.
    base_mean_shap = np.mean(np.stack(shap_baseline_acc), axis=0)
    eng_mean_shap = np.mean(np.stack(shap_engineered_acc), axis=0)
    base_df = pd.DataFrame({"feature": baseline_cols, "mean_abs_shap_baseline": base_mean_shap})
    eng_df = pd.DataFrame({"feature": engineered_cols, "mean_abs_shap_engineered": eng_mean_shap})
    merged = base_df.merge(eng_df, on="feature", how="outer")
    merged = merged.sort_values("mean_abs_shap_engineered", ascending=False)
    merged.to_csv(OUT / "step2_shap_ranking.csv", index=False)

    print("\nTop 20 features by mean|SHAP| in engineered model:")
    print(merged.head(20).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    parents = ["station_price_cents", "cycle_pct_through",
               "cycle_last_min_cents", "cycle_last_max_cents",
               "station_minus_last_min_cents", "station_minus_last_max_cents"]
    print(f"\nEngineered feature + parents (mean|SHAP|):")
    focus = merged[merged.feature.isin([NEW_COL] + parents)].sort_values(
        "mean_abs_shap_engineered", ascending=False
    )
    print(focus.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    meta = {
        "seeds": list(SEEDS),
        "baseline_n_features": len(baseline_cols),
        "engineered_n_features": len(engineered_cols),
        "engineered_col": NEW_COL,
        "formula": "station_price - (last_min + pct_through * (last_max - last_min))",
    }
    (OUT / "step2_meta.json").write_text(json.dumps(meta, indent=2))

    total_s = time.perf_counter() - overall_t0
    print(f"\n[total wall] {total_s:.1f}s")


if __name__ == "__main__":
    main()
