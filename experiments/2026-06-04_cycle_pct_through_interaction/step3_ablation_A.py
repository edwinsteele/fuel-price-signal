"""Ablation A: does the engineered phase-residual truly replace its siblings?

Three configs, 3 seeds each, single val window:
- baseline:  50 Phase 4 features.
- additive:  + station_minus_expected_phase_price (51 features). Mirrors step 2.
- ablationA: + station_minus_expected_phase_price
              − station_minus_last_min_cents
              − station_minus_last_max_cents             (49 features)

Hypothesis: the engineered residual is the diagonal projection of the
saddle that the two sibling features carry. If it really absorbs that
signal, ablationA's logloss should not regress versus baseline (and may
match or beat additive). If ablationA regresses materially, the siblings
carry information that the engineered formula's diagonal projection drops
— in which case true replacement sacrifices signal.

Per [[feedback-instrument-walltime]] all wall times are logged.
Per [[feedback-load-features-helper]] uses load_features().
"""

from __future__ import annotations

import json
import pathlib
import time

import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier

from fuel_signal import evaluate as _ev
from fuel_signal.features import FEATURE_COLUMNS, LGA_FEATURE_COLUMNS, load_features

OUT = pathlib.Path(__file__).parent
SEEDS = (42, 43, 44)
NEW_COL = "station_minus_expected_phase_price"
DROP_FOR_ABLATION_A = (
    "station_minus_last_min_cents",
    "station_minus_last_max_cents",
)


def add_engineered(df: pd.DataFrame) -> pd.DataFrame:
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

    p_val = model.predict_proba(X_val)[:, 1]
    ll = _ev.log_loss(y_val, p_val)
    br = _ev.brier(y_val, p_val)
    print(f"  [{label} fit] {fit_s:.1f}s  logloss={ll:.6f}  brier={br:.6f}")
    return {"model": model, "logloss": ll, "brier": br,
            "X_val": X_val, "fit_s": fit_s}


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
    print("Loading features (load_features) …")
    t0 = time.perf_counter()
    df = load_features()
    df["price_date"] = pd.to_datetime(df["price_date"])
    print(f"  [load_features] {time.perf_counter() - t0:.1f}s  rows={len(df):,}")

    df = add_engineered(df)

    train, val, _test = _ev.split(df)
    print(f"  train={len(train):,}  val={len(val):,}")

    baseline_cols = FEATURE_COLUMNS + LGA_FEATURE_COLUMNS
    additive_cols = baseline_cols + [NEW_COL]
    ablation_cols = [c for c in additive_cols if c not in DROP_FOR_ABLATION_A]
    assert len(baseline_cols) == 50
    assert len(additive_cols) == 51
    assert len(ablation_cols) == 49
    print(f"  configs: baseline={len(baseline_cols)} additive={len(additive_cols)} ablationA={len(ablation_cols)}")

    rows = []
    shap_b_acc = []
    shap_a_acc = []  # additive
    shap_x_acc = []  # ablationA

    for seed in SEEDS:
        print(f"\n=== seed {seed} ===")
        b = fit_and_score(train, val, baseline_cols, seed, "baseline")
        a = fit_and_score(train, val, additive_cols, seed, "additive")
        x = fit_and_score(train, val, ablation_cols, seed, "ablationA")

        rng = np.random.default_rng(seed)
        n_shap = min(8000, b["X_val"].shape[0])
        idx = rng.choice(b["X_val"].shape[0], size=n_shap, replace=False)
        b_shap, _ = mean_abs_shap(b["model"], b["X_val"][idx], "baseline")
        a_shap, _ = mean_abs_shap(a["model"], a["X_val"][idx], "additive")
        x_shap, _ = mean_abs_shap(x["model"], x["X_val"][idx], "ablationA")
        shap_b_acc.append(b_shap)
        shap_a_acc.append(a_shap)
        shap_x_acc.append(x_shap)

        rows.append({
            "seed": seed,
            "baseline_logloss": b["logloss"],
            "additive_logloss": a["logloss"],
            "ablationA_logloss": x["logloss"],
            "delta_additive_vs_baseline": a["logloss"] - b["logloss"],
            "delta_ablationA_vs_baseline": x["logloss"] - b["logloss"],
            "delta_ablationA_vs_additive": x["logloss"] - a["logloss"],
            "baseline_brier": b["brier"],
            "additive_brier": a["brier"],
            "ablationA_brier": x["brier"],
            "baseline_fit_s": b["fit_s"],
            "additive_fit_s": a["fit_s"],
            "ablationA_fit_s": x["fit_s"],
        })

    results = pd.DataFrame(rows)
    results.to_csv(OUT / "step3_results.csv", index=False)
    print("\nPer-seed:")
    print(results[[
        "seed", "baseline_logloss", "additive_logloss", "ablationA_logloss",
        "delta_additive_vs_baseline", "delta_ablationA_vs_baseline",
    ]].to_string(index=False, float_format=lambda v: f"{v:.6f}"))

    def summary(col: str, label: str) -> tuple[float, float]:
        m, s = results[col].mean(), results[col].std(ddof=1)
        print(f"  {label:<35s} mean={m:+.6f}  seed_std={s:.6f}  |m|/s={abs(m)/max(s,1e-9):.2f}")
        return float(m), float(s)

    print("\nDeltas (single val window, 3 seeds):")
    m_a, s_a = summary("delta_additive_vs_baseline", "additive − baseline")
    m_x, s_x = summary("delta_ablationA_vs_baseline", "ablationA − baseline")
    m_xa, s_xa = summary("delta_ablationA_vs_additive", "ablationA − additive")

    # SHAP rankings averaged across seeds.
    base_shap = np.mean(np.stack(shap_b_acc), axis=0)
    add_shap = np.mean(np.stack(shap_a_acc), axis=0)
    abl_shap = np.mean(np.stack(shap_x_acc), axis=0)
    base_df = pd.DataFrame({"feature": baseline_cols, "mean_abs_shap_baseline": base_shap})
    add_df = pd.DataFrame({"feature": additive_cols, "mean_abs_shap_additive": add_shap})
    abl_df = pd.DataFrame({"feature": ablation_cols, "mean_abs_shap_ablationA": abl_shap})
    merged = base_df.merge(add_df, on="feature", how="outer").merge(abl_df, on="feature", how="outer")
    merged = merged.sort_values("mean_abs_shap_ablationA", ascending=False)
    merged.to_csv(OUT / "step3_shap_ranking.csv", index=False)

    focus_features = [
        NEW_COL,
        "station_price_cents",
        "cycle_pct_through",
        "cycle_last_min_cents",
        "cycle_last_max_cents",
        "station_minus_last_min_cents",
        "station_minus_last_max_cents",
        "station_minus_sydney_avg_cents",
        "station_minus_lga_mean_cents",
        "stickiness_score",
    ]
    print("\nFocus features (mean|SHAP|, by ablationA rank):")
    focus = merged[merged.feature.isin(focus_features)].sort_values(
        "mean_abs_shap_ablationA", ascending=False
    )
    print(focus.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\nTop 15 features in ablationA model:")
    print(merged.head(15).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    meta = {
        "seeds": list(SEEDS),
        "baseline_n_features": len(baseline_cols),
        "additive_n_features": len(additive_cols),
        "ablationA_n_features": len(ablation_cols),
        "engineered_col": NEW_COL,
        "dropped_for_ablation_A": list(DROP_FOR_ABLATION_A),
        "formula": "station_price - (last_min + pct_through * (last_max - last_min))",
        "additive_delta_mean": m_a, "additive_delta_seed_std": s_a,
        "ablationA_delta_mean": m_x, "ablationA_delta_seed_std": s_x,
        "ablationA_vs_additive_mean": m_xa,
        "ablationA_vs_additive_seed_std": s_xa,
    }
    (OUT / "step3_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\n[total wall] {time.perf_counter() - overall_t0:.1f}s")


if __name__ == "__main__":
    main()
