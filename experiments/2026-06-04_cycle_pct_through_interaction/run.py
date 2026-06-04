"""SHAP interaction surfaces for cycle_pct_through.

Tests the 'phase modulator' hypothesis: cycle_pct_through doesn't just have a
main effect, it gates how the model reads station_price vs Sydney-cycle
anchors. If the hypothesis holds the (cycle_pct_through × cents_feature)
interaction surface should be a saddle (sign flips with pct_through).

Also computes interaction matrices for two control pairings:
- cycle_pct_through × stickiness_score  (non-Sydney-cycle anchor, lower
  partner score: 4.2%)
- cycle_pct_through × cycle_peak_count   (Sydney series but slow-changing
  integer, low partner score: 0.9%)

Subsamples val to keep TreeExplainer.shap_interaction_values tractable
(O(n * M^2 * trees)).
"""

from __future__ import annotations

import json
import pathlib

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

from fuel_signal import evaluate as _ev  # noqa: E402
from fuel_signal.features import load_features  # noqa: E402

OUT = pathlib.Path(__file__).parent
MODEL_PATH = pathlib.Path("data/models/lgbm_phase4.joblib")
N_SAMPLE = 10_000
SEED = 0

MAIN = "cycle_pct_through"
TARGETS = [
    "station_minus_last_max_cents",   # ~10.24% partner score
    "station_minus_last_min_cents",   # ~10.24%
    "station_price_cents",            # ~10.28%
    "station_minus_sydney_avg_cents", # ~10.30%
    "station_minus_lga_mean_cents",   # ~8.90%  (LGA anchor, weaker)
    "stickiness_score",               # ~4.25%  (non-Sydney-cycle)
    "cycle_peak_count",               # ~0.89%  (Sydney series, slow)
]


def main() -> None:
    bundle = joblib.load(MODEL_PATH)
    pipeline = bundle["pipeline"]
    cols = bundle["feature_columns"]
    print(f"Model: {MODEL_PATH.name}  features: {len(cols)}")

    print("Loading features (parquet cache via load_features) + reproducing val split…")
    df = load_features()
    df["price_date"] = pd.to_datetime(df["price_date"])
    _train, val, _test = _ev.split(df)
    X_val = val[cols].to_numpy(dtype=float)
    print(f"  val rows: {X_val.shape[0]:,}")

    rng = np.random.default_rng(SEED)
    idx = rng.choice(X_val.shape[0], size=min(N_SAMPLE, X_val.shape[0]), replace=False)
    Xs = X_val[idx]
    print(f"  subsample: {Xs.shape[0]:,}")

    clf = pipeline.named_steps["clf"] if hasattr(pipeline, "named_steps") else pipeline
    explainer = shap.TreeExplainer(clf.booster_)
    print("Computing shap_interaction_values …")
    iv = explainer.shap_interaction_values(Xs)
    if isinstance(iv, list):
        iv = iv[1] if len(iv) == 2 else iv[0]
    iv = np.asarray(iv)
    print(f"  iv.shape: {iv.shape}")
    np.save(OUT / "iv.npy", iv)
    np.save(OUT / "Xs.npy", Xs)

    main_idx = cols.index(MAIN)

    summary = []
    for target in TARGETS:
        if target not in cols:
            print(f"  skip {target}: not in model")
            continue
        t_idx = cols.index(target)

        # iv is symmetric: iv[:, i, j] = iv[:, j, i] = signed interaction
        # contribution to log-odds.
        pair = iv[:, main_idx, t_idx]
        m_val = Xs[:, main_idx]
        t_val = Xs[:, t_idx]
        mask = np.isfinite(m_val) & np.isfinite(t_val) & np.isfinite(pair)
        m_val, t_val, pair = m_val[mask], t_val[mask], pair[mask]
        n = len(pair)

        mean_abs = float(np.mean(np.abs(pair)))

        # Saddle test: split target at median, split main (pct_through) at
        # median. Report mean signed interaction in each of 4 quadrants. A
        # saddle shows alternating signs along a diagonal.
        m_med = float(np.median(m_val))
        t_med = float(np.median(t_val))
        q = {}
        for mlab, mlo in (("lo", m_val < m_med), ("hi", m_val >= m_med)):
            for tlab, tlo in (("lo", t_val < t_med), ("hi", t_val >= t_med)):
                cell = pair[mlo & tlo]
                q[f"{mlab}_{tlab}"] = float(np.mean(cell)) if len(cell) else float("nan")

        saddle = abs((q["lo_lo"] + q["hi_hi"]) - (q["lo_hi"] + q["hi_lo"]))
        summary.append({
            "target": target,
            "n": n,
            "mean_abs_interaction": mean_abs,
            "q_lo_lo": q["lo_lo"],
            "q_lo_hi": q["lo_hi"],
            "q_hi_lo": q["hi_lo"],
            "q_hi_hi": q["hi_hi"],
            "saddle_score": saddle,
            "main_median": m_med,
            "target_median": t_med,
        })

        # Plot.
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
        # Left: 2D hexbin of mean interaction over (target, pct_through) grid.
        hb = ax1.hexbin(t_val, m_val, C=pair, reduce_C_function=np.mean,
                        gridsize=40, cmap="coolwarm",
                        vmin=-max(abs(np.percentile(pair, 1)), abs(np.percentile(pair, 99))),
                        vmax= max(abs(np.percentile(pair, 1)), abs(np.percentile(pair, 99))))
        ax1.axhline(m_med, color="k", lw=0.5, alpha=0.4)
        ax1.axvline(t_med, color="k", lw=0.5, alpha=0.4)
        ax1.set_xlabel(target)
        ax1.set_ylabel(MAIN)
        ax1.set_title(f"mean SHAP interaction (log-odds)\n|S|={mean_abs:.4f}  saddle={saddle:.4f}")
        plt.colorbar(hb, ax=ax1)

        # Right: dependence — interaction vs target, coloured by pct_through.
        c_min = float(np.percentile(m_val, 1))
        c_max = float(np.percentile(m_val, 99))
        sc = ax2.scatter(t_val, pair, c=m_val, cmap="coolwarm",
                         vmin=c_min, vmax=c_max, s=2, alpha=0.5, edgecolors="none")
        ax2.axhline(0, color="k", lw=0.5, alpha=0.4)
        ax2.set_xlabel(target)
        ax2.set_ylabel(f"SHAP interaction ({MAIN} × {target})")
        ax2.set_title(f"signed interaction vs {target}, coloured by {MAIN}")
        plt.colorbar(sc, ax=ax2, label=MAIN)

        fig.suptitle(f"{MAIN} × {target}  (n={n:,})", fontsize=11)
        fig.tight_layout()
        out_png = OUT / f"iv_{target}.png"
        fig.savefig(out_png, dpi=110)
        plt.close(fig)
        print(f"  saved {out_png.name}")

    pd.DataFrame(summary).to_csv(OUT / "summary.csv", index=False)
    print("\nQuadrant means of signed SHAP interaction (log-odds units):")
    sdf = pd.DataFrame(summary)
    print(sdf[["target", "mean_abs_interaction", "q_lo_lo", "q_lo_hi",
               "q_hi_lo", "q_hi_hi", "saddle_score"]]
          .to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    meta = {
        "model_path": str(MODEL_PATH),
        "n_sample": int(N_SAMPLE),
        "seed": SEED,
        "main": MAIN,
        "targets": TARGETS,
        "iv_shape": list(iv.shape),
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
