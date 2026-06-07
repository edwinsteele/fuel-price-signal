"""SHAP analysis CLI for fuel-signal models.

Loads a fitted joblib model bundle and features CSV, computes TreeExplainer SHAP
values on the requested split, and emits:
  - shap_values.npy            (n_rows, n_features)
  - X_val.npy                  (n_rows, n_features) feature matrix for the split
  - feature_columns.json       ordered list of feature names
  - summary.csv                per-feature: mean_abs_shap, rank, r, nan_fraction
  - partner_scores.csv         per (feature, partner): approx interaction scores
  - dependence/<feature>.png   one scatter per feature

Usage::

    uv run python -m fuel_signal.shap_report \\
        --model data/models/lgbm.joblib \\
        --features data/features.csv \\
        --split val \\
        --output experiments/shap_<phase>/
"""

from __future__ import annotations

import json
import pathlib
import warnings
from typing import Literal

import click
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

from fuel_signal import evaluate as _ev  # noqa: E402

DEFAULT_MODEL = pathlib.Path("data/models/lgbm.joblib")
DEFAULT_FEATURES_CSV = pathlib.Path("data/features.csv")

Split = Literal["train", "val", "test"]

_MAX_SCATTER_POINTS = 5_000


def _load_split(df: pd.DataFrame, split: Split) -> pd.DataFrame:
    train, val, test = _ev.split(df)
    return {"train": train, "val": val, "test": test}[split]


def compute_shap(model: object, X: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Run TreeExplainer on X; return (n_rows, n_features) array."""
    explainer = shap.TreeExplainer(model)
    with warnings.catch_warnings():
        # SHAP warns that binary-classifier output changed to list[ndarray]; we
        # already handle both forms below, so the warning is noise.
        warnings.filterwarnings(
            "ignore",
            message="LightGBM binary classifier with TreeExplainer shap values output",
            category=UserWarning,
        )
        raw = explainer.shap_values(X)
    if isinstance(raw, list):
        return raw[1]
    return raw


def build_summary(
    feature_columns: list[str],
    X: np.ndarray,
    shap_values: np.ndarray,
) -> pd.DataFrame:
    """Return per-feature summary DataFrame.

    Columns: feature, mean_abs_shap, rank, r, nan_fraction.
    r is the signed Pearson r(feature_values, shap_values), or NaN
    when the feature is all-NaN or has zero variance.
    """
    n_feat = len(feature_columns)
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    r_values = []
    nan_fracs = []
    for i in range(n_feat):
        vals = X[:, i]
        nan_mask = np.isnan(vals)
        nan_fracs.append(float(nan_mask.mean()))
        v = vals[~nan_mask]
        s = shap_values[~nan_mask, i]
        if len(v) < 2 or np.std(v) == 0 or np.std(s) == 0:
            r_values.append(float("nan"))
        else:
            r_values.append(float(np.corrcoef(v, s)[0, 1]))

    df = pd.DataFrame({
        "feature": feature_columns,
        "mean_abs_shap": mean_abs,
        "r": r_values,
        "nan_fraction": nan_fracs,
    })
    df = df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    df.insert(2, "rank", range(1, len(df) + 1))
    return df


def approx_interaction_scores(main_idx: int, sv: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Compute SHAP approximate interaction scores for one feature against all others.

    Replicates shap.utils.approximate_interactions so we get raw scores, not just rank.
    """
    x_main = X[:, main_idx]
    order = np.argsort(x_main)
    shap_main = sv[order, main_idx]
    inc = int(np.clip(len(x_main) / 10.0, 1, 50))
    scores = np.zeros(X.shape[1], dtype=np.float64)
    for j in range(X.shape[1]):
        if j == main_idx:
            continue
        partner = X[order, j].astype(np.float64)
        s = 0.0
        for k in range(0, len(x_main), inc):
            a = shap_main[k : k + inc]
            b = partner[k : k + inc]
            mask = ~np.isnan(b)
            if mask.sum() < 3:
                continue
            a = a[mask]
            b = b[mask]
            if np.std(a) == 0 or np.std(b) == 0:
                continue
            s += abs(np.corrcoef(a, b)[0, 1])
        scores[j] = s
    return scores


def compute_partner_scores(
    feature_columns: list[str],
    X: np.ndarray,
    sv: np.ndarray,
) -> pd.DataFrame:
    """Compute approx interaction scores for all (feature, partner) pairs.

    Returns DataFrame with columns: feature, partner, score, pct_of_top, pct_of_total.
    Zero-score partners are omitted to keep the file compact.
    """
    rows = []
    for i, feat in enumerate(feature_columns):
        scores = approx_interaction_scores(i, sv, X)
        total = float(scores.sum())
        top = float(scores.max())
        for j, partner in enumerate(feature_columns):
            if j == i or scores[j] == 0:
                continue
            rows.append({
                "feature": feat,
                "partner": partner,
                "score": float(scores[j]),
                "pct_of_top": float(scores[j] / top) if top > 0 else 0.0,
                "pct_of_total": float(scores[j] / total) if total > 0 else 0.0,
            })
    return pd.DataFrame(rows, columns=["feature", "partner", "score", "pct_of_top", "pct_of_total"])


def save_dependence_plots(
    feature_columns: list[str],
    X: np.ndarray,
    shap_values: np.ndarray,
    out_dir: pathlib.Path,
) -> None:
    """Save one dependence scatter PNG per feature into out_dir.

    Uses shap.dependence_plot with interaction_index="auto" so each scatter is
    coloured by whichever other feature explains the most variance in the SHAP
    values. The colourbar label names the interaction feature. All-NaN features
    get a plain "all NaN" placeholder. Rows are subsampled to _MAX_SCATTER_POINTS
    so plots remain readable and generation stays fast on large val splits.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    if X.shape[0] > _MAX_SCATTER_POINTS:
        sample_idx = rng.choice(X.shape[0], _MAX_SCATTER_POINTS, replace=False)
        X_plot = X[sample_idx]
        sv_plot = shap_values[sample_idx]
    else:
        X_plot = X
        sv_plot = shap_values

    for i, feat in enumerate(feature_columns):
        nan_mask = np.isnan(X_plot[:, i])
        safe_name = feat.replace("/", "_")

        if nan_mask.all():
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, "all NaN", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(feat, fontsize=9)
            fig.tight_layout()
            fig.savefig(out_dir / f"{safe_name}.png", dpi=100)
            plt.close(fig)
            continue

        # shap.dependence_plot opens its own figure; capture and close it.
        shap.dependence_plot(
            i,
            sv_plot,
            X_plot,
            feature_names=feature_columns,
            interaction_index="auto",
            show=False,
        )
        fig = plt.gcf()
        fig.set_size_inches(7, 4)
        plt.tight_layout()
        fig.savefig(out_dir / f"{safe_name}.png", dpi=100)
        plt.close(fig)


def run_shap_report(
    model_path: pathlib.Path,
    features_csv: pathlib.Path,
    split: Split,
    output_dir: pathlib.Path,
) -> dict:
    """Compute SHAP on the requested split; write artifacts to output_dir.

    Returns a dict with 'shap_values' (ndarray), 'summary' (DataFrame),
    'feature_columns' (list), and 'n_rows' (int).
    """
    bundle = joblib.load(model_path)
    model = bundle["pipeline"]
    feature_columns: list[str] = bundle["feature_columns"]

    df = pd.read_csv(features_csv)
    split_df = _load_split(df, split)
    if split_df.empty:
        raise ValueError(f"Split '{split}' is empty after applying canonical date boundaries.")

    X_df = split_df[feature_columns]
    sv = compute_shap(model, X_df)
    X = X_df.to_numpy(dtype=float)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "shap_values.npy", sv)
    np.save(output_dir / "X_val.npy", X)
    with open(output_dir / "feature_columns.json", "w") as fh:
        json.dump(feature_columns, fh)

    summary = build_summary(feature_columns, X, sv)
    summary.to_csv(output_dir / "summary.csv", index=False)

    partner_scores = compute_partner_scores(feature_columns, X, sv)
    partner_scores.to_csv(output_dir / "partner_scores.csv", index=False)

    dep_dir = output_dir / "dependence"
    save_dependence_plots(feature_columns, X, sv, dep_dir)

    return {
        "shap_values": sv,
        "summary": summary,
        "feature_columns": feature_columns,
        "n_rows": len(split_df),
    }


@click.command("shap_report")
@click.option(
    "--model",
    "model_path",
    default=str(DEFAULT_MODEL),
    show_default=True,
    help="Path to fitted joblib model bundle (pipeline + feature_columns).",
)
@click.option(
    "--features",
    "features_csv",
    default=str(DEFAULT_FEATURES_CSV),
    show_default=True,
    help="Path to features.csv produced by `python -m fuel_signal.features`.",
)
@click.option(
    "--split",
    "split",
    type=click.Choice(["train", "val", "test"]),
    default="val",
    show_default=True,
    help="Which canonical split to evaluate SHAP on.",
)
@click.option(
    "--output",
    "output_dir",
    required=True,
    help="Directory to write shap_values.npy, summary.csv, and dependence/ PNGs.",
)
def main(model_path: str, features_csv: str, split: str, output_dir: str) -> None:
    """Compute SHAP values for a fitted model; write summary.csv and dependence PNGs."""
    mp = pathlib.Path(model_path)
    if not mp.exists():
        raise click.ClickException(f"Model not found: {model_path}")

    fp = pathlib.Path(features_csv)
    if not fp.exists():
        raise click.ClickException(
            f"Features CSV not found: {features_csv}. "
            "Run 'uv run python -m fuel_signal.features' first."
        )

    out = pathlib.Path(output_dir)
    click.echo(f"Loading model from {mp}")
    click.echo(f"Loading features from {fp}")
    click.echo(f"Split: {split}")

    result = run_shap_report(mp, fp, split, out)  # type: ignore[arg-type]

    summary = result["summary"]
    click.echo(f"\nSHAP summary — {split} split  (n={result['n_rows']:,})")
    click.echo(f"{'rank':>4}  {'feature':<45} {'mean|SHAP|':>10}  {'r':>6}  {'nan%':>5}")
    click.echo("─" * 78)
    for _, row in summary.head(25).iterrows():
        r_val = row["r"]
        r_str = f"{r_val:+.2f}" if r_val == r_val else "   ?"
        click.echo(
            f"{int(row['rank']):>4}  {row['feature']:<45} "
            f"{row['mean_abs_shap']:>10.4f}  {r_str:>6}  {row['nan_fraction']:>5.1%}"
        )
    if len(summary) > 25:
        click.echo(f"     … {len(summary) - 25} more features (see summary.csv)")

    click.echo(f"\nArtifacts written to {out}/")
    click.echo(f"  shap_values.npy   ({result['shap_values'].shape[0]:,} × {result['shap_values'].shape[1]})")
    click.echo(f"  X_val.npy         ({result['shap_values'].shape[0]:,} × {result['shap_values'].shape[1]})")
    click.echo(f"  feature_columns.json  ({len(result['feature_columns'])} features)")
    click.echo(f"  summary.csv       ({len(summary)} features)")
    click.echo("  partner_scores.csv")
    click.echo(f"  dependence/       ({len(result['feature_columns'])} PNGs)")


if __name__ == "__main__":
    main()
