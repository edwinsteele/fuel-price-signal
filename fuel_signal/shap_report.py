"""SHAP analysis CLI for fuel-signal models.

Loads a fitted joblib model bundle and features CSV, computes TreeExplainer SHAP
values on the requested split, and emits:
  - shap_values.npy            (n_rows, n_features)
  - summary.csv                per-feature: mean_abs_shap, rank, sign_of_r, nan_fraction
  - dependence/<feature>.png   one scatter per feature

Usage::

    uv run python -m fuel_signal.shap_report \\
        --model data/models/lgbm.joblib \\
        --features data/features.csv \\
        --split val \\
        --output experiments/shap_<phase>/
"""

from __future__ import annotations

import pathlib
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


def compute_shap(model: object, X: np.ndarray) -> np.ndarray:
    """Run TreeExplainer on X; return (n_rows, n_features) array."""
    explainer = shap.TreeExplainer(model)
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

    Columns: feature, mean_abs_shap, rank, sign_of_r, nan_fraction.
    sign_of_r is the sign of Pearson r(feature_values, shap_values), or NaN
    when the feature is all-NaN or has zero variance.
    """
    n_feat = len(feature_columns)
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    signs = []
    nan_fracs = []
    for i in range(n_feat):
        vals = X[:, i]
        nan_mask = np.isnan(vals)
        nan_fracs.append(float(nan_mask.mean()))
        v = vals[~nan_mask]
        s = shap_values[~nan_mask, i]
        if len(v) < 2 or np.std(v) == 0 or np.std(s) == 0:
            signs.append(float("nan"))
        else:
            r = float(np.corrcoef(v, s)[0, 1])
            signs.append(float(np.sign(r)))

    df = pd.DataFrame({
        "feature": feature_columns,
        "mean_abs_shap": mean_abs,
        "sign_of_r": signs,
        "nan_fraction": nan_fracs,
    })
    df = df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    df.insert(2, "rank", range(1, len(df) + 1))
    return df


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
    get a plain "all NaN" placeholder.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, feat in enumerate(feature_columns):
        vals = X[:, i]
        nan_mask = np.isnan(vals)

        safe_name = feat.replace("/", "_")
        if nan_mask.all():
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, "all NaN", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(feat, fontsize=9)
            fig.tight_layout()
            fig.savefig(out_dir / f"{safe_name}.png", dpi=100)
            plt.close(fig)
            continue

        # shap.dependence_plot opens its own figure; capture it.
        shap.dependence_plot(
            i,
            shap_values,
            X,
            feature_names=feature_columns,
            interaction_index="auto",
            show=False,
        )
        plt.gcf().set_size_inches(7, 4)
        plt.tight_layout()
        plt.savefig(out_dir / f"{safe_name}.png", dpi=100)
        plt.close("all")


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

    X = split_df[feature_columns].to_numpy(dtype=float)
    sv = compute_shap(model, X)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "shap_values.npy", sv)

    summary = build_summary(feature_columns, X, sv)
    summary.to_csv(output_dir / "summary.csv", index=False)

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
    click.echo(f"{'rank':>4}  {'feature':<45} {'mean|SHAP|':>10}  {'sign_r':>6}  {'nan%':>5}")
    click.echo("─" * 78)
    for _, row in summary.head(25).iterrows():
        sign_str = "+" if row["sign_of_r"] > 0 else ("-" if row["sign_of_r"] < 0 else "?")
        click.echo(
            f"{int(row['rank']):>4}  {row['feature']:<45} "
            f"{row['mean_abs_shap']:>10.4f}  {sign_str:>6}  {row['nan_fraction']:>5.1%}"
        )
    if len(summary) > 25:
        click.echo(f"     … {len(summary) - 25} more features (see summary.csv)")

    click.echo(f"\nArtifacts written to {out}/")
    click.echo(f"  shap_values.npy  ({result['shap_values'].shape[0]:,} × {result['shap_values'].shape[1]})")
    click.echo(f"  summary.csv      ({len(summary)} features)")
    click.echo(f"  dependence/      ({len(result['feature_columns'])} PNGs)")


if __name__ == "__main__":
    main()
