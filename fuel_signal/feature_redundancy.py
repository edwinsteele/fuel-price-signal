"""Feature redundancy and decomposition analysis from SHAP.

Two analyses in one pass over a fitted model + split:

#1 Redundancy clustering
    Cluster features by row-wise correlation of their SHAP value columns.
    Features in the same cluster contribute the same signal to the model,
    even when their raw values are not linearly correlated.

#2 Decomposition candidates
    Rank features by how diffuse their SHAP interaction mass is across
    partners. A feature whose interaction mass is spread across many
    partners is likely carrying multiple distinct signals — it is a
    candidate to be decomposed into separate engineered features.

Usage::

    uv run python -m fuel_signal.feature_redundancy \\
        --model data/models/lgbm.joblib \\
        --features data/features.csv \\
        --split val \\
        --output experiments/redundancy_<tag>/ \\
        --cluster-threshold 0.5 \\
        --interaction-sample 3000
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
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage  # noqa: E402
from scipy.spatial.distance import squareform  # noqa: E402

from fuel_signal import evaluate as _ev  # noqa: E402
from fuel_signal.cv_report import run_paired_cv as _run_paired_cv  # noqa: E402
from fuel_signal.shap_report import compute_shap  # noqa: E402

DEFAULT_MODEL = pathlib.Path("data/models/lgbm.joblib")
DEFAULT_FEATURES_CSV = pathlib.Path("data/features.csv")

Split = Literal["train", "val", "test"]


# ---------------------------------------------------------------------------
# #1 Redundancy clustering
# ---------------------------------------------------------------------------

def shap_correlation_matrix(sv: np.ndarray) -> np.ndarray:
    """Return F×F Pearson r between SHAP value columns.

    Zero-variance columns yield NaN. `cluster_features` then maps NaN → r = 0
    → distance d = 1 (the maximum), so such columns form singleton clusters
    rather than collapsing into a neighbour.
    """
    n_feat = sv.shape[1]
    corr = np.full((n_feat, n_feat), np.nan, dtype=np.float64)
    stds = np.std(sv, axis=0)
    for i in range(n_feat):
        if stds[i] == 0:
            continue
        for j in range(i, n_feat):
            if stds[j] == 0:
                continue
            if i == j:
                corr[i, j] = 1.0
                continue
            r = float(np.corrcoef(sv[:, i], sv[:, j])[0, 1])
            corr[i, j] = r
            corr[j, i] = r
    return corr


def cluster_features(
    corr: np.ndarray,
    threshold: float,
    method: str = "average",
) -> tuple[np.ndarray, np.ndarray]:
    """Hierarchical clustering on 1 − |corr| distance.

    Returns (cluster_ids, linkage_matrix). Features with NaN correlation (zero
    SHAP variance) are placed at maximum distance from all others, so they
    form their own singleton clusters.
    """
    d = 1.0 - np.abs(np.nan_to_num(corr, nan=0.0))
    np.fill_diagonal(d, 0.0)
    d = (d + d.T) / 2.0  # enforce symmetry against float roundoff
    condensed = squareform(d, checks=False)
    Z = linkage(condensed, method=method)
    labels = fcluster(Z, t=threshold, criterion="distance")
    return labels, Z


def build_cluster_table(
    feature_columns: list[str],
    mean_abs_shap: np.ndarray,
    cluster_ids: np.ndarray,
) -> pd.DataFrame:
    """Return per-feature cluster assignment with sibling list per row."""
    df = pd.DataFrame({
        "feature": feature_columns,
        "cluster_id": cluster_ids.astype(int),
        "mean_abs_shap": mean_abs_shap,
    })
    siblings: list[str] = []
    for _, row in df.iterrows():
        same = df[(df["cluster_id"] == row["cluster_id"]) & (df["feature"] != row["feature"])]
        siblings.append(",".join(same["feature"].tolist()))
    df["siblings"] = siblings
    df = df.sort_values(
        ["cluster_id", "mean_abs_shap"], ascending=[True, False]
    ).reset_index(drop=True)
    return df


def save_dendrogram(
    feature_columns: list[str],
    Z: np.ndarray,
    threshold: float,
    out_path: pathlib.Path,
) -> None:
    fig, ax = plt.subplots(figsize=(max(8, 0.4 * len(feature_columns)), 5))
    dendrogram(
        Z,
        labels=feature_columns,
        color_threshold=threshold,
        leaf_rotation=90,
        ax=ax,
    )
    ax.axhline(threshold, color="grey", linestyle="--", linewidth=0.8)
    ax.set_ylabel("1 − |corr(SHAP)|")
    ax.set_title("Feature redundancy by SHAP-column correlation")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# #2 Decomposition candidates via SHAP interactions
# ---------------------------------------------------------------------------

def compute_interaction_matrix(
    model: object,
    X: np.ndarray,
    sample_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """Return (F×F mean |interaction|, n_rows_used).

    Subsamples X to keep TreeExplainer.shap_interaction_values tractable: it is
    O(rows × trees × depth²), much slower than plain SHAP.
    """
    if X.shape[0] > sample_size:
        idx = rng.choice(X.shape[0], sample_size, replace=False)
        Xs = X[idx]
    else:
        Xs = X

    explainer = shap.TreeExplainer(model)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="LightGBM binary classifier with TreeExplainer shap values output",
            category=UserWarning,
        )
        iv = explainer.shap_interaction_values(Xs)
    if isinstance(iv, list):
        iv = iv[1]
    return np.mean(np.abs(iv), axis=0), Xs.shape[0]


def decomposition_scores(
    feature_columns: list[str],
    interaction_matrix: np.ndarray,
) -> pd.DataFrame:
    """Per-feature decomposition diagnostics derived from the F×F interaction matrix.

    For each feature i, off-diagonal row i gives the |SHAP interaction| mass it
    shares with every other feature. We summarise that distribution with:

    - main_effect_share: diag[i] / (diag[i] + Σ off-diag) — how much of the
      feature's contribution is unmodulated.
    - total_partner_mass: Σ off-diag[i] — absolute scale of interactions.
    - entropy_norm: normalised Shannon entropy of partner-mass distribution
      across F−1 partners, in [0, 1]. High = diffuse (multi-signal candidate).
    - top1/2/3 partner + share: concentration tells.
    - n_partners_ge_5pct: how many partners carry ≥5% of partner mass.
    """
    F = interaction_matrix.shape[0]
    main = np.diag(interaction_matrix).astype(float)
    rows = []
    log_denom = np.log(F - 1) if F > 2 else 1.0
    for i, feat in enumerate(feature_columns):
        partners = interaction_matrix[i].astype(float).copy()
        partners[i] = 0.0
        total_partner = float(partners.sum())
        total = float(main[i] + total_partner)
        main_share = float(main[i] / total) if total > 0 else float("nan")

        if total_partner == 0:
            entropy_norm = 0.0
            top_idx = [-1, -1, -1]
            top_share = [0.0, 0.0, 0.0]
            n_ge_5 = 0
        else:
            p = partners / total_partner
            nz = p[p > 0]
            entropy = float(-(nz * np.log(nz)).sum())
            entropy_norm = float(entropy / log_denom) if log_denom > 0 else 0.0
            # Exclude self before ranking so top-N never reports feature i,
            # even in degenerate cases with very few non-zero partners.
            order = [int(o) for o in np.argsort(-partners) if int(o) != i]
            top_idx = [order[k] if k < len(order) else -1 for k in range(3)]
            top_share = [
                float(partners[order[k]] / total_partner) if k < len(order) else 0.0
                for k in range(3)
            ]
            n_ge_5 = int((p >= 0.05).sum())

        def _name(j: int) -> str:
            return feature_columns[j] if j >= 0 else ""

        rows.append({
            "feature": feat,
            "main_effect": main[i],
            "total_partner_mass": total_partner,
            "main_effect_share": main_share,
            "entropy_norm": entropy_norm,
            "n_partners_ge_5pct": n_ge_5,
            "top1_partner": _name(top_idx[0]),
            "top1_share": top_share[0],
            "top2_partner": _name(top_idx[1]),
            "top2_share": top_share[1],
            "top3_partner": _name(top_idx[2]),
            "top3_share": top_share[2],
        })
    df = pd.DataFrame(rows)
    return df.sort_values(
        ["entropy_norm", "total_partner_mass"], ascending=[False, False]
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Paired walk-forward CV per candidate
# ---------------------------------------------------------------------------

_CV_COLS = [
    "paired_cv_median_delta",
    "paired_cv_worst_fold_delta",
    "paired_cv_fold_wins",
    "paired_cv_csv",
]


def _cv_nan_row() -> dict:
    return {
        "paired_cv_median_delta": float("nan"),
        "paired_cv_worst_fold_delta": float("nan"),
        "paired_cv_fold_wins": "",
        "paired_cv_csv": "",
    }


def _cv_summary(results: list[dict], csv_rel_path: str) -> dict:
    if not results:
        return _cv_nan_row()
    deltas = np.array([r["delta"] for r in results])
    n_wins = int((deltas < 0).sum())
    return {
        "paired_cv_median_delta": float(np.median(deltas)),
        "paired_cv_worst_fold_delta": float(np.max(deltas)),
        "paired_cv_fold_wins": f"{n_wins}/{len(results)}",
        "paired_cv_csv": csv_rel_path,
    }


def _add_nan_cv_cols(df: pd.DataFrame) -> pd.DataFrame:
    nan_row = _cv_nan_row()
    for col in _CV_COLS:
        df[col] = nan_row[col]
    return df


def _run_cluster_cv(
    cluster_table: pd.DataFrame,
    df: pd.DataFrame,
    model_path: pathlib.Path,
    output_dir: pathlib.Path,
    *,
    cv_seed: int,
    cv_train_min_days: int,
    cv_val_days: int,
    cv_step_days: int,
) -> pd.DataFrame:
    """Add paired_cv_* columns to cluster_table — one CV run per unique cluster."""
    cv_dir = output_dir / "cv_clusters"
    cv_dir.mkdir(parents=True, exist_ok=True)

    cluster_cv: dict[int, dict] = {}
    for cid in sorted(cluster_table["cluster_id"].unique()):
        mask = cluster_table["cluster_id"] == cid
        rep = (
            cluster_table[mask]
            .sort_values("mean_abs_shap", ascending=False)
            .iloc[0]["feature"]
        )
        csv_name = f"cluster_{cid}.csv"
        csv_rel = f"cv_clusters/{csv_name}"
        fold_results = _run_paired_cv(
            df,
            model_path,
            drop_features=[rep],
            seed=cv_seed,
            train_min_days=cv_train_min_days,
            val_days=cv_val_days,
            step_days=cv_step_days,
        )
        pd.DataFrame(fold_results).to_csv(cv_dir / csv_name, index=False)
        cluster_cv[cid] = _cv_summary(fold_results, csv_rel)

    for col in _CV_COLS:
        cluster_table[col] = cluster_table["cluster_id"].map(
            lambda cid, c=col: cluster_cv[cid][c]
        )
    return cluster_table


def _run_decomp_cv(
    decomp: pd.DataFrame,
    df: pd.DataFrame,
    model_path: pathlib.Path,
    output_dir: pathlib.Path,
    *,
    cv_seed: int,
    cv_train_min_days: int,
    cv_val_days: int,
    cv_step_days: int,
) -> pd.DataFrame:
    """Add paired_cv_* columns to decomp — one CV run per feature."""
    if decomp.empty:
        return _add_nan_cv_cols(decomp)

    cv_dir = output_dir / "cv_decomp"
    cv_dir.mkdir(parents=True, exist_ok=True)

    cv_rows: list[dict] = []
    for pos, (_, row) in enumerate(decomp.iterrows()):
        feat = row["feature"]
        csv_name = f"{pos}.csv"
        csv_rel = f"cv_decomp/{csv_name}"
        fold_results = _run_paired_cv(
            df,
            model_path,
            drop_features=[feat],
            seed=cv_seed,
            train_min_days=cv_train_min_days,
            val_days=cv_val_days,
            step_days=cv_step_days,
        )
        pd.DataFrame(fold_results).to_csv(cv_dir / csv_name, index=False)
        cv_rows.append({"feature": feat, **_cv_summary(fold_results, csv_rel)})

    cv_df = pd.DataFrame(cv_rows).set_index("feature")
    for col in _CV_COLS:
        decomp[col] = decomp["feature"].map(cv_df[col])
    return decomp


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _load_split(df: pd.DataFrame, split: Split) -> pd.DataFrame:
    train, val, test = _ev.split(df)
    return {"train": train, "val": val, "test": test}[split]


def run_redundancy_report(
    model_path: pathlib.Path,
    features_csv: pathlib.Path,
    split: Split,
    output_dir: pathlib.Path,
    cluster_threshold: float,
    interaction_sample: int,
    seed: int,
    *,
    skip_paired_cv: bool = False,
    cv_seed: int = 42,
    cv_train_min_days: int = 1825,
    cv_val_days: int = 90,
    cv_step_days: int = 90,
) -> dict:
    bundle = joblib.load(model_path)
    model = bundle["pipeline"]
    feature_columns: list[str] = bundle["feature_columns"]

    df = pd.read_csv(features_csv)
    split_df = _load_split(df, split)
    if split_df.empty:
        raise ValueError(f"Split '{split}' is empty after canonical date boundaries.")

    X = split_df[feature_columns].to_numpy(dtype=float)
    sv = compute_shap(model, X)
    mean_abs = np.mean(np.abs(sv), axis=0)

    corr = shap_correlation_matrix(sv)
    cluster_ids, Z = cluster_features(corr, cluster_threshold)
    cluster_table = build_cluster_table(feature_columns, mean_abs, cluster_ids)

    rng = np.random.default_rng(seed)
    interaction_matrix, n_used = compute_interaction_matrix(
        model, X, interaction_sample, rng
    )
    decomp = decomposition_scores(feature_columns, interaction_matrix)

    output_dir.mkdir(parents=True, exist_ok=True)

    cv_kwargs = dict(
        cv_seed=cv_seed,
        cv_train_min_days=cv_train_min_days,
        cv_val_days=cv_val_days,
        cv_step_days=cv_step_days,
    )
    if skip_paired_cv:
        cluster_table = _add_nan_cv_cols(cluster_table)
        decomp = _add_nan_cv_cols(decomp)
    else:
        cluster_table = _run_cluster_cv(
            cluster_table, df, model_path, output_dir, **cv_kwargs
        )
        decomp = _run_decomp_cv(
            decomp, df, model_path, output_dir, **cv_kwargs
        )

    pd.DataFrame(corr, index=feature_columns, columns=feature_columns).to_csv(
        output_dir / "shap_corr.csv"
    )
    cluster_table.to_csv(output_dir / "clusters.csv", index=False)
    save_dendrogram(feature_columns, Z, cluster_threshold, output_dir / "dendrogram.png")
    pd.DataFrame(
        interaction_matrix, index=feature_columns, columns=feature_columns
    ).to_csv(output_dir / "interaction_matrix.csv")
    decomp.to_csv(output_dir / "decomposition_candidates.csv", index=False)
    with open(output_dir / "feature_columns.json", "w") as fh:
        json.dump(feature_columns, fh)
    with open(output_dir / "params.json", "w") as fh:
        json.dump({
            "model": str(model_path),
            "features": str(features_csv),
            "split": split,
            "cluster_threshold": cluster_threshold,
            "interaction_sample": interaction_sample,
            "interaction_rows_used": n_used,
            "seed": seed,
            "skip_paired_cv": skip_paired_cv,
            "cv_seed": cv_seed,
            "cv_train_min_days": cv_train_min_days,
            "cv_val_days": cv_val_days,
            "cv_step_days": cv_step_days,
            "n_features": len(feature_columns),
            "n_rows_split": int(split_df.shape[0]),
        }, fh, indent=2)

    return {
        "feature_columns": feature_columns,
        "clusters": cluster_table,
        "decomposition": decomp,
        "n_clusters": int(cluster_table["cluster_id"].nunique()),
        "n_rows_interaction": n_used,
        "n_rows_split": int(split_df.shape[0]),
    }


@click.command("feature_redundancy")
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
    type=click.Choice(["train", "val", "test"]),
    default="val",
    show_default=True,
    help="Canonical split to analyse.",
)
@click.option(
    "--output",
    "output_dir",
    required=True,
    help="Directory to write redundancy + decomposition artifacts.",
)
@click.option(
    "--cluster-threshold",
    type=float,
    default=0.5,
    show_default=True,
    help="Distance cutoff for fcluster (1 − |corr|). 0.3 ≈ |r|≥0.7 sibling threshold.",
)
@click.option(
    "--interaction-sample",
    type=click.IntRange(min=1),
    default=3000,
    show_default=True,
    help="Rows to subsample for shap_interaction_values (cost scales linearly).",
)
@click.option(
    "--seed",
    type=int,
    default=0,
    show_default=True,
    help="RNG seed for interaction subsampling.",
)
@click.option(
    "--skip-paired-cv",
    is_flag=True,
    default=False,
    help=(
        "Skip paired walk-forward CV per candidate. Leaves paired_cv_* columns as NaN. "
        "Use for a fast SHAP-only screening pass when wall time is a concern."
    ),
)
@click.option(
    "--cv-seed",
    type=int,
    default=42,
    show_default=True,
    help="RNG seed used when re-training each fold during paired CV.",
)
def main(
    model_path: str,
    features_csv: str,
    split: str,
    output_dir: str,
    cluster_threshold: float,
    interaction_sample: int,
    seed: int,
    skip_paired_cv: bool,
    cv_seed: int,
) -> None:
    """Compute SHAP redundancy clusters + decomposition candidates."""
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
    if skip_paired_cv:
        click.echo("Paired CV: skipped (--skip-paired-cv)")
    else:
        click.echo(f"Paired CV: enabled (cv_seed={cv_seed}) — may take several minutes")

    result = run_redundancy_report(
        mp, fp, split, out,  # type: ignore[arg-type]
        cluster_threshold=cluster_threshold,
        interaction_sample=interaction_sample,
        seed=seed,
        skip_paired_cv=skip_paired_cv,
        cv_seed=cv_seed,
    )

    clusters = result["clusters"]
    click.echo(
        f"\nRedundancy clusters — {result['n_clusters']} clusters across "
        f"{len(clusters)} features (threshold {cluster_threshold})"
    )
    click.echo(f"{'cluster':>7}  {'feature':<45} {'mean|SHAP|':>10}")
    click.echo("─" * 66)
    for _, row in clusters.iterrows():
        click.echo(
            f"{int(row['cluster_id']):>7}  {row['feature']:<45} "
            f"{row['mean_abs_shap']:>10.4f}"
        )

    decomp = result["decomposition"]
    click.echo(
        f"\nDecomposition candidates — ranked by entropy_norm "
        f"(interaction sample n={result['n_rows_interaction']:,})"
    )
    click.echo(
        f"{'feature':<45} {'entr':>5} {'partners≥5%':>11}  top1"
    )
    click.echo("─" * 78)
    for _, row in decomp.head(15).iterrows():
        click.echo(
            f"{row['feature']:<45} {row['entropy_norm']:>5.2f} "
            f"{int(row['n_partners_ge_5pct']):>11}  "
            f"{row['top1_partner']} ({row['top1_share']:.0%})"
        )

    click.echo(f"\nArtifacts written to {out}/")
    click.echo("  shap_corr.csv")
    click.echo("  clusters.csv")
    click.echo("  dendrogram.png")
    click.echo("  interaction_matrix.csv")
    click.echo("  decomposition_candidates.csv")
    click.echo("  feature_columns.json")
    click.echo("  params.json")
    if not skip_paired_cv:
        click.echo("  cv_clusters/  (per-fold CSVs, one per cluster)")
        click.echo("  cv_decomp/    (per-fold CSVs, one per feature)")


if __name__ == "__main__":
    main()
