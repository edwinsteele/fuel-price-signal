"""Production-scale benchmark for streaming WFCV folds instead of retaining all 14.

Run each measured variant in its own process; max RSS is process-lifetime state.
See README.md in this directory for the command sequence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import platform
import resource
import subprocess
import time

import numpy as np
import pandas as pd

from experiments.lib.cohorts import hard_quantile_mask
from experiments.lib.fit import fit_score, per_row_log_loss
from experiments.lib.rowpreds import RowPredCollector
from experiments.pipeline.runner import BASELINE_ARM, CANDIDATE_ARM, _run_wfcv_screen
from experiments.pipeline.shock_folds import load_cached_shock_folds
from fuel_signal import evaluate as _ev
from fuel_signal.features import baseline_fingerprint, load_features

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = pathlib.Path(__file__).parent
ARTIFACTS = OUT / "artifacts"
BATCH_DIR = ROOT / "experiments" / "batches" / "batch1"
FEATURES_PATH = BATCH_DIR / "features.csv"
BASELINE_COLUMNS_PATH = BATCH_DIR / "baseline_columns.json"
BENCH_COLUMN = "tgp_delta_7d"


def _git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _max_rss_mib() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports KiB.
    divisor = 1024**2 if platform.system() == "Darwin" else 1024
    return float(value / divisor)


def _seed_tuple(raw: str) -> tuple[int, ...]:
    seeds = tuple(int(x.strip()) for x in raw.split(",") if x.strip())
    if not seeds:
        raise ValueError("--seeds must contain at least one integer")
    return seeds


def _seed_slug(seeds: tuple[int, ...]) -> str:
    return "-".join(str(seed) for seed in seeds)


def _profile_path(stage: str, variant: str, seeds: tuple[int, ...] = ()) -> pathlib.Path:
    suffix = f"_{_seed_slug(seeds)}" if seeds else ""
    return ARTIFACTS / f"{stage}_{variant}{suffix}_profile.json"


def _rows_path(variant: str, seeds: tuple[int, ...]) -> pathlib.Path:
    return ARTIFACTS / f"wfcv_{variant}_{_seed_slug(seeds)}_rows.parquet"


def _rowpreds_path(variant: str, seeds: tuple[int, ...]) -> pathlib.Path:
    return ARTIFACTS / f"wfcv_{variant}_{_seed_slug(seeds)}_rowpreds.parquet"


def _index_digest(frame: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(frame.index.to_series(), index=False).to_numpy()
    return hashlib.sha256(hashed.tobytes()).hexdigest()[:16]


def _fold_signature(fold_idx: int, train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
    train_dates = pd.to_datetime(train_df["price_date"])
    val_dates = pd.to_datetime(val_df["price_date"])
    return {
        "fold": fold_idx,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "train_start": train_dates.min().strftime("%Y-%m-%d"),
        "train_end": train_dates.max().strftime("%Y-%m-%d"),
        "val_start": val_dates.min().strftime("%Y-%m-%d"),
        "val_end": val_dates.max().strftime("%Y-%m-%d"),
        "train_index_digest": _index_digest(train_df),
        "val_index_digest": _index_digest(val_df),
    }


def _write_json(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def _load_frame_and_contract() -> tuple[pd.DataFrame, list[str], frozenset[int]]:
    frame = load_features(FEATURES_PATH)
    baseline_columns = json.loads(BASELINE_COLUMNS_PATH.read_text())
    if BENCH_COLUMN not in frame.columns:
        raise ValueError(f"benchmark column {BENCH_COLUMN!r} is absent from frozen features")
    if BENCH_COLUMN in baseline_columns:
        raise ValueError(f"benchmark column {BENCH_COLUMN!r} unexpectedly entered the baseline")
    shock_folds = load_cached_shock_folds(
        BATCH_DIR,
        baseline_fingerprint=baseline_fingerprint(baseline_columns),
        seed=42,
        outer_fold_params={},
    )
    if shock_folds is None:
        raise ValueError(
            "batch1 has no compatible shock_folds.json; this benchmark requires the normal "
            "cached-regime path so streaming can label each fold immediately"
        )
    return frame, baseline_columns, shock_folds


def run_fold_construction(variant: str) -> dict:
    overall_t0 = time.perf_counter()
    frame, baseline_columns, shock_folds = _load_frame_and_contract()
    load_s = time.perf_counter() - overall_t0

    fold_t0 = time.perf_counter()
    signatures: list[dict] = []
    if variant == "current":
        # Reproduce folds.py's retention shape: list(...) plus another container retaining
        # every train/validation frame until the phase completes.
        folds = list(_ev.walk_forward_folds(frame))
        retained = {
            i: (train_df, val_df)
            for i, (train_df, val_df) in enumerate(folds, start=1)
            if not val_df.empty
        }
        for fold_idx, (train_df, val_df) in retained.items():
            signatures.append(_fold_signature(fold_idx, train_df, val_df))
    else:
        for fold_idx, (train_df, val_df) in enumerate(_ev.walk_forward_folds(frame), start=1):
            if val_df.empty:
                continue
            signatures.append(_fold_signature(fold_idx, train_df, val_df))
    fold_s = time.perf_counter() - fold_t0

    return {
        "stage": "folds",
        "variant": variant,
        "git_sha": _git_sha(),
        "batch_dir": str(BATCH_DIR),
        "rows": len(frame),
        "columns": len(frame.columns),
        "baseline_fingerprint": baseline_fingerprint(baseline_columns),
        "shock_folds": sorted(shock_folds),
        "load_seconds": load_s,
        "stage_seconds": fold_s,
        "total_seconds": time.perf_counter() - overall_t0,
        "max_rss_mib": _max_rss_mib(),
        "folds": signatures,
    }


def _run_streaming_wfcv(
    frame: pd.DataFrame,
    baseline_columns: list[str],
    candidate_columns: list[str],
    seeds: tuple[int, ...],
    shock_folds: frozenset[int],
) -> tuple[pd.DataFrame, RowPredCollector]:
    """Equivalent to runner._run_wfcv_screen, consuming each canonical fold eagerly."""
    rows: list[dict] = []
    collector = RowPredCollector(pd.DataFrame())

    for fold_idx, (train_df, val_df) in enumerate(_ev.walk_forward_folds(frame), start=1):
        if val_df.empty:
            continue

        y = val_df["label"].to_numpy(dtype=int)
        baseline_ll, baseline_p, baseline_fit_s = fit_score(
            train_df, val_df, baseline_columns, seeds[0]
        )
        baseline_prl = per_row_log_loss(y, baseline_p)
        hard25_mask = hard_quantile_mask(baseline_prl, 0.75)
        regime = "shock" if fold_idx in shock_folds else "normal"
        val_dates = pd.to_datetime(val_df["price_date"])

        collector.ident_base = pd.DataFrame(
            {
                "fold": np.int8(fold_idx),
                "station_code": val_df["station_code"].to_numpy(),
                "price_date": val_dates.to_numpy(),
                "label": y.astype(np.int8),
                "is_hard25": hard25_mask.astype(np.int8),
                "cycle_pct_through": val_df["cycle_pct_through"].to_numpy(),
                BENCH_COLUMN: val_df[BENCH_COLUMN].to_numpy(),
            }
        )

        for run_name, columns in (
            (BASELINE_ARM, baseline_columns),
            (CANDIDATE_ARM, candidate_columns),
        ):
            for seed in seeds:
                if run_name == BASELINE_ARM and seed == seeds[0]:
                    ll, probabilities, fit_s = baseline_ll, baseline_p, baseline_fit_s
                else:
                    ll, probabilities, fit_s = fit_score(train_df, val_df, columns, seed)
                row_loss = per_row_log_loss(y, probabilities)
                rows.append(
                    {
                        "fold": fold_idx,
                        "regime": regime,
                        "run": run_name,
                        "seed": seed,
                        "ll_all": ll,
                        "ll_hard25": (
                            float(row_loss[hard25_mask].mean())
                            if hard25_mask.any()
                            else float("nan")
                        ),
                        "fit_s": fit_s,
                    }
                )
                collector.add(run_name, seed, probabilities)
        print(f"[streaming] fold {fold_idx:>2} {regime:<6} done", flush=True)

    return pd.DataFrame(rows), collector


def run_wfcv(variant: str, seeds: tuple[int, ...]) -> dict:
    overall_t0 = time.perf_counter()
    frame, baseline_columns, shock_folds = _load_frame_and_contract()
    load_s = time.perf_counter() - overall_t0
    candidate_columns = [*baseline_columns, BENCH_COLUMN]

    screen_t0 = time.perf_counter()
    if variant == "current":
        rows, collector = _run_wfcv_screen(
            frame,
            baseline_columns,
            candidate_columns,
            seeds=seeds,
            axis_series=None,
            outer_fold_params={},
            verbose=True,
            persist_columns=[BENCH_COLUMN],
            shock_folds=shock_folds,
        )
    else:
        rows, collector = _run_streaming_wfcv(
            frame, baseline_columns, candidate_columns, seeds, shock_folds
        )
    screen_s = time.perf_counter() - screen_t0

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(_rows_path(variant, seeds), index=False)
    write_t0 = time.perf_counter()
    rowpreds = collector.to_parquet(_rowpreds_path(variant, seeds))
    write_s = time.perf_counter() - write_t0

    return {
        "stage": "wfcv",
        "variant": variant,
        "git_sha": _git_sha(),
        "batch_dir": str(BATCH_DIR),
        "rows": len(frame),
        "columns": len(frame.columns),
        "baseline_fingerprint": baseline_fingerprint(baseline_columns),
        "benchmark_column": BENCH_COLUMN,
        "seeds": list(seeds),
        "shock_folds": sorted(shock_folds),
        "n_folds": int(rows["fold"].nunique()),
        "n_fits": len(rows),
        "n_rowpreds": len(rowpreds),
        "load_seconds": load_s,
        "screen_seconds": screen_s,
        "rowpred_write_seconds": write_s,
        "sum_fit_seconds": float(rows["fit_s"].sum()),
        "total_seconds": time.perf_counter() - overall_t0,
        "max_rss_mib": _max_rss_mib(),
    }


def compare_folds() -> dict:
    current = json.loads(_profile_path("folds", "current").read_text())
    streaming = json.loads(_profile_path("folds", "streaming").read_text())
    if current["folds"] != streaming["folds"]:
        raise AssertionError("fold signatures differ between current and streaming variants")
    return {
        "stage": "folds",
        "parity": "exact",
        "n_folds": len(current["folds"]),
        "current_stage_seconds": current["stage_seconds"],
        "streaming_stage_seconds": streaming["stage_seconds"],
        "wall_speedup": current["stage_seconds"] / streaming["stage_seconds"],
        "current_max_rss_mib": current["max_rss_mib"],
        "streaming_max_rss_mib": streaming["max_rss_mib"],
        "peak_rss_reduction_pct": (
            (current["max_rss_mib"] - streaming["max_rss_mib"])
            / current["max_rss_mib"]
            * 100
        ),
    }


def compare_wfcv(seeds: tuple[int, ...]) -> dict:
    current_rows = pd.read_parquet(_rows_path("current", seeds))
    streaming_rows = pd.read_parquet(_rows_path("streaming", seeds))
    pd.testing.assert_frame_equal(
        current_rows.drop(columns="fit_s"),
        streaming_rows.drop(columns="fit_s"),
        check_exact=True,
    )

    current_preds = pd.read_parquet(_rowpreds_path("current", seeds))
    streaming_preds = pd.read_parquet(_rowpreds_path("streaming", seeds))
    pd.testing.assert_frame_equal(current_preds, streaming_preds, check_exact=True)

    current = json.loads(_profile_path("wfcv", "current", seeds).read_text())
    streaming = json.loads(_profile_path("wfcv", "streaming", seeds).read_text())
    return {
        "stage": "wfcv",
        "seeds": list(seeds),
        "parity": "exact",
        "n_folds": current["n_folds"],
        "n_fits": current["n_fits"],
        "n_rowpreds": current["n_rowpreds"],
        "current_screen_seconds": current["screen_seconds"],
        "streaming_screen_seconds": streaming["screen_seconds"],
        "screen_speedup": current["screen_seconds"] / streaming["screen_seconds"],
        "current_sum_fit_seconds": current["sum_fit_seconds"],
        "streaming_sum_fit_seconds": streaming["sum_fit_seconds"],
        "current_max_rss_mib": current["max_rss_mib"],
        "streaming_max_rss_mib": streaming["max_rss_mib"],
        "peak_rss_reduction_pct": (
            (current["max_rss_mib"] - streaming["max_rss_mib"])
            / current["max_rss_mib"]
            * 100
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=("folds", "wfcv", "compare-folds", "compare-wfcv")
    )
    parser.add_argument("variant", nargs="?", choices=("current", "streaming"))
    parser.add_argument("--seeds", default="42")
    args = parser.parse_args()
    seeds = _seed_tuple(args.seeds)

    if args.stage in {"folds", "wfcv"} and args.variant is None:
        parser.error(f"{args.stage} requires a variant: current or streaming")
    if args.stage.startswith("compare") and args.variant is not None:
        parser.error(f"{args.stage} does not accept a variant")

    if args.stage == "folds":
        payload = run_fold_construction(args.variant)
        path = _profile_path("folds", args.variant)
    elif args.stage == "wfcv":
        payload = run_wfcv(args.variant, seeds)
        path = _profile_path("wfcv", args.variant, seeds)
    elif args.stage == "compare-folds":
        payload = compare_folds()
        path = ARTIFACTS / "comparison_folds.json"
    else:
        payload = compare_wfcv(seeds)
        path = ARTIFACTS / f"comparison_wfcv_{_seed_slug(seeds)}.json"

    _write_json(path, payload)
    print(json.dumps(payload, indent=2), flush=True)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
