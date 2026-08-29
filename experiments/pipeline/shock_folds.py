"""Batch-level shock-fold calibration (fps-3tu follow-up) — the empirical shock/normal
split, computed once per batch and persisted, instead of re-derived by every candidate.

`experiments.lib.folds.compute_shock_folds` is deterministic given (data, baseline
columns, seed): every candidate in a batch shares the same frozen `features.csv` and the
same locked baseline columns, so every candidate's own live derivation of the shock-fold
set is byte-identical to every other's — the underlying baseline log-loss fit was already
happening per candidate run regardless (`_run_wfcv_screen` needs it for R0-vs-candidate
deltas), so the live path costs nothing extra, but computing the SAME thing five times
over is still redundant in spirit. This module does it once and writes the result to
`<batch_dir>/shock_folds.json`; `runner.py` reads it when present and falls back to the
live per-run derivation otherwise (a batch frozen before this file existed, or one whose
cache was deleted).

Deliberately NOT a hardcoded Python constant (the exact failure mode fps-3tu replaced):
a batch-scoped file, keyed by `baseline_fingerprint`, is re-derived automatically for the
next batch and refuses to grade a batch whose baseline has since been relocked, rather
than silently going stale the way `SHOCK_FOLDS` did.

Usage (recompute for an already-frozen batch):
  PYTHONPATH=. uv run python -m experiments.pipeline.shock_folds <batch-name> --force
"""
from __future__ import annotations

import json
import pathlib
import time
from datetime import datetime, timezone

import click

from experiments.lib.constants import SEEDS, SHOCK_QUANTILE
from experiments.lib.folds import DEFAULT_OUTER_FOLD_PARAMS, iter_folds_with_baseline_fit
from experiments.lib.io import current_git_sha, to_jsonable
from experiments.pipeline.batch_freeze import (
    BASELINE_COLUMNS_FILENAME,
    DEFAULT_BATCHES_DIR,
    check_baseline_contract,
)
from fuel_signal.features import baseline_fingerprint, load_features

SHOCK_FOLDS_FILENAME = "shock_folds.json"


def compute_shock_folds_for_batch(
    batch_dir: pathlib.Path,
    *,
    quantile: float = SHOCK_QUANTILE,
    seed: int = SEEDS[0],
    train_min_days: int = DEFAULT_OUTER_FOLD_PARAMS["train_min_days"],
    val_days: int = DEFAULT_OUTER_FOLD_PARAMS["val_days"],
    step_days: int = DEFAULT_OUTER_FOLD_PARAMS["step_days"],
    out_name: str = SHOCK_FOLDS_FILENAME,
    force: bool = False,
    verbose: bool = True,
) -> dict:
    """Fit the baseline once per WFCV fold against this batch's own frozen data, derive
    the empirical shock-fold set, and persist it to `<batch_dir>/<out_name>`.

    Reuses `iter_folds_with_baseline_fit`'s own live-derivation path (no override passed
    in) rather than re-implementing the fit loop — this call's whole purpose is to run
    that derivation exactly once and cache the result, not to compute it differently.

    Raises FileExistsError if the output file already exists and `force` is not set —
    same convention as `noise_floor.compute_noise_floor`: every dossier built so far may
    already be reading this file, so overwriting it is opt-in.
    """
    batch_dir = pathlib.Path(batch_dir)
    out_path = batch_dir / out_name
    if out_path.exists() and not force:
        raise FileExistsError(
            f"{out_path} already exists. Pass force=True (CLI: --force) only if you intend "
            "to replace the shock-fold set every dossier built against this batch so far "
            "may already be reading."
        )

    frame = load_features(batch_dir / "features.csv")  # .parquet sibling, batch_freeze convention
    check_baseline_contract(batch_dir, frame)
    baseline_columns = json.loads((batch_dir / BASELINE_COLUMNS_FILENAME).read_text())

    t0 = time.perf_counter()
    fold_ll: dict[int, float] = {}
    shock: set[int] = set()
    for fold_idx, regime, _train_df, _val_df, ll, *_rest in iter_folds_with_baseline_fit(
        frame, baseline_columns, seed=seed,
        train_min_days=train_min_days, val_days=val_days, step_days=step_days,
        shock_quantile=quantile,
    ):
        fold_ll[fold_idx] = ll
        if regime == "shock":
            shock.add(fold_idx)
        if verbose:
            print(f"[shock_folds] fold {fold_idx:>2} {regime:<6} ll={ll:.4f}", flush=True)
    wall_seconds = time.perf_counter() - t0

    payload = {
        "shock_folds": sorted(shock),
        "shock_quantile": quantile,
        "fold_baseline_ll": {str(k): v for k, v in sorted(fold_ll.items())},
        "n_folds": len(fold_ll),
        "seed": seed,
        "outer_fold_params": {
            "train_min_days": train_min_days, "val_days": val_days, "step_days": step_days,
        },
        # Identity of the baseline this was derived against (fps-cf8 pattern) — runner.py
        # refuses to trust a cached set whose fingerprint doesn't match the run's own.
        "n_baseline_columns": len(baseline_columns),
        "baseline_fingerprint": baseline_fingerprint(baseline_columns),
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": current_git_sha(),
        "wall_seconds": wall_seconds,
    }
    out_path.write_text(json.dumps(to_jsonable(payload), indent=2, default=str) + "\n")
    if verbose:
        print(f"[shock_folds] wrote {sorted(shock)} -> {out_path} ({wall_seconds:.1f}s)", flush=True)
    return payload


def load_cached_shock_folds(
    batch_dir: pathlib.Path | None,
    *,
    baseline_fingerprint: str,
    seed: int = SEEDS[0],
    outer_fold_params: dict | None = None,
    quantile: float = SHOCK_QUANTILE,
) -> frozenset[int] | None:
    """This batch's cached shock-fold set, or None if unavailable/untrustworthy.

    Compares the FULL config the cache was computed under — baseline_fingerprint, seed,
    the resolved outer fold geometry, and quantile — not baseline_fingerprint alone
    (review finding on PR #350: `seed` and `shock_quantile` were already recorded in the
    payload but never checked, and a run launched with non-default outer fold geometry —
    reachable via runner.py's CLI `--outer-*-days` flags — would silently apply
    shock-fold INDICES derived under a completely different fold windowing; a smoke-
    tested repro confirmed this on a synthetic batch). Mirrors
    `experiments.lib.realised._baseline_cache_fingerprint`'s exact-dict-equality
    pattern, the direct in-repo precedent for "what does a cache need to match to be
    safely reusable" thirty lines from this module's own cache read.
    `outer_fold_params` may be a caller's PARTIAL override dict (e.g. from
    runner.py's CLI, which only sets the keys actually overridden) — resolved against
    `DEFAULT_OUTER_FOLD_PARAMS` before comparing, since the payload always stores the
    fully-resolved geometry it was computed at.

    None (rather than raising) on every mismatch or failure mode — a missing batch_dir,
    a missing cache file, or any config disagreement — because the caller (runner.py)
    always has a correct, if redundant, fallback: derive it live for this run instead. A
    stale cache must never be trusted silently (same "cannot be shown to match, so
    cannot be trusted" rule dossier_tables._noise_band applies to noise_floor.json), but
    it also must never turn into a hard failure — that would make a routine baseline
    re-lock, or even an ordinary smoke run at custom fold geometry, break until someone
    remembers to re-run this module's CLI.
    """
    if batch_dir is None:
        return None
    path = pathlib.Path(batch_dir) / SHOCK_FOLDS_FILENAME
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    resolved_outer = {**DEFAULT_OUTER_FOLD_PARAMS, **(outer_fold_params or {})}
    expected = {
        "baseline_fingerprint": baseline_fingerprint,
        "seed": seed,
        "outer_fold_params": resolved_outer,
        "shock_quantile": quantile,
    }
    actual = {
        "baseline_fingerprint": payload.get("baseline_fingerprint"),
        "seed": payload.get("seed"),
        "outer_fold_params": payload.get("outer_fold_params"),
        "shock_quantile": payload.get("shock_quantile"),
    }
    if actual != expected:
        return None
    raw = payload.get("shock_folds")
    if raw is None:
        return None
    return frozenset(int(f) for f in raw)


@click.command("shock_folds")
@click.argument("batch_name")
@click.option(
    "--batches-dir", default=str(DEFAULT_BATCHES_DIR), show_default=True,
    help="Parent directory for batch snapshots.",
)
@click.option(
    "--quantile", default=SHOCK_QUANTILE, show_default=True,
    help="Fraction of folds (by count, ceil-rounded) tagged shock — the top quantile by "
    "the baseline's own val-fold log-loss.",
)
@click.option(
    "--out-name", default=SHOCK_FOLDS_FILENAME, show_default=True,
    help="Output filename under the batch dir.",
)
@click.option(
    "--force", is_flag=True, default=False,
    help="Overwrite an existing output file for this batch. Every run graded against this "
    "batch so far may already be reading it — only pass this if you intend to replace it.",
)
def main(batch_name: str, batches_dir: str, quantile: float, out_name: str, force: bool) -> None:
    """Compute and persist the empirical shock-fold set for an already-frozen batch."""
    batch_dir = pathlib.Path(batches_dir) / batch_name
    payload = compute_shock_folds_for_batch(batch_dir, quantile=quantile, out_name=out_name, force=force)
    click.echo(f"Wrote shock_folds={payload['shock_folds']} for batch '{batch_name}' -> {batch_dir / out_name}")


if __name__ == "__main__":
    main()
