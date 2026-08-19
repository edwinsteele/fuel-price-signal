"""Noise-floor calibration (fps-3jj.9) — R0-vs-R0 pipeline noise, persisted once per batch.

The dossier (experiments/pipeline/dossier_tables.py's `_noise_band()`) already reads
`<batch_dir>/noise_floor.json` and grades every candidate's realised delta against it;
until this module ran, that file didn't exist and every dossier reported
`facts["noise_band"]["available"] = False`. This module is the producer that satisfies
that contract — the required key (`"deltas_cpl_held": [<float>, ...]`) and filename
(`NOISE_FLOOR_FILENAME`) are fixed by dossier_tables.py, not redefined here. The payload
also carries provenance (seeds, fold_subset, a "partial" flag, timestamp, git SHA) that
dossier_tables.py doesn't require but a human debugging a batch later will want.

Why this shape (see fps-3jj.9's bd description for the full history): a placebo arm
compares a noisy number against a noisy ruler; a permuted-column null needs a
permutation axis matching the axis the candidate claims to carry information on, and
for a market-wide series like tgp_delta_7d (one value per date, shared by every
station) a within-date permutation is a no-op — so neither approach gives a single
universal ruler. The owner's replacement, implemented here: fit the SAME frozen
baseline (same 54 columns, same data) at two different seeds and take the realised-CPL
delta. That delta is pure pipeline fit noise by construction — no definition of an
"uninformative column" is needed, so the market-wide-series problem never arises.

Limit (explicit in the bd issue, repeated here so a reader of noise_floor.json alone
gets it too): this measures fit instability from the seed alone, not the specific
perturbation of adding a candidate column. A candidate delta inside this floor is
noise; one just outside it is not necessarily signal — a floor, not the true band.

`SEEDS_A`/`SEEDS_B` pair index-wise (42↔47, 43↔48, ...) into `len(SEEDS_A)` independent
draws, each costing one single-arm (baseline only, no candidate) full walk-forward fit
per seed — 10 fits total for the 5-pair default. Because a batch is frozen (data + DB
pinned), every draw is deterministic, so this runs once at batch-setup time (right
after `batch_freeze.py`) and is cached to disk from then on.

Usage (batch setup, once, after batch_freeze.py and before the launch routine claims
candidate 1):
  PYTHONPATH=. uv run python -m experiments.pipeline.noise_floor <batch-name>
"""
from __future__ import annotations

import json
import pathlib
import time
from collections.abc import Iterable
from datetime import datetime, timezone

import click

from experiments.lib.constants import SEEDS
from experiments.lib.io import current_git_sha
from experiments.lib.realised import ArmSpec, run_paired_realised_backtest
from experiments.pipeline.batch_freeze import (
    BASELINE_COLUMNS_FILENAME,
    DEFAULT_BATCHES_DIR,
    FROZEN_DB_FILENAME,
    check_baseline_contract,
)
from experiments.pipeline.dossier_tables import NOISE_FLOOR_FILENAME
from experiments.pipeline.runner import DEFAULT_INNER_FOLD_PARAMS
from fuel_signal.features import load_features

BASELINE_ARM_NAME = "baseline"

# The starting idea recorded on fps-3jj.9: fit R0 with SEEDS (42..46), fit it again
# with a disjoint second group, take the realised-CPL delta. Paired index-wise here
# (42 vs 47, 43 vs 48, ...) rather than as two 5-seed ensembles, so the result is a
# BAND of 5 draws (the acceptance criterion) rather than a single point.
SEEDS_A: tuple[int, ...] = SEEDS
SEEDS_B: tuple[int, ...] = (47, 48, 49, 50, 51)


def _baseline_cpl_held(
    frame,
    baseline_columns: list[str],
    seed: int,
    *,
    outer_fold_params: dict,
    inner_fold_params: dict,
    fold_subset: Iterable[int] | None,
    station_codes: list[int] | None,
    db_path: pathlib.Path,
    verbose: bool,
) -> float:
    """Fit the baseline alone (no candidate arm) at `seed`; return its pooled cpl_held.

    held_tau=None means the baseline's own per-fold τ IS the held τ (see
    experiments/lib/realised.py's module docstring), so with a single arm cpl_held
    equals cpl_own — this is exactly the R0 fit + realised-backtest path every
    candidate run pays for, just without a candidate arm alongside it.
    """
    result = run_paired_realised_backtest(
        [ArmSpec(BASELINE_ARM_NAME, frame, feature_columns=baseline_columns)],
        baseline_columns,
        seed=seed,
        held_tau=None,
        station_codes=station_codes,
        outer_fold_params=outer_fold_params,
        inner_fold_params=inner_fold_params,
        fold_subset=fold_subset,
        db_path=db_path,
        collect_fills=False,
        verbose=verbose,
    )
    return float(result.aggregate.set_index("arm").loc[BASELINE_ARM_NAME, "cpl_held"])


def compute_noise_floor(
    batch_dir: pathlib.Path,
    *,
    seeds_a: tuple[int, ...] = SEEDS_A,
    seeds_b: tuple[int, ...] = SEEDS_B,
    outer_fold_params: dict | None = None,
    inner_fold_params: dict | None = None,
    fold_subset: Iterable[int] | None = None,
    station_codes: list[int] | None = None,
    force: bool = False,
    verbose: bool = True,
) -> dict:
    """Compute `len(seeds_a)` R0-vs-R0 realised-CPL deltas and persist them to
    `<batch_dir>/noise_floor.json`.

    Reads the frozen batch's own features + baseline column contract (declared,
    never discovered — same `baseline_columns.json` every candidate run reads,
    fps-sa1) and its cloned DB, so this reproduces the exact fit + realised-backtest
    path a candidate run takes, minus the candidate arm. Also re-checks that
    contract against the current code (`check_baseline_contract`, same call every
    candidate run makes) so a code-side FEATURE_COLUMNS drift since freeze is
    caught here too, not just silently fit against a stale list.

    Returns the written payload. Raises:
      - ValueError if seeds_a/seeds_b aren't the same length (they pair 1:1 into
        one draw per pair, so a length mismatch means some seed has no partner),
        or if either group has a repeated seed, or the two groups overlap — a
        shared/repeated seed pairs a fit against itself and writes a zero delta,
        silently narrowing the floor.
      - FileExistsError if `<batch_dir>/noise_floor.json` already exists and
        `force` is not set — a full calibration already sitting there is the
        ruler every dossier so far was graded against; overwriting it silently
        (e.g. from a re-run) would move that ruler under them after the fact.
      - BaselineContractMismatch (via check_baseline_contract) on column drift.
    """
    if len(seeds_a) != len(seeds_b):
        raise ValueError(
            f"seeds_a and seeds_b must pair 1:1; got {len(seeds_a)} vs {len(seeds_b)} seeds."
        )
    if len(set(seeds_a)) != len(seeds_a) or len(set(seeds_b)) != len(seeds_b):
        raise ValueError(f"seeds_a and seeds_b must each be free of repeats; got {seeds_a}, {seeds_b}.")
    overlap = set(seeds_a) & set(seeds_b)
    if overlap:
        raise ValueError(
            f"seeds_a and seeds_b must be disjoint — a shared seed {sorted(overlap)} pairs a fit "
            "against itself and writes a zero delta, silently narrowing the floor."
        )

    batch_dir = pathlib.Path(batch_dir)
    out_path = batch_dir / NOISE_FLOOR_FILENAME
    if out_path.exists() and not force:
        raise FileExistsError(
            f"{out_path} already exists. Every dossier written so far was graded against it — "
            "pass force=True (CLI: --force) only if you intend to replace that ruler for the "
            "whole batch, not just for this run."
        )

    outer_fold_params = dict(outer_fold_params or {})
    # Merged, not replaced — same rationale as runner.py's run_candidate: a caller
    # overriding one key (say val_days) still gets a workable train_min_days.
    inner_fold_params = {**DEFAULT_INNER_FOLD_PARAMS, **(inner_fold_params or {})}

    frame = load_features(batch_dir / "features.csv")  # .parquet sibling, batch_freeze convention
    check_baseline_contract(batch_dir, frame)
    baseline_columns = json.loads((batch_dir / BASELINE_COLUMNS_FILENAME).read_text())
    db_path = batch_dir / FROZEN_DB_FILENAME

    t0 = time.perf_counter()
    deltas: list[float] = []
    for i, (seed_a, seed_b) in enumerate(zip(seeds_a, seeds_b, strict=True), start=1):
        kwargs = dict(
            outer_fold_params=outer_fold_params,
            inner_fold_params=inner_fold_params,
            fold_subset=fold_subset,
            station_codes=station_codes,
            db_path=db_path,
            verbose=verbose,
        )
        cpl_a = _baseline_cpl_held(frame, baseline_columns, seed_a, **kwargs)
        cpl_b = _baseline_cpl_held(frame, baseline_columns, seed_b, **kwargs)
        delta = cpl_b - cpl_a
        deltas.append(delta)
        if verbose:
            print(
                f"[noise_floor] pair {i}/{len(seeds_a)} (seed {seed_a} vs {seed_b}): "
                f"delta_cpl_held={delta:+.4f}",
                flush=True,
            )

    # fold_subset is documented as an iteration/smoke speed-up, not a real calibration —
    # a partial-fold floor is not comparable to effect_delta_cpl_held, which always pools
    # every fold. Tagging it "partial" (rather than just omitting the metadata) means
    # dossier_tables._noise_band() can refuse it outright instead of silently grading
    # every candidate against a ruler that only covers some folds.
    payload = {
        "deltas_cpl_held": deltas,
        "seeds_a": list(seeds_a),
        "seeds_b": list(seeds_b),
        "fold_subset": list(fold_subset) if fold_subset is not None else None,
        "partial": fold_subset is not None,
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": current_git_sha(),
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    if verbose:
        print(
            f"[noise_floor] wrote {len(deltas)} draws to {out_path} "
            f"in {time.perf_counter() - t0:.1f}s",
            flush=True,
        )
    return payload


def _parse_fold_subset(value: str | None) -> list[int] | None:
    """Comma-separated 1-indexed fold numbers, e.g. '1,2'. None -> None (every fold)."""
    if not value:
        return None
    folds: list[int] = []
    for token in value.split(","):
        try:
            fold = int(token)
        except ValueError:
            raise click.BadParameter(
                f"{token!r} is not an integer — expected comma-separated 1-indexed fold "
                "numbers, e.g. '1,2'."
            ) from None
        if fold < 1:
            raise click.BadParameter(f"fold {fold} is not 1-indexed — folds start at 1.")
        folds.append(fold)
    return folds


@click.command("noise_floor")
@click.argument("batch_name")
@click.option(
    "--batches-dir", default=str(DEFAULT_BATCHES_DIR), show_default=True,
    help="Parent directory for batch snapshots.",
)
@click.option(
    "--fold-subset", default=None,
    help="Comma-separated 1-indexed outer folds, e.g. '1,2'. Iteration/smoke speed-up ONLY — "
    "the result covers a partial fold set and is written as 'partial': true, which the "
    "dossier refuses to grade candidates against.",
)
@click.option(
    "--force", is_flag=True, default=False,
    help="Overwrite an existing noise_floor.json for this batch. Every dossier written so "
    "far was graded against it — only pass this if you intend to replace that ruler.",
)
def main(batch_name: str, batches_dir: str, fold_subset: str | None, force: bool) -> None:
    """Compute and persist the noise-floor calibration for an already-frozen batch."""
    batch_dir = pathlib.Path(batches_dir) / batch_name
    subset = _parse_fold_subset(fold_subset)
    compute_noise_floor(batch_dir, fold_subset=subset, force=force)
    click.echo(f"Wrote noise floor for batch '{batch_name}' -> {batch_dir / NOISE_FLOOR_FILENAME}")


if __name__ == "__main__":
    main()
