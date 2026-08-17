"""One-shot candidate runner (fps-3jj.4) — the core: everything else launches it,
validates its input, or reads its output.

Given a frozen batch (fps-3jj.1: `experiments/pipeline/batch_freeze.py`) and one
candidate module (fps-3jj.2's module format — see `experiments/candidates/TEMPLATE.py`
for a worked example), runs the WFCV log-loss screen
(`experiments/lib/folds.py` — descriptive colour) and the paired realised-CPL
backtest (`experiments/lib/realised.py` — the arbiter), then emits results.json +
rowpreds.parquet + fills.parquet and optionally posts a self-reported bd comment.

R0 caching across a batch's nights (the parent design's "fit it once, cache the
predictions") is deliberately NOT implemented here (DECIDED 2026-08-17, fps-3jj.4
discussion): `run_paired_realised_backtest` couples the candidate's held-tau score
to the baseline's own per-fold tau from the SAME call, so caching R0 correctly
needs either extending `experiments/lib/realised.py` with a cached-baseline seam
or accepting an approximation that scores the candidate at a stale tau. Both are
real design decisions with correctness consequences for the arbiter, deferred to
a follow-up bead (dormant until batch 2 makes the tradeoff concrete) rather than
guessed at here. Every run refits both arms.

Outcome taxonomy (five codes — see fps-3jj):
  rejected            — ran to completion. The default outcome; results.json's
                         effect_resolved / zone carry the actual verdict. A
                         graduate candidate still closes "rejected" — graduation
                         is a separate, human-initiated PR.
  disqualified        — the differential PIT test caught a leak.
  aborted_candidate   — add_columns/add_axis raised, an INPUTS/COLUMNS
                         declaration was wrong, all-NaN output, a COLUMNS/
                         baseline name collision, a degenerate zero-variance
                         fit, the WFCV screen itself raised, or a total
                         extra_feature_provider (station_code, date) key-format
                         miss. All deterministic and attributable to THIS
                         candidate — repeatable; do not retry.
  aborted_pipeline    — the run never got to say anything about the candidate
                         because the pipeline itself was misconfigured: a
                         realised-backtest config error (bad ArmSpec /
                         fold_subset / inner_fold_params for this batch). Every
                         candidate against this batch would fail identically,
                         so the candidate is NOT to blame and must go back on
                         the queue once the config is fixed (fps-g31). Retrying
                         unchanged is pointless; retrying after a fix is the
                         whole point. Retry once.
  aborted_environment — the frozen batch's baseline contract drifted underneath
                         it, or the DB-backed realised backtest raised something
                         genuinely unexpected (DB/disk/OOM/interrupted). Retry
                         once.

aborted_pipeline and aborted_environment are RETRYABLE_STATUSES: the claim goes
back to the queue once (launch.py's stale sweep, MAX_RETRIES=1). A second
retryable abort of the same claim blocks it instead of releasing it again —
a released-but-unassigned claim keeps its original creation date and would
otherwise win `bd ready`'s oldest-first ordering forever, starving every
other queued candidate (fps-rtd). Every terminal status posts a bd
comment — an abort that says nothing on the bead is indistinguishable from a run
that never happened, which is exactly how fps-32p sat stuck for a day (fps-g31).

Two CONFIDENCE fields (fps-3jj.4 DECIDED 2026-08-17):
  CONFIDENCE_EFFECT resolves mechanically as pooled delta_cpl_held < 0 (does it
    move the arbiter, at the baseline's own held tau — the #254 clean-
    attribution cut).
  CONFIDENCE_ZONE resolves ONLY when CONFIDENCE_EFFECT resolves true (a
    candidate that never moved the arbiter has no zone to be right about —
    scoring it would calibrate zone-accuracy on noise). Grades TARGET's named
    cells against the rest using pooled delta_cpl_OWN from the fills ledger
    (fills only carry each arm's own-tau fills — held-tau fills aren't
    collected by run_paired_realised_backtest), grouped by fold
    (TARGET["folds"]), the built-in "regime" axis (SHOCK_FOLDS shock-vs-normal),
    or a candidate's own add_axis label looked up per (station_code, date). A
    fill counts as "in target" if it matches EITHER a named fold OR the axis —
    the parent design's "conflation is harmless" call. Cells below
    min_row_cell_n are excluded, never reported as findings (fps-3jj slicing-
    axes decision).
"""
from __future__ import annotations

import inspect
import json
import pathlib
import subprocess
import time
from dataclasses import dataclass

import click
import numpy as np
import pandas as pd

from experiments.lib.aggregate import aggregate_with_deltas
from experiments.lib.cohorts import hard_quantile_mask
from experiments.lib.constants import SEEDS, SHOCK_FOLDS
from experiments.lib.fit import fit_score, per_row_log_loss
from experiments.lib.folds import iter_folds_with_baseline_fit
from experiments.lib.gates import seed_variance_gate
from experiments.lib.io import current_git_sha, to_jsonable
from experiments.lib.pit_test import PitLeakError
from experiments.lib.realised import ArmSpec, run_paired_realised_backtest
from experiments.lib.rowpreds import RowPredCollector
from experiments.lib.zones import pooled_cpl
from experiments.pipeline.batch_freeze import (
    BASELINE_COLUMNS_FILENAME,
    FROZEN_DB_FILENAME,
    BaselineContractMismatch,
    check_baseline_contract,
)
from experiments.pipeline.validate import (
    AllNaNColumnError,
    CandidateImportError,
    MissingInputColumnsError,
    MissingOutputColumnError,
    RestrictedFrameViolation,
    load_candidate_module,
    validate_candidate,
)
from fuel_signal import evaluate as _ev
from fuel_signal.features import load_features

STATUS_REJECTED = "rejected"
STATUS_DISQUALIFIED = "disqualified"
STATUS_ABORTED_CANDIDATE = "aborted_candidate"
STATUS_ABORTED_PIPELINE = "aborted_pipeline"
STATUS_ABORTED_ENVIRONMENT = "aborted_environment"

# Statuses where the candidate never got a fair hearing, so its claim must go
# back on the queue rather than be consumed. Read by launch.find_stale_claims.
RETRYABLE_STATUSES = frozenset({STATUS_ABORTED_PIPELINE, STATUS_ABORTED_ENVIRONMENT})

BASELINE_ARM = "R0"
CANDIDATE_ARM = "candidate"
PASS_CRITERION_FILENAME = "pass_criterion.json"
RESULTS_FILENAME = "results.json"

# walk_forward_folds' own defaults, read off the signature rather than restated,
# so a change there can't silently invalidate _check_fold_geometry below.
_WFCV_DEFAULTS = {
    name: param.default
    for name, param in inspect.signature(_ev.walk_forward_folds).parameters.items()
    if param.default is not inspect.Parameter.empty
}

# Below this many fills, a target/other cell is excluded from zone grading
# rather than reported as a finding (fps-3jj slicing-axes decision).
DEFAULT_MIN_ROW_CELL_N = 30

_WFCV_KWARGS = ("train_min_days", "val_days", "step_days")

# Inner (within-fold) walk-forward used by the realised backtest to build OOF
# predictions for calibration + tau selection.
#
# This MUST be set explicitly and MUST be smaller than the outer grid. The
# library default (fuel_signal.evaluate.walk_forward_folds, train_min_days=1825)
# is the same value the OUTER folds use, and outer fold 1's train window is
# exactly train_min_days long by construction — so an inner walk-forward at 1825
# can never fit a single fold inside it, and _train_calibrate_select_tau raises
# "no OOF folds over fold-train" on the very first fold. That is candidate-
# independent: it killed every run until fps-g31. Verified on batch0:
#
#   outer fold  1: train 1825d -> 0 inner folds at 1825, 8 at 1095
#   outer fold  2: train 1915d -> 0 inner folds at 1825, 9 at 1095
#   outer fold 14: train 2995d -> 12 inner folds at 1825, 21 at 1095
#
# 1095 (3y) is the value every prior realised-arbiter experiment passed by hand:
# experiments/2026-06-20_leading_indicators/realised_tgp.py:55,
# experiments/2026-06-18_late_descent_gate1/realised_by_regime.py:57,
# experiments/2026-06-19_headroom_map/headroom_map.py:118. Matching them keeps
# pipeline runs comparable to the hand-run experiments they're benchmarked
# against, so don't retune it casually.
DEFAULT_INNER_FOLD_PARAMS = {"train_min_days": 1095, "val_days": 90, "step_days": 90}


class FoldGeometryError(ValueError):
    """The inner walk-forward cannot fit inside outer fold 1's train window.

    Raised BEFORE any fitting, so a misconfigured run fails in a second with a
    usable message instead of ~9 minutes later inside the arbiter with
    "realised: no OOF folds over fold-train".
    """


def _check_fold_geometry(outer_fold_params: dict, inner_fold_params: dict) -> None:
    """Enforce the inner-vs-outer invariant the CLI help only advises about.

    Outer fold 1's train window is exactly `train_min_days` long, and
    walk_forward_folds needs

        train_min_days + buffer_days + val_days <= span

    to yield even one fold. So the real constraint is not "inner < outer" — at
    the defaults the boundary is 1095 + 7 + 90 = 1192, and an outer
    train_min_days of 1100 would pass a naive inner<outer check while still
    producing zero inner folds. Verified empirically: outer=1192 gives 1 inner
    fold, outer=1191 gives 0.

    Exposing --outer-train-min-days made this reachable from the CLI, which is
    what makes the check worth having: getting it wrong reproduces fps-g31
    exactly, and the resulting abort is now retryable, so it would re-queue
    nightly rather than fail once loudly.
    """
    outer_train = outer_fold_params.get("train_min_days", _WFCV_DEFAULTS["train_min_days"])
    inner_train = inner_fold_params.get("train_min_days", _WFCV_DEFAULTS["train_min_days"])
    inner_val = inner_fold_params.get("val_days", _WFCV_DEFAULTS["val_days"])
    buffer_days = inner_fold_params.get("buffer_days", _WFCV_DEFAULTS["buffer_days"])

    needed = inner_train + buffer_days + inner_val
    if needed > outer_train:
        raise FoldGeometryError(
            f"inner fold params don't fit inside outer fold 1's train window: "
            f"inner needs train_min_days({inner_train}) + buffer_days({buffer_days}) + "
            f"val_days({inner_val}) = {needed} days, but outer fold 1's train span is "
            f"train_min_days({outer_train}). Lower --inner-train-min-days/--inner-val-days "
            f"or raise --outer-train-min-days; the realised backtest would otherwise abort "
            f"on fold 1 with 'no OOF folds over fold-train' (fps-g31)."
        )


def read_run_status(out_dir: pathlib.Path) -> str | None:
    """The status recorded in out_dir/results.json, or None if it can't be read.

    Shared by launch.py (should this claim go back on the queue?) and
    dossier_tables.py (is this run finished enough to write up?) so the two
    can't drift on what counts as a finished run.

    Deliberately total: a missing file, a truncated write, a file that isn't
    valid UTF-8, or a results.json that isn't a JSON object all read as "no
    status". Both callers run unattended overnight, and an exception here would
    take down the whole nightly sweep on the strength of one malformed file.

    `ValueError`, not `json.JSONDecodeError`: read_text() raises
    UnicodeDecodeError on invalid UTF-8 — a write interrupted by SIGKILL/OOM
    partway through a multi-byte sequence, or on-disk corruption. That is a
    ValueError subclass and NOT an OSError, so it would escape a narrower
    except and defeat the totality this function exists to provide.
    JSONDecodeError is itself a ValueError subclass, so one clause covers both.
    """
    results_path = pathlib.Path(out_dir) / RESULTS_FILENAME
    try:
        parsed = json.loads(results_path.read_text())
    except (ValueError, OSError):
        return None
    if not isinstance(parsed, dict):
        return None
    status = parsed.get("status")
    return status if isinstance(status, str) else None


def default_out_dir(candidate_path: pathlib.Path) -> pathlib.Path:
    """Per-candidate run directory: candidate_path with its .py suffix stripped.

    Candidate modules are flat siblings within a batch dir
    (experiments/candidates/<batch>/<name>.py — see generator.md's Filing
    section), so candidate_path.parent is the whole BATCH directory, shared by
    every candidate filed against it. Using that as out_dir meant candidate
    2+ in a batch overwrote candidate 1's run.log/results.json/rowpreds.parquet/
    fills.parquet in place, and dossier_tables.py's find_pending_runs (any dir
    with results.json and no README.md) would then treat the directory as
    already-dossiered and never surface the overwriting candidate (fps-icv).
    Stripping .py gives each candidate its own subdirectory instead.
    """
    return candidate_path.with_suffix("")


@dataclass
class RunResult:
    status: str
    candidate_name: str
    wall_seconds: float
    results: dict
    error: str | None = None
    results_path: pathlib.Path | None = None
    rowpreds_path: pathlib.Path | None = None
    fills_path: pathlib.Path | None = None


def record_pass_criterion(batch_dir: pathlib.Path, criterion: str, **extra) -> pathlib.Path:
    """Pre-register a batch's pass criterion BEFORE any candidate in it runs.

    A workflow step at batch setup, not something a run can infer for itself —
    call once, before the first run_candidate() for the batch. run_candidate()
    reads this file (if present) and carries it into every run's results.json
    so the dossier routine (fps-3jj.6) can grade against it without re-deriving
    it after the fact. Batch 1's criterion (fps-3jj, tgp_delta_7d): "sign plus
    concentration — pooled realised CPL delta negative AND gains appearing in
    the expensive high-headroom folds. Magnitude explicitly not part of the
    test."
    """
    path = pathlib.Path(batch_dir) / PASS_CRITERION_FILENAME
    path.write_text(json.dumps({"criterion": criterion, **extra}, indent=2))
    return path


def run_candidate(
    batch_dir: pathlib.Path,
    candidate_path: pathlib.Path,
    *,
    out_dir: pathlib.Path | None = None,
    seeds: tuple[int, ...] = SEEDS,
    realised_seed: int = SEEDS[0],
    outer_fold_params: dict | None = None,
    inner_fold_params: dict | None = None,
    fold_subset=None,
    station_codes: list[int] | None = None,
    min_row_cell_n: int = DEFAULT_MIN_ROW_CELL_N,
    bead_id: str | None = None,
    verbose: bool = True,
) -> RunResult:
    """Validate, fit, and score one candidate against a frozen batch. Never retries."""
    t0 = time.perf_counter()
    batch_dir = pathlib.Path(batch_dir)
    candidate_path = pathlib.Path(candidate_path)
    out_dir = pathlib.Path(out_dir) if out_dir is not None else default_out_dir(candidate_path)
    outer_fold_params = dict(outer_fold_params or {})
    # Merged, not replaced: a caller overriding one key (say val_days) still gets
    # a workable train_min_days rather than silently falling back to the library
    # default that cannot fit inside outer fold 1. See DEFAULT_INNER_FOLD_PARAMS.
    inner_fold_params = {**DEFAULT_INNER_FOLD_PARAMS, **(inner_fold_params or {})}

    # Fail in a second, not ~9 minutes in. Raised rather than _finish()'d: a run
    # this misconfigured shouldn't leave a results.json at all, because that
    # file is the queue key for both downstream consumers (launch's claim sweep,
    # dossier's scan) and a caller who got the geometry wrong wants a traceback
    # they can act on, not a status code buried in an artifact.
    _check_fold_geometry(outer_fold_params, inner_fold_params)

    # A run dir is keyed on the candidate, so a re-run reuses it — and a
    # retryable abort now sends candidates back for exactly that (fps-g31).
    # Nothing else clears this file: run_candidate only ever WRITES results.json
    # (at _finish, or on completion), so a previous attempt's verdict would sit
    # on disk for this entire run. launch.find_stale_claims reads it with no age
    # gate, so an overlapping launch invocation would see the OLD status,
    # release this in-flight run's claim, and launch a second runner into the
    # same out_dir — two processes racing on results.json/rowpreds and both
    # appending to one run.log. Clearing it up front makes an in-flight run look
    # like what it is: started, no verdict yet.
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / RESULTS_FILENAME).unlink(missing_ok=True)

    try:
        candidate = load_candidate_module(candidate_path)
    except CandidateImportError as exc:
        return _finish(
            STATUS_ABORTED_CANDIDATE, candidate_path.stem, t0, out_dir,
            error=str(exc), bead_id=bead_id,
        )

    name = candidate.NAME
    frame = load_features(batch_dir / "features.csv")
    baseline_columns = json.loads((batch_dir / BASELINE_COLUMNS_FILENAME).read_text())
    db_path = batch_dir / FROZEN_DB_FILENAME

    try:
        check_baseline_contract(batch_dir, frame)
    except BaselineContractMismatch as exc:
        return _finish(STATUS_ABORTED_ENVIRONMENT, name, t0, out_dir, error=str(exc), bead_id=bead_id)

    try:
        validate_candidate(candidate, frame)
    except PitLeakError as exc:
        return _finish(STATUS_DISQUALIFIED, name, t0, out_dir, error=str(exc), bead_id=bead_id)
    except (
        RestrictedFrameViolation,
        MissingInputColumnsError,
        MissingOutputColumnError,
        AllNaNColumnError,
    ) as exc:
        return _finish(STATUS_ABORTED_CANDIDATE, name, t0, out_dir, error=str(exc), bead_id=bead_id)
    except Exception as exc:  # noqa: BLE001 — any other add_columns/add_axis failure is candidate-caused
        # validate_candidate calls add_columns/add_axis several times internally
        # (restricted-frame check, differential PIT truncations); the two
        # excepts above only cover the failure modes those calls are DESIGNED
        # to raise. Anything else (a ValueError/TypeError from the candidate's
        # own arithmetic) must still resolve to a status code, not an
        # uncaught traceback with no results.json.
        return _finish(
            STATUS_ABORTED_CANDIDATE, name, t0, out_dir,
            error=f"validation raised: {exc!r}", bead_id=bead_id,
        )

    overlap = set(candidate.COLUMNS) & set(baseline_columns)
    if overlap:
        return _finish(
            STATUS_ABORTED_CANDIDATE, name, t0, out_dir,
            error=f"{name}: COLUMNS overlaps the baseline contract: {sorted(overlap)}",
            bead_id=bead_id,
        )

    try:
        candidate_output = candidate.add_columns(frame)
        candidate_frame = frame.copy()
        for col in candidate.COLUMNS:
            candidate_frame[col] = candidate_output[col]
    except Exception as exc:  # noqa: BLE001 — candidate-caused, see outcome taxonomy
        return _finish(
            STATUS_ABORTED_CANDIDATE, name, t0, out_dir,
            error=f"add_columns raised: {exc!r}", bead_id=bead_id,
        )
    candidate_cols = list(baseline_columns) + list(candidate.COLUMNS)

    add_axis = getattr(candidate, "add_axis", None)
    axis_series = None
    axis_lookup = None
    if add_axis is not None:
        try:
            axis_series = add_axis(frame).rename("axis")
        except Exception as exc:  # noqa: BLE001 — candidate-caused, see outcome taxonomy
            return _finish(
                STATUS_ABORTED_CANDIDATE, name, t0, out_dir,
                error=f"add_axis raised: {exc!r}", bead_id=bead_id,
            )
        axis_lookup = _build_axis_lookup(frame, axis_series)

    try:
        df_rows, collector = _run_wfcv_screen(
            candidate_frame, baseline_columns, candidate_cols,
            seeds=seeds, axis_series=axis_series, outer_fold_params=outer_fold_params, verbose=verbose,
            persist_columns=list(candidate.COLUMNS),
        )
    except Exception as exc:  # noqa: BLE001 — deliberately broad, see outcome taxonomy below
        # Nothing here except LightGBM fits on the candidate's own columns; any
        # exception (bad dtype, inf from a degenerate transform, etc.) is
        # candidate-caused by construction. The outcome taxonomy (module
        # docstring) is explicit that aborted_candidate covers "add_columns
        # threw" generically, not a fixed exception allowlist — an unattended
        # pipeline running unreviewed LLM-authored code can't predict every
        # failure shape in advance. The exception isn't swallowed: repr(exc) is
        # preserved in results.json's error field and RunResult.error.
        return _finish(
            STATUS_ABORTED_CANDIDATE, name, t0, out_dir,
            error=f"WFCV screen raised: {exc!r}",
            bead_id=bead_id,
        )

    if len(seeds) < 2:
        # seed_variance_gate needs >=2 samples per (fold, run) cell to compute
        # a std (ddof=1 on n=1 divides by zero -> NaN cohort median -> raises).
        # `seeds` is an exposed parameter and a single-seed smoke run is a
        # natural use case; that should skip the gate, not deterministically
        # abort as aborted_candidate for a reason that has nothing to do with
        # the candidate.
        seed_var_summary, seed_var_flags = {}, []
        if verbose:
            print(f"[runner] seeds={seeds}: seed-variance gate skipped (needs >=2 seeds)", flush=True)
    else:
        try:
            seed_var_summary, seed_var_flags = seed_variance_gate(
                df_rows, {"all": "ll_all", "hard25": "ll_hard25"}
            )
        except ValueError as exc:
            return _finish(
                STATUS_ABORTED_CANDIDATE, name, t0, out_dir,
                error=f"seed-variance gate: {exc}", bead_id=bead_id,
            )

    try:
        fold_run = aggregate_with_deltas(
            df_rows, {"all": "ll_all", "hard25": "ll_hard25"}, baseline_run=BASELINE_ARM
        )
    except ValueError as exc:
        return _finish(STATUS_ABORTED_CANDIDATE, name, t0, out_dir, error=f"WFCV aggregation: {exc}", bead_id=bead_id)

    provider = _make_lookup_provider(candidate_frame, list(candidate.COLUMNS))
    arms = [
        ArmSpec(BASELINE_ARM, candidate_frame, feature_columns=baseline_columns),
        ArmSpec(CANDIDATE_ARM, candidate_frame, feature_columns=candidate_cols, extra_feature_provider=provider),
    ]
    try:
        realised = run_paired_realised_backtest(
            arms, baseline_columns,
            seed=realised_seed, station_codes=station_codes,
            outer_fold_params=outer_fold_params, inner_fold_params=inner_fold_params,
            fold_subset=fold_subset, db_path=db_path, collect_fills=True, verbose=verbose,
        )
    except ValueError as exc:
        # experiments/lib/realised.py's own explicit ValueErrors are all
        # config-shaped (duplicate/reserved arm names, mismatched arm index,
        # empty feature_columns, or inner_fold_params too large for this
        # batch's date range to yield any OOF fold) — deterministic and NOT a
        # DB/disk/environment failure, so aborted_environment's "retry once"
        # would just fail identically on the same batch.
        #
        # It isn't the CANDIDATE's fault either: every one of these is a
        # property of how the RUN was configured, so every candidate against
        # this batch fails identically. That is not hypothetical — it is what
        # fps-g31 was: the whole queue drained into this branch one candidate
        # per night, each recorded as though the feature had been tried and
        # found wanting. aborted_pipeline says what actually happened and puts
        # the claim back on the queue.
        return _finish(
            STATUS_ABORTED_PIPELINE, name, t0, out_dir,
            error=f"realised backtest config error (not a DB/environment failure): {exc!r}",
            bead_id=bead_id,
        )
    except Exception as exc:  # noqa: BLE001 — genuinely unexpected: DB/disk/OOM/interrupted
        # By this point the candidate's own columns already passed validation
        # AND the WFCV screen, and the ValueError branch above has already
        # peeled off realised.py's known config-shaped failures — anything
        # left really is DB/disk/interrupted, the aborted_environment outcome
        # (retry once). repr(exc) is preserved in results.json, not swallowed.
        return _finish(
            STATUS_ABORTED_ENVIRONMENT, name, t0, out_dir,
            error=f"realised backtest raised: {exc!r}", bead_id=bead_id,
        )

    provider_health_error = _check_provider_health(provider)
    if provider_health_error is not None:
        # A (station_code, date) key-format mismatch is deterministic — same
        # reasoning as the ValueError split above (finding #3): aborted_candidate
        # ("repeatable, don't retry"), not aborted_environment ("retry once",
        # which would just fail identically on this same batch).
        return _finish(STATUS_ABORTED_CANDIDATE, name, t0, out_dir, error=provider_health_error, bead_id=bead_id)

    # Write the expensive artifacts BEFORE grading: a bug in the grading step
    # (which is comparatively cheap — DataFrame groupbys over already-computed
    # results) must not discard a completed multi-hour backtest.
    out_dir.mkdir(parents=True, exist_ok=True)
    rowpreds_path = out_dir / "rowpreds.parquet"
    collector.to_parquet(rowpreds_path)
    fills_path = out_dir / "fills.parquet"
    realised.fills.to_parquet(fills_path, index=False)

    target = getattr(candidate, "TARGET", None)
    effect_resolved, effect_delta, zone, grading_error = _grade_run(
        realised, target, axis_lookup, min_row_cell_n=min_row_cell_n
    )

    wall_seconds = time.perf_counter() - t0
    results = {
        "status": STATUS_REJECTED,
        "candidate": {
            "name": name,
            "hypothesis": getattr(candidate, "HYPOTHESIS", None),
            "predicted_signature": getattr(candidate, "PREDICTED_SIGNATURE", None),
            "confidence_effect": getattr(candidate, "CONFIDENCE_EFFECT", None),
            "confidence_zone": getattr(candidate, "CONFIDENCE_ZONE", None),
            "columns": list(candidate.COLUMNS),
            "inputs": list(candidate.INPUTS),
            "prior_art": getattr(candidate, "PRIOR_ART", None),
            "target": target,
        },
        "effect_resolved": effect_resolved,
        "effect_delta_cpl_held": effect_delta,
        "zone": zone,
        "grading_error": grading_error,
        "aggregate": realised.aggregate.to_dict(orient="records"),
        "fold_run_deltas": fold_run[
            ["fold", "regime", "run", "delta_ll_all_median", "delta_ll_hard25_median"]
        ].to_dict(orient="records"),
        "seed_variance": {"summary": seed_var_summary, "flags": seed_var_flags},
        "error": None,
        "meta": {
            "batch_dir": str(batch_dir),
            "seeds": list(seeds),
            "realised_seed": realised_seed,
            "n_windows": realised.meta["n_windows"],
            "realised_wall_seconds": realised.meta["total_wall_seconds"],
            "wall_seconds": wall_seconds,
            "pass_criterion": _read_pass_criterion(batch_dir),
            "extra_feature_provider_hits": provider.stats["hits"],
            "extra_feature_provider_misses": provider.stats["misses"],
            "git_sha": current_git_sha(),
            "bead_id": bead_id,
        },
    }
    results_path = out_dir / RESULTS_FILENAME
    results_path.write_text(json.dumps(to_jsonable(results), indent=2, default=str))

    if bead_id:
        post_bd_comment(bead_id, _summarise_for_comment(results))

    return RunResult(
        status=STATUS_REJECTED, candidate_name=name, wall_seconds=wall_seconds, results=results,
        results_path=results_path, rowpreds_path=rowpreds_path, fills_path=fills_path,
    )


def _run_wfcv_screen(
    candidate_frame: pd.DataFrame,
    baseline_columns: list[str],
    candidate_cols: list[str],
    *,
    seeds: tuple[int, ...],
    axis_series: pd.Series | None,
    outer_fold_params: dict,
    verbose: bool,
    persist_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, RowPredCollector]:
    """The WFCV log-loss screen (descriptive colour) — R0 vs candidate, all seeds.

    Mirrors experiments/TEMPLATE_paired_wfcv.py's per-(fold, run, seed) loop:
    R0's SEEDS[0] fit is reused from iter_folds_with_baseline_fit rather than
    refit. Persists the axis column into rowpreds' ident_base when the
    candidate declared one (fps-3jj.4 acceptance criterion).

    persist_columns (fps-3jj.6): the candidate's own engineered COLUMNS, and
    cycle_pct_through when present, are copied into rowpreds' ident_base too —
    the dossier routine's "candidate over time with its NaN band" and
    "feature vs cycle_pct_through" plots have no other artifact to read a raw
    row-level column value from (rowpreds/fills otherwise carry only
    probabilities and fill economics). Optional and additive: omitting it
    reproduces the pre-fps-3jj.6 rowpreds schema exactly.
    """
    wfcv_kwargs = {k: v for k, v in outer_fold_params.items() if k in _WFCV_KWARGS}
    rows: list[dict] = []
    collector = RowPredCollector(pd.DataFrame())

    for fold_idx, regime, train_df, val_df, ll0, p0, t_fit0, prl0 in iter_folds_with_baseline_fit(
        candidate_frame, baseline_columns, seed=seeds[0], **wfcv_kwargs
    ):
        vd = pd.to_datetime(val_df["price_date"])
        y = val_df["label"].to_numpy(dtype=int)
        hard25_mask = hard_quantile_mask(prl0, 0.75)

        ident: dict[str, object] = {
            "fold": np.int8(fold_idx),
            "station_code": val_df["station_code"].to_numpy(),
            "price_date": vd.to_numpy(),
            "label": y.astype(np.int8),
            "is_hard25": hard25_mask.astype(np.int8),
        }
        if axis_series is not None:
            ident["axis"] = axis_series.loc[val_df.index].to_numpy()
        if "cycle_pct_through" in val_df.columns:
            ident["cycle_pct_through"] = val_df["cycle_pct_through"].to_numpy()
        for col in persist_columns or ():
            if col in val_df.columns:
                ident[col] = val_df[col].to_numpy()
        collector.ident_base = pd.DataFrame(ident)

        for run_name, cols in ((BASELINE_ARM, baseline_columns), (CANDIDATE_ARM, candidate_cols)):
            for seed in seeds:
                if run_name == BASELINE_ARM and seed == seeds[0]:
                    ll, p, t_fit = ll0, p0, t_fit0
                else:
                    ll, p, t_fit = fit_score(train_df, val_df, cols, seed)
                prl = per_row_log_loss(y, p)
                rows.append({
                    "fold": fold_idx, "regime": regime, "run": run_name, "seed": seed,
                    "ll_all": ll,
                    "ll_hard25": float(prl[hard25_mask].mean()) if hard25_mask.any() else float("nan"),
                    "fit_s": t_fit,
                })
                collector.add(run_name, seed, p)
        if verbose:
            print(f"[wfcv] fold {fold_idx:>2} {regime:<6} done", flush=True)

    return pd.DataFrame(rows), collector


def _build_axis_lookup(frame: pd.DataFrame, axis_series: pd.Series) -> pd.DataFrame:
    """(station_code, date, axis) lookup table for _resolve_zone's fills merge.

    reindex, not to_numpy() straight off axis_series: add_axis is allowed to
    return its Series in a different row order than frame
    (differential_pit_test compares by index label, not position — see
    pit_test.py's docstring), so a candidate that sorts internally would
    otherwise pair frame's row i with axis_series's row i, silently
    mismatching station_code/date to the wrong axis label. This must align by
    index label, matching every other use of axis_series in this module
    (candidate_frame[col] assignment in run_candidate, and
    axis_series.loc[val_df.index] in _run_wfcv_screen).
    """
    return pd.DataFrame({
        "station_code": frame["station_code"].to_numpy(),
        "date": pd.to_datetime(frame["price_date"]).dt.strftime("%Y-%m-%d"),
        "axis": axis_series.reindex(frame.index).to_numpy(),
    })


def _read_pass_criterion(batch_dir: pathlib.Path) -> dict | None:
    path = pathlib.Path(batch_dir) / PASS_CRITERION_FILENAME
    return json.loads(path.read_text()) if path.exists() else None


def _make_lookup_provider(candidate_frame: pd.DataFrame, columns: list[str], date_column: str = "price_date"):
    """Build a realised-backtest extra_feature_provider from a precomputed candidate frame.

    Both train (fit) and live (replay) read the SAME precomputed table by
    (station_code, date) — there's no second computation path to diverge from
    the first (the #270 "two sites must agree" hazard doesn't apply here, since
    add_columns runs exactly once, offline, and this is a pure lookup over its
    output).

    The returned function carries a `.stats` dict ({"hits": int, "misses":
    int}), mutated in place on every call — the caller inspects it after the
    backtest to catch a systemic key-format mismatch (see run_candidate's
    post-backtest check) rather than silently reporting "no effect".
    """
    keyed = candidate_frame[["station_code", date_column, *columns]].copy()
    keyed["_date_key"] = pd.to_datetime(keyed[date_column]).dt.strftime("%Y-%m-%d")
    lookup = keyed.set_index(["station_code", "_date_key"])[columns]
    stats = {"hits": 0, "misses": 0}

    def provider(as_of: str, station_code: int, station_price: float) -> dict[str, float | None]:
        try:
            row = lookup.loc[(station_code, as_of)]
        except KeyError:
            stats["misses"] += 1
            return dict.fromkeys(columns)
        stats["hits"] += 1
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]
        return {c: (None if pd.isna(row[c]) else float(row[c])) for c in columns}

    provider.stats = stats
    return provider


def _check_provider_health(provider) -> str | None:
    """None if the extra_feature_provider looked healthy during the replay it
    just ran through; an error message if every lookup missed.

    A 100% miss means the candidate arm was scored with its added column(s)
    silently NaN for the entire backtest — that reads as a legitimate
    effect_resolved: false ("the feature does nothing") when it's actually a
    (station_code, date) key-format mismatch (features.csv's price_date
    representation has already varied between INTEGER YYYYMMDD and ISO
    strings — batch_freeze.py's _snapshot_date handles both for exactly this
    reason). Fail loudly instead of reporting a false negative. The caller
    maps this to aborted_candidate, not aborted_environment: the mismatch is
    deterministic for this (batch, candidate) pair, so a retry fails
    identically — same reasoning as the realised-backtest ValueError split.
    """
    if provider.stats["misses"] > 0 and provider.stats["hits"] == 0:
        return (
            f"extra_feature_provider missed on all {provider.stats['misses']} lookups "
            "during the realised backtest replay — likely a (station_code, date) "
            "key-format mismatch between the frozen batch and the runner's lookup, "
            "not a legitimate 'no effect' result."
        )
    return None


def _grade_run(
    realised, target: dict | None, axis_lookup: pd.DataFrame | None, *, min_row_cell_n: int,
) -> tuple[bool | None, float | None, dict, str | None]:
    """CONFIDENCE_EFFECT / CONFIDENCE_ZONE grading, isolated so a bug here
    can't discard an already-completed (and already artifact-written)
    backtest. Returns (effect_resolved, effect_delta, zone, grading_error);
    on failure the first three are None/a "not resolved" zone dict and
    grading_error carries repr(exc).
    """
    try:
        effect_resolved, effect_delta = _resolve_effect(realised.aggregate, BASELINE_ARM, CANDIDATE_ARM)
        if not target:
            zone = {"resolved": None, "reason": "candidate declared no TARGET"}
        elif not effect_resolved:
            zone = {"resolved": None, "reason": "CONFIDENCE_EFFECT did not resolve true — scored conditionally"}
        else:
            zone = _resolve_zone(
                target, realised.fills, BASELINE_ARM, CANDIDATE_ARM, axis_lookup, min_row_cell_n=min_row_cell_n
            )
        return effect_resolved, effect_delta, zone, None
    except Exception as exc:  # noqa: BLE001 — a grading bug must not discard a completed backtest
        grading_error = repr(exc)
        return None, None, {"resolved": None, "reason": f"grading raised: {grading_error}"}, grading_error


def _resolve_effect(aggregate: pd.DataFrame, baseline_name: str, candidate_name: str) -> tuple[bool, float]:
    by_arm = aggregate.set_index("arm")
    delta = float(by_arm.loc[candidate_name, "cpl_held"] - by_arm.loc[baseline_name, "cpl_held"])
    return delta < 0, delta


def _resolve_zone(
    target: dict,
    fills: pd.DataFrame,
    baseline_name: str,
    candidate_name: str,
    axis_lookup: pd.DataFrame | None,
    *,
    min_row_cell_n: int = DEFAULT_MIN_ROW_CELL_N,
) -> dict:
    """Grade TARGET's named cells against the rest, on pooled delta_cpl_OWN.

    fills only carries each arm's own-tau fills (run_paired_realised_backtest
    doesn't collect held-tau fills), so this is an own-tau economics grade, not
    the held-tau number effect_resolved uses.
    """
    folds_wanted = set(target.get("folds") or [])
    axis_name = target.get("axis")
    expect = set(target.get("expect_concentration_in") or [])

    if not folds_wanted and not (axis_name and expect):
        return {"resolved": None, "reason": "TARGET declared neither folds nor axis/expect_concentration_in"}

    in_target = pd.Series(False, index=fills.index)
    if folds_wanted:
        in_target |= fills["fold"].isin(folds_wanted)
    if axis_name and expect:
        if axis_name == "regime":
            regime = fills["fold"].map(lambda f: "shock" if f in SHOCK_FOLDS else "normal")
            in_target |= regime.isin(expect)
        elif axis_lookup is None:
            return {"resolved": None, "reason": f"TARGET axis {axis_name!r} needs add_axis; candidate has none"}
        else:
            tagged = fills.merge(axis_lookup, on=["station_code", "date"], how="left", validate="many_to_one")
            in_target |= tagged["axis"].isin(expect).to_numpy()

    if not in_target.any() or in_target.all():
        return {"resolved": None, "reason": "TARGET's named cells matched none or all of the fills"}

    def _cell_delta(mask: pd.Series) -> tuple[float | None, int, int]:
        cand = fills[(fills["arm"] == candidate_name) & mask]
        base = fills[(fills["arm"] == baseline_name) & mask]
        if len(cand) < min_row_cell_n or len(base) < min_row_cell_n:
            return None, len(cand), len(base)
        return pooled_cpl(cand) - pooled_cpl(base), len(cand), len(base)

    target_delta, n_target_cand, n_target_base = _cell_delta(in_target)
    other_delta, n_other_cand, n_other_base = _cell_delta(~in_target)
    if target_delta is None or other_delta is None:
        return {
            "resolved": None, "reason": "a cell is below min_row_cell_n",
            "min_row_cell_n": min_row_cell_n,
            "n_target_candidate_fills": n_target_cand, "n_target_baseline_fills": n_target_base,
            "n_other_candidate_fills": n_other_cand, "n_other_baseline_fills": n_other_base,
        }
    return {
        "resolved": bool(target_delta < other_delta),
        "target_delta_cpl_own": target_delta,
        "other_delta_cpl_own": other_delta,
        "n_target_candidate_fills": n_target_cand,
        "n_other_candidate_fills": n_other_cand,
    }


def _finish(
    status: str, name: str, t0: float, out_dir: pathlib.Path, *, error: str,
    bead_id: str | None = None,
) -> RunResult:
    """Write a status-only results.json, self-report to the bead, and return early.

    Covers every non-completing outcome (validation, candidate, pipeline,
    environment). The bd comment is not optional polish: before fps-g31 this
    path returned without posting anything, so an aborted run was
    indistinguishable on the bead from a run that never started — fps-32p
    aborted at 22:07 and nothing said so until a human read run.log the next
    day. Whatever else changes here, keep something landing on the bead.
    """
    wall_seconds = time.perf_counter() - t0
    results = {"status": status, "candidate": {"name": name}, "error": error, "meta": {"wall_seconds": wall_seconds}}
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / RESULTS_FILENAME
    results_path.write_text(json.dumps(to_jsonable(results), indent=2, default=str))
    if bead_id:
        retryable = status in RETRYABLE_STATUSES
        post_bd_comment(
            bead_id,
            f"[pipeline] {name}: {status} (wall={wall_seconds:.1f}s)\n{error}\n\n"
            + (
                "This is a pipeline/environment fault, not a verdict on the candidate — "
                "the claim will be released for a re-run by the next launch sweep."
                if retryable
                else "Terminal for this candidate; not retried."
            ),
        )
    return RunResult(
        status=status, candidate_name=name, wall_seconds=wall_seconds, results=results,
        error=error, results_path=results_path,
    )


def post_bd_comment(issue_id: str, text: str) -> None:
    """Self-reported bd comment — `bd comment <id> --stdin`.

    Best-effort: called after results.json/rowpreds/fills are already written,
    so a `bd` CLI hiccup (missing binary, network) must not raise past a
    successful run and lose the artifacts' return value. Prints instead.
    """
    try:
        subprocess.run(["bd", "comment", issue_id, "--stdin"], input=text, text=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"[runner] post_bd_comment({issue_id!r}) failed, continuing: {exc}", flush=True)


def _summarise_for_comment(results: dict) -> str:
    c = results["candidate"]
    effect_resolved = results["effect_resolved"]
    if effect_resolved is None:
        # _grade_run's exception path: effect_resolved/effect_delta_cpl_held
        # are None, not False — "did not move the arbiter" would misreport a
        # grading failure as a real (negative) finding.
        grading_error = results.get("grading_error")
        effect = f"grading failed — {grading_error}" if grading_error else "effect not resolved"
    else:
        effect = "moved the arbiter" if effect_resolved else "did not move the arbiter"
    delta = results["effect_delta_cpl_held"]
    delta_str = f"{delta:+.4f}" if delta is not None else "n/a"
    lines = [
        f"[pipeline] {c['name']}: {results['status']} — {effect} "
        f"(delta_cpl_held={delta_str})",
    ]
    zone = results["zone"]
    if zone.get("resolved") is not None:
        verdict = "concentrated where predicted" if zone["resolved"] else "did NOT concentrate where predicted"
        lines.append(
            f"Zone: {verdict} (target={zone.get('target_delta_cpl_own', float('nan')):+.4f} "
            f"vs other={zone.get('other_delta_cpl_own', float('nan')):+.4f})"
        )
    else:
        lines.append(f"Zone: not resolved — {zone.get('reason', '')}")
    meta = results["meta"]
    lines.append(
        f"wall={meta.get('wall_seconds', float('nan')):.1f}s  "
        f"windows={meta.get('n_windows')}  seeds={meta.get('seeds')}"
    )
    return "\n".join(lines)


@click.command("runner")
@click.option(
    "--batch-dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path),
    help="Frozen batch directory (see experiments/pipeline/batch_freeze.py).",
)
@click.option(
    "--candidate", "candidate_path", required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
    help="Candidate module .py file.",
)
@click.option("--bead-id", default=None, help="bd issue ID to post the self-reported comment to.")
@click.option(
    "--outer-train-min-days", type=int, default=None,
    help="Outer walk-forward train_min_days (default: fuel_signal.evaluate's 1825).",
)
@click.option("--outer-val-days", type=int, default=None, help="Outer walk-forward val_days.")
@click.option("--outer-step-days", type=int, default=None, help="Outer walk-forward step_days.")
@click.option(
    "--inner-train-min-days", type=int,
    default=DEFAULT_INNER_FOLD_PARAMS["train_min_days"], show_default=True,
    help="Within-fold OOF walk-forward train_min_days. MUST be smaller than the outer "
         "grid's — see DEFAULT_INNER_FOLD_PARAMS.",
)
@click.option(
    "--inner-val-days", type=int,
    default=DEFAULT_INNER_FOLD_PARAMS["val_days"], show_default=True,
    help="Within-fold OOF walk-forward val_days.",
)
@click.option(
    "--inner-step-days", type=int,
    default=DEFAULT_INNER_FOLD_PARAMS["step_days"], show_default=True,
    help="Within-fold OOF walk-forward step_days.",
)
def main(
    batch_dir: pathlib.Path,
    candidate_path: pathlib.Path,
    bead_id: str | None,
    outer_train_min_days: int | None,
    outer_val_days: int | None,
    outer_step_days: int | None,
    inner_train_min_days: int,
    inner_val_days: int,
    inner_step_days: int,
) -> None:
    """Entry point for the launch routine (fps-3jj.5) to invoke this as a detached subprocess."""
    outer_fold_params = {
        k: v
        for k, v in (
            ("train_min_days", outer_train_min_days),
            ("val_days", outer_val_days),
            ("step_days", outer_step_days),
        )
        if v is not None
    }
    result = run_candidate(
        batch_dir, candidate_path, bead_id=bead_id,
        outer_fold_params=outer_fold_params,
        inner_fold_params={
            "train_min_days": inner_train_min_days,
            "val_days": inner_val_days,
            "step_days": inner_step_days,
        },
    )
    click.echo(f"{result.candidate_name}: {result.status} (wall={result.wall_seconds:.1f}s)")


if __name__ == "__main__":
    main()
