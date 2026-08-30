"""Tests for experiments/pipeline/dossier_tables.py (fps-3jj.6).

No real batch/DB exists in this checkout (batches are local-only, gitignored — see the
parent design's "the pipeline is gated on the Mac being awake"), so these tests build
results.json/rowpreds.parquet/fills.parquet fixtures by hand, matching the exact schema
experiments/pipeline/runner.py (fps-3jj.4) actually writes, rather than running a live
run_candidate() end to end.
"""
from __future__ import annotations

import bisect
import json
import os
import pathlib

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from experiments.lib.constants import ROW_AXIS_ECONOMICS_CAVEAT, SHOCK_FOLDS
from experiments.lib.flips import summarise_regret
from experiments.pipeline import dossier_tables as dt
from experiments.pipeline.runner import BASELINE_ARM, CANDIDATE_ARM, RETRYABLE_STATUSES

SHOCK = {1, 4}
FOLDS = [1, 2, 3, 4]
N_DATES = 20
DATES = pd.date_range("2026-01-01", periods=N_DATES, freq="D")


def _fold_run_deltas() -> list[dict]:
    rows = []
    for fold in FOLDS:
        regime = "shock" if fold in SHOCK else "normal"
        for run in (BASELINE_ARM, CANDIDATE_ARM):
            rows.append({
                "fold": fold, "regime": regime, "run": run,
                "delta_ll_all_median": 0.0 if run == BASELINE_ARM else (0.01 if fold in SHOCK else -0.02),
                "delta_ll_hard25_median": 0.0 if run == BASELINE_ARM else (0.02 if fold in SHOCK else -0.03),
            })
    return rows


def _make_results(batch_dir: pathlib.Path, *, status: str = "graded", target=None,
                   inputs=None, columns=None, error=None, bead_id="fps-test.1",
                   tank_params: str | None = "50/3.571/7d/10%",
                   effect_delta_cpl_held: float = -0.05,
                   omit_shock_folds: bool = False) -> dict:
    meta = {
        "batch_dir": str(batch_dir), "seeds": [42, 43, 44], "realised_seed": 42,
        "n_windows": 4, "realised_wall_seconds": 12.3, "wall_seconds": 20.0,
        "pass_criterion": {"criterion": "sign plus concentration"},
        "extra_feature_provider_hits": 100, "extra_feature_provider_misses": 0,
        "git_sha": "abc1234", "bead_id": bead_id,
        "n_baseline_columns": 54, "baseline_fingerprint": "54:deadbeef1234",
        "tank_params": tank_params,
    }
    if not omit_shock_folds:
        # A results.json predating fps-3tu's shock-fold field simply never had the key —
        # not present-but-null — so the legacy-path fixture omits it entirely rather than
        # setting it to None, matching what an actual old file on disk looks like.
        meta["shock_folds"] = sorted(SHOCK)
    return {
        "status": status,
        "candidate": {
            "name": "cand", "hypothesis": "test hypothesis",
            "predicted_signature": "helps normal folds, not shock",
            "confidence_effect": 0.6, "confidence_zone": 0.3,
            "columns": columns if columns is not None else ["cand_col"],
            "inputs": inputs if inputs is not None else ["price_date"],
            "prior_art": None, "mechanism_family": "test-family", "target": target,
        },
        "effect_resolved": True if status == "graded" else None,
        "effect_delta_cpl_held": effect_delta_cpl_held if status == "graded" else None,
        "zone": (
            {"resolved": True, "target_delta_cpl_own": -0.1, "other_delta_cpl_own": 0.02}
            if status == "graded" else {"resolved": None}
        ),
        "grading_error": None,
        "aggregate": [
            {"arm": BASELINE_ARM, "cpl_own": 4.20, "cpl_held": 4.25, "saving_own_pct": 3.0, "saving_held_pct": 2.8},
            {"arm": CANDIDATE_ARM, "cpl_own": 4.15, "cpl_held": 4.20, "saving_own_pct": 3.2, "saving_held_pct": 3.0},
        ] if status == "graded" else None,
        "fold_run_deltas": _fold_run_deltas() if status == "graded" else None,
        "seed_variance": {
            "summary": {"all": {"cohort_median_seed_std": 0.01, "n_cells": 8, "n_flagged_gt_5x": 1}},
            "flags": [
                {"cohort": "all", "fold": 4, "run": "candidate", "seed_std": 0.09, "ratio_vs_cohort_median": 9.0}
            ],
        } if status == "graded" else {},
        "error": error,
        "meta": meta,
    }


def _make_fills(*, low_n_fold: int | None = None, n_per_cell: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for fold in FOLDS:
        n = 10 if fold == low_n_fold else n_per_cell
        for arm in (BASELINE_ARM, CANDIDATE_ARM):
            bump = -1.0 if arm == CANDIDATE_ARM else 0.0  # candidate slightly cheaper
            for i in range(n):
                litres = 40.0
                price = 150.0 + rng.normal() + bump
                rows.append({
                    "fold": fold, "arm": arm, "own_tau": 0.25,
                    "date": DATES[i % N_DATES].strftime("%Y-%m-%d"),
                    "station_code": i % 4,
                    "price": price, "litres": litres,
                    "spend_cents": price * litres, "emergency": False,
                })
    return pd.DataFrame(rows)


def _make_rowpreds(*, with_axis: bool = False, with_cycle: bool = False,
                    candidate_col: str | None = "cand_col") -> pd.DataFrame:
    rng = np.random.default_rng(1)
    # label is a property of (fold, station_code, price_date) alone — fixed at labelling time,
    # not re-drawn per (run, seed) — same invariant the real runner's ident_base encodes (one
    # label per validation row, reused across every run/seed in that fold). Drawing it once per
    # key here (rather than inside the run/seed loop) keeps the fixture honest for
    # _plot_tau_sweep's "label=first" aggregation, which relies on exactly this invariant.
    label_by_key = {
        (fold, station, i): int(rng.uniform() < 0.5)
        for fold in FOLDS for station in range(4) for i in range(N_DATES)
    }
    axis_by_station = {station: ("A" if station % 2 == 0 else "B") for station in range(4)}
    rows = []
    for fold in FOLDS:
        for run in (BASELINE_ARM, CANDIDATE_ARM):
            for seed in (42, 43, 44):
                for i in range(N_DATES):
                    for station in range(4):
                        row = {
                            "fold": fold, "station_code": station, "price_date": DATES[i],
                            "label": label_by_key[(fold, station, i)], "is_hard25": int(i % 4 == 0),
                            "run": run, "seed": seed,
                            "proba": float(np.clip(rng.uniform(), 0.01, 0.99)),
                        }
                        if with_axis:
                            row["axis"] = axis_by_station[station]
                        if with_cycle:
                            row["cycle_pct_through"] = float(rng.uniform(0, 1.3))
                        if candidate_col:
                            row[candidate_col] = (
                                float("nan") if rng.uniform() < 0.05 else float(rng.normal())
                            )
                        rows.append(row)
    return pd.DataFrame(rows)


def _write_run(
    tmp_path, *, status="graded", target=None, inputs=None, columns=None,
    with_axis=False, with_cycle=False, low_n_fold=None, batch_extras=None,
    tank_params: str | None = "50/3.571/7d/10%", fills_df: pd.DataFrame | None = None,
    effect_delta_cpl_held: float = -0.05, omit_shock_folds: bool = False,
) -> tuple[pathlib.Path, pathlib.Path]:
    batch_dir = tmp_path / "batch1"
    batch_dir.mkdir(exist_ok=True)
    (batch_dir / dt.FREEZE_MANIFEST_FILENAME).write_text(json.dumps({"snapshot_date": "2026-08-15"}))
    (batch_dir / dt.BASELINE_COLUMNS_FILENAME).write_text(
        json.dumps(["station_price_cents", "cycle_pct_through"])
    )
    if batch_extras:
        batch_extras(batch_dir)

    run_dir = tmp_path / "run1"
    run_dir.mkdir(exist_ok=True)
    results = _make_results(
        batch_dir, status=status, target=target, inputs=inputs, columns=columns,
        tank_params=tank_params, effect_delta_cpl_held=effect_delta_cpl_held,
        omit_shock_folds=omit_shock_folds,
    )
    (run_dir / dt.RESULTS_FILENAME).write_text(json.dumps(results))
    if status == "graded":
        rowpreds = _make_rowpreds(
            with_axis=with_axis, with_cycle=with_cycle, candidate_col=(columns or ["cand_col"])[0]
        )
        rowpreds.to_parquet(run_dir / dt.ROWPREDS_FILENAME)
        fills = fills_df if fills_df is not None else _make_fills(low_n_fold=low_n_fold)
        fills.to_parquet(run_dir / dt.FILLS_FILENAME)
    return run_dir, batch_dir


# ── work queue ────────────────────────────────────────────────────────────────

def test_find_pending_runs_skips_in_progress_and_already_dossiered(tmp_path):
    run_dir, _ = _write_run(tmp_path)
    (tmp_path / "in_progress").mkdir()  # no results.json at all
    done_dir = tmp_path / "done"
    done_dir.mkdir()
    (done_dir / dt.RESULTS_FILENAME).write_text("{}")
    (done_dir / dt.README_FILENAME).write_text("# done")

    pending = dt.find_pending_runs(tmp_path)

    assert pending == [run_dir]


def test_find_pending_runs_surfaces_deliberate_rerun(tmp_path):
    """fps-0yd: a candidate re-run after its prior dossiering still gets picked up.

    Mirror image of fps-g31's abort-then-succeed case: here the run dir already has a
    README from an earlier dossiering, and a human deliberately re-runs the candidate
    (e.g. after a contract fix), overwriting results.json with a fresh verdict. Requiring
    "no README.md" alone would hide that re-run forever; the results.json being NEWER than
    the stale README is what should make it pending again.
    """
    run_dir = tmp_path / "rerun"
    run_dir.mkdir()
    readme = run_dir / dt.README_FILENAME
    readme.write_text("# stale")
    results = run_dir / dt.RESULTS_FILENAME
    results.write_text(json.dumps({"status": "graded"}))
    # Force an unambiguous mtime ordering regardless of filesystem timestamp resolution.
    old = readme.stat().st_mtime - 10
    os.utime(readme, (old, old))

    assert dt.find_pending_runs(tmp_path) == [run_dir]


def test_find_pending_runs_skips_readme_newer_than_results(tmp_path):
    """A README written AFTER the current results.json (the ordinary post-dossier state)
    stays out of the queue — only a results.json newer than its README counts as pending."""
    run_dir = tmp_path / "dossiered"
    run_dir.mkdir()
    (run_dir / dt.RESULTS_FILENAME).write_text(json.dumps({"status": "graded"}))
    (run_dir / dt.README_FILENAME).write_text("# done")

    assert dt.find_pending_runs(tmp_path) == []


@pytest.mark.parametrize("status", sorted(RETRYABLE_STATUSES))
def test_find_pending_runs_skips_retryable_aborts(tmp_path, status):
    """fps-g31: dossiering a retryable abort permanently hides its successful re-run.

    A run dir is keyed on the candidate, so the re-run reuses it. Write a README
    for the aborted attempt and find_pending_runs (which requires no README)
    will never surface the re-run that overwrote results.json with a real
    verdict.
    """
    aborted = tmp_path / "aborted"
    aborted.mkdir()
    (aborted / dt.RESULTS_FILENAME).write_text(json.dumps({"status": status, "error": "bad config"}))

    assert dt.find_pending_runs(tmp_path) == []


@pytest.mark.parametrize("status", ["graded", "disqualified", "aborted_candidate"])
def test_find_pending_runs_keeps_terminal_statuses(tmp_path, status):
    """Only retryable runs are held back — real verdicts still get written up."""
    run_dir = tmp_path / "terminal"
    run_dir.mkdir()
    (run_dir / dt.RESULTS_FILENAME).write_text(json.dumps({"status": status}))

    assert dt.find_pending_runs(tmp_path) == [run_dir]


def test_find_pending_runs_keeps_unreadable_results_json(tmp_path):
    """An unparseable results.json stays in the queue: process_run's own try/except
    logs and skips it, which surfaces the problem. Silently dropping it here would
    hide a malformed run instead."""
    run_dir = tmp_path / "truncated"
    run_dir.mkdir()
    (run_dir / dt.RESULTS_FILENAME).write_text("{ truncated mid-write")

    assert dt.find_pending_runs(tmp_path) == [run_dir]


# Stale-claim recovery is NOT this module's job — see the module docstring. It used to be
# (find_stale_claims/release_stale_claim, backed by a claim.json contract nothing ever wrote),
# removed once launch.py (fps-3jj.5, merged) turned out to already own this correctly via
# bd `in_progress` + a required traceback tail — tested in tests/test_launch.py, not here.


# ── facts.json ────────────────────────────────────────────────────────────────

def test_build_facts_graded_run_has_all_blocks_and_suppresses_thin_cells(tmp_path):
    run_dir, batch_dir = _write_run(tmp_path, low_n_fold=1)

    facts = dt.build_facts(run_dir)

    assert facts["candidate"]["name"] == "cand"
    assert facts["provenance"]["batch"] == "batch1"
    assert facts["provenance"]["snapshot_date"] == "2026-08-15"
    assert facts["provenance"]["bead_id"] == "fps-test.1"
    assert facts["provenance"]["git_sha"] == "abc1234"
    assert facts["provenance"]["git_sha_is_run_time"] is True
    assert facts["provenance"]["n_baseline_columns"] == 54
    assert facts["provenance"]["baseline_fingerprint"] == "54:deadbeef1234"
    assert facts["provenance"]["tank_params"] == "50/3.571/7d/10%"
    assert facts["provenance"]["realised_seed"] == 42

    assert facts["headline"]["realised"]["delta_cpl_held"] == -0.05
    assert "NOT the arbiter" in facts["headline"]["wfcv_log_loss"]["label"]

    per_fold = {row["fold"]: row for row in facts["breakdowns"]["per_fold"]}
    assert per_fold[1]["suppressed"] is True  # low_n_fold
    assert per_fold[1]["delta_cpl_own"] is None
    assert per_fold[2]["suppressed"] is False
    assert per_fold[2]["delta_cpl_own"] is not None
    assert per_fold[1]["regime"] == "shock"
    assert per_fold[2]["regime"] == "normal"

    regimes = {row["regime"]: row for row in facts["breakdowns"]["per_regime"]}
    assert set(regimes) == {"shock", "normal"}

    assert facts["breakdowns"]["per_axis"] is None  # no add_axis declared
    # No row-level axis, so no unidentified cell to warn about. per_fold/per_regime
    # cut on whole folds (independent simulations) and must NOT carry the caveat.
    assert facts["breakdowns"]["per_axis_identification_note"] is None

    assert facts["seed_flags"] == [
        {"cohort": "all", "fold": 4, "run": "candidate", "seed_std": 0.09, "ratio_vs_cohort_median": 9.0}
    ]

    assert facts["validation"]["pit_test"].startswith("passed")
    assert facts["validation"]["candidate_column_nan_rate"]["cand_col"] > 0
    assert facts["validation"]["extra_feature_provider_hits"] == 100
    assert facts["validation"]["extra_feature_provider_misses"] == 0
    assert facts["validation"]["extra_feature_provider_miss_rate"] == 0.0

    assert facts["grading"]["pending"] is True
    assert facts["grading"]["predicted_signature"] == "helps normal folds, not shock"

    assert facts["noise_band"]["available"] is False
    assert "fps-3jj.9" in facts["noise_band"]["reason"]

    # No batch.json in this batch dir — degrades gracefully rather than raising.
    assert facts["redundancy"]["available"] is False


def test_build_facts_legacy_run_without_shock_folds_degrades_regime_facts(tmp_path):
    """Review finding on PR #350: the whole legacy path (a results.json predating
    fps-3tu's empirical shock-fold field) shipped uncovered — _make_results always
    stamped shock_folds and nothing popped it, so a later refactor reintroducing the old
    constant as a silent fallback would have passed every existing test green."""
    run_dir, _ = _write_run(tmp_path, low_n_fold=1, omit_shock_folds=True)

    facts = dt.build_facts(run_dir)

    per_fold = {row["fold"]: row for row in facts["breakdowns"]["per_fold"]}
    assert all(row["regime"] is None for row in per_fold.values())
    # Everything else in per_fold is still computed normally — only regime degrades.
    assert per_fold[2]["delta_cpl_own"] is not None

    assert facts["breakdowns"]["per_regime"] == {
        "computed": False,
        "reason": "this run's results.json predates fps-3tu's empirical shock-fold set "
        "(meta.shock_folds) — re-run the candidate to backfill it.",
    }
    # per_fold's bare None regime and per_regime's computed:false share ONE explanation,
    # surfaced at the breakdowns level so a reader hitting either symptom finds it.
    assert facts["breakdowns"]["per_fold_regime_unavailable_reason"] == facts["breakdowns"]["per_regime"]["reason"]


def test_build_facts_validation_provider_miss_rate_partial(tmp_path):
    """fps-qbv: batch1's live incident — 607/6300 (9.6%) misses passed
    `_check_provider_health` silently (it only errors at 100%), so the partial
    coverage never reached a dossier. Assert the rate is carried, not just the
    raw counts."""
    run_dir, _ = _write_run(tmp_path)
    results = json.loads((run_dir / dt.RESULTS_FILENAME).read_text())
    results["meta"]["extra_feature_provider_hits"] = 5693
    results["meta"]["extra_feature_provider_misses"] = 607
    (run_dir / dt.RESULTS_FILENAME).write_text(json.dumps(results))

    facts = dt.build_facts(run_dir)

    assert facts["validation"]["extra_feature_provider_hits"] == 5693
    assert facts["validation"]["extra_feature_provider_misses"] == 607
    assert facts["validation"]["extra_feature_provider_miss_rate"] == pytest.approx(607 / 6300)


def test_build_facts_provenance_and_validation_degrade_gracefully_on_legacy_meta(tmp_path):
    """Older results.json predate realised_seed and the provider stats — must read
    as None, not KeyError."""
    run_dir, _ = _write_run(tmp_path)
    results = json.loads((run_dir / dt.RESULTS_FILENAME).read_text())
    del results["meta"]["realised_seed"]
    del results["meta"]["extra_feature_provider_hits"]
    del results["meta"]["extra_feature_provider_misses"]
    (run_dir / dt.RESULTS_FILENAME).write_text(json.dumps(results))

    facts = dt.build_facts(run_dir)

    assert facts["provenance"]["realised_seed"] is None
    assert facts["validation"]["extra_feature_provider_hits"] is None
    assert facts["validation"]["extra_feature_provider_misses"] is None
    assert facts["validation"]["extra_feature_provider_miss_rate"] is None


def _batch_record_payload(rows: list[dict]) -> str:
    return json.dumps({"block_r2": rows})


def test_build_facts_redundancy_available_when_batch_json_has_matching_candidate(tmp_path):
    """batch.json lives in the candidate MODULES' parent dir — `run_dir.parent` — not in
    `meta["batch_dir"]` (the runner's frozen-artifact dir, written to a sibling `batch1/`
    by `_write_run` for freeze.json/baseline_columns.json/noise_floor.json). fps-1l1's
    review of this PR caught exactly this confusion live: passing `meta["batch_dir"]`
    always misses the real file. Written directly at `run_dir.parent`, deliberately not
    via the `batch_extras` hook (which targets `meta["batch_dir"]`), so this test cannot
    pass by accident the way it did before the fix.
    """
    run_dir, _ = _write_run(tmp_path)
    (run_dir.parent / "batch.json").write_text(_batch_record_payload([
        {
            "candidate": "cand", "mechanism_family": "test-family", "n_columns": 1,
            "n_predictors_used": 49, "predictors_dropped": ["bayside"],
            "block_r2": 0.42, "max_column_r2": 0.855,
            "per_column_r2": {"cand_col": 0.855},
        },
        {
            "candidate": "other_cand", "mechanism_family": "test-family", "n_columns": 1,
            "n_predictors_used": 49, "predictors_dropped": [],
            "block_r2": 0.10, "max_column_r2": 0.10,
            "per_column_r2": {"other_col": 0.10},
        },
    ]))

    facts = dt.build_facts(run_dir)

    assert facts["redundancy"]["available"] is True
    assert facts["redundancy"]["block_r2"] == 0.42
    assert facts["redundancy"]["max_column_r2"] == 0.855
    assert facts["redundancy"]["per_column_r2"] == {"cand_col": 0.855}
    assert facts["redundancy"]["n_predictors_used"] == 49
    assert facts["redundancy"]["predictors_dropped"] == ["bayside"]


def test_build_facts_redundancy_unavailable_when_batch_json_only_in_meta_batch_dir(tmp_path):
    """Regression for the fps-1l1 finding: a batch.json placed in `meta["batch_dir"]`
    (where the runner writes freeze.json etc.) must NOT be found — that is never where
    `redundancy.write_batch_record` puts it in production. Proves the fix reads from
    `run_dir.parent`, not `meta["batch_dir"]`."""
    def add_batch_record_in_wrong_place(batch_dir):
        (batch_dir / "batch.json").write_text(_batch_record_payload([
            {"candidate": "cand", "block_r2": 0.42, "max_column_r2": 0.855,
             "per_column_r2": {"cand_col": 0.855}, "n_predictors_used": 49,
             "predictors_dropped": []},
        ]))

    run_dir, _ = _write_run(tmp_path, batch_extras=add_batch_record_in_wrong_place)

    facts = dt.build_facts(run_dir)

    assert facts["redundancy"]["available"] is False
    assert str(run_dir.parent) in facts["redundancy"]["reason"]  # looked in the right dir...
    assert "batch1" not in facts["redundancy"]["reason"]  # ...not meta["batch_dir"]


def test_build_facts_redundancy_unavailable_when_candidate_missing_from_batch_record(tmp_path):
    """A candidate filed after the batch record was written (or renamed since) has no
    row to find — refuse rather than silently return another candidate's numbers."""
    run_dir, _ = _write_run(tmp_path)
    (run_dir.parent / "batch.json").write_text(_batch_record_payload([
        {"candidate": "some_other_cand", "block_r2": 0.1, "max_column_r2": 0.1,
         "per_column_r2": {}, "n_predictors_used": 49, "predictors_dropped": []},
    ]))

    facts = dt.build_facts(run_dir)

    assert facts["redundancy"]["available"] is False
    # repr(), not a bare substring check: "cand" is a substring of the reason template's
    # own word "candidates" and would pass unconditionally (fps-1l1 review).
    assert repr("cand") in facts["redundancy"]["reason"]
    assert "block_r2" not in facts["redundancy"]


def test_build_facts_raises_when_graded_run_has_no_tank_params(tmp_path):
    """fps-15c: status=='graded' means headline/breakdowns are about to carry a
    realised CPL — refuse rather than write a facts.json with no cadence stamp.
    Covers an old results.json that predates the field AND a hand-written
    mechanism that built its own results.json and skipped the shared path."""
    run_dir, _ = _write_run(tmp_path, tank_params=None)

    with pytest.raises(ValueError, match="no tank_params"):
        dt.build_facts(run_dir)


def test_build_facts_per_axis_present_when_add_axis_declared(tmp_path):
    run_dir, _ = _write_run(tmp_path, with_axis=True)

    facts = dt.build_facts(run_dir)

    per_axis = facts["breakdowns"]["per_axis"]
    assert per_axis is not None
    assert {row["axis"] for row in per_axis} <= {"A", "B"}
    assert all(row["n_fills"] >= 0 for row in per_axis)

    # fps-grp: a per-row axis cuts THROUGH a window, so its CPL cells allocate a
    # path-coupled cost to a sub-period and are not identified. The caveat must ship
    # in the same dict as the numbers — a dossier reader must not be able to meet the
    # delta without it — and must say so in terms the reader can act on.
    note = facts["breakdowns"]["per_axis_identification_note"]
    assert note == ROW_AXIS_ECONOMICS_CAVEAT
    assert "NOT IDENTIFIED" in note
    assert "never as a finding" in note


def test_build_facts_non_graded_status_sets_status_note_and_no_breakdowns(tmp_path):
    run_dir, _ = _write_run(tmp_path, status="disqualified")
    results = json.loads((run_dir / dt.RESULTS_FILENAME).read_text())
    results["error"] = "differential PIT test failed: mismatch at date 2026-01-05"
    (run_dir / dt.RESULTS_FILENAME).write_text(json.dumps(results))

    facts = dt.build_facts(run_dir)

    assert facts["headline"] is None
    assert facts["breakdowns"] is None
    assert "status='disqualified'" in facts["status_note"]
    assert facts["validation"]["pit_test"].startswith("FAILED")


def test_build_facts_noise_band_available_when_batch_has_calibration(tmp_path):
    noise_deltas = list(np.random.default_rng(2).normal(0, 0.02, size=20))

    def add_noise_floor(batch_dir):
        (batch_dir / dt.NOISE_FLOOR_FILENAME).write_text(json.dumps({
            "deltas_cpl_held": noise_deltas, "baseline_fingerprint": "54:deadbeef1234",
            "null_method": dt.NULL_METHOD_PLACEBO_COLUMN, "tank_params": "50/3.571/7d/10%",
        }))

    run_dir, _ = _write_run(tmp_path, batch_extras=add_noise_floor)

    facts = dt.build_facts(run_dir)

    band = facts["noise_band"]
    assert band["available"] is True
    assert band["n_draws"] == 20
    assert band["candidate_delta_cpl_held"] == -0.05
    # -0.05 is a strong (cheaper) result against a noise band centred on 0 with std 0.02 — nearly
    # every noise draw should be worse (higher/less-negative) than the candidate, so this must
    # read as a HIGH percentile, not near 0 (the pre-fix, inverted, sign gave ~0 for this case).
    expected_percentile = sum(d > -0.05 for d in noise_deltas) / len(noise_deltas) * 100
    assert band["candidate_percentile_better_than_noise"] == pytest.approx(expected_percentile)
    assert band["candidate_percentile_better_than_noise"] > 90
    # fps-3jj.17: the mechanical rejected/inconclusive split (docs/routines/dossier.md § Step 1.4)
    # reads this field directly rather than the dossier session computing a t-critical value
    # itself — single-sourced from family_wise_z_threshold(n_candidates=1, n_draws=20).
    assert band["single_candidate_z_threshold"] == pytest.approx(
        dt.family_wise_z_threshold(n_candidates=1, n_draws=20)
    )
    assert band["single_candidate_z_threshold"] > 0


@pytest.mark.parametrize("noise_deltas", [
    [0.01] * 20,
    [0.0] * 20,
])
def test_build_facts_noise_band_degenerate_canonical_floor_is_not_estimable(tmp_path, noise_deltas):
    """fps-tnz: `band_std > 0` alone does not catch "every draw landed on the same value" —
    np.std of twenty identical 0.01s is 1.78e-18, not 0.0, because 0.01 has no exact binary
    representation. Left unguarded, that sails past a `> 0` check and turns
    candidate_z_vs_band into a ~1e18 z, which docs/routines/dossier.md step 3's mechanical
    `-t < z < t` split reads as a confident REJECT on a bank that carries no information at
    all. The exact-0.0 case must stay green too — it was already caught before this fix."""
    def add_noise_floor(batch_dir):
        (batch_dir / dt.NOISE_FLOOR_FILENAME).write_text(json.dumps({
            "deltas_cpl_held": noise_deltas, "baseline_fingerprint": "54:deadbeef1234",
            "null_method": dt.NULL_METHOD_PLACEBO_COLUMN, "tank_params": "50/3.571/7d/10%",
        }))

    run_dir, _ = _write_run(tmp_path, batch_extras=add_noise_floor)

    facts = dt.build_facts(run_dir)

    band = facts["noise_band"]
    assert band["available"] is True
    assert band["candidate_z_vs_band"] is None
    assert band["single_candidate_z_threshold"] is None


def test_build_facts_noise_band_refuses_a_stale_pre_awz_seed_swap_floor(tmp_path):
    """fps-awz: this PR changed what noise_floor.json's deltas MEASURE (placebo-column null,
    not seed-swap) without changing baseline_fingerprint — batch0's already-committed
    noise_floor.json (fingerprint-valid, no null_method key at all, predates this field
    entirely) must be refused, not silently graded against the OLD, differently-shaped band
    this rework exists to replace."""
    noise_deltas = list(np.random.default_rng(2).normal(0, 0.02, size=5))

    def add_noise_floor(batch_dir):
        (batch_dir / dt.NOISE_FLOOR_FILENAME).write_text(json.dumps({
            "deltas_cpl_held": noise_deltas, "baseline_fingerprint": "54:deadbeef1234",
            # No "null_method" key — the exact shape of a pre-fps-awz noise_floor.json.
        }))

    run_dir, _ = _write_run(tmp_path, batch_extras=add_noise_floor)

    facts = dt.build_facts(run_dir)

    band = facts["noise_band"]
    assert band["available"] is False
    assert "null_method" in band["reason"]


def test_build_facts_noise_band_refuses_a_mismatched_baseline_fingerprint(tmp_path):
    """fps-cf8: a floor computed against a different baseline must not silently grade
    this run — the exact failure class (fps-sa1, fps-zci) the noise floor exists to
    guard other candidates against."""
    noise_deltas = list(np.random.default_rng(2).normal(0, 0.02, size=20))

    def add_noise_floor(batch_dir):
        (batch_dir / dt.NOISE_FLOOR_FILENAME).write_text(json.dumps({
            "deltas_cpl_held": noise_deltas, "baseline_fingerprint": "54:wrongwrongwr",
        }))

    run_dir, _ = _write_run(tmp_path, batch_extras=add_noise_floor)

    facts = dt.build_facts(run_dir)

    band = facts["noise_band"]
    assert band["available"] is False
    assert "fingerprint" in band["reason"]


def test_build_facts_noise_band_refuses_a_mismatched_tank_params(tmp_path):
    """fps-v8o: the consuming-side half of fps-15c's stamping work — a floor computed at
    one cadence must not silently grade a run at a different cadence, the same failure
    class the baseline_fingerprint check guards along the feature-set axis."""
    noise_deltas = list(np.random.default_rng(2).normal(0, 0.02, size=20))

    def add_noise_floor(batch_dir):
        (batch_dir / dt.NOISE_FLOOR_FILENAME).write_text(json.dumps({
            "deltas_cpl_held": noise_deltas, "baseline_fingerprint": "54:deadbeef1234",
            "null_method": dt.NULL_METHOD_PLACEBO_COLUMN, "tank_params": "50/3.571/1d/10%",
        }))

    run_dir, _ = _write_run(tmp_path, batch_extras=add_noise_floor, tank_params="50/3.571/7d/10%")

    facts = dt.build_facts(run_dir)

    band = facts["noise_band"]
    assert band["available"] is False
    assert "tank_params" in band["reason"]


def test_build_facts_noise_band_refuses_a_floor_with_no_tank_params(tmp_path):
    """fps-v8o: a pre-fps-v8o floor (no tank_params at all) cannot be shown to match
    the run's cadence, so it must not be treated as a pass by default — same rule
    the fingerprint check already applies to a pre-fps-cf8 floor."""
    noise_deltas = list(np.random.default_rng(2).normal(0, 0.02, size=20))

    def add_noise_floor(batch_dir):
        (batch_dir / dt.NOISE_FLOOR_FILENAME).write_text(json.dumps({
            "deltas_cpl_held": noise_deltas, "baseline_fingerprint": "54:deadbeef1234",
            "null_method": dt.NULL_METHOD_PLACEBO_COLUMN,
            # No "tank_params" key — the exact shape of a pre-fps-v8o noise_floor.json.
        }))

    run_dir, _ = _write_run(tmp_path, batch_extras=add_noise_floor, tank_params="50/3.571/7d/10%")

    facts = dt.build_facts(run_dir)

    band = facts["noise_band"]
    assert band["available"] is False
    assert "tank_params" in band["reason"]


def test_noise_band_refuses_a_floor_and_run_that_both_lack_tank_params(tmp_path):
    """review of PR #349: `_bank_admissibility` applies fps-v8o's "cannot be shown to match,
    so cannot be trusted" rule symmetrically — a run with no tank_params of its own can't show
    a match against a floor that also lacks the field, even though the two look equal
    (`None == None`). Unreachable via `build_facts` itself (fps-15c already refuses a graded
    run missing meta.tank_params before `_noise_band` is ever called — see
    test_build_facts_raises_when_graded_run_has_no_tank_params), so this calls `_noise_band`
    directly, the same way retrospective.py's `_batch_noise_summary` does — except with
    check_fingerprint=True, since a hypothetical caller other than build_facts/retrospective.py
    could reach this state."""
    batch_dir = tmp_path / "batch1"
    batch_dir.mkdir()
    (batch_dir / dt.NOISE_FLOOR_FILENAME).write_text(json.dumps({
        "deltas_cpl_held": [0.01, 0.02, -0.01], "baseline_fingerprint": "54:deadbeef1234",
        "null_method": dt.NULL_METHOD_PLACEBO_COLUMN,
        # No "tank_params" key.
    }))
    results = {
        "candidate": {"columns": ["a"]},
        "effect_delta_cpl_held": -0.05,
        "meta": {"baseline_fingerprint": "54:deadbeef1234"},  # no tank_params either
    }

    band = dt._noise_band(results, batch_dir)

    assert band["available"] is False
    assert "neither side" in band["reason"]
    # Must not be self-contradictory: the old wording, applied to this case, would read
    # "(None) does not match this run's (None)" — a claimed mismatch between two equal values.
    assert "(None) does not match this run's (None)" not in band["reason"]


# ── decision flips (fps-gez) ───────────────────────────────────────────────────

def _diverging_fills(fold=1) -> pd.DataFrame:
    """One clean flip: baseline buys expensive, candidate buys the same litres
    cheaper, on different dates — exactly the case `summarise_flips` should
    turn into a non-null `flip_cpl_delta`."""
    return pd.DataFrame([
        {"fold": fold, "arm": BASELINE_ARM, "own_tau": 0.25, "date": "2026-01-01",
         "station_code": 100, "price": 200.0, "litres": 10.0, "spend_cents": 2000.0,
         "emergency": False},
        {"fold": fold, "arm": CANDIDATE_ARM, "own_tau": 0.25, "date": "2026-01-02",
         "station_code": 100, "price": 150.0, "litres": 10.0, "spend_cents": 1500.0,
         "emergency": False},
    ])


def test_decision_flips_not_computed_below_arity_2(tmp_path):
    run_dir, _ = _write_run(tmp_path, columns=["one_col"], fills_df=_diverging_fills())
    facts = dt.build_facts(run_dir)
    flips = facts["decision_flips"]
    assert flips["computed"] is False
    assert "1 column" in flips["reason"]


def test_decision_flips_not_computed_when_noise_band_unavailable(tmp_path):
    # No batch_extras -> no noise_floor.json -> noise_band["available"] is False.
    run_dir, _ = _write_run(tmp_path, columns=["col_a", "col_b"], fills_df=_diverging_fills())
    facts = dt.build_facts(run_dir)
    flips = facts["decision_flips"]
    assert flips["computed"] is False
    assert "noise band unavailable" in flips["reason"]


def test_decision_flips_not_computed_when_band_std_not_estimable(tmp_path):
    def add_noise_floor(batch_dir):
        (batch_dir / dt.NOISE_FLOOR_FILENAME).write_text(json.dumps({
            "deltas_cpl_held": [-0.05], "baseline_fingerprint": "54:deadbeef1234",
            "null_method": dt.NULL_METHOD_PLACEBO_COLUMN, "tank_params": "50/3.571/7d/10%",
            "n_placebo_columns": 2,
        }))

    run_dir, _ = _write_run(
        tmp_path, columns=["col_a", "col_b"], batch_extras=add_noise_floor,
        fills_df=_diverging_fills(),
    )
    facts = dt.build_facts(run_dir)
    flips = facts["decision_flips"]
    assert flips["computed"] is False
    assert "null" in flips["reason"]


def test_decision_flips_not_computed_on_a_clean_reject(tmp_path):
    # Noise band centred far better than the candidate's own -0.05 delta, with a
    # tiny std -> candidate reads as clearly the WRONG sign vs. this batch's noise.
    def add_noise_floor(batch_dir):
        (batch_dir / dt.NOISE_FLOOR_FILENAME).write_text(json.dumps({
            "deltas_cpl_held": [-1.01, -1.0, -0.99, -1.0, -1.0],
            "baseline_fingerprint": "54:deadbeef1234",
            "null_method": dt.NULL_METHOD_PLACEBO_COLUMN, "tank_params": "50/3.571/7d/10%",
            "n_placebo_columns": 2,
        }))

    run_dir, _ = _write_run(
        tmp_path, columns=["col_a", "col_b"], batch_extras=add_noise_floor,
        fills_df=_diverging_fills(),
    )
    facts = dt.build_facts(run_dir)
    band = facts["noise_band"]
    assert band["candidate_z_vs_band"] >= band["single_candidate_z_threshold"]  # sanity: is a clean reject
    flips = facts["decision_flips"]
    assert flips["computed"] is False
    assert "clean reject" in flips["reason"]


def test_decision_flips_computed_when_clearing_noise_on_the_good_side(tmp_path):
    # Mirror of the clean-reject test above, flipped: a tight noise band centred
    # near 0 with the candidate's -0.05 delta far BELOW it (more negative = better)
    # -> z <= -t, clearing noise on the good side. dossier.md is explicit this
    # case still gets the flip breakdown computed (a human may want to look
    # closer at exactly the candidates that cleared) — it must not be silently
    # folded into the "clean reject" (z >= t) skip branch by a future sign bug.
    def add_noise_floor(batch_dir):
        (batch_dir / dt.NOISE_FLOOR_FILENAME).write_text(json.dumps({
            "deltas_cpl_held": [0.01, 0.0, -0.01, 0.0, 0.0],
            "baseline_fingerprint": "54:deadbeef1234",
            "null_method": dt.NULL_METHOD_PLACEBO_COLUMN, "tank_params": "50/3.571/7d/10%",
            "n_placebo_columns": 2,
        }))

    run_dir, _ = _write_run(
        tmp_path, columns=["col_a", "col_b"], batch_extras=add_noise_floor,
        fills_df=_diverging_fills(),
    )
    facts = dt.build_facts(run_dir)
    band = facts["noise_band"]
    assert band["candidate_z_vs_band"] <= -band["single_candidate_z_threshold"]  # sanity: clears on the good side

    flips = facts["decision_flips"]
    assert flips["computed"] is True
    assert flips["n_flips"] == 2


def test_decision_flips_computed_when_inside_the_noise_band(tmp_path):
    # Wide noise band (std ~0.79) around the candidate's -0.05 delta -> z is tiny,
    # well inside (-t, t) regardless of n_draws -> the trigger fires.
    def add_noise_floor(batch_dir):
        (batch_dir / dt.NOISE_FLOOR_FILENAME).write_text(json.dumps({
            "deltas_cpl_held": [-1.0, -0.5, 0.0, 0.5, 1.0],
            "baseline_fingerprint": "54:deadbeef1234",
            "null_method": dt.NULL_METHOD_PLACEBO_COLUMN, "tank_params": "50/3.571/7d/10%",
            "n_placebo_columns": 2,
        }))

    run_dir, _ = _write_run(
        tmp_path, columns=["col_a", "col_b"], batch_extras=add_noise_floor,
        fills_df=_diverging_fills(fold=1),
    )
    facts = dt.build_facts(run_dir)
    band = facts["noise_band"]
    assert -band["single_candidate_z_threshold"] < band["candidate_z_vs_band"] < band["single_candidate_z_threshold"]

    flips = facts["decision_flips"]
    assert flips["computed"] is True
    assert flips["n_flips"] == 2
    # 50/3.571/7d/10% (this fixture's default tank_params) derives a half-tank-life of 7 days
    # (50/3.571/2) — which, for THIS tank_params, exactly EQUALS its own 7-day evaluation
    # cadence. cascade_window_days floors the window strictly above the cadence (PR #347
    # review finding #2: an unguarded window at or below the cadence is a structural
    # degenerate case, not derived-vs-hardcoded colour), so the actual window here is 8, not
    # the coincidental 7 the half-life formula alone would give.
    assert flips["cascade_window_days"] == 8
    row = next(r for r in flips["per_fold"] if r["fold"] == 1)
    assert row["regime"] == "shock"
    assert row["n_baseline_only"] == 1
    assert row["n_candidate_only"] == 1
    assert row["n_flips"] == 2
    # 2026-01-01 and 2026-01-02, one day apart, same station -> well within the 8-day window
    # -> one cascade, one decision, even though n_flips is 2.
    assert row["n_decisions"] == 1
    assert row["litres_baseline"] == pytest.approx(10.0)
    assert row["litres_candidate"] == pytest.approx(10.0)
    assert row["flip_cpl_baseline"] == pytest.approx(200.0)
    assert row["flip_cpl_candidate"] == pytest.approx(150.0)
    assert row["flip_cpl_delta"] == pytest.approx(-50.0)
    assert row["flip_cpl_delta_reason"] is None
    assert len(flips["rows"]) == 2
    # This fixture's only fold has 2 flip fills total, below min_row_cell_n (30) — the whole
    # fold's delta_cpl_own is suppressed, so nothing is attributable and the entire headline
    # delta lands in the composition residual rather than being silently dropped.
    assert flips["run_contribution_total_cpl"] == pytest.approx(0.0)
    assert flips["composition_residual_cpl"] == pytest.approx(facts["headline"]["realised"]["delta_cpl_held"])
    assert row["run_contribution_cpl"] is None


def test_decision_flips_refuses_for_a_legacy_run_without_shock_folds(tmp_path):
    """Review finding on PR #350: this early refusal (facts["provenance"]["shock_folds"]
    is None) was the other half of the legacy path that shipped uncovered — everything
    upstream of it (arity 2+, noise band not a clean reject) is identical to the
    computed-True fixture above, isolating this branch specifically."""
    def add_noise_floor(batch_dir):
        (batch_dir / dt.NOISE_FLOOR_FILENAME).write_text(json.dumps({
            "deltas_cpl_held": [-1.0, -0.5, 0.0, 0.5, 1.0],
            "baseline_fingerprint": "54:deadbeef1234",
            "null_method": dt.NULL_METHOD_PLACEBO_COLUMN, "tank_params": "50/3.571/7d/10%",
            "n_placebo_columns": 2,
        }))

    run_dir, _ = _write_run(
        tmp_path, columns=["col_a", "col_b"], batch_extras=add_noise_floor,
        fills_df=_diverging_fills(fold=1), omit_shock_folds=True,
    )
    facts = dt.build_facts(run_dir)
    band = facts["noise_band"]
    assert -band["single_candidate_z_threshold"] < band["candidate_z_vs_band"] < band["single_candidate_z_threshold"]

    flips = facts["decision_flips"]
    assert flips == {
        "computed": False,
        "reason": "this run's results.json predates fps-3tu's empirical shock-fold set "
        "(meta.shock_folds) — flip/regret detail is graded per (fold, regime) and there is "
        "no trustworthy regime tag to use; re-run the candidate to backfill it.",
    }


# ── decision flips: fps-6yi regression (fps-e1w) ────────────────────────────────

# fps-6yi (stickiness_phase_saddle, batch1) real committed data, reproduced verbatim from
# experiments/candidates/batch1/stickiness_phase_saddle/facts.json's decision_flips.rows
# (the 303 real flip fills, committed) plus this run's fills.parquet (gitignored, read once
# locally to derive the padding below — not re-read at test time). Padding rows share a
# dedicated per-fold/per-arm station+date key so diff_fills treats them as MATCHING (not
# flips), giving each fold's TOTAL litres/spend the exact values the real run had, so
# delta_cpl_own and the litres-weighted shift-share reconcile to the real headline exactly.
_FPS_6YI_FLIP_ROWS = [
    # (fold, station_code, date, price, litres, bought_by)
    (2, 414, '2022-03-07', 184.9, 21.428571, 'baseline'),
    (2, 18517, '2022-03-07', 184.9, 21.428571, 'baseline'),
    (2, 18517, '2022-04-03', 189.9, 21.428571, 'baseline'),
    (2, 18517, '2022-04-07', 169.9, 3.571429, 'baseline'),
    (2, 18517, '2022-04-28', 187.7, 3.571429, 'baseline'),
    (2, 429, '2022-02-05', 167.7, 32.142857, 'baseline'),
    (2, 585, '2022-03-06', 174.9, 28.571429, 'baseline'),
    (2, 261, '2022-03-05', 174.7, 17.857143, 'baseline'),
    (2, 261, '2022-03-06', 174.7, 3.571429, 'baseline'),
    (2, 261, '2022-03-07', 174.7, 3.571429, 'baseline'),
    (2, 261, '2022-03-20', 191.9, 21.428571, 'baseline'),
    (2, 261, '2022-03-26', 195.4, 21.428571, 'baseline'),
    (3, 414, '2022-05-06', 177.9, 32.142857, 'baseline'),
    (3, 414, '2022-05-19', 209.9, 21.428571, 'baseline'),
    (3, 414, '2022-05-25', 206.9, 21.428571, 'baseline'),
    (3, 414, '2022-05-31', 202.9, 21.428571, 'baseline'),
    (3, 414, '2022-06-06', 199.9, 46.428571, 'baseline'),
    (3, 414, '2022-06-07', 199.9, 3.571429, 'baseline'),
    (3, 18517, '2022-05-16', 217.9, 21.428571, 'baseline'),
    (3, 18517, '2022-05-22', 209.9, 21.428571, 'baseline'),
    (3, 18517, '2022-05-28', 206.9, 21.428571, 'baseline'),
    (3, 18517, '2022-06-03', 199.9, 21.428571, 'baseline'),
    (3, 18517, '2022-06-06', 199.9, 35.714286, 'baseline'),
    (3, 18517, '2022-06-07', 199.9, 3.571429, 'baseline'),
    (3, 429, '2022-05-14', 172.9, 3.571429, 'baseline'),
    (3, 429, '2022-05-17', 172.9, 3.571429, 'baseline'),
    (3, 429, '2022-05-30', 197.9, 21.428571, 'baseline'),
    (3, 429, '2022-07-07', 199.7, 42.857143, 'baseline'),
    (3, 429, '2022-07-08', 199.7, 3.571429, 'baseline'),
    (3, 429, '2022-07-09', 199.7, 3.571429, 'baseline'),
    (3, 429, '2022-07-14', 199.7, 3.571429, 'baseline'),
    (3, 585, '2022-07-12', 199.9, 39.285714, 'baseline'),
    (3, 261, '2022-07-11', 187.9, 3.571429, 'baseline'),
    (4, 414, '2022-08-17', 193.9, 10.714286, 'baseline'),
    (4, 414, '2022-08-30', 193.9, 21.428571, 'baseline'),
    (4, 414, '2022-10-01', 185.9, 21.428571, 'baseline'),
    (4, 414, '2022-10-25', 185.9, 21.428571, 'baseline'),
    (4, 18517, '2022-08-17', 193.9, 14.285714, 'baseline'),
    (4, 18517, '2022-08-30', 193.9, 21.428571, 'baseline'),
    (4, 18517, '2022-09-02', 179.9, 35.714286, 'baseline'),
    (4, 18517, '2022-09-11', 159.9, 3.571429, 'baseline'),
    (4, 429, '2022-09-02', 177.7, 21.428571, 'baseline'),
    (4, 429, '2022-09-08', 163.9, 21.428571, 'baseline'),
    (4, 429, '2022-09-11', 159.9, 3.571429, 'baseline'),
    (4, 585, '2022-08-14', 193.9, 3.571429, 'baseline'),
    (4, 585, '2022-08-18', 193.9, 3.571429, 'baseline'),
    (4, 585, '2022-08-28', 189.9, 35.714286, 'baseline'),
    (4, 585, '2022-08-31', 177.9, 10.714286, 'baseline'),
    (5, 18517, '2022-11-09', 205.9, 35.714286, 'baseline'),
    (5, 18517, '2022-11-10', 205.9, 3.571429, 'baseline'),
    (5, 18517, '2022-11-11', 205.9, 3.571429, 'baseline'),
    (5, 18517, '2022-11-30', 189.9, 21.428571, 'baseline'),
    (5, 429, '2023-01-01', 202.9, 28.571429, 'baseline'),
    (5, 429, '2023-01-14', 184.5, 21.428571, 'baseline'),
    (6, 414, '2023-03-14', 185.9, 3.571429, 'baseline'),
    (6, 414, '2023-03-21', 185.9, 10.714286, 'baseline'),
    (6, 414, '2023-03-22', 185.9, 3.571429, 'baseline'),
    (6, 414, '2023-04-04', 185.9, 21.428571, 'baseline'),
    (6, 414, '2023-04-10', 185.9, 21.428571, 'baseline'),
    (6, 414, '2023-04-16', 185.9, 21.428571, 'baseline'),
    (6, 414, '2023-04-22', 185.9, 21.428571, 'baseline'),
    (6, 414, '2023-04-28', 185.9, 21.428571, 'baseline'),
    (6, 18517, '2023-04-25', 193.9, 35.714286, 'baseline'),
    (6, 429, '2023-03-03', 177.9, 3.571429, 'baseline'),
    (6, 429, '2023-03-04', 177.9, 3.571429, 'baseline'),
    (6, 429, '2023-03-05', 177.9, 3.571429, 'baseline'),
    (6, 429, '2023-03-06', 177.9, 3.571429, 'baseline'),
    (6, 261, '2023-04-20', 179.9, 28.571429, 'baseline'),
    (6, 261, '2023-04-21', 179.9, 3.571429, 'baseline'),
    (6, 261, '2023-04-22', 179.9, 3.571429, 'baseline'),
    (6, 261, '2023-04-23', 179.7, 3.571429, 'baseline'),
    (6, 261, '2023-04-24', 177.9, 3.571429, 'baseline'),
    (7, 414, '2023-05-06', 185.9, 28.571429, 'baseline'),
    (7, 414, '2023-05-07', 185.9, 3.571429, 'baseline'),
    (7, 414, '2023-05-08', 185.9, 3.571429, 'baseline'),
    (7, 414, '2023-05-09', 185.9, 3.571429, 'baseline'),
    (7, 18517, '2023-04-29', 186.9, 25.0, 'baseline'),
    (7, 18517, '2023-05-02', 184.9, 3.571429, 'baseline'),
    (8, 414, '2023-08-19', 215.9, 21.428571, 'baseline'),
    (8, 414, '2023-08-25', 199.9, 21.428571, 'baseline'),
    (8, 414, '2023-08-31', 199.9, 21.428571, 'baseline'),
    (8, 414, '2023-09-06', 227.9, 21.428571, 'baseline'),
    (8, 414, '2023-09-12', 227.9, 21.428571, 'baseline'),
    (8, 414, '2023-09-18', 224.9, 21.428571, 'baseline'),
    (8, 414, '2023-09-24', 219.9, 21.428571, 'baseline'),
    (8, 414, '2023-09-30', 213.9, 21.428571, 'baseline'),
    (8, 414, '2023-10-06', 207.9, 21.428571, 'baseline'),
    (8, 414, '2023-10-12', 195.9, 21.428571, 'baseline'),
    (8, 18517, '2023-09-28', 223.9, 21.428571, 'baseline'),
    (8, 18517, '2023-10-04', 209.9, 21.428571, 'baseline'),
    (8, 18517, '2023-10-10', 205.9, 21.428571, 'baseline'),
    (8, 18517, '2023-10-16', 205.9, 21.428571, 'baseline'),
    (8, 585, '2023-07-28', 177.9, 25.0, 'baseline'),
    (8, 585, '2023-07-29', 177.9, 3.571429, 'baseline'),
    (8, 585, '2023-07-30', 177.9, 3.571429, 'baseline'),
    (8, 585, '2023-07-31', 176.9, 3.571429, 'baseline'),
    (8, 585, '2023-08-01', 174.9, 3.571429, 'baseline'),
    (8, 261, '2023-10-23', 178.7, 42.857143, 'baseline'),
    (10, 429, '2024-03-03', 227.9, 21.428571, 'baseline'),
    (10, 429, '2024-03-09', 223.9, 21.428571, 'baseline'),
    (10, 429, '2024-03-15', 219.9, 21.428571, 'baseline'),
    (10, 429, '2024-03-21', 207.9, 21.428571, 'baseline'),
    (10, 429, '2024-03-27', 197.9, 21.428571, 'baseline'),
    (10, 261, '2024-02-21', 189.9, 21.428571, 'baseline'),
    (10, 261, '2024-03-09', 217.9, 21.428571, 'baseline'),
    (10, 261, '2024-03-15', 202.9, 21.428571, 'baseline'),
    (10, 261, '2024-03-21', 200.5, 21.428571, 'baseline'),
    (10, 261, '2024-03-27', 197.9, 21.428571, 'baseline'),
    (11, 18517, '2024-05-07', 223.9, 32.142857, 'baseline'),
    (11, 18517, '2024-05-08', 223.9, 3.571429, 'baseline'),
    (12, 414, '2024-08-06', 195.9, 35.714286, 'baseline'),
    (12, 429, '2024-08-06', 195.9, 35.714286, 'baseline'),
    (12, 429, '2024-09-11', 179.9, 28.571429, 'baseline'),
    (13, 414, '2025-01-11', 187.9, 3.571429, 'baseline'),
    (13, 414, '2025-01-12', 187.9, 3.571429, 'baseline'),
    (13, 18517, '2025-01-15', 213.9, 3.571429, 'baseline'),
    (13, 585, '2024-11-21', 207.9, 21.428571, 'baseline'),
    (13, 585, '2024-11-27', 207.9, 21.428571, 'baseline'),
    (14, 585, '2025-02-18', 179.9, 28.571429, 'baseline'),
    (14, 585, '2025-02-19', 179.9, 3.571429, 'baseline'),
    (2, 414, '2022-03-04', 184.9, 35.714286, 'candidate'),
    (2, 414, '2022-03-10', 184.9, 21.428571, 'candidate'),
    (2, 414, '2022-03-22', 184.9, 3.571429, 'candidate'),
    (2, 414, '2022-03-23', 184.9, 3.571429, 'candidate'),
    (2, 414, '2022-03-24', 184.9, 3.571429, 'candidate'),
    (2, 414, '2022-03-25', 184.9, 3.571429, 'candidate'),
    (2, 414, '2022-03-26', 184.9, 3.571429, 'candidate'),
    (2, 414, '2022-03-27', 184.9, 3.571429, 'candidate'),
    (2, 414, '2022-03-28', 184.9, 3.571429, 'candidate'),
    (2, 414, '2022-03-29', 184.9, 3.571429, 'candidate'),
    (2, 414, '2022-03-31', 189.9, 7.142857, 'candidate'),
    (2, 414, '2022-04-03', 182.9, 3.571429, 'candidate'),
    (2, 414, '2022-04-23', 187.9, 3.571429, 'candidate'),
    (2, 414, '2022-04-25', 187.9, 3.571429, 'candidate'),
    (2, 18517, '2022-03-03', 184.9, 32.142857, 'candidate'),
    (2, 18517, '2022-03-04', 184.9, 3.571429, 'candidate'),
    (2, 18517, '2022-03-10', 184.9, 21.428571, 'candidate'),
    (2, 18517, '2022-03-22', 184.9, 3.571429, 'candidate'),
    (2, 18517, '2022-03-23', 184.9, 3.571429, 'candidate'),
    (2, 18517, '2022-03-24', 184.9, 3.571429, 'candidate'),
    (2, 18517, '2022-03-25', 184.9, 3.571429, 'candidate'),
    (2, 18517, '2022-03-26', 184.9, 3.571429, 'candidate'),
    (2, 18517, '2022-03-27', 184.9, 3.571429, 'candidate'),
    (2, 18517, '2022-03-28', 184.9, 3.571429, 'candidate'),
    (2, 18517, '2022-03-29', 184.9, 3.571429, 'candidate'),
    (2, 18517, '2022-03-31', 189.9, 7.142857, 'candidate'),
    (2, 18517, '2022-04-04', 177.9, 14.285714, 'candidate'),
    (2, 429, '2022-02-03', 167.7, 25.0, 'candidate'),
    (2, 585, '2022-04-07', 163.4, 10.714286, 'candidate'),
    (2, 585, '2022-05-02', 187.9, 17.857143, 'candidate'),
    (2, 261, '2022-02-24', 180.7, 32.142857, 'candidate'),
    (2, 261, '2022-02-25', 178.9, 3.571429, 'candidate'),
    (2, 261, '2022-03-10', 184.9, 35.714286, 'candidate'),
    (2, 261, '2022-03-23', 197.9, 21.428571, 'candidate'),
    (2, 261, '2022-03-29', 191.7, 21.428571, 'candidate'),
    (2, 261, '2022-03-31', 183.9, 32.142857, 'candidate'),
    (2, 261, '2022-04-02', 177.9, 3.571429, 'candidate'),
    (2, 261, '2022-04-07', 162.4, 10.714286, 'candidate'),
    (3, 414, '2022-05-10', 179.9, 21.428571, 'candidate'),
    (3, 414, '2022-05-16', 213.9, 21.428571, 'candidate'),
    (3, 414, '2022-05-22', 206.9, 21.428571, 'candidate'),
    (3, 414, '2022-05-28', 206.9, 21.428571, 'candidate'),
    (3, 414, '2022-06-03', 199.9, 21.428571, 'candidate'),
    (3, 18517, '2022-05-11', 174.9, 28.571429, 'candidate'),
    (3, 18517, '2022-05-24', 206.9, 21.428571, 'candidate'),
    (3, 18517, '2022-05-30', 202.9, 21.428571, 'candidate'),
    (3, 18517, '2022-06-05', 199.9, 21.428571, 'candidate'),
    (3, 429, '2022-05-29', 199.9, 21.428571, 'candidate'),
    (3, 429, '2022-06-04', 195.9, 21.428571, 'candidate'),
    (3, 429, '2022-07-06', 199.7, 39.285714, 'candidate'),
    (3, 585, '2022-05-11', 173.9, 3.571429, 'candidate'),
    (3, 261, '2022-05-12', 169.9, 32.142857, 'candidate'),
    (3, 261, '2022-06-08', 184.7, 39.285714, 'candidate'),
    (3, 261, '2022-06-09', 183.5, 3.571429, 'candidate'),
    (4, 414, '2022-08-27', 193.9, 21.428571, 'candidate'),
    (4, 414, '2022-09-08', 185.9, 21.428571, 'candidate'),
    (4, 414, '2022-09-19', 185.9, 3.571429, 'candidate'),
    (4, 414, '2022-09-20', 185.9, 3.571429, 'candidate'),
    (4, 414, '2022-10-03', 185.9, 21.428571, 'candidate'),
    (4, 414, '2022-10-04', 185.9, 28.571429, 'candidate'),
    (4, 414, '2022-10-05', 185.9, 3.571429, 'candidate'),
    (4, 414, '2022-10-06', 185.9, 3.571429, 'candidate'),
    (4, 414, '2022-10-08', 185.9, 3.571429, 'candidate'),
    (4, 414, '2022-10-09', 185.9, 3.571429, 'candidate'),
    (4, 414, '2022-10-10', 185.9, 3.571429, 'candidate'),
    (4, 414, '2022-10-11', 185.9, 3.571429, 'candidate'),
    (4, 414, '2022-10-12', 185.9, 3.571429, 'candidate'),
    (4, 414, '2022-10-14', 185.9, 3.571429, 'candidate'),
    (4, 414, '2022-10-15', 185.9, 3.571429, 'candidate'),
    (4, 414, '2022-10-16', 185.9, 3.571429, 'candidate'),
    (4, 414, '2022-10-17', 185.9, 3.571429, 'candidate'),
    (4, 414, '2022-10-18', 185.9, 3.571429, 'candidate'),
    (4, 414, '2022-10-20', 185.9, 3.571429, 'candidate'),
    (4, 18517, '2022-08-26', 193.9, 21.428571, 'candidate'),
    (4, 18517, '2022-09-01', 183.9, 21.428571, 'candidate'),
    (4, 18517, '2022-10-04', 185.9, 32.142857, 'candidate'),
    (4, 18517, '2022-10-05', 185.9, 3.571429, 'candidate'),
    (4, 18517, '2022-10-06', 185.9, 3.571429, 'candidate'),
    (4, 18517, '2022-10-07', 185.9, 3.571429, 'candidate'),
    (4, 18517, '2022-10-09', 185.9, 3.571429, 'candidate'),
    (4, 429, '2022-08-31', 179.9, 39.285714, 'candidate'),
    (4, 429, '2022-09-03', 177.7, 10.714286, 'candidate'),
    (4, 429, '2022-10-04', 185.9, 42.857143, 'candidate'),
    (4, 429, '2022-10-06', 185.9, 3.571429, 'candidate'),
    (4, 585, '2022-08-30', 185.9, 21.428571, 'candidate'),
    (4, 585, '2022-09-01', 177.9, 32.142857, 'candidate'),
    (4, 585, '2022-09-10', 165.9, 3.571429, 'candidate'),
    (4, 585, '2022-09-11', 165.9, 3.571429, 'candidate'),
    (4, 585, '2022-10-04', 187.9, 39.285714, 'candidate'),
    (4, 585, '2022-10-05', 187.9, 3.571429, 'candidate'),
    (4, 261, '2022-10-04', 187.9, 32.142857, 'candidate'),
    (4, 261, '2022-10-05', 187.9, 3.571429, 'candidate'),
    (4, 261, '2022-10-07', 177.9, 7.142857, 'candidate'),
    (5, 414, '2022-11-23', 185.9, 28.571429, 'candidate'),
    (5, 414, '2022-11-24', 185.9, 3.571429, 'candidate'),
    (5, 414, '2022-11-25', 185.9, 3.571429, 'candidate'),
    (5, 414, '2022-11-26', 185.9, 3.571429, 'candidate'),
    (5, 414, '2022-11-27', 185.9, 3.571429, 'candidate'),
    (5, 414, '2022-12-13', 185.9, 3.571429, 'candidate'),
    (5, 414, '2022-12-28', 185.9, 3.571429, 'candidate'),
    (5, 414, '2022-12-29', 185.9, 3.571429, 'candidate'),
    (5, 414, '2022-12-31', 185.9, 7.142857, 'candidate'),
    (5, 414, '2023-01-01', 185.9, 3.571429, 'candidate'),
    (5, 414, '2023-01-05', 185.9, 3.571429, 'candidate'),
    (5, 414, '2023-01-08', 185.9, 10.714286, 'candidate'),
    (5, 18517, '2022-11-12', 205.9, 21.428571, 'candidate'),
    (5, 18517, '2022-11-18', 199.9, 21.428571, 'candidate'),
    (5, 18517, '2022-11-28', 189.9, 39.285714, 'candidate'),
    (5, 18517, '2023-01-13', 191.9, 42.857143, 'candidate'),
    (5, 18517, '2023-01-15', 191.9, 3.571429, 'candidate'),
    (5, 18517, '2023-01-16', 191.9, 3.571429, 'candidate'),
    (5, 429, '2023-01-06', 194.9, 21.428571, 'candidate'),
    (5, 429, '2023-01-12', 186.5, 21.428571, 'candidate'),
    (5, 585, '2022-12-01', 186.9, 28.571429, 'candidate'),
    (5, 585, '2023-01-19', 177.9, 3.571429, 'candidate'),
    (6, 414, '2023-03-20', 185.9, 7.142857, 'candidate'),
    (6, 414, '2023-04-02', 185.9, 21.428571, 'candidate'),
    (6, 414, '2023-04-08', 185.9, 21.428571, 'candidate'),
    (6, 414, '2023-04-14', 185.9, 21.428571, 'candidate'),
    (6, 414, '2023-04-20', 185.9, 21.428571, 'candidate'),
    (6, 414, '2023-04-26', 185.9, 21.428571, 'candidate'),
    (6, 18517, '2023-04-28', 186.9, 21.428571, 'candidate'),
    (6, 429, '2023-04-28', 182.5, 35.714286, 'candidate'),
    (7, 414, '2023-06-06', 165.9, 3.571429, 'candidate'),
    (7, 414, '2023-06-07', 165.9, 3.571429, 'candidate'),
    (7, 414, '2023-06-08', 165.9, 3.571429, 'candidate'),
    (7, 414, '2023-06-09', 165.9, 3.571429, 'candidate'),
    (7, 414, '2023-06-20', 165.9, 28.571429, 'candidate'),
    (7, 414, '2023-06-21', 165.9, 3.571429, 'candidate'),
    (7, 18517, '2023-06-24', 187.9, 42.857143, 'candidate'),
    (7, 429, '2023-06-23', 177.9, 32.142857, 'candidate'),
    (7, 261, '2023-06-24', 177.5, 32.142857, 'candidate'),
    (7, 261, '2023-06-25', 175.5, 3.571429, 'candidate'),
    (8, 414, '2023-08-04', 178.9, 28.571429, 'candidate'),
    (8, 414, '2023-08-08', 199.9, 7.142857, 'candidate'),
    (8, 414, '2023-08-21', 209.9, 21.428571, 'candidate'),
    (8, 414, '2023-08-27', 199.9, 21.428571, 'candidate'),
    (8, 414, '2023-09-02', 199.9, 21.428571, 'candidate'),
    (8, 414, '2023-09-08', 227.9, 21.428571, 'candidate'),
    (8, 414, '2023-09-14', 225.9, 21.428571, 'candidate'),
    (8, 414, '2023-09-20', 224.9, 21.428571, 'candidate'),
    (8, 414, '2023-09-26', 219.9, 21.428571, 'candidate'),
    (8, 414, '2023-10-02', 209.9, 21.428571, 'candidate'),
    (8, 414, '2023-10-08', 207.9, 21.428571, 'candidate'),
    (8, 414, '2023-10-14', 195.9, 21.428571, 'candidate'),
    (8, 18517, '2023-09-26', 223.9, 39.285714, 'candidate'),
    (8, 18517, '2023-10-09', 205.9, 21.428571, 'candidate'),
    (8, 18517, '2023-10-15', 205.9, 21.428571, 'candidate'),
    (8, 18517, '2023-10-21', 195.9, 46.428571, 'candidate'),
    (9, 585, '2024-01-19', 180.7, 42.857143, 'candidate'),
    (9, 261, '2024-01-13', 178.5, 28.571429, 'candidate'),
    (9, 261, '2024-01-17', 177.9, 7.142857, 'candidate'),
    (10, 414, '2024-02-20', 191.9, 32.142857, 'candidate'),
    (10, 414, '2024-02-21', 191.9, 3.571429, 'candidate'),
    (10, 414, '2024-03-30', 199.9, 35.714286, 'candidate'),
    (10, 429, '2024-02-24', 187.5, 39.285714, 'candidate'),
    (10, 429, '2024-02-27', 187.5, 3.571429, 'candidate'),
    (10, 429, '2024-03-11', 223.9, 21.428571, 'candidate'),
    (10, 429, '2024-03-17', 219.9, 21.428571, 'candidate'),
    (10, 429, '2024-03-23', 199.9, 21.428571, 'candidate'),
    (10, 429, '2024-03-29', 197.9, 21.428571, 'candidate'),
    (10, 585, '2024-02-20', 189.9, 35.714286, 'candidate'),
    (10, 585, '2024-02-21', 189.9, 3.571429, 'candidate'),
    (10, 261, '2024-02-20', 189.9, 42.857143, 'candidate'),
    (10, 261, '2024-02-23', 189.5, 10.714286, 'candidate'),
    (10, 261, '2024-02-24', 189.5, 3.571429, 'candidate'),
    (10, 261, '2024-02-26', 189.5, 3.571429, 'candidate'),
    (10, 261, '2024-02-27', 189.5, 3.571429, 'candidate'),
    (10, 261, '2024-03-11', 217.9, 21.428571, 'candidate'),
    (10, 261, '2024-03-17', 202.9, 21.428571, 'candidate'),
    (10, 261, '2024-03-23', 200.5, 21.428571, 'candidate'),
    (10, 261, '2024-03-29', 194.5, 21.428571, 'candidate'),
    (11, 414, '2024-05-09', 205.9, 39.285714, 'candidate'),
    (11, 414, '2024-06-28', 192.9, 3.571429, 'candidate'),
    (11, 414, '2024-06-29', 192.9, 3.571429, 'candidate'),
    (11, 414, '2024-07-02', 189.9, 3.571429, 'candidate'),
    (11, 18517, '2024-05-09', 205.9, 39.285714, 'candidate'),
    (12, 414, '2024-10-18', 172.9, 35.714286, 'candidate'),
    (12, 261, '2024-08-07', 189.9, 39.285714, 'candidate'),
    (13, 585, '2024-11-20', 172.9, 42.857143, 'candidate'),
    (14, 429, '2025-02-20', 179.9, 3.571429, 'candidate'),
    (14, 261, '2025-02-17', 177.5, 42.857143, 'candidate'),
    (14, 261, '2025-02-19', 177.5, 3.571429, 'candidate'),
]

_FPS_6YI_PADDING = [
    # (fold, arm, litres, spend_cents) -- one matching (non-flip) fill per fold per arm,
    # padding that fold's total litres/spend to the real run's values.
    (1, 'R0', 1539.2857, 257712.5),
    (1, 'candidate', 1539.2857, 257712.5),
    (2, 'R0', 1453.5714, 255819.2857),
    (2, 'candidate', 1246.4286, 218660.0),
    (3, 'R0', 1317.8571, 254876.7857),
    (3, 'candidate', 1353.5714, 263462.5),
    (4, 'R0', 1403.5714, 241301.7857),
    (4, 'candidate', 1182.1429, 200993.2143),
    (5, 'R0', 1560.7143, 290215.3571),
    (5, 'candidate', 1389.2857, 258336.0714),
    (6, 'R0', 1425.0, 269542.1429),
    (6, 'candidate', 1460.7143, 275739.2857),
    (7, 'R0', 1482.1429, 264166.0714),
    (7, 'candidate', 1392.8571, 249235.7143),
    (8, 'R0', 1307.1429, 258862.8571),
    (8, 'candidate', 1310.7143, 258650.3571),
    (9, 'R0', 1710.7143, 335413.2143),
    (9, 'candidate', 1632.1429, 321421.0714),
    (10, 'R0', 1357.1429, 275705.7143),
    (10, 'candidate', 1182.1429, 242344.6429),
    (11, 'R0', 1500.0, 303878.5714),
    (11, 'candidate', 1446.4286, 293187.5),
    (12, 'R0', 1550.0, 284266.4286),
    (12, 'candidate', 1575.0, 289851.0714),
    (13, 'R0', 1660.7143, 309671.7857),
    (13, 'candidate', 1671.4286, 311795.7143),
    (14, 'R0', 1671.4286, 314477.1429),
    (14, 'candidate', 1653.5714, 311324.6429),
]

_FPS_6YI_HEADLINE_DELTA_CPL_HELD = -0.07349643460619859


#: The real run's smallest fold had 84 fills per arm, well clear of DEFAULT_MIN_ROW_CELL_N (30)
#: — none of its 14 folds' delta_cpl_own were suppressed (breakdowns.per_fold's `suppressed` is
#: False throughout). Splitting each fold/arm's padding total across this many equal-sized rows
#: (same shared key, so still non-flip) reproduces that "never suppressed" property here too,
#: rather than collapsing each pad into the single row a bare total would need only 1 row for.
_PAD_ROWS_PER_ARM = 40


def _fps_6yi_fills() -> pd.DataFrame:
    rows = []
    for fold, station, date, price, litres, bought_by in _FPS_6YI_FLIP_ROWS:
        arm = BASELINE_ARM if bought_by == "baseline" else CANDIDATE_ARM
        rows.append({
            "fold": fold, "arm": arm, "own_tau": 0.25, "date": date,
            "station_code": station, "price": price, "litres": litres,
            "spend_cents": price * litres, "emergency": False,
        })
    for fold, arm, litres, spend_cents in _FPS_6YI_PADDING:
        # A dedicated per-fold station+date shared by BOTH arms -> diff_fills treats every one
        # of these as a MATCHING key (present on both sides), never a flip, regardless of its
        # price/litres — split into _PAD_ROWS_PER_ARM identical rows (not 1) so n_fills clears
        # DEFAULT_MIN_ROW_CELL_N and delta_cpl_own isn't suppressed, matching the real run.
        for _ in range(_PAD_ROWS_PER_ARM):
            rows.append({
                "fold": fold, "arm": arm, "own_tau": 0.25, "date": "2020-01-01",
                "station_code": 900000 + fold,
                "price": (spend_cents / litres) if litres else 0.0,
                "litres": litres / _PAD_ROWS_PER_ARM, "spend_cents": spend_cents / _PAD_ROWS_PER_ARM,
                "emergency": False,
            })
    return pd.DataFrame(rows)


def test_decision_flips_reproduces_fps_6yi_stickiness_phase_saddle_numbers(tmp_path):
    """Regression test (fps-e1w) against the real committed run that motivated this rework —
    experiments/candidates/batch1/stickiness_phase_saddle/facts.json (fps-6yi) and its
    post-dossier review addendum (that run's own README.md, quoted in fps-e1w). Every expected
    number below was computed independently from the real fills.parquet (gitignored, not part
    of this checkout) against the formulas fps-e1w specifies, before being reproduced here
    against the actual summarise_flips / _attach_run_contributions implementation.
    """
    def add_noise_floor(batch_dir):
        (batch_dir / dt.NOISE_FLOOR_FILENAME).write_text(json.dumps({
            "deltas_cpl_held": [-1.0, -0.5, 0.0, 0.5, 1.0],
            "baseline_fingerprint": "54:deadbeef1234",
            "null_method": dt.NULL_METHOD_PLACEBO_COLUMN, "tank_params": "50/3.571/1d/10%",
            "n_placebo_columns": 2,
        }))

    run_dir, _ = _write_run(
        tmp_path, columns=["col_a", "col_b"], batch_extras=add_noise_floor,
        tank_params="50/3.571/1d/10%", fills_df=_fps_6yi_fills(),
        effect_delta_cpl_held=_FPS_6YI_HEADLINE_DELTA_CPL_HELD,
    )
    facts = dt.build_facts(run_dir)
    flips = facts["decision_flips"]
    assert flips["computed"] is True
    assert flips["n_flips"] == 303
    assert flips["cascade_window_days"] == 7

    # fold: (n_baseline_only, n_candidate_only, n_decisions, litres_baseline, litres_candidate,
    #        flip_cpl_delta, run_contribution_cpl)
    expected = {
        1: (0, 0, 0, 0.0, 0.0, None, 0.0),
        2: (12, 38, 13, 200.0, 425.0, 1.7422268907562568, 0.055842627815360535),
        3: (21, 16, 10, 396.428571, 360.714286, -6.378271340647672, -0.037839515753530416),
        4: (15, 39, 16, 250.0, 478.571429, 0.8838805970149792, 0.043706015781614635),
        5: (6, 22, 9, 114.285714, 285.714286, -7.367499999999978, -0.0007760360080724506),
        6: (18, 8, 6, 217.857143, 171.428571, -0.1538251366120278, -0.01826674602054821),
        7: (6, 10, 7, 67.857143, 157.142857, -9.27033492822963, 0.010398882508147125),
        8: (20, 16, 5, 382.142857, 378.571429, 2.5837065773231984, 0.0013968648145325775),
        9: (0, 3, 2, 0.0, 78.571429, None, 0.005339127735527151),
        10: (10, 20, 6, 214.285714, 389.285714, -10.337981651376168, -0.03780847431320868),
        11: (2, 5, 3, 35.714286, 89.285714, -19.680000000000035, -0.019711314604999634),
        12: (3, 2, 5, 100.0, 75.0, -9.523809523809518, 0.003787055719386593),
        13: (5, 1, 3, 53.571429, 42.857143, -32.73333333333335, -0.06441098866987288),
        14: (2, 3, 3, 32.142857, 50.0, -2.2285714285714846, -0.0022349837032429095),
    }
    by_fold = {row["fold"]: row for row in flips["per_fold"]}
    assert set(by_fold) == set(expected)
    resolvable_folds = 0
    for fold, (n_b, n_c, n_dec, lb, lc, delta, contrib) in expected.items():
        row = by_fold[fold]
        assert row["n_baseline_only"] == n_b, fold
        assert row["n_candidate_only"] == n_c, fold
        assert row["n_flips"] == n_b + n_c, fold
        assert row["n_decisions"] == n_dec, fold
        assert row["litres_baseline"] == pytest.approx(lb, abs=1e-3), fold
        assert row["litres_candidate"] == pytest.approx(lc, abs=1e-3), fold
        if delta is None:
            assert row["flip_cpl_delta"] is None, fold
            assert row["flip_cpl_delta_reason"] in ("no flips", "one arm only"), fold
        else:
            assert row["flip_cpl_delta"] == pytest.approx(delta), fold
            # fps-6yi's own finding, quoted in fps-e1w: 0 of 12 resolvable folds print a delta
            # larger than twice its own SE — every one must be marked unresolved.
            assert row["flip_cpl_delta_interval"] is not None, fold
            assert row["flip_cpl_delta_inside_own_se"] is True, fold
            resolvable_folds += 1
        assert row["run_contribution_cpl"] == pytest.approx(contrib, abs=1e-6), fold
    assert resolvable_folds == 12

    # Worked figures from the post-dossier review addendum, quoted verbatim in fps-e1w: fold
    # 13's -32.73 c/L flip delta contributes -0.0644 c/L to the run; folds 2 and 4 hand back
    # +0.0995 between them; the total is -0.0606, reconciling to the -0.0735 headline via a
    # -0.0129 composition residual.
    assert by_fold[13]["run_contribution_cpl"] == pytest.approx(-0.0644, abs=5e-4)
    assert by_fold[2]["run_contribution_cpl"] + by_fold[4]["run_contribution_cpl"] == pytest.approx(
        0.0995, abs=5e-4
    )
    assert flips["run_contribution_total_cpl"] == pytest.approx(-0.0606, abs=5e-4)
    assert flips["composition_residual_cpl"] == pytest.approx(-0.0129, abs=5e-4)
    assert facts["headline"]["realised"]["delta_cpl_held"] == pytest.approx(_FPS_6YI_HEADLINE_DELTA_CPL_HELD)
    # The reconciliation acceptance criterion, asserted rather than eyeballed: contribution
    # total + residual must equal the headline EXACTLY (composition_residual_cpl is defined as
    # the difference, so this is a construction check, not a coincidence).
    assert (
        flips["run_contribution_total_cpl"] + flips["composition_residual_cpl"]
        == pytest.approx(facts["headline"]["realised"]["delta_cpl_held"])
    )

    # fold 13's 2*SE specifically (docs/routines/dossier.md's worked example: -32.73 sits
    # inside a 2*SE of ~41.05 either side — DECISION-level, not fill-level: fold 13 is 6 flips
    # collapsed into 3 decisions, so its effective n is sized on 3 draws, not 6, per PR #347
    # review finding #1). Full interval width is 2*41.05.
    lo, hi = by_fold[13]["flip_cpl_delta_interval"]
    assert hi - lo == pytest.approx(2 * 41.05, abs=0.1)


# ── plots ─────────────────────────────────────────────────────────────────────

def test_plot_realised_cpl_by_fold_titles_a_legacy_run_explicitly(tmp_path, monkeypatch):
    """Review finding on PR #350: the fourth uncovered legacy branch — shock_folds=None
    must produce a title that says the shock/normal shading is unavailable, not a plot
    that silently renders unshaded with no indication why."""
    import matplotlib.axes

    titles: list[str] = []
    original_set_title = matplotlib.axes.Axes.set_title

    def capture_set_title(self, label, *a, **kw):
        titles.append(label)
        return original_set_title(self, label, *a, **kw)

    monkeypatch.setattr(matplotlib.axes.Axes, "set_title", capture_set_title)

    run_dir, _ = _write_run(tmp_path)
    fills = pd.read_parquet(run_dir / dt.FILLS_FILENAME)

    dt._plot_realised_cpl_by_fold(run_dir, fills, "cand", None)
    assert "legacy run" in titles[-1]

    dt._plot_realised_cpl_by_fold(run_dir, fills, "cand", [1, 4])
    assert "shaded = shock fold" in titles[-1]


def test_make_plots_writes_four_always_plots(tmp_path, capsys):
    run_dir, batch_dir = _write_run(tmp_path)
    facts = dt.build_facts(run_dir)

    written = dt.make_plots(run_dir, facts, batch_dir)

    for expected in (
        "per_fold_delta_bars.png", "seed_mean_vs_median.png",
        "realised_cpl_by_fold.png", "tau_sweep.png", "candidate_over_time.png",
    ):
        assert expected in written
        assert (run_dir / expected).exists()
    assert "axis_breakdown.png" not in written
    assert "cycle_phase_breakdown.png" not in written
    assert "external_series_overlay.png" not in written
    assert "inconsistent labels" not in capsys.readouterr().out  # clean fixture, no false alarm


def test_plot_tau_sweep_warns_on_inconsistent_label_per_key(tmp_path, capsys):
    run_dir, batch_dir = _write_run(tmp_path)
    facts = dt.build_facts(run_dir)
    rowpreds = pd.read_parquet(run_dir / dt.ROWPREDS_FILENAME)
    # Flip the label on exactly one row of an otherwise-duplicated (fold, station_code,
    # price_date) key for the candidate arm, so that key now disagrees with itself.
    cand_mask = rowpreds["run"] == dt.CANDIDATE_ARM
    first_idx = rowpreds[cand_mask].index[0]
    rowpreds.loc[first_idx, "label"] = 1 - rowpreds.loc[first_idx, "label"]

    written = dt._plot_tau_sweep(run_dir, rowpreds, facts["candidate"]["name"])

    assert written == "tau_sweep.png"
    assert "inconsistent labels" in capsys.readouterr().out


def test_make_plots_conditional_plots_route_off_declared_metadata(tmp_path):
    def add_features_parquet(batch_dir):
        frame = pd.DataFrame({
            "price_date": [d.strftime("%Y-%m-%d") for d in DATES],
            "tgp_delta_7d": np.linspace(-0.5, 0.5, N_DATES),
        })
        frame.to_parquet(batch_dir / dt.FROZEN_FEATURES_FILENAME)

    run_dir, batch_dir = _write_run(
        tmp_path, with_axis=True, with_cycle=True,
        target={"axis": "regime", "expect_concentration_in": ["shock"]},
        inputs=["cycle_pct_through", "tgp_delta_7d"],
        batch_extras=add_features_parquet,
    )
    facts = dt.build_facts(run_dir)

    written = dt.make_plots(run_dir, facts, batch_dir)

    assert "axis_breakdown.png" in written
    assert "cycle_phase_breakdown.png" in written
    assert "external_series_overlay.png" in written
    for name in ("axis_breakdown.png", "cycle_phase_breakdown.png", "external_series_overlay.png"):
        assert (run_dir / name).exists()


def test_parse_price_date_handles_iso_strings():
    s = pd.Series(["2026-01-01", "2026-01-02"])
    parsed = dt._parse_price_date(s)
    assert pd.api.types.is_datetime64_any_dtype(parsed)
    assert list(parsed.dt.strftime("%Y-%m-%d")) == ["2026-01-01", "2026-01-02"]


def test_parse_price_date_handles_integer_yyyymmdd():
    # batch_freeze.py's own _snapshot_date docstring documents this as the DB-native form —
    # features.parquet's price_date is not guaranteed to be an ISO string.
    s = pd.Series([20260101, 20260102])
    parsed = dt._parse_price_date(s)
    assert pd.api.types.is_datetime64_any_dtype(parsed)
    assert list(parsed.dt.strftime("%Y-%m-%d")) == ["2026-01-01", "2026-01-02"]


def test_plot_external_overlay_handles_integer_price_date(tmp_path):
    # Regression: the plot used to pass a raw features.parquet price_date column straight to
    # ax.plot with no datetime conversion — fine by luck when the fixture used ISO strings
    # (matplotlib still renders, just badly), but an INTEGER YYYYMMDD column (the DB-native form)
    # would plot as a numeric axis with no date semantics at all. Assert the function still runs
    # and produces the plot either way.
    def add_features_parquet(batch_dir):
        frame = pd.DataFrame({
            "price_date": [int(d.strftime("%Y%m%d")) for d in DATES],
            "tgp_delta_7d": np.linspace(-0.5, 0.5, N_DATES),
        })
        frame.to_parquet(batch_dir / dt.FROZEN_FEATURES_FILENAME)

    run_dir, batch_dir = _write_run(
        tmp_path, inputs=["tgp_delta_7d"], batch_extras=add_features_parquet,
    )
    facts = dt.build_facts(run_dir)

    written = dt._plot_external_overlay(run_dir, facts, batch_dir, facts["candidate"]["name"])

    assert written == "external_series_overlay.png"
    assert (run_dir / "external_series_overlay.png").exists()


def test_make_plots_returns_empty_when_run_did_not_score(tmp_path):
    run_dir, batch_dir = _write_run(tmp_path, status="aborted_candidate")
    facts = dt.build_facts(run_dir)

    assert dt.make_plots(run_dir, facts, batch_dir) == []


# ── orchestration ─────────────────────────────────────────────────────────────

def test_build_facts_leaves_the_baseline_fingerprint_none_for_older_runs(tmp_path):
    """A run that predates the field reports None rather than a guess (fps-zci).

    Unlike git_sha, there is no safe fallback: substituting today's constants would
    assert that an old run used today's baseline, which is exactly the claim that was
    false for every batch0 run. The absence is the signal.
    """
    run_dir, _ = _write_run(tmp_path)
    results = json.loads((run_dir / dt.RESULTS_FILENAME).read_text())
    del results["meta"]["n_baseline_columns"]
    del results["meta"]["baseline_fingerprint"]
    (run_dir / dt.RESULTS_FILENAME).write_text(json.dumps(results))

    facts = dt.build_facts(run_dir)

    assert facts["provenance"]["n_baseline_columns"] is None
    assert facts["provenance"]["baseline_fingerprint"] is None


def test_process_run_writes_facts_json_with_plots_list(tmp_path):
    run_dir, _ = _write_run(tmp_path)

    facts_path = dt.process_run(run_dir)

    assert facts_path == run_dir / dt.FACTS_FILENAME
    facts = json.loads(facts_path.read_text())
    assert "per_fold_delta_bars.png" in facts["plots"]


def _strict_json_loads(text: str):
    """json.loads with the NaN/Infinity extension disabled — Python's default is permissive on
    both dumps AND loads, so a plain round-trip through json.loads would NOT catch a bare `NaN`
    token; this is what an RFC-8259-strict reader (jq, most non-Python consumers) actually sees.
    """
    def _reject(token):
        raise ValueError(f"non-finite JSON constant {token!r} — invalid per RFC 8259")

    return json.loads(text, parse_constant=_reject)


def test_process_run_facts_json_is_strictly_valid_even_with_a_single_draw_noise_band(tmp_path):
    # Regression: a noise_floor.json with exactly one draw takes _noise_band's `deltas.size == 1`
    # branch, where band_std is float("nan") — process_run used to write that straight through
    # raw json.dumps, which happily emits the literal `NaN` token (not valid JSON per RFC 8259).
    # facts.json is the sole input to the downstream Claude session and to any non-Python reader.
    def add_noise_floor(batch_dir):
        (batch_dir / dt.NOISE_FLOOR_FILENAME).write_text(json.dumps({
            "deltas_cpl_held": [0.01], "baseline_fingerprint": "54:deadbeef1234",
            "null_method": dt.NULL_METHOD_PLACEBO_COLUMN, "tank_params": "50/3.571/7d/10%",
        }))

    run_dir, _ = _write_run(tmp_path, batch_extras=add_noise_floor)

    facts_path = dt.process_run(run_dir)
    raw_text = facts_path.read_text()

    facts = _strict_json_loads(raw_text)  # must not raise
    assert facts["noise_band"]["available"] is True
    assert facts["noise_band"]["band_std_delta_cpl_held"] is None  # to_jsonable maps NaN -> null
    # single_candidate_z_threshold needs a t-critical value (df = n_draws - 1 >= 1), same guard
    # as candidate_z_vs_band itself — n_draws=1 can't estimate a std, so both are None (fps-3jj.17).
    assert facts["noise_band"]["single_candidate_z_threshold"] is None


# ── CLI robustness ────────────────────────────────────────────────────────────

def test_main_scan_isolates_one_broken_run_from_the_rest(tmp_path):
    # Regression: process_run used to be called with no per-run guard in the --scan loop, so one
    # malformed run (e.g. a results.json without the fields build_facts expects) would raise past
    # the loop and skip every other pending run in the same unattended overnight pass.
    good_root = tmp_path / "good"
    good_root.mkdir()
    good_dir, _ = _write_run(good_root)
    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    (bad_dir / dt.RESULTS_FILENAME).write_text(json.dumps({"status": "graded"}))  # missing "candidate" etc

    result = CliRunner().invoke(dt.main, ["--scan", str(tmp_path)])

    assert result.exit_code == 0
    assert (good_dir / dt.FACTS_FILENAME).exists()
    assert not (bad_dir / dt.FACTS_FILENAME).exists()
    assert "bad" in result.output and "failed" in result.output


# ── per-axis merge safety ────────────────────────────────────────────────────

def test_breakdowns_per_axis_raises_on_duplicate_station_date_key_with_conflicting_axis(tmp_path):
    # Regression: the fills<->axis_lookup merge had no `validate="many_to_one"` — a candidate
    # whose add_axis produced two different labels for the same (station_code, date) key would
    # silently fan out the merge (inflating n_fills and skewing pooled_cpl) instead of raising.
    run_dir, _ = _write_run(tmp_path, with_axis=True)
    results = json.loads((run_dir / dt.RESULTS_FILENAME).read_text())
    rowpreds = pd.read_parquet(run_dir / dt.ROWPREDS_FILENAME)
    fills = pd.read_parquet(run_dir / dt.FILLS_FILENAME)

    # Inject a genuine conflict: duplicate one axis row for an existing (station_code, price_date)
    # key with the opposite label.
    dup = rowpreds.iloc[[0]].copy()
    dup["axis"] = "B" if rowpreds.iloc[0]["axis"] == "A" else "A"
    rowpreds = pd.concat([rowpreds, dup], ignore_index=True)

    with pytest.raises(Exception, match="many_to_one|Merge"):
        dt._breakdowns(results, rowpreds, fills)


def test_breakdowns_per_axis_reports_coverage_note_and_unmatched_count(tmp_path):
    run_dir, _ = _write_run(tmp_path, with_axis=True)
    facts = dt.build_facts(run_dir)

    note = facts["breakdowns"]["per_axis_coverage_note"]
    assert note is not None
    assert "WFCV validation-window" in note


def test_mechanism_family_survives_into_facts(tmp_path):
    """generator.md makes MECHANISM_FAMILY a required DISCLOSURE, whose purpose is that a
    same-family batch becomes visible downstream rather than a gate the generator learns
    to game. Until fps-3jj.11 nothing read it off the module at all, so it lived only in
    a source file and the disclosure was inert. This asserts the wire is connected.
    """
    run_dir, _ = _write_run(tmp_path)

    facts = dt.build_facts(run_dir)

    assert facts["candidate"]["mechanism_family"] == "test-family"


# ── noise-band arity guard (fps-3jj.14) ───────────────────────────────────────

def _floor_with(arity=None, n=20):
    deltas = list(np.random.default_rng(7).normal(0, 0.02, size=n))
    payload = {
        "deltas_cpl_held": deltas, "baseline_fingerprint": "54:deadbeef1234",
        "null_method": dt.NULL_METHOD_PLACEBO_COLUMN, "tank_params": "50/3.571/7d/10%",
    }
    if arity is not None:
        payload["n_placebo_columns"] = arity

    def _write(batch_dir):
        (batch_dir / dt.NOISE_FLOOR_FILENAME).write_text(json.dumps(payload))

    return _write


def test_noise_band_refuses_a_multi_column_candidate_against_a_1_column_floor(tmp_path):
    """The bias this bead exists to close: a 3-column arm has more chances for the fit to
    find something than the 1-column placebo arm the band was built from, so the band is
    narrow in the CANDIDATE's favour by an unmeasured amount."""
    run_dir, _ = _write_run(
        tmp_path, columns=["a", "b", "c"], batch_extras=_floor_with(arity=1)
    )

    band = dt.build_facts(run_dir)["noise_band"]

    assert band["available"] is False
    assert "1-column null" in band["reason"]
    assert "3 columns" in band["reason"]
    # The message must be actionable — it is read by an unattended 05:06 dossier session.
    # BOTH halves have to be there. An earlier revision said only "compute a floor at this
    # arity ... then point this batch's ruler at it", which names no mechanism: _noise_band
    # reads only NOISE_FLOOR_FILENAME and nothing reads an arity-suffixed side-file, so
    # following it produces a file that changes nothing. Promotion is a rename; the rename
    # IS the remediation.
    assert "--arity 3" in band["reason"]
    assert "--out-name noise_floor_k3.json" in band["reason"]
    assert "mv noise_floor.json noise_floor_k1.json" in band["reason"]
    assert "mv noise_floor_k3.json noise_floor.json" in band["reason"]
    # And it must steer away from the tempting-but-destructive alternative.
    assert "--force" in band["reason"]
    # n_draws is derived from the floor being REPLACED, not from a pool size (fps-3jj.21:
    # the pools no longer bind, so the wider floor is computed at the same certainty as the
    # one it supersedes — which is also what makes the two comparable).
    assert "--n-draws 20" in band["reason"]


def test_noise_band_grades_a_wide_candidate_instead_of_refusing_it(tmp_path):
    """fps-3jj.25 removed the arity cap. A candidate far wider than the ruler's own arity used
    to get `available: false` and no verdict at all — a night of compute wasted, then an
    unbounded nightly re-pick. It must now be GRADED, against a floor of at least its own
    arity, with the reuse cost carried by the threshold rather than by a refusal."""
    wide = [f"c{i}" for i in range(12)]
    run_dir, _ = _write_run(
        tmp_path, columns=wide, batch_extras=_floor_with(arity=12)
    )

    band = dt.build_facts(run_dir)["noise_band"]

    assert band["available"] is True
    assert band["candidate_n_columns"] == 12
    assert band["single_candidate_bar_cpl_held"] is not None


def test_noise_band_uses_effective_draws_not_nominal_for_the_threshold(tmp_path):
    """The correction that replaced the cap. A floor stamped with fewer EFFECTIVE draws than
    nominal must produce a strictly harder bar — that is the whole mechanism by which a wide
    candidate is priced rather than turned away. Passing the nominal count would act as though
    the band were better estimated than it is: a bar too EASY, the bias this exists to remove."""
    (tmp_path / "nominal").mkdir()
    (tmp_path / "reused").mkdir()

    def _floor(effective):
        base = _floor_with(arity=1, n=20)

        def _write(batch_dir):
            base(batch_dir)
            path = batch_dir / dt.NOISE_FLOOR_FILENAME
            payload = json.loads(path.read_text())
            if effective is not None:
                payload["effective_n_draws"] = effective
            path.write_text(json.dumps(payload))

        return _write

    full = _write_run(tmp_path / "nominal", columns=["a"], batch_extras=_floor(20.0))[0]
    thin = _write_run(tmp_path / "reused", columns=["a"], batch_extras=_floor(9.04))[0]

    full_band = dt.build_facts(full)["noise_band"]
    thin_band = dt.build_facts(thin)["noise_band"]

    assert full_band["effective_n_draws"] == pytest.approx(20.0)
    assert thin_band["effective_n_draws"] == pytest.approx(9.04)
    # Both bands are identical; only the effective draw count differs. delta_cpl_held is a
    # COST, so "harder" is a MORE NEGATIVE bar.
    assert thin_band["single_candidate_bar_cpl_held"] < full_band["single_candidate_bar_cpl_held"]
    assert thin_band["single_candidate_z_threshold"] > full_band["single_candidate_z_threshold"]


def test_noise_band_reconstructs_effective_draws_from_an_unstamped_reusing_floor(tmp_path):
    """Review of PR #336. fps-3jj.21 landed reuse-capable `candidate_sequence` one PR BEFORE
    the `effective_n_draws` stamp existed, so a floor built from main in that window can carry
    real reuse and no stamp. Assuming nominal there would grade a ~17.5-draw bank at 20 —
    silent, and in the too-easy direction. `placebo_draws` records the source column of every
    member, so the overlap is RECOVERABLE and must be recovered, not assumed."""
    reused = {
        "n_placebo_columns": 2,
        # 4 draws x 2 columns over 4 distinct columns — heavy, deliberate overlap.
        "placebo_draws": [
            {"source_column": c, "block_seed": i, "self_correlation": 0.0}
            for i, c in enumerate(["a", "b", "a", "b", "c", "d", "c", "d"])
        ],
    }

    def _write(batch_dir):
        _floor_with(arity=2, n=4)(batch_dir)
        path = batch_dir / dt.NOISE_FLOOR_FILENAME
        payload = json.loads(path.read_text())
        payload.update(reused)
        payload.pop("effective_n_draws", None)
        path.write_text(json.dumps(payload))

    run_dir, _ = _write_run(tmp_path, columns=["a", "b"], batch_extras=_write)
    band = dt.build_facts(run_dir)["noise_band"]

    assert band["available"] is True
    assert band["effective_n_draws"] < 4.0, "reuse was assumed away instead of reconstructed"


def test_noise_band_separates_the_reuse_charge_from_the_thinness_charge(tmp_path):
    """Review of PR #336. `bar_draw_count_shift_cpl_held` exists to isolate band THINNESS —
    `experiments/2026-08-23_placebo_arity/` used exactly this quantity to attribute two thirds
    of batch1's k=1 -> k=3 bar move to draw count rather than arity. Once the threshold took
    the EFFECTIVE count, deriving the shift from it would have folded an arity-driven reuse
    effect into the very number meant to separate them. So the thinness shift is computed on
    NOMINAL draws and the reuse charge is emitted alongside; the two must span the whole
    distance from the large-sample bar to the applied one."""
    def _write(batch_dir):
        _floor_with(arity=1, n=20)(batch_dir)
        path = batch_dir / dt.NOISE_FLOOR_FILENAME
        payload = json.loads(path.read_text())
        payload["effective_n_draws"] = 9.04
        path.write_text(json.dumps(payload))

    run_dir, _ = _write_run(tmp_path, columns=["a"], batch_extras=_write)
    band = dt.build_facts(run_dir)["noise_band"]

    std = band["band_std_delta_cpl_held"]
    thinness = band["bar_draw_count_shift_cpl_held"]
    reuse = band["bar_reuse_shift_cpl_held"]

    # The thinness charge must be the NOMINAL-draw one, untouched by reuse.
    assert thinness == pytest.approx(
        (dt._LARGE_SAMPLE_Z - band["single_candidate_z_threshold_nominal"]) * std
    )
    assert reuse < 0, "reuse must make the bar harder, i.e. more negative"
    # Together they span large-sample bar -> applied bar, with nothing double-counted.
    assert thinness + reuse == pytest.approx(
        (dt._LARGE_SAMPLE_Z - band["single_candidate_z_threshold"]) * std
    )


def test_thinness_shift_survives_a_band_too_reused_to_estimate(tmp_path):
    """Review of PR #336. `bar_draw_count_shift_cpl_held` is a function of the NOMINAL draw
    count alone and is documented as "a property of this floor alone, comparable across floors".
    Gating it on `band_estimable` — which carries the reuse condition `n_eff >= 2` — let a REUSE
    property null out a quantity documented as independent of reuse, silently re-coupling the
    two things the split had just separated.

    Latent rather than live (rho_bar caps at TEXTURE_ICC_BOUND, so n_eff cannot fall under 2 at
    20 draws), but a 5-draw fully-overlapping bank reaches n_eff 1.95 and would lose a thinness
    value its nominal 5 supports perfectly well."""
    def _write(batch_dir):
        _floor_with(arity=1, n=5)(batch_dir)
        path = batch_dir / dt.NOISE_FLOOR_FILENAME
        payload = json.loads(path.read_text())
        payload["effective_n_draws"] = 1.95  # every draw overlapping every other
        path.write_text(json.dumps(payload))

    run_dir, _ = _write_run(tmp_path, columns=["a"], batch_extras=_write)
    band = dt.build_facts(run_dir)["noise_band"]

    # The band itself cannot be graded — too few effective draws for a t-critical.
    assert band["single_candidate_z_threshold"] is None
    # But the NOMINAL-draw quantity is unaffected and must still be reported.
    assert band["single_candidate_z_threshold_nominal"] is not None
    assert band["bar_draw_count_shift_cpl_held"] is not None
    # The reuse charge needs both, so it is legitimately None here.
    assert band["bar_reuse_shift_cpl_held"] is None


def test_noise_band_reports_no_reuse_charge_for_a_reuse_free_bank(tmp_path):
    run_dir, _ = _write_run(tmp_path, columns=["a"], batch_extras=_floor_with(arity=1, n=20))
    band = dt.build_facts(run_dir)["noise_band"]
    assert band["bar_reuse_shift_cpl_held"] == pytest.approx(0.0)


def test_noise_band_falls_back_to_nominal_draws_on_a_pre_fps_3jj_25_floor(tmp_path):
    """A floor written before effective_n_draws existed predates column reuse entirely — the
    pools could not supply it — so its nominal count IS its effective count. Falling back is
    exact here, not optimistic, which is why this one does NOT follow the "cannot be shown to
    match, so cannot be trusted" rule the fingerprint checks apply."""
    run_dir, _ = _write_run(tmp_path, columns=["a"], batch_extras=_floor_with(arity=1, n=20))

    band = dt.build_facts(run_dir)["noise_band"]

    assert band["available"] is True
    assert band["effective_n_draws"] == pytest.approx(20.0)


def test_noise_band_reports_the_bar_in_cents_per_litre(tmp_path):
    """fps-3jj.21's rendering half: `candidate_z_vs_band` against a z threshold is a
    comparison in band-standard-deviation units, which says nothing about whether one
    candidate's ruler is stricter than another's. The same bar in c/L is directly comparable —
    against a batch-mate, and against results.csv's own 0.03-0.26 c/L history."""
    run_dir, _ = _write_run(tmp_path, columns=["a"], batch_extras=_floor_with(arity=1))

    band = dt.build_facts(run_dir)["noise_band"]

    mean, std = band["band_mean_delta_cpl_held"], band["band_std_delta_cpl_held"]
    expected = mean - band["single_candidate_z_threshold"] * std
    assert band["single_candidate_bar_cpl_held"] == pytest.approx(expected)
    # delta_cpl_held is a COST, so the bar must sit BELOW the band's centre.
    assert band["single_candidate_bar_cpl_held"] < mean


def test_noise_band_separates_band_thinness_from_band_width(tmp_path):
    """Two thirds of batch1's k=1 -> k=3 bar move was the t-penalty for estimating the band
    from 10 draws instead of 20, not the arity effect it was being read as
    (`experiments/2026-08-23_placebo_arity/`). A dossier that prints only the bar cannot
    distinguish those, so the draw-count component is emitted separately — and a floor built
    from fewer draws must carry a strictly larger one."""
    (tmp_path / "thin").mkdir()
    (tmp_path / "thick").mkdir()
    thin = _write_run(tmp_path / "thin", columns=["a"], batch_extras=_floor_with(arity=1, n=5))[0]
    thick = _write_run(tmp_path / "thick", columns=["a"], batch_extras=_floor_with(arity=1, n=40))[0]

    thin_band = dt.build_facts(thin)["noise_band"]
    thick_band = dt.build_facts(thick)["noise_band"]

    # Reported as a signed move on a cost, so "harder" is MORE NEGATIVE.
    assert thin_band["bar_draw_count_shift_cpl_held"] < 0
    assert thin_band["bar_draw_count_shift_cpl_held"] < thick_band["bar_draw_count_shift_cpl_held"]


def test_noise_band_discloses_how_much_wider_the_ruler_is(tmp_path):
    """A floor ABOVE the run's arity is allowed (it can only raise the bar) but it means a
    candidate failing against it has not been shown to fail against its OWN arity's band. The
    boolean said that a difference existed; the excess says how big it is."""
    run_dir, _ = _write_run(tmp_path, columns=["a"], batch_extras=_floor_with(arity=3))

    band = dt.build_facts(run_dir)["noise_band"]

    assert band["floor_arity_exceeds_run"] is True
    assert band["floor_arity_excess"] == 2


def test_noise_band_treats_a_floor_with_no_arity_key_as_arity_1(tmp_path):
    """A floor predating fps-3jj.14 WAS arity 1 — the parameter did not exist. So its
    absence pins the value rather than leaving it unknown, and it keeps grading 1-column
    candidates instead of being refused wholesale."""
    (tmp_path / "single").mkdir()
    (tmp_path / "multi").mkdir()
    single = _write_run(tmp_path / "single", columns=["a"], batch_extras=_floor_with(arity=None))[0]
    assert dt.build_facts(single)["noise_band"]["available"] is True

    multi = _write_run(tmp_path / "multi", columns=["a", "b"], batch_extras=_floor_with(arity=None))[0]
    multi_band = dt.build_facts(multi)["noise_band"]
    assert multi_band["available"] is False
    assert "1-column null" in multi_band["reason"]


def test_noise_band_refuses_a_floor_declaring_zero_placebo_columns(tmp_path):
    """review of PR #349: `n_placebo_columns: 0` is a degenerate ruler, not evidence of arity
    1 — only a MISSING key means arity 1 (a floor predating fps-3jj.14 entirely, where the
    parameter did not exist yet). A defensive `... or 1` fallback would silently promote a
    bank explicitly claiming 0 placebo columns into a valid-looking arity-1 ruler, exactly the
    bias fps-3jj.14 exists to catch."""
    run_dir, _ = _write_run(tmp_path, columns=["a"], batch_extras=_floor_with(arity=0))

    band = dt.build_facts(run_dir)["noise_band"]

    assert band["available"] is False
    assert band["reason_code"] == dt.NOISE_BAND_REFUSAL_ARITY


def test_noise_band_allows_a_wider_floor_and_discloses_it(tmp_path):
    """One-sided on purpose. A floor at arity >= the run's can only be as wide or wider,
    i.e. a HARDER bar — refusing it would force one ~2h calibration per distinct arity in a
    batch (batch1 alone has arities 3,3,3,2,2) to buy a tighter bar nobody needs."""
    run_dir, _ = _write_run(
        tmp_path, columns=["a", "b"], batch_extras=_floor_with(arity=3)
    )

    band = dt.build_facts(run_dir)["noise_band"]

    assert band["available"] is True
    assert band["n_placebo_columns"] == 3
    assert band["candidate_n_columns"] == 2
    assert band["floor_arity_exceeds_run"] is True


def test_noise_band_matching_arity_is_not_flagged_as_wider(tmp_path):
    run_dir, _ = _write_run(
        tmp_path, columns=["a", "b", "c"], batch_extras=_floor_with(arity=3)
    )

    band = dt.build_facts(run_dir)["noise_band"]

    assert band["available"] is True
    assert band["n_placebo_columns"] == 3
    assert band["candidate_n_columns"] == 3
    assert band["floor_arity_exceeds_run"] is False


def test_arity_refusal_carries_a_machine_readable_code(tmp_path):
    """The dossier routine maps `available: false` to `outcome: rejected` — dead ground the
    next generator won't re-propose. An arity refusal must NOT reach that branch: the run
    graded fine and becomes a real measurement once a wide-enough ruler exists. The routine
    branches on this code rather than pattern-matching the prose reason, because fps-3jj.17
    made the ledger outcome mechanical rather than a 05:06 judgement call."""
    run_dir, _ = _write_run(
        tmp_path, columns=["a", "b", "c"], batch_extras=_floor_with(arity=1)
    )

    band = dt.build_facts(run_dir)["noise_band"]

    assert band["available"] is False
    assert band["reason_code"] == dt.NOISE_BAND_REFUSAL_ARITY
    # The prose must ALSO say it, for the human reading the dossier rather than the branch.
    assert "NOT rejected" in band["reason"]


def test_other_refusals_carry_no_arity_code(tmp_path):
    """The carve-out is specific. A missing floor really does mean 'no ruler, and none is
    coming without human action', so it keeps falling through to the rejected rule."""
    run_dir, _ = _write_run(tmp_path, columns=["a"])  # no noise_floor.json at all

    band = dt.build_facts(run_dir)["noise_band"]

    assert band["available"] is False
    assert "reason_code" not in band


# ── comparable sibling noise banks (fps-30p) ──────────────────────────────────

def _banks(canonical: dict, **siblings: dict):
    """Write a canonical noise_floor.json plus named sibling banks into the batch dir."""
    def _write(batch_dir):
        (batch_dir / dt.NOISE_FLOOR_FILENAME).write_text(json.dumps(canonical))
        for name, payload in siblings.items():
            (batch_dir / f"noise_floor_{name}.json").write_text(json.dumps(payload))
    return _write


def _bank(*, n=20, arity=2, sd=0.02, seed=7, **overrides):
    payload = {
        "deltas_cpl_held": list(np.random.default_rng(seed).normal(0, sd, size=n)),
        "baseline_fingerprint": "54:deadbeef1234",
        "null_method": dt.NULL_METHOD_PLACEBO_COLUMN,
        "tank_params": "50/3.571/7d/10%",
        "n_placebo_columns": arity,
    }
    payload.update(overrides)
    return payload


def test_noise_band_scores_the_candidate_against_every_comparable_sibling_bank(tmp_path):
    """fps-30p: the motivating failure was a `not_tested` line saying a differently-shaped
    floor "could resolve it either way" while a comparable bank sat unread in the same
    directory. Siblings must be scored, not just noticed."""
    run_dir, _ = _write_run(tmp_path, columns=["a", "b"], batch_extras=_banks(
        _bank(arity=2, n=10, seed=1),
        k1=_bank(arity=1, n=20, seed=2),
        icc=_bank(arity=1, n=32, seed=3, null_method="placebo_column_pinned_source"),
    ))

    band = dt.build_facts(run_dir)["noise_band"]
    assert band["available"] is True
    by_name = {b["name"]: b for b in band["comparable_banks"]}
    assert set(by_name) == {"noise_floor_k1.json", "noise_floor_icc.json"}

    for name, arity, n in (("noise_floor_k1.json", 1, 20), ("noise_floor_icc.json", 1, 32)):
        b = by_name[name]
        assert b["comparable"] is True
        assert (b["n_placebo_columns"], b["n_draws"]) == (arity, n)
        # Scored against the SAME delta the canonical bank graded.
        expected = (band["candidate_delta_cpl_held"] - b["band_mean_delta_cpl_held"]) / b[
            "band_std_delta_cpl_held"
        ]
        assert b["candidate_z_vs_band"] == pytest.approx(expected)
        assert b["single_candidate_z_threshold"] > 0

    # A differing null_method is REPORTED, never refused — a pinned-source bank measures a
    # genuinely different null and its z is still worth seeing, provided the reader is told.
    assert by_name["noise_floor_icc.json"]["null_method"] == "placebo_column_pinned_source"


def test_noise_band_canonical_grade_is_untouched_by_siblings(tmp_path):
    """Siblings are corroboration only. The ledger outcome keys on the canonical bank
    (docs/routines/dossier.md step 3's mechanical `-t < z < t`), so a sibling that
    disagrees must not move `candidate_z_vs_band` or the threshold beside it."""
    canonical = _bank(arity=2, n=10, seed=1)
    (tmp_path / "alone").mkdir()
    (tmp_path / "sibs").mkdir()
    alone, _ = _write_run(tmp_path / "alone", columns=["a", "b"], batch_extras=_banks(canonical))
    withsibs, _ = _write_run(tmp_path / "sibs", columns=["a", "b"], batch_extras=_banks(
        canonical, k1=_bank(arity=1, n=20, sd=0.5, seed=9),
    ))

    a = dt.build_facts(alone)["noise_band"]
    b = dt.build_facts(withsibs)["noise_band"]

    assert a["comparable_banks"] == []
    assert len(b["comparable_banks"]) == 1
    for key in ("candidate_z_vs_band", "single_candidate_z_threshold",
                "band_mean_delta_cpl_held", "band_std_delta_cpl_held", "n_draws"):
        assert a[key] == pytest.approx(b[key]), key


@pytest.mark.parametrize("overrides,expected", [
    ({"baseline_fingerprint": "54:other"}, "baseline_fingerprint"),
    ({"tank_params": "50/3.571/1d/10%"}, "tank_params"),
    ({"partial": True}, "--fold-subset"),
    ({"deltas_cpl_held": [0.01]}, "only 1 draw"),
])
def test_noise_band_lists_non_comparable_siblings_with_a_reason(tmp_path, overrides, expected):
    """Listed-with-a-reason, not dropped: "no sibling answered this" and "no sibling was
    looked at" must not be confusable by a reader of the committed facts.json."""
    run_dir, _ = _write_run(tmp_path, columns=["a", "b"], batch_extras=_banks(
        _bank(arity=2, n=10, seed=1), bad=_bank(**overrides),
    ))

    banks = dt.build_facts(run_dir)["noise_band"]["comparable_banks"]

    assert [b["name"] for b in banks] == ["noise_floor_bad.json"]
    assert banks[0]["comparable"] is False
    assert expected in banks[0]["reason"]
    assert "candidate_z_vs_band" not in banks[0]


def test_noise_band_arity_refusal_names_an_existing_wide_enough_bank(tmp_path):
    """The refusal tells an operator to spend ~2h computing a wider ruler. If one already
    exists beside the canonical bank — which is exactly what the `mv`-don't-`--force`
    promote step causes to accumulate — say so first."""
    run_dir, _ = _write_run(tmp_path, columns=["a", "b", "c"], batch_extras=_banks(
        _bank(arity=1, n=20, seed=1), k3=_bank(arity=3, n=20, seed=2),
    ))

    band = dt.build_facts(run_dir)["noise_band"]

    assert band["available"] is False
    assert band["reason_code"] == dt.NOISE_BAND_REFUSAL_ARITY
    assert "may already exist" in band["reason"]
    assert "noise_floor_k3.json (3 columns, 20 draws)" in band["reason"]
    # The recompute instructions must survive — the existing bank is a hint, not a promise.
    assert "--arity 3" in band["reason"]


def test_noise_band_arity_refusal_stays_quiet_when_no_sibling_is_wide_enough(tmp_path):
    run_dir, _ = _write_run(tmp_path, columns=["a", "b", "c"], batch_extras=_banks(
        _bank(arity=1, n=20, seed=1), k2=_bank(arity=2, n=20, seed=2),
    ))

    band = dt.build_facts(run_dir)["noise_band"]

    assert band["available"] is False
    assert "may already exist" not in band["reason"]


# ── sibling-bank robustness (review of PR #345) ───────────────────────────────

@pytest.mark.parametrize("overrides,expected", [
    ({"deltas_cpl_held": [0.01] * 20}, "same value"),
    ({"deltas_cpl_held": [0.0] * 20}, "same value"),
])
def test_noise_band_sibling_with_no_estimable_band_is_not_comparable(tmp_path, overrides, expected):
    """A bank whose identity matches but whose band cannot be estimated corroborates nothing.
    The canonical path already treats present-but-null z/t as ungradeable
    (docs/routines/dossier.md), so a sibling in that state must not read as comparable."""
    run_dir, _ = _write_run(tmp_path, columns=["a", "b"], batch_extras=_banks(
        _bank(arity=2, n=10, seed=1), flat=_bank(**overrides),
    ))

    banks = dt.build_facts(run_dir)["noise_band"]["comparable_banks"]

    assert [b["name"] for b in banks] == ["noise_floor_flat.json"]
    assert banks[0]["comparable"] is False
    assert expected in banks[0]["reason"]
    assert "candidate_z_vs_band" not in banks[0]


def test_noise_band_and_sibling_path_agree_on_a_degenerate_bank(tmp_path):
    """Documents agreement on the ONE axis fps-tnz had already shared before fps-1x5
    (`_band_std_usable`) — put the IDENTICAL degenerate bank on both paths and confirm the
    same estimability verdict. Note this does NOT prove the two paths share code: it still
    passes against the pre-fps-1x5 duplication (fps-tnz's fix already landed on both copies
    by the time this PR started), so it is not itself the "would fail under the old
    duplication" regression test fps-1x5's acceptance criteria ask for — see
    test_noise_band_and_sibling_scores_are_both_derived_from_the_shared_score_bank below for
    that, which proves sharing directly rather than inferring it from agreement (review of
    PR #349)."""
    degenerate = [0.01] * 20
    run_dir, _ = _write_run(tmp_path, columns=["a", "b"], batch_extras=_banks(
        _bank(arity=2, deltas_cpl_held=degenerate),
        twin=_bank(arity=2, deltas_cpl_held=degenerate),
    ))

    band = dt.build_facts(run_dir)["noise_band"]

    # Canonical: still "available" (docs/routines/dossier.md's present-but-null rule) but
    # ungradeable — no z, no threshold.
    assert band["available"] is True
    assert band["candidate_z_vs_band"] is None
    assert band["single_candidate_z_threshold"] is None

    # The IDENTICAL deltas as a sibling: refused outright, but for the same underlying
    # reason a degenerate canonical bank is ungradeable — every draw landed on the same value.
    sib = band["comparable_banks"][0]
    assert sib["comparable"] is False
    assert "same value" in sib["reason"]


def test_noise_band_and_sibling_scores_are_both_derived_from_the_shared_score_bank(tmp_path):
    """fps-1x5's actual unification target, proven directly rather than inferred from
    agreement (review of PR #349): `_noise_band` and `_comparable_noise_banks` must compute
    their band statistics by calling the SAME `_score_bank`, not two independently written
    computations that happen to produce the same numbers today. Computes the expected
    band_mean/std/effective_n_draws/z independently via a bare `_score_bank` call on the exact
    same deltas the fixture bank carries, and asserts BOTH callers' facts.json numbers equal
    it exactly — so a future fork that reintroduces even a subtly different computation in one
    caller (a different ddof, a rounding step, a stale copy of a `_resolve_effective_n_draws`
    call) shows up as a mismatch here, unlike the agreement test above."""
    deltas = list(np.random.default_rng(11).normal(0, 0.03, size=15))
    run_dir, _ = _write_run(tmp_path, columns=["a", "b"], batch_extras=_banks(
        _bank(arity=2, deltas_cpl_held=deltas),
        twin=_bank(arity=2, deltas_cpl_held=deltas),
    ))

    facts = dt.build_facts(run_dir)
    band = facts["noise_band"]
    sib = band["comparable_banks"][0]

    expected = dt._score_bank(
        np.asarray(deltas, dtype=float), band["candidate_delta_cpl_held"],
        {"deltas_cpl_held": deltas},
    )

    for observed in (band, sib):
        assert observed["band_mean_delta_cpl_held"] == pytest.approx(expected["band_mean"])
        assert observed["band_std_delta_cpl_held"] == pytest.approx(expected["band_std"])
        assert observed["effective_n_draws"] == pytest.approx(expected["effective_n_draws"])
        assert observed["candidate_z_vs_band"] == pytest.approx(expected["candidate_z_vs_band"])
        assert observed["single_candidate_z_threshold"] == pytest.approx(
            expected["single_candidate_z_threshold"]
        )


def test_noise_band_arity_refusal_ignores_a_wide_but_ungradeable_sibling(tmp_path):
    """The refusal must not advise promoting a ruler that still cannot grade once promoted —
    the arity filter keys on `comparable`, so an unestimable wide bank has to fail that gate."""
    run_dir, _ = _write_run(tmp_path, columns=["a", "b", "c"], batch_extras=_banks(
        _bank(arity=1, n=20, seed=1),
        k3=_bank(arity=3, deltas_cpl_held=[0.02] * 20),   # wide enough, but zero-variance
    ))

    band = dt.build_facts(run_dir)["noise_band"]

    assert band["available"] is False
    assert band["reason_code"] == dt.NOISE_BAND_REFUSAL_ARITY
    assert "may already exist" not in band["reason"]
    assert "--arity 3" in band["reason"]


@pytest.mark.parametrize("overrides", [
    {"n_placebo_columns": "three"},
    {"deltas_cpl_held": ["not", "numbers"]},
    {"deltas_cpl_held": {"nested": "object"}},
    # review of PR #349: `_score_bank` (called for `effective_n_draws` via
    # `_resolve_effective_n_draws`) has to stay inside the same try as the coercions above —
    # it raises ValueError on a non-numeric stamp exactly like `float(bank["deltas_cpl_held"])`
    # does, and moving it outside the try (as an earlier revision of this PR did) let a junk
    # side-file raise straight out of build_facts instead of being skipped.
    {"effective_n_draws": "twenty"},
])
def test_noise_band_malformed_sibling_is_skipped_not_fatal(tmp_path, overrides):
    """A stray or half-written side-file matching noise_floor*.json must not stop build_facts
    producing a facts.json at all — the canonical floor here is perfectly good."""
    run_dir, _ = _write_run(tmp_path, columns=["a", "b"], batch_extras=_banks(
        _bank(arity=2, n=10, seed=1), junk=_bank(**overrides),
    ))

    facts = dt.build_facts(run_dir)  # must not raise
    band = facts["noise_band"]

    assert band["available"] is True
    assert band["candidate_z_vs_band"] is not None  # canonical grading unaffected
    junk = [b for b in band["comparable_banks"] if b["name"] == "noise_floor_junk.json"]
    assert len(junk) == 1
    assert junk[0]["comparable"] is False
    assert "unreadable" in junk[0]["reason"]


# ── promote-hint correctness + arity disclosure (peer review of PR #345) ──────

def test_arity_refusal_hint_omits_a_bank_the_canonical_check_would_refuse(tmp_path):
    """The hint recommends a `mv`. _comparable_noise_banks deliberately RELAXES null_method
    so a pinned-source bank's z can still be reported as corroboration — that relaxation
    must not leak into a promote recommendation, because _noise_band's own null_method check
    is unconditional and would refuse the promoted file on the very next run."""
    run_dir, _ = _write_run(tmp_path, columns=["a", "b", "c"], batch_extras=_banks(
        _bank(arity=1, n=20, seed=1),
        k3=_bank(arity=3, n=20, seed=2),
        icc=_bank(arity=3, n=20, seed=3, null_method="placebo_column_pinned_source"),
    ))

    reason = dt.build_facts(run_dir)["noise_band"]["reason"]

    assert "noise_floor_k3.json" in reason
    assert "noise_floor_icc.json" not in reason


def test_arity_refusal_hint_does_not_run_into_the_following_sentence(tmp_path):
    """Reproduced by review: the hint was concatenated with no separator, rendering as
    '...step (2) alone.noise_floor.json is a 1-column null...'. The message is read by an
    unattended dossier session and pasted into a README, so it has to be legible."""
    run_dir, _ = _write_run(tmp_path, columns=["a", "b", "c"], batch_extras=_banks(
        _bank(arity=1, n=20, seed=1), k3=_bank(arity=3, n=20, seed=2),
    ))

    reason = dt.build_facts(run_dir)["noise_band"]["reason"]

    assert "alone.noise_floor.json" not in reason
    assert "alone. noise_floor.json" in reason


@pytest.mark.parametrize("sibling_arity,expected", [(1, True), (2, False), (4, False)])
def test_comparable_bank_discloses_when_it_is_narrower_than_the_run(
    tmp_path, sibling_arity, expected
):
    """Siblings are not REFUSED on arity — a narrower bank is exactly what answers "would a
    different arity move this call" (fps-6yi was resolved by an arity-1 bank against a
    2-column run), so excluding it would delete the use case. Disclose instead, mirroring
    `floor_arity_exceeds_run` on the canonical side: a narrower bank's z is biased in the
    candidate's favour and is a lower bound on the bar, never a verdict."""
    run_dir, _ = _write_run(tmp_path, columns=["a", "b"], batch_extras=_banks(
        _bank(arity=2, n=10, seed=1), sib=_bank(arity=sibling_arity, n=20, seed=2),
    ))

    banks = dt.build_facts(run_dir)["noise_band"]["comparable_banks"]

    assert banks[0]["comparable"] is True
    assert banks[0]["narrower_than_run"] is expected


# ── timing regret: fps-6yi regression (fps-2js) ─────────────────────────────────

#: Real `daily_prices` rows (E10) for the five stations fps-6yi's flips touch, covering every
#: date the regret windows below walk — regenerated by the query in this file's sibling comment:
#:   SELECT price_date, price_decicents FROM daily_prices
#:    WHERE station_code=? AND fuel_type_id=(SELECT id FROM fuel_types WHERE code='E10')
#: The price DB itself is ~500 MB and gitignored, so the windows are committed instead of the
#: source. Verified against the batch's own frozen clone AND the live DB: all three produce
#: identical regret numbers for these flips.
#:
#: HORIZON-BOUNDED (PR #348 review): the rows stop at last_flip + 13 days, so this fixture can
#: only support H <= 13. Because `price_at` forward-fills, raising the horizon does NOT fail —
#: it silently carries the last committed price off the end of the CSV and the "matches the
#: real DB" property quietly stops holding. Regenerate the fixture before testing a larger H.
_FPS_6YI_PRICES_CSV = pathlib.Path(__file__).parent / "data" / "fps_6yi_flip_window_prices.csv"


class _CsvPrices:
    """`flips.StationPriceSource` over the committed fixture, with the same forward-fill
    `bisect_right(...) - 1` lookup `dossier_tables._DbStationPrices` uses against sqlite."""

    def __init__(self, path: pathlib.Path):
        frame = pd.read_csv(path, dtype={"station_code": int, "price_date": str})
        self._dates, self._prices = {}, {}
        for code, group in frame.groupby("station_code"):
            ordered = group.sort_values("price_date")
            self._dates[int(code)] = ordered["price_date"].tolist()
            self._prices[int(code)] = ordered["price_cents"].astype(float).tolist()

    def price_at(self, station_code, as_of):
        dates = self._dates.get(int(station_code))
        if not dates:
            return None
        idx = bisect.bisect_right(dates, as_of) - 1
        return self._prices[int(station_code)][idx] if idx >= 0 else None

    def is_observed(self, station_code, as_of):
        dates = self._dates.get(int(station_code)) or []
        idx = bisect.bisect_left(dates, as_of)
        return idx < len(dates) and dates[idx] == as_of


def _fps_6yi_flips() -> pd.DataFrame:
    """The 303 real flip rows, in diff_fills' own output shape."""
    return pd.DataFrame([
        {"fold": fold, "station_code": station, "date": date, "price": price,
         "litres": litres, "spend_cents": price * litres, "bought_by": bought_by}
        for fold, station, date, price, litres, bought_by in _FPS_6YI_FLIP_ROWS
    ])


def test_summarise_regret_reproduces_the_fps_6yi_prototype_at_its_own_horizon():
    """The prototype behind fps-2js published 6.22 / 6.21 c/L for a litres-weighted 7-day
    regret over all flips (stickiness_phase_saddle/README.md § 2). Pinning it here proves the
    shipped implementation is the SAME computation, not merely a similar-looking one — the
    shipped horizon differs (13, the tank's feasible wait) and this is the only assertion that
    ties the two together.
    """
    out = summarise_regret(
        _fps_6yi_flips(), _CsvPrices(_FPS_6YI_PRICES_CSV), SHOCK_FOLDS,
        horizon_days=7, cadence_days=1, window_days=7,
    )
    assert out["all"]["regret_cpl_baseline"] == pytest.approx(6.2242214, abs=1e-6)
    assert out["all"]["regret_cpl_candidate"] == pytest.approx(6.2089820, abs=1e-6)


def test_summarise_regret_reproduces_fps_6yi_at_the_shipped_horizon():
    """The full fps-6yi regret table at the shipped tank-derived horizon (13 days)."""
    out = summarise_regret(
        _fps_6yi_flips(), _CsvPrices(_FPS_6YI_PRICES_CSV), SHOCK_FOLDS,
        horizon_days=13, cadence_days=1, window_days=7,
    )

    # Every flip is scored: dark days are forward-filled, never dropped.
    assert out["n_scored"] == 303 and out["n_unscored"] == 0
    # The station 414 outage — non-random (it thins folds 4-7), reported even though
    # forward-filling means it no longer biases the estimate.
    assert out["dark_fill_days"] == 56
    assert out["dark_fill_folds"] == [4, 5, 6, 7]
    # The bead's own verification: every flip's executed price matches its DB price.
    assert out["n_price_mismatch"] == 0
    # Same cascade collapse the flip table reports for this run (fps-e1w).
    assert out["all"]["n_decisions"] == 88

    all_row = out["all"]
    assert all_row["regret_cpl_baseline"] == pytest.approx(10.0259515, abs=1e-6)
    assert all_row["regret_cpl_candidate"] == pytest.approx(9.3822754, abs=1e-6)
    assert all_row["regret_cpl_delta"] == pytest.approx(-0.6436761, abs=1e-6)
    # Cascade-widened, not the naive per-fill band: 2*SE = ±4.24, so even the pooled run-level
    # row cannot resolve a -0.64 difference. fps-2js is a legibility fix, not a power fix, and
    # this assertion is what stops it being sold as one.
    assert all_row["regret_cpl_delta_se"] == pytest.approx(2.1206768, abs=1e-6)
    assert all_row["regret_cpl_delta_inside_own_se"] is True

    # Fold 13 is the cell the whole rework exists for: its flip_cpl_delta prints -32.73 c/L
    # off a single candidate fill. Under regret that same fill reads 0.00 — the candidate
    # bought the exact optimum reachable from where it stood.
    fold13 = next(row for row in out["per_fold"] if row["fold"] == 13)
    assert fold13["n_baseline"] == 5 and fold13["n_candidate"] == 1
    assert fold13["regret_cpl_baseline"] == pytest.approx(5.7333333, abs=1e-6)
    assert fold13["regret_cpl_candidate"] == 0.0

    # And the honest headline: like flip_cpl_delta before it, no fold resolves its own delta.
    resolvable = [r for r in out["per_fold"] if r["regret_cpl_delta_inside_own_se"] is not None]
    assert len(resolvable) == 12
    assert all(r["regret_cpl_delta_inside_own_se"] for r in resolvable)


def _regret_fills() -> pd.DataFrame:
    return pd.DataFrame([
        {"fold": 1, "arm": BASELINE_ARM, "own_tau": 0.25, "date": "2026-01-01",
         "station_code": 100, "price": 180.0, "litres": 10.0, "spend_cents": 1800.0, "emergency": False},
        {"fold": 1, "arm": CANDIDATE_ARM, "own_tau": 0.25, "date": "2026-01-05",
         "station_code": 100, "price": 170.0, "litres": 10.0, "spend_cents": 1700.0, "emergency": False},
    ])


def test_attach_regret_without_a_freeze_manifest_says_so(tmp_path):
    out = dt._attach_regret(_regret_fills(), "50/3.571/1d/10%", 7, frozenset({4}), None)
    assert out["computed"] is False
    assert "freeze manifest" in out["reason"]


def test_attach_regret_with_an_absent_price_db_is_skipped_not_fatal(tmp_path):
    """The frozen price DB is gitignored and ~500 MB, so a checkout without it must still build
    every other table — this path is why regret is optional-with-a-reason rather than a raise."""
    (tmp_path / dt.FREEZE_MANIFEST_FILENAME).write_text(json.dumps({"source_db": "fuel_signal.db"}))
    out = dt._attach_regret(_regret_fills(), "50/3.571/1d/10%", 7, frozenset({4}), tmp_path)
    assert out["computed"] is False
    assert dt.FROZEN_DB_FILENAME in out["reason"] and "not present" in out["reason"]


def test_attach_regret_reads_the_batchs_frozen_db_not_freeze_jsons_source_db(tmp_path):
    """Regression (PR #348 review finding #1): `source_db` records where the freeze READ from
    at freeze time — a repo-root-relative path to the LIVE database, which keeps changing.
    The run was graded against the batch's frozen clone (`runner.py`'s `db_path`), so regret
    must replay that file; resolving `source_db` instead scored later, possibly revised prices
    and issued a WAL write against the live production DB on every dossier build.
    """
    (tmp_path / dt.FREEZE_MANIFEST_FILENAME).write_text(
        json.dumps({"source_db": "/somewhere/else/live.db"})
    )
    assert dt._resolve_source_db(tmp_path) == tmp_path / dt.FROZEN_DB_FILENAME
    # freeze.json's presence is still the gate: an unfrozen directory is not a batch.
    assert dt._resolve_source_db(tmp_path / "no_manifest_here") is None


def test_attach_regret_with_no_differing_fills_says_so(tmp_path):
    (tmp_path / dt.FROZEN_DB_FILENAME).touch()
    (tmp_path / dt.FREEZE_MANIFEST_FILENAME).write_text(json.dumps({"source_db": "fuel_signal.db"}))
    identical = _regret_fills().iloc[[0]]
    both = pd.concat([identical, identical.assign(arm=CANDIDATE_ARM)], ignore_index=True)
    out = dt._attach_regret(both, "50/3.571/1d/10%", 7, frozenset({4}), tmp_path)
    assert out["computed"] is False
    assert "no differing fills" in out["reason"]
