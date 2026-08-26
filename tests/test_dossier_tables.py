"""Tests for experiments/pipeline/dossier_tables.py (fps-3jj.6).

No real batch/DB exists in this checkout (batches are local-only, gitignored — see the
parent design's "the pipeline is gated on the Mac being awake"), so these tests build
results.json/rowpreds.parquet/fills.parquet fixtures by hand, matching the exact schema
experiments/pipeline/runner.py (fps-3jj.4) actually writes, rather than running a live
run_candidate() end to end.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from experiments.lib.constants import ROW_AXIS_ECONOMICS_CAVEAT
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
                   tank_params: str | None = "50/3.571/7d/10%") -> dict:
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
        "effect_delta_cpl_held": -0.05 if status == "graded" else None,
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
        "meta": {
            "batch_dir": str(batch_dir), "seeds": [42, 43, 44], "realised_seed": 42,
            "n_windows": 4, "realised_wall_seconds": 12.3, "wall_seconds": 20.0,
            "pass_criterion": {"criterion": "sign plus concentration"},
            "extra_feature_provider_hits": 100, "extra_feature_provider_misses": 0,
            "git_sha": "abc1234", "bead_id": bead_id,
            "n_baseline_columns": 54, "baseline_fingerprint": "54:deadbeef1234",
            "tank_params": tank_params,
        },
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
        tank_params=tank_params,
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

    assert facts["grading"]["pending"] is True
    assert facts["grading"]["predicted_signature"] == "helps normal folds, not shock"

    assert facts["noise_band"]["available"] is False
    assert "fps-3jj.9" in facts["noise_band"]["reason"]


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
    row = next(r for r in flips["per_fold"] if r["fold"] == 1)
    assert row["regime"] == "shock"
    assert row["n_baseline_only"] == 1
    assert row["n_candidate_only"] == 1
    assert row["flip_cpl_baseline"] == pytest.approx(200.0)
    assert row["flip_cpl_candidate"] == pytest.approx(150.0)
    assert row["flip_cpl_delta"] == pytest.approx(-50.0)
    assert len(flips["rows"]) == 2


# ── plots ─────────────────────────────────────────────────────────────────────

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


def test_noise_band_refuses_a_candidate_wider_than_the_arity_cap(tmp_path):
    """Above `placebo.MAX_RULER_ARITY` there is no remediation to spell out, because no ruler
    can be built that wide: a draw needs `arity` distinct source columns, so wide draws are
    forced to overlap (two 35-column draws out of 49 columns must share 21 of them, by
    pigeonhole), their deltas stop being independent samples of the null, and the band's std
    understates the true spread — a bar that is too EASY. The message must say that instead of
    sending the reader to a command that would itself raise, and must still NOT read as a
    rejection (fps-3jj.17: the run completed and graded fine)."""
    wide = [f"c{i}" for i in range(dt.MAX_RULER_ARITY + 1)]
    run_dir, _ = _write_run(tmp_path, columns=wide, batch_extras=_floor_with(arity=1))

    band = dt.build_facts(run_dir)["noise_band"]

    assert band["available"] is False
    assert band["reason_code"] == dt.NOISE_BAND_REFUSAL_ARITY
    assert f"above MAX_RULER_ARITY ({dt.MAX_RULER_ARITY})" in band["reason"]
    assert "fps-3jj.22" in band["reason"]
    assert "NOT rejected" in band["reason"]
    # It must not hand over a recipe that cannot work.
    assert "--arity" not in band["reason"]


def test_noise_band_arity_cap_is_not_bypassed_by_an_equally_wide_floor(tmp_path):
    """Review finding (PR #335). The cap check sat INSIDE the `run_arity > floor_arity`
    branch, so a floor whose own arity already met or exceeded the run's — a pre-fps-3jj.21
    wide floor, or a hand-built one — slipped a 5+ column candidate straight through to a
    graded band the change declares meaningless. Precisely the case where a bad band already
    exists on disk and looks gradeable, so the guard must not depend on the floor at all."""
    wide = [f"c{i}" for i in range(dt.MAX_RULER_ARITY + 2)]
    run_dir, _ = _write_run(
        tmp_path, columns=wide, batch_extras=_floor_with(arity=len(wide))
    )

    band = dt.build_facts(run_dir)["noise_band"]

    assert band["available"] is False
    assert band["reason_code"] == dt.NOISE_BAND_REFUSAL_ARITY
    assert f"above MAX_RULER_ARITY ({dt.MAX_RULER_ARITY})" in band["reason"]


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
    assert thin_band["bar_draw_count_penalty_cpl_held"] < 0
    assert thin_band["bar_draw_count_penalty_cpl_held"] < thick_band["bar_draw_count_penalty_cpl_held"]


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
