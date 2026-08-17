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
import subprocess
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from experiments.pipeline import dossier_tables as dt
from experiments.pipeline.runner import BASELINE_ARM, CANDIDATE_ARM

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


def _make_results(batch_dir: pathlib.Path, *, status: str = "rejected", target=None,
                   inputs=None, columns=None, error=None, bead_id="fps-test.1") -> dict:
    return {
        "status": status,
        "candidate": {
            "name": "cand", "hypothesis": "test hypothesis",
            "predicted_signature": "helps normal folds, not shock",
            "confidence_effect": 0.6, "confidence_zone": 0.3,
            "columns": columns if columns is not None else ["cand_col"],
            "inputs": inputs if inputs is not None else ["price_date"],
            "prior_art": None, "target": target,
        },
        "effect_resolved": True if status == "rejected" else None,
        "effect_delta_cpl_held": -0.05 if status == "rejected" else None,
        "zone": (
            {"resolved": True, "target_delta_cpl_own": -0.1, "other_delta_cpl_own": 0.02}
            if status == "rejected" else {"resolved": None}
        ),
        "grading_error": None,
        "aggregate": [
            {"arm": BASELINE_ARM, "cpl_own": 4.20, "cpl_held": 4.25, "saving_own_pct": 3.0, "saving_held_pct": 2.8},
            {"arm": CANDIDATE_ARM, "cpl_own": 4.15, "cpl_held": 4.20, "saving_own_pct": 3.2, "saving_held_pct": 3.0},
        ] if status == "rejected" else None,
        "fold_run_deltas": _fold_run_deltas() if status == "rejected" else None,
        "seed_variance": {
            "summary": {"all": {"cohort_median_seed_std": 0.01, "n_cells": 8, "n_flagged_gt_5x": 1}},
            "flags": [
                {"cohort": "all", "fold": 4, "run": "candidate", "seed_std": 0.09, "ratio_vs_cohort_median": 9.0}
            ],
        } if status == "rejected" else {},
        "error": error,
        "meta": {
            "batch_dir": str(batch_dir), "seeds": [42, 43, 44], "realised_seed": 42,
            "n_windows": 4, "realised_wall_seconds": 12.3, "wall_seconds": 20.0,
            "pass_criterion": {"criterion": "sign plus concentration"},
            "extra_feature_provider_hits": 100, "extra_feature_provider_misses": 0,
            "git_sha": "abc1234", "bead_id": bead_id,
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
    rows = []
    for fold in FOLDS:
        for run in (BASELINE_ARM, CANDIDATE_ARM):
            for seed in (42, 43, 44):
                for i in range(N_DATES):
                    for station in range(4):
                        row = {
                            "fold": fold, "station_code": station, "price_date": DATES[i],
                            "label": int(rng.uniform() < 0.5), "is_hard25": int(i % 4 == 0),
                            "run": run, "seed": seed,
                            "proba": float(np.clip(rng.uniform(), 0.01, 0.99)),
                        }
                        if with_axis:
                            row["axis"] = "A" if station % 2 == 0 else "B"
                        if with_cycle:
                            row["cycle_pct_through"] = float(rng.uniform(0, 1.3))
                        if candidate_col:
                            row[candidate_col] = (
                                float("nan") if rng.uniform() < 0.05 else float(rng.normal())
                            )
                        rows.append(row)
    return pd.DataFrame(rows)


def _write_run(
    tmp_path, *, status="rejected", target=None, inputs=None, columns=None,
    with_axis=False, with_cycle=False, low_n_fold=None, batch_extras=None,
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
    results = _make_results(batch_dir, status=status, target=target, inputs=inputs, columns=columns)
    (run_dir / dt.RESULTS_FILENAME).write_text(json.dumps(results))
    if status == "rejected":
        rowpreds = _make_rowpreds(
            with_axis=with_axis, with_cycle=with_cycle, candidate_col=(columns or ["cand_col"])[0]
        )
        rowpreds.to_parquet(run_dir / dt.ROWPREDS_FILENAME)
        _make_fills(low_n_fold=low_n_fold).to_parquet(run_dir / dt.FILLS_FILENAME)
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


def test_find_stale_claims_flags_old_claim_without_results(tmp_path):
    run_dir = tmp_path / "stale_run"
    run_dir.mkdir()
    started = datetime.now(timezone.utc) - timedelta(hours=20)
    (run_dir / dt.CLAIM_FILENAME).write_text(
        json.dumps({"bead_id": "fps-x.1", "candidate": "cand", "started_at": started.isoformat()})
    )
    (run_dir / dt.RUN_LOG_FILENAME).write_text("...\nTraceback (most recent call last):\nValueError: boom\n")

    fresh_dir = tmp_path / "fresh_run"
    fresh_dir.mkdir()
    (fresh_dir / dt.CLAIM_FILENAME).write_text(
        json.dumps({"bead_id": "fps-x.2", "candidate": "cand2",
                    "started_at": datetime.now(timezone.utc).isoformat()})
    )

    done_dir = tmp_path / "done_run"
    done_dir.mkdir()
    (done_dir / dt.CLAIM_FILENAME).write_text(
        json.dumps({"bead_id": "fps-x.3", "candidate": "cand3", "started_at": started.isoformat()})
    )
    (done_dir / dt.RESULTS_FILENAME).write_text("{}")

    stale = dt.find_stale_claims(tmp_path)

    assert len(stale) == 1
    assert stale[0]["run_dir"] == run_dir
    assert stale[0]["bead_id"] == "fps-x.1"
    assert "ValueError" in stale[0]["traceback_tail"]


def test_release_stale_claim_invokes_bd_comment_assign_update(monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    entry = {
        "run_dir": pathlib.Path("/tmp/x"), "bead_id": "fps-x.1", "candidate": "cand",
        "started_at": "2026-08-15T00:00:00+00:00", "age_hours": 20.0, "traceback_tail": "boom",
    }

    dt.release_stale_claim(entry)

    assert calls[0][:2] == ["bd", "comment"]
    assert calls[1] == ["bd", "assign", "fps-x.1", ""]
    assert calls[2] == ["bd", "update", "fps-x.1", "--status", "open"]


def test_release_stale_claim_skips_when_no_bead_id(monkeypatch, capsys):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("should not shell out"))
    dt.release_stale_claim({"run_dir": pathlib.Path("/tmp/x"), "bead_id": None, "age_hours": 1, "started_at": "x"})
    assert "no bead_id" in capsys.readouterr().out


def test_release_stale_claim_dry_run_does_not_shell_out(monkeypatch, capsys):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("should not shell out"))
    dt.release_stale_claim(
        {"run_dir": pathlib.Path("/tmp/x"), "bead_id": "fps-x.1", "age_hours": 20.0,
         "started_at": "2026-08-15T00:00:00+00:00", "traceback_tail": "boom"},
        dry_run=True,
    )
    assert "DRY RUN" in capsys.readouterr().out


# ── facts.json ────────────────────────────────────────────────────────────────

def test_build_facts_rejected_run_has_all_blocks_and_suppresses_thin_cells(tmp_path):
    run_dir, batch_dir = _write_run(tmp_path, low_n_fold=1)

    facts = dt.build_facts(run_dir)

    assert facts["candidate"]["name"] == "cand"
    assert facts["provenance"]["batch"] == "batch1"
    assert facts["provenance"]["snapshot_date"] == "2026-08-15"
    assert facts["provenance"]["bead_id"] == "fps-test.1"
    assert facts["provenance"]["git_sha"] == "abc1234"
    assert facts["provenance"]["git_sha_is_run_time"] is True

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

    assert facts["seed_flags"] == [
        {"cohort": "all", "fold": 4, "run": "candidate", "seed_std": 0.09, "ratio_vs_cohort_median": 9.0}
    ]

    assert facts["validation"]["pit_test"].startswith("passed")
    assert facts["validation"]["candidate_column_nan_rate"]["cand_col"] > 0

    assert facts["grading"]["pending"] is True
    assert facts["grading"]["predicted_signature"] == "helps normal folds, not shock"

    assert facts["noise_band"]["available"] is False
    assert "fps-3jj.9" in facts["noise_band"]["reason"]


def test_build_facts_per_axis_present_when_add_axis_declared(tmp_path):
    run_dir, _ = _write_run(tmp_path, with_axis=True)

    facts = dt.build_facts(run_dir)

    per_axis = facts["breakdowns"]["per_axis"]
    assert per_axis is not None
    assert {row["axis"] for row in per_axis} <= {"A", "B"}
    assert all(row["n_fills"] >= 0 for row in per_axis)


def test_build_facts_non_rejected_status_sets_status_note_and_no_breakdowns(tmp_path):
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
    def add_noise_floor(batch_dir):
        deltas = list(np.random.default_rng(2).normal(0, 0.02, size=20))
        (batch_dir / dt.NOISE_FLOOR_FILENAME).write_text(json.dumps({"deltas_cpl_held": deltas}))

    run_dir, _ = _write_run(tmp_path, batch_extras=add_noise_floor)

    facts = dt.build_facts(run_dir)

    band = facts["noise_band"]
    assert band["available"] is True
    assert band["n_draws"] == 20
    assert band["candidate_delta_cpl_held"] == -0.05


# ── plots ─────────────────────────────────────────────────────────────────────

def test_make_plots_writes_four_always_plots(tmp_path):
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


def test_make_plots_returns_empty_when_run_did_not_score(tmp_path):
    run_dir, batch_dir = _write_run(tmp_path, status="aborted_candidate")
    facts = dt.build_facts(run_dir)

    assert dt.make_plots(run_dir, facts, batch_dir) == []


# ── orchestration ─────────────────────────────────────────────────────────────

def test_process_run_writes_facts_json_with_plots_list(tmp_path):
    run_dir, _ = _write_run(tmp_path)

    facts_path = dt.process_run(run_dir)

    assert facts_path == run_dir / dt.FACTS_FILENAME
    facts = json.loads(facts_path.read_text())
    assert "per_fold_delta_bars.png" in facts["plots"]
