"""Tests for experiments/pipeline/launch.py — the nightly launch routine (fps-3jj.5).

Everything here is unit-testable without a real bd db or a real detached process:
subprocess.run/Popen are monkeypatched to record calls and return canned JSON. The
real cross-process detachment guarantee (launch_detached survives its caller
exiting) was verified manually — see fps-3jj.5's PR description, not reproduced here.
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
from datetime import datetime, timedelta, timezone

import pytest
from click.testing import CliRunner

import experiments.pipeline.launch as launch_module
from experiments.pipeline.launch import (
    RESULTS_FILENAME,
    RUN_LOG_FILENAME,
    STALE_AFTER,
    CandidateRefError,
    build_runner_cmd,
    claim_next_candidate,
    find_stale_claims,
    launch_detached,
    main,
    parse_candidate_ref,
    release_stale_claim,
)
from experiments.pipeline.runner import default_out_dir

# ── parse_candidate_ref ──────────────────────────────────────────────────────

def test_parse_candidate_ref_happy_path():
    description = (
        "## Candidate\n"
        "Batch: experiments/batches/2026-08-20_batch1\n"
        "Module: experiments/candidates/2026-08-20_batch1/tgp_delta_7d.py\n"
    )
    batch_dir, candidate_path = parse_candidate_ref(description)
    assert batch_dir.name == "2026-08-20_batch1"
    assert candidate_path.name == "tgp_delta_7d.py"
    assert candidate_path.parent.name == "2026-08-20_batch1"


@pytest.mark.parametrize("description", ["", "no structured fields here", "Batch: only-batch-line\n"])
def test_parse_candidate_ref_raises_on_malformed_description(description):
    with pytest.raises(CandidateRefError):
        parse_candidate_ref(description)


@pytest.mark.parametrize(
    "module_line",
    [
        "Module: experiments/candidates/batch1/../../../etc/passwd",
        "Module: /etc/passwd",
    ],
)
def test_parse_candidate_ref_rejects_path_outside_experiments(module_line):
    description = f"Batch: experiments/batches/batch1\n{module_line}\n"
    with pytest.raises(CandidateRefError, match="outside experiments/"):
        parse_candidate_ref(description)


# ── build_runner_cmd ─────────────────────────────────────────────────────────

def test_build_runner_cmd():
    batch_dir = pathlib.Path("batch1")
    cmd = build_runner_cmd(batch_dir, batch_dir / "c.py", "fps-x")
    assert cmd == [
        "uv", "run", "python", "-m", "experiments.pipeline.runner",
        "--batch-dir", "batch1",
        "--candidate", "batch1/c.py",
        "--bead-id", "fps-x",
    ]


# ── default_out_dir (fps-icv regression) ───────────────────────────────────────

def test_default_out_dir_gives_distinct_dirs_for_two_candidates_in_one_batch():
    batch_dir = pathlib.Path("experiments/candidates/batch1")
    candidate_a = batch_dir / "tgp_delta_7d.py"
    candidate_b = batch_dir / "another_candidate.py"

    out_dir_a = default_out_dir(candidate_a)
    out_dir_b = default_out_dir(candidate_b)

    assert out_dir_a != out_dir_b
    assert out_dir_a.parent == batch_dir
    assert out_dir_b.parent == batch_dir
    assert out_dir_a.name == "tgp_delta_7d"
    assert out_dir_b.name == "another_candidate"


# ── find_stale_claims (fault injection) ──────────────────────────────────────

def _issue(description: str, started_at: str) -> dict:
    return {"id": "fps-exp-1", "description": description, "started_at": started_at, "status": "in_progress"}


def _description_for(batch_rel: str, module_rel: str) -> str:
    return f"Batch: {batch_rel}\nModule: {module_rel}\n"


def _completed_process(stdout: str) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=["bd"], returncode=0, stdout=stdout, stderr="")


def _fake_repo_root(tmp_path, monkeypatch) -> pathlib.Path:
    """parse_candidate_ref enforces containment under EXPERIMENTS_ROOT, so
    fault-injection fixtures need a fake repo root, not an arbitrary tmp dir.
    """
    monkeypatch.setattr(launch_module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(launch_module, "EXPERIMENTS_ROOT", tmp_path / "experiments")
    return tmp_path


def test_find_stale_claims_flags_old_traceback_with_no_results(tmp_path, monkeypatch):
    repo_root = _fake_repo_root(tmp_path, monkeypatch)
    candidate_path = repo_root / "experiments" / "candidates" / "batch1" / "tgp_delta_7d.py"
    candidate_path.parent.mkdir(parents=True)
    candidate_path.write_text("NAME = 'x'")
    out_dir = default_out_dir(candidate_path)
    out_dir.mkdir(parents=True)
    (out_dir / RUN_LOG_FILENAME).write_text(
        "some output\nTraceback (most recent call last):\n  File \"x.py\", line 1\nValueError: boom\n"
    )
    description = _description_for(
        "experiments/batches/batch1", "experiments/candidates/batch1/tgp_delta_7d.py"
    )
    old_claim = (datetime.now(timezone.utc) - STALE_AFTER - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    issue = _issue(description, old_claim)

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _completed_process(json.dumps([issue])))

    stale = find_stale_claims()
    assert len(stale) == 1
    assert stale[0]["issue"]["id"] == "fps-exp-1"
    assert "ValueError: boom" in stale[0]["traceback_tail"]


def test_find_stale_claims_ignores_recent_claim(tmp_path, monkeypatch):
    repo_root = _fake_repo_root(tmp_path, monkeypatch)
    candidate_path = repo_root / "experiments" / "candidates" / "batch1" / "tgp_delta_7d.py"
    out_dir = default_out_dir(candidate_path)
    out_dir.mkdir(parents=True)
    (out_dir / RUN_LOG_FILENAME).write_text("Traceback (most recent call last):\nValueError: boom\n")
    description = _description_for(
        "experiments/batches/batch1", "experiments/candidates/batch1/tgp_delta_7d.py"
    )
    recent_claim = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    issue = _issue(description, recent_claim)

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _completed_process(json.dumps([issue])))

    assert find_stale_claims() == []


def test_find_stale_claims_ignores_completed_run(tmp_path, monkeypatch):
    repo_root = _fake_repo_root(tmp_path, monkeypatch)
    candidate_path = repo_root / "experiments" / "candidates" / "batch1" / "tgp_delta_7d.py"
    out_dir = default_out_dir(candidate_path)
    out_dir.mkdir(parents=True)
    (out_dir / RUN_LOG_FILENAME).write_text("Traceback (most recent call last):\nValueError: boom\n")
    (out_dir / RESULTS_FILENAME).write_text("{}")
    description = _description_for(
        "experiments/batches/batch1", "experiments/candidates/batch1/tgp_delta_7d.py"
    )
    old_claim = (datetime.now(timezone.utc) - STALE_AFTER - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    issue = _issue(description, old_claim)

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _completed_process(json.dumps([issue])))

    assert find_stale_claims() == []


def test_find_stale_claims_ignores_non_traceback_log(tmp_path, monkeypatch):
    repo_root = _fake_repo_root(tmp_path, monkeypatch)
    candidate_path = repo_root / "experiments" / "candidates" / "batch1" / "tgp_delta_7d.py"
    out_dir = default_out_dir(candidate_path)
    out_dir.mkdir(parents=True)
    (out_dir / RUN_LOG_FILENAME).write_text("still fitting fold 7...\n")
    description = _description_for(
        "experiments/batches/batch1", "experiments/candidates/batch1/tgp_delta_7d.py"
    )
    old_claim = (datetime.now(timezone.utc) - STALE_AFTER - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    issue = _issue(description, old_claim)

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _completed_process(json.dumps([issue])))

    assert find_stale_claims() == []


# ── release_stale_claim ───────────────────────────────────────────────────────

def test_release_stale_claim_comments_unassigns_and_reopens(monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return _completed_process("")

    monkeypatch.setattr(subprocess, "run", fake_run)
    release_stale_claim({"id": "fps-exp-1"}, "Traceback (most recent call last):\nValueError: boom")

    commands = [c[0] for c in calls]
    assert commands[0][:2] == ["bd", "comment"]
    assert "boom" in calls[0][1]["input"]
    assert commands[1] == ["bd", "assign", "fps-exp-1", ""]
    assert commands[2] == ["bd", "update", "fps-exp-1", "--status", "open"]


# ── claim_next_candidate ──────────────────────────────────────────────────────

def test_claim_next_candidate_returns_none_when_queue_empty(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _completed_process("[]"))
    assert claim_next_candidate() is None


def test_claim_next_candidate_returns_and_pushes_claimed_issue(monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[:2] == ["bd", "ready"]:
            return _completed_process(json.dumps([{"id": "fps-exp-2"}]))
        return _completed_process("")

    monkeypatch.setattr(subprocess, "run", fake_run)
    issue = claim_next_candidate()
    assert issue == {"id": "fps-exp-2"}
    assert ["bd", "dolt", "push"] in calls


# ── launch_detached ────────────────────────────────────────────────────────────

def test_launch_detached_starts_new_session_with_redirected_io(tmp_path, monkeypatch):
    captured = {}

    class FakeProcess:
        pid = 4242

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    out_dir = tmp_path / "out"
    pid = launch_detached(["echo", "hi"], out_dir)

    assert pid == 4242
    assert captured["kwargs"]["start_new_session"] is True
    assert captured["kwargs"]["stdout"].name == str(out_dir / RUN_LOG_FILENAME)
    assert captured["kwargs"]["stdin"].name == os.devnull
    assert captured["kwargs"]["stderr"] is subprocess.STDOUT
    assert captured["kwargs"]["cwd"] == launch_module.REPO_ROOT
    assert out_dir.exists()


# ── main() CLI orchestration ──────────────────────────────────────────────────

def test_main_exits_quietly_when_queue_empty(monkeypatch):
    monkeypatch.setattr(launch_module, "sync_pull", lambda: None)
    monkeypatch.setattr(launch_module, "recover_stale_claims", lambda: [])
    monkeypatch.setattr(launch_module, "claim_next_candidate", lambda: None)

    result = CliRunner().invoke(main, [])

    assert result.exit_code == 0
    assert "no experiment work ready" in result.output


def test_main_aborts_claim_on_malformed_candidate_reference(monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return _completed_process("")

    monkeypatch.setattr(launch_module, "sync_pull", lambda: None)
    monkeypatch.setattr(launch_module, "recover_stale_claims", lambda: [])
    monkeypatch.setattr(
        launch_module, "claim_next_candidate",
        lambda: {"id": "fps-exp-9", "description": "no structured fields here"},
    )
    monkeypatch.setattr(subprocess, "run", fake_run)

    result = CliRunner().invoke(main, [])

    assert result.exit_code == 0
    assert "aborted" in result.output
    assert ["bd", "assign", "fps-exp-9", ""] in calls
    assert ["bd", "update", "fps-exp-9", "--status", "open"] in calls
