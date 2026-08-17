"""Tests for experiments/pipeline/launch.py — the nightly launch routine (fps-3jj.5).

Everything here is unit-testable without a real bd db or a real detached process:
subprocess.run/Popen are monkeypatched to record calls and return canned JSON. The
real cross-process detachment guarantee (launch_detached survives its caller
exiting) was verified manually — see fps-3jj.5's PR description, not reproduced here.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
from datetime import datetime, timedelta, timezone

import pytest

from experiments.pipeline.launch import (
    RESULTS_FILENAME,
    RUN_LOG_FILENAME,
    STALE_AFTER,
    CandidateRefError,
    build_runner_cmd,
    claim_next_candidate,
    find_stale_claims,
    launch_detached,
    parse_candidate_ref,
    release_stale_claim,
)

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


# ── find_stale_claims (fault injection) ──────────────────────────────────────

def _issue(description: str, started_at: str) -> dict:
    return {"id": "fps-exp-1", "description": description, "started_at": started_at, "status": "in_progress"}


def _description_for(batch_dir, candidate_path) -> str:
    return f"Batch: {batch_dir}\nModule: {candidate_path}\n"


def _completed_process(stdout: str) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=["bd"], returncode=0, stdout=stdout, stderr="")


def test_find_stale_claims_flags_old_traceback_with_no_results(tmp_path, monkeypatch):
    out_dir = tmp_path / "candidates" / "batch1"
    out_dir.mkdir(parents=True)
    (out_dir / RUN_LOG_FILENAME).write_text(
        "some output\nTraceback (most recent call last):\n  File \"x.py\", line 1\nValueError: boom\n"
    )
    candidate_path = out_dir / "tgp_delta_7d.py"
    candidate_path.write_text("NAME = 'x'")
    description = _description_for(tmp_path / "batches" / "batch1", candidate_path)
    old_claim = (datetime.now(timezone.utc) - STALE_AFTER - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    issue = _issue(description, old_claim)

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _completed_process(json.dumps([issue])))

    stale = find_stale_claims()
    assert len(stale) == 1
    assert stale[0]["issue"]["id"] == "fps-exp-1"
    assert "ValueError: boom" in stale[0]["traceback_tail"]


def test_find_stale_claims_ignores_recent_claim(tmp_path, monkeypatch):
    out_dir = tmp_path / "candidates" / "batch1"
    out_dir.mkdir(parents=True)
    (out_dir / RUN_LOG_FILENAME).write_text("Traceback (most recent call last):\nValueError: boom\n")
    candidate_path = out_dir / "tgp_delta_7d.py"
    description = _description_for(tmp_path / "batches" / "batch1", candidate_path)
    recent_claim = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    issue = _issue(description, recent_claim)

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _completed_process(json.dumps([issue])))

    assert find_stale_claims() == []


def test_find_stale_claims_ignores_completed_run(tmp_path, monkeypatch):
    out_dir = tmp_path / "candidates" / "batch1"
    out_dir.mkdir(parents=True)
    (out_dir / RUN_LOG_FILENAME).write_text("Traceback (most recent call last):\nValueError: boom\n")
    (out_dir / RESULTS_FILENAME).write_text("{}")
    candidate_path = out_dir / "tgp_delta_7d.py"
    description = _description_for(tmp_path / "batches" / "batch1", candidate_path)
    old_claim = (datetime.now(timezone.utc) - STALE_AFTER - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    issue = _issue(description, old_claim)

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _completed_process(json.dumps([issue])))

    assert find_stale_claims() == []


def test_find_stale_claims_ignores_non_traceback_log(tmp_path, monkeypatch):
    out_dir = tmp_path / "candidates" / "batch1"
    out_dir.mkdir(parents=True)
    (out_dir / RUN_LOG_FILENAME).write_text("still fitting fold 7...\n")
    candidate_path = out_dir / "tgp_delta_7d.py"
    description = _description_for(tmp_path / "batches" / "batch1", candidate_path)
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
    assert out_dir.exists()
