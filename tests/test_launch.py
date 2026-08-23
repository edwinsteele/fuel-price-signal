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
    MAX_RETRIES,
    RESULTS_FILENAME,
    RETRY_COUNT_METADATA_KEY,
    RUN_LOG_FILENAME,
    STALE_AFTER,
    CandidateRefError,
    build_runner_cmd,
    claim_next_candidate,
    find_stale_claims,
    launch_detached,
    main,
    parse_candidate_ref,
    recover_stale_claims,
    release_stale_claim,
)
from experiments.pipeline.runner import RETRYABLE_STATUSES, default_out_dir

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

def _issue(description: str, started_at: str, *, metadata: dict | None = None) -> dict:
    issue = {"id": "fps-exp-1", "description": description, "started_at": started_at, "status": "in_progress"}
    if metadata is not None:
        issue["metadata"] = metadata
    return issue


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
    assert stale[0]["action"] == "release"
    assert stale[0]["retry_count"] == 1


def test_find_stale_claims_blocks_crashed_run_once_retry_budget_exhausted(tmp_path, monkeypatch):
    """fps-rtd PR #304 review finding #1: the crashed-mid-run (traceback) shape
    must be budgeted too, not just the RETRYABLE_STATUSES shape.

    STALE_AFTER only delays the FIRST release (it re-satisfies itself between
    any two nightly runs), so without this a candidate that crashes the same
    deterministic way every attempt re-wins `bd ready --sort oldest` forever
    -- the exact starvation fps-rtd was filed about, just reached through a
    traceback instead of a RETRYABLE_STATUSES result.
    """
    repo_root = _fake_repo_root(tmp_path, monkeypatch)
    candidate_path = repo_root / "experiments" / "candidates" / "batch1" / "tgp_delta_7d.py"
    out_dir = default_out_dir(candidate_path)
    out_dir.mkdir(parents=True)
    (out_dir / RUN_LOG_FILENAME).write_text("Traceback (most recent call last):\nValueError: boom\n")
    description = _description_for(
        "experiments/batches/batch1", "experiments/candidates/batch1/tgp_delta_7d.py"
    )
    old_claim = (datetime.now(timezone.utc) - STALE_AFTER - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    issue = _issue(description, old_claim, metadata={RETRY_COUNT_METADATA_KEY: MAX_RETRIES})

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _completed_process(json.dumps([issue])))

    stale = find_stale_claims()
    assert len(stale) == 1
    assert stale[0]["action"] == "block"
    assert "retry_count" not in stale[0]


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


@pytest.mark.parametrize("status", sorted(RETRYABLE_STATUSES))
def test_find_stale_claims_reclaims_retryable_abort(tmp_path, monkeypatch, status):
    """fps-g31: an aborted run leaves a results.json, which used to read as "done".

    fps-32p aborted with aborted_candidate at 22:07 and stayed in_progress
    indefinitely: results.json existing short-circuited the sweep, and there was
    no traceback for the fallback path to find either. A retryable status must
    release the claim, and must do so with no age gate -- results.json existing
    is proof the run is over.
    """
    repo_root = _fake_repo_root(tmp_path, monkeypatch)
    candidate_path = repo_root / "experiments" / "candidates" / "batch1" / "tgp_delta_7d.py"
    out_dir = default_out_dir(candidate_path)
    out_dir.mkdir(parents=True)
    (out_dir / RESULTS_FILENAME).write_text(json.dumps({"status": status, "error": "bad config"}))
    description = _description_for(
        "experiments/batches/batch1", "experiments/candidates/batch1/tgp_delta_7d.py"
    )
    recent_claim = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat().replace("+00:00", "Z")
    issue = _issue(description, recent_claim)

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _completed_process(json.dumps([issue])))

    stale = find_stale_claims()
    assert len(stale) == 1
    assert status in stale[0]["traceback_tail"]
    assert stale[0]["action"] == "release"
    assert stale[0]["retry_count"] == 1


@pytest.mark.parametrize("malformed_metadata", [["not", "a", "dict"], "also not a dict", -5])
def test_find_stale_claims_treats_malformed_metadata_as_zero_retries(tmp_path, monkeypatch, malformed_metadata):
    """_retry_count must stay total: a non-dict `metadata`, or a hand-edited
    negative retry_count, reads as 0 rather than raising or granting extra
    retries via `retry_count + 1` landing back at/below 0.
    """
    repo_root = _fake_repo_root(tmp_path, monkeypatch)
    candidate_path = repo_root / "experiments" / "candidates" / "batch1" / "tgp_delta_7d.py"
    out_dir = default_out_dir(candidate_path)
    out_dir.mkdir(parents=True)
    (out_dir / RESULTS_FILENAME).write_text(json.dumps({"status": "aborted_pipeline", "error": "bad config"}))
    description = _description_for(
        "experiments/batches/batch1", "experiments/candidates/batch1/tgp_delta_7d.py"
    )
    recent_claim = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat().replace("+00:00", "Z")
    issue = _issue(description, recent_claim)
    # metadata isn't always a dict of {key: int} on the wire -- inject the
    # malformed shape directly rather than through the metadata= kwarg, which
    # only ever builds well-formed dicts.
    metadata_value = (
        {RETRY_COUNT_METADATA_KEY: malformed_metadata} if not isinstance(malformed_metadata, (list, str))
        else malformed_metadata
    )
    issue["metadata"] = metadata_value

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _completed_process(json.dumps([issue])))

    stale = find_stale_claims()
    assert len(stale) == 1
    assert stale[0]["action"] == "release"
    assert stale[0]["retry_count"] == 1


def test_find_stale_claims_honours_numeric_string_retry_count(tmp_path, monkeypatch):
    """fps-rtd PR #304 review finding #5: a retry_count stored as a numeric
    STRING (possible via the `--metadata` JSON form, unlike `--set-metadata`
    which always writes a real int) must be honoured as its int value, not
    zeroed like a genuinely non-numeric value -- see _retry_count's docstring.
    """
    repo_root = _fake_repo_root(tmp_path, monkeypatch)
    candidate_path = repo_root / "experiments" / "candidates" / "batch1" / "tgp_delta_7d.py"
    out_dir = default_out_dir(candidate_path)
    out_dir.mkdir(parents=True)
    (out_dir / RESULTS_FILENAME).write_text(json.dumps({"status": "aborted_pipeline", "error": "bad config"}))
    description = _description_for(
        "experiments/batches/batch1", "experiments/candidates/batch1/tgp_delta_7d.py"
    )
    recent_claim = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat().replace("+00:00", "Z")
    issue = _issue(description, recent_claim, metadata={RETRY_COUNT_METADATA_KEY: str(MAX_RETRIES)})

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _completed_process(json.dumps([issue])))

    stale = find_stale_claims()
    assert len(stale) == 1
    assert stale[0]["action"] == "block"  # "1" parses to 1, same budget outcome as int 1


def test_find_stale_claims_blocks_once_retry_budget_exhausted(tmp_path, monkeypatch):
    """fps-rtd: a second retryable abort of the same claim must block, not release.

    A released-but-unassigned issue keeps its original bd creation date, so an
    unbounded release always wins `bd ready --sort oldest` and starves the rest
    of the queue forever. MAX_RETRIES bounds this.
    """
    repo_root = _fake_repo_root(tmp_path, monkeypatch)
    candidate_path = repo_root / "experiments" / "candidates" / "batch1" / "tgp_delta_7d.py"
    out_dir = default_out_dir(candidate_path)
    out_dir.mkdir(parents=True)
    (out_dir / RESULTS_FILENAME).write_text(json.dumps({"status": "aborted_pipeline", "error": "bad config"}))
    description = _description_for(
        "experiments/batches/batch1", "experiments/candidates/batch1/tgp_delta_7d.py"
    )
    recent_claim = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat().replace("+00:00", "Z")
    issue = _issue(description, recent_claim, metadata={RETRY_COUNT_METADATA_KEY: MAX_RETRIES})

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _completed_process(json.dumps([issue])))

    stale = find_stale_claims()
    assert len(stale) == 1
    assert stale[0]["action"] == "block"
    assert "budget" in stale[0]["traceback_tail"]
    assert "retry_count" not in stale[0]


@pytest.mark.parametrize("status", ["graded", "disqualified", "aborted_candidate"])
def test_find_stale_claims_leaves_terminal_status_alone(tmp_path, monkeypatch, status):
    """A verdict legitimately consumes the claim -- only retryable aborts come back."""
    repo_root = _fake_repo_root(tmp_path, monkeypatch)
    candidate_path = repo_root / "experiments" / "candidates" / "batch1" / "tgp_delta_7d.py"
    out_dir = default_out_dir(candidate_path)
    out_dir.mkdir(parents=True)
    (out_dir / RESULTS_FILENAME).write_text(json.dumps({"status": status}))
    description = _description_for(
        "experiments/batches/batch1", "experiments/candidates/batch1/tgp_delta_7d.py"
    )
    old_claim = (datetime.now(timezone.utc) - STALE_AFTER - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    issue = _issue(description, old_claim)

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _completed_process(json.dumps([issue])))

    assert find_stale_claims() == []


@pytest.mark.parametrize("status", ["graded", "disqualified", "aborted_candidate"])
def test_find_stale_claims_resets_retry_metadata_on_terminal_verdict_with_spent_budget(
    tmp_path, monkeypatch, status,
):
    """fps-rtd PR #304 review finding #3: a spent retry_count must not outlive
    its cycle. Without this, a candidate that spent its one retry and then
    eventually succeeded would be born already at budget the next time a
    human manually re-queues this SAME bead (e.g. against a re-frozen batch)
    -- its first retryable abort in that later, unrelated cycle would block
    immediately instead of getting the one retry the docstring promises.
    """
    repo_root = _fake_repo_root(tmp_path, monkeypatch)
    candidate_path = repo_root / "experiments" / "candidates" / "batch1" / "tgp_delta_7d.py"
    out_dir = default_out_dir(candidate_path)
    out_dir.mkdir(parents=True)
    (out_dir / RESULTS_FILENAME).write_text(json.dumps({"status": status}))
    description = _description_for(
        "experiments/batches/batch1", "experiments/candidates/batch1/tgp_delta_7d.py"
    )
    old_claim = (datetime.now(timezone.utc) - STALE_AFTER - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    issue = _issue(description, old_claim, metadata={RETRY_COUNT_METADATA_KEY: 1})

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _completed_process(json.dumps([issue])))

    stale = find_stale_claims()
    assert len(stale) == 1
    assert stale[0]["action"] == "reset_retry"


@pytest.mark.parametrize("written", ["{ truncated mid-write", "[]", "null", '"a bare string"'])
def test_find_stale_claims_ignores_unparseable_results_json(tmp_path, monkeypatch, written):
    """Releasing a claim on the strength of a file we couldn't read is the worse guess.

    The non-object cases are valid JSON: a bare `.get("status")` on them raises
    AttributeError, which is uncaught and would take down the entire nightly
    sweep rather than skipping one bad run.
    """
    repo_root = _fake_repo_root(tmp_path, monkeypatch)
    candidate_path = repo_root / "experiments" / "candidates" / "batch1" / "tgp_delta_7d.py"
    out_dir = default_out_dir(candidate_path)
    out_dir.mkdir(parents=True)
    (out_dir / RESULTS_FILENAME).write_text(written)
    description = _description_for(
        "experiments/batches/batch1", "experiments/candidates/batch1/tgp_delta_7d.py"
    )
    old_claim = (datetime.now(timezone.utc) - STALE_AFTER - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    issue = _issue(description, old_claim)

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
    assert len(commands) == 3  # no --set-metadata call when retry_count isn't given


def test_release_stale_claim_records_retry_count_metadata(monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return _completed_process("")

    monkeypatch.setattr(subprocess, "run", fake_run)
    release_stale_claim({"id": "fps-exp-1"}, "retryable abort", retry_count=1)

    assert calls == [
        ["bd", "comment", "fps-exp-1", "--stdin"],
        ["bd", "assign", "fps-exp-1", ""],
        ["bd", "update", "fps-exp-1", "--set-metadata", f"{RETRY_COUNT_METADATA_KEY}=1"],
        ["bd", "update", "fps-exp-1", "--status", "open"],
    ]


def test_release_stale_claim_writes_metadata_before_status_open(monkeypatch):
    """fps-rtd PR #304 review finding #2: if a `bd` call partway through this
    sequence fails, the safer stuck state is "looks already-retried" (blocks
    one cycle early) rather than "counter never advanced" (the same fault
    gets released and retried forever) -- so --set-metadata must land before
    --status open, not after.
    """
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return _completed_process("")

    monkeypatch.setattr(subprocess, "run", fake_run)
    release_stale_claim({"id": "fps-exp-1"}, "retryable abort", retry_count=1)

    metadata_idx = calls.index(["bd", "update", "fps-exp-1", "--set-metadata", f"{RETRY_COUNT_METADATA_KEY}=1"])
    status_idx = calls.index(["bd", "update", "fps-exp-1", "--status", "open"])
    assert metadata_idx < status_idx


def test_block_exhausted_claim_comments_unassigns_and_blocks(monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return _completed_process("")

    monkeypatch.setattr(subprocess, "run", fake_run)
    launch_module.block_exhausted_claim({"id": "fps-exp-1"}, "retry budget exhausted")

    commands = [c[0] for c in calls]
    assert commands[0][:2] == ["bd", "comment"]
    assert "exhausted" in calls[0][1]["input"]
    assert commands[1] == ["bd", "assign", "fps-exp-1", ""]
    assert commands[2] == ["bd", "update", "fps-exp-1", "--status", "blocked"]


def test_clear_retry_metadata_unsets_metadata_without_touching_status(monkeypatch):
    """fps-rtd PR #304 review finding #3."""
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return _completed_process("")

    monkeypatch.setattr(subprocess, "run", fake_run)
    launch_module.clear_retry_metadata({"id": "fps-exp-1"}, "terminal verdict, spent retry")

    commands = [c[0] for c in calls]
    assert commands[0][:2] == ["bd", "comment"]
    assert commands[1] == ["bd", "update", "fps-exp-1", "--unset-metadata", RETRY_COUNT_METADATA_KEY]
    assert not any(cmd[:2] == ["bd", "assign"] or "--status" in cmd for cmd in commands)


# ── _abort_claim: pre-launch validation failures share the budget (fps-rtd PR #304 review finding #4) ──

def test_abort_claim_releases_with_retry_count_on_first_failure(monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return _completed_process("")

    monkeypatch.setattr(subprocess, "run", fake_run)
    launch_module._abort_claim({"id": "fps-exp-9"}, "malformed candidate reference: boom")

    assert ["bd", "update", "fps-exp-9", "--set-metadata", f"{RETRY_COUNT_METADATA_KEY}=1"] in calls
    assert ["bd", "update", "fps-exp-9", "--status", "open"] in calls
    assert ["bd", "update", "fps-exp-9", "--status", "blocked"] not in calls
    assert ["bd", "dolt", "push"] in calls


def test_abort_claim_blocks_once_retry_budget_exhausted(monkeypatch):
    """A candidate whose description/module is broken in a way that will never
    self-correct must not burn the nightly slot forever either -- same
    starvation shape as the runtime abort paths, just reached through
    pre-launch validation.
    """
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return _completed_process("")

    monkeypatch.setattr(subprocess, "run", fake_run)
    issue = {"id": "fps-exp-9", "metadata": {RETRY_COUNT_METADATA_KEY: MAX_RETRIES}}
    launch_module._abort_claim(issue, "malformed candidate reference: boom")

    assert ["bd", "update", "fps-exp-9", "--status", "open"] not in calls
    assert ["bd", "update", "fps-exp-9", "--status", "blocked"] in calls


# ── recover_stale_claims: retry budget end-to-end (fps-rtd) ──────────────────

def test_recover_stale_claims_second_consecutive_retryable_completion_does_not_release(
    tmp_path, monkeypatch,
):
    """Acceptance criterion: two consecutive retryable completions of the same
    claim; the second does not release.
    """
    repo_root = _fake_repo_root(tmp_path, monkeypatch)
    candidate_path = repo_root / "experiments" / "candidates" / "batch1" / "tgp_delta_7d.py"
    out_dir = default_out_dir(candidate_path)
    out_dir.mkdir(parents=True)
    (out_dir / RESULTS_FILENAME).write_text(json.dumps({"status": "aborted_pipeline", "error": "bad config"}))
    description = _description_for(
        "experiments/batches/batch1", "experiments/candidates/batch1/tgp_delta_7d.py"
    )
    recent_claim = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat().replace("+00:00", "Z")

    # First completion: fresh claim, no prior retries.
    issue_round_1 = _issue(description, recent_claim)
    calls = []

    def fake_run_round_1(cmd, **kwargs):
        calls.append(cmd)
        if cmd[:2] == ["bd", "list"]:
            return _completed_process(json.dumps([issue_round_1]))
        return _completed_process("")

    monkeypatch.setattr(subprocess, "run", fake_run_round_1)
    recover_stale_claims()
    assert ["bd", "update", "fps-exp-1", "--status", "open"] in calls
    assert ["bd", "update", "fps-exp-1", "--set-metadata", f"{RETRY_COUNT_METADATA_KEY}=1"] in calls

    # Second completion: the claim now carries retry_count=1 (budget spent).
    issue_round_2 = _issue(description, recent_claim, metadata={RETRY_COUNT_METADATA_KEY: 1})
    calls.clear()

    def fake_run_round_2(cmd, **kwargs):
        calls.append(cmd)
        if cmd[:2] == ["bd", "list"]:
            return _completed_process(json.dumps([issue_round_2]))
        return _completed_process("")

    monkeypatch.setattr(subprocess, "run", fake_run_round_2)
    recover_stale_claims()

    assert ["bd", "update", "fps-exp-1", "--status", "open"] not in calls
    assert ["bd", "update", "fps-exp-1", "--status", "blocked"] in calls


def test_exhausted_candidate_does_not_prevent_a_second_candidate_from_being_claimed(
    tmp_path, monkeypatch,
):
    """Acceptance criterion: a retryably-failing candidate does not prevent a
    second queued candidate from being claimed.

    Once fps-exp-1's retry budget is spent, recover_stale_claims blocks it
    (status=blocked) rather than reopening it -- it never issues the
    `--status open` call that would put it back in front of fps-exp-2 in
    `bd ready`'s oldest-first ordering. claim_next_candidate then goes on to
    claim fps-exp-2, as it would once bd's own `ready` query stops returning
    the blocked issue.
    """
    repo_root = _fake_repo_root(tmp_path, monkeypatch)
    candidate_path = repo_root / "experiments" / "candidates" / "batch1" / "tgp_delta_7d.py"
    out_dir = default_out_dir(candidate_path)
    out_dir.mkdir(parents=True)
    (out_dir / RESULTS_FILENAME).write_text(json.dumps({"status": "aborted_environment", "error": "disk full"}))
    description = _description_for(
        "experiments/batches/batch1", "experiments/candidates/batch1/tgp_delta_7d.py"
    )
    recent_claim = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat().replace("+00:00", "Z")
    exhausted_issue = _issue(description, recent_claim, metadata={RETRY_COUNT_METADATA_KEY: MAX_RETRIES})

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[:2] == ["bd", "list"]:
            return _completed_process(json.dumps([exhausted_issue]))
        if cmd[:2] == ["bd", "ready"]:
            return _completed_process(json.dumps([{"id": "fps-exp-2"}]))
        return _completed_process("")

    monkeypatch.setattr(subprocess, "run", fake_run)

    recover_stale_claims()
    assert ["bd", "update", "fps-exp-1", "--status", "open"] not in calls
    assert ["bd", "update", "fps-exp-1", "--status", "blocked"] in calls

    claimed = claim_next_candidate()
    assert claimed == {"id": "fps-exp-2"}


def test_recover_stale_claims_dispatches_reset_retry_action(tmp_path, monkeypatch):
    """fps-rtd PR #304 review finding #3, end to end through recover_stale_claims."""
    repo_root = _fake_repo_root(tmp_path, monkeypatch)
    candidate_path = repo_root / "experiments" / "candidates" / "batch1" / "tgp_delta_7d.py"
    out_dir = default_out_dir(candidate_path)
    out_dir.mkdir(parents=True)
    (out_dir / RESULTS_FILENAME).write_text(json.dumps({"status": "graded"}))
    description = _description_for(
        "experiments/batches/batch1", "experiments/candidates/batch1/tgp_delta_7d.py"
    )
    old_claim = (datetime.now(timezone.utc) - STALE_AFTER - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    issue = _issue(description, old_claim, metadata={RETRY_COUNT_METADATA_KEY: 1})

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[:2] == ["bd", "list"]:
            return _completed_process(json.dumps([issue]))
        return _completed_process("")

    monkeypatch.setattr(subprocess, "run", fake_run)

    recover_stale_claims()
    assert ["bd", "update", "fps-exp-1", "--unset-metadata", RETRY_COUNT_METADATA_KEY] in calls
    assert not any(
        cmd[:2] == ["bd", "assign"] or (cmd[:2] == ["bd", "update"] and "--status" in cmd) for cmd in calls
    )


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
