"""Tests for experiments/pipeline/launch.py — the nightly launch routine (fps-3jj.5).

Everything here is unit-testable without touching GitHub or spawning a real detached
process: subprocess.run/Popen are monkeypatched to record calls and return canned
`gh --json` output. The real cross-process detachment guarantee (launch_detached
survives its caller exiting) was verified manually — see fps-3jj.5's PR description,
not reproduced here.

Ported from bd to `gh` in the 2026-09 tracker cutover. The three state-model changes
that mattered — assignee-is-the-claim, `blocked` as a non-exclusive label, and
`retry_count` as the boolean `retried` label — each have their own test below, because
each is a place where the old bd semantics were free and the GitHub ones have to be
written explicitly.
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
    BLOCKED_LABEL,
    EXPERIMENT_LABEL,
    MAX_RETRIES,
    RESULTS_FILENAME,
    RETRIED_LABEL,
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
    body = (
        "## Candidate\n"
        "Batch: experiments/batches/2026-08-20_batch1\n"
        "Module: experiments/candidates/2026-08-20_batch1/tgp_delta_7d.py\n"
    )
    batch_dir, candidate_path = parse_candidate_ref(body)
    assert batch_dir.name == "2026-08-20_batch1"
    assert candidate_path.name == "tgp_delta_7d.py"
    assert candidate_path.parent.name == "2026-08-20_batch1"


@pytest.mark.parametrize("body", ["", "no structured fields here", "Batch: only-batch-line\n"])
def test_parse_candidate_ref_raises_on_malformed_body(body):
    with pytest.raises(CandidateRefError):
        parse_candidate_ref(body)


@pytest.mark.parametrize(
    "module_line",
    [
        "Module: experiments/candidates/batch1/../../../etc/passwd",
        "Module: /etc/passwd",
    ],
)
def test_parse_candidate_ref_rejects_path_outside_experiments(module_line):
    body = f"Batch: experiments/batches/batch1\n{module_line}\n"
    with pytest.raises(CandidateRefError, match="outside experiments/"):
        parse_candidate_ref(body)


# ── build_runner_cmd ─────────────────────────────────────────────────────────

def test_build_runner_cmd():
    batch_dir = pathlib.Path("batch1")
    cmd = build_runner_cmd(batch_dir, batch_dir / "c.py", "42")
    assert cmd == [
        "uv", "run", "python", "-m", "experiments.pipeline.runner",
        "--batch-dir", "batch1",
        "--candidate", "batch1/c.py",
        "--bead-id", "42",
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


# ── fixtures ─────────────────────────────────────────────────────────────────

def _issue(
    body: str,
    updated_at: str,
    *,
    number: int = 1,
    retried: bool = False,
    blocked: bool = False,
    assigned: bool = True,
    created_at: str = "2026-09-01T00:00:00Z",
) -> dict:
    """A `gh issue list --json number,title,body,labels,assignees,createdAt,updatedAt` row.

    Defaults to a live claim: assigned (the claim), `experiment`-labelled, not blocked,
    retry budget unspent.
    """
    names = [EXPERIMENT_LABEL]
    if retried:
        names.append(RETRIED_LABEL)
    if blocked:
        names.append(BLOCKED_LABEL)
    return {
        "number": number,
        "title": f"candidate {number}",
        "body": body,
        "labels": [{"name": name} for name in names],
        "assignees": [{"login": "edwinsteele"}] if assigned else [],
        "createdAt": created_at,
        "updatedAt": updated_at,
    }


def _body_for(batch_rel: str, module_rel: str) -> str:
    return f"Batch: {batch_rel}\nModule: {module_rel}\n"


def _completed_process(stdout: str) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=["gh"], returncode=0, stdout=stdout, stderr="")


def _listing(*issues: dict):
    """A subprocess.run stand-in that answers `gh issue list` with `issues` and every
    other `gh` call with empty output."""
    def fake_run(cmd, **kwargs):
        if cmd[:3] == ["gh", "issue", "list"]:
            return _completed_process(json.dumps(list(issues)))
        return _completed_process("")
    return fake_run


def _recording(*issues: dict):
    """As _listing, but also returns the list every command was appended to."""
    calls: list[list[str]] = []
    inner = _listing(*issues)

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return inner(cmd, **kwargs)
    return fake_run, calls


def _fake_repo_root(tmp_path, monkeypatch) -> pathlib.Path:
    """parse_candidate_ref enforces containment under EXPERIMENTS_ROOT, so
    fault-injection fixtures need a fake repo root, not an arbitrary tmp dir.
    """
    monkeypatch.setattr(launch_module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(launch_module, "EXPERIMENTS_ROOT", tmp_path / "experiments")
    return tmp_path


def _stale_fixture(tmp_path, monkeypatch, *, results: dict | str | None = None, run_log: str | None = None):
    """Lay down a candidate's out_dir with the given artifacts; return its body line pair."""
    repo_root = _fake_repo_root(tmp_path, monkeypatch)
    candidate_path = repo_root / "experiments" / "candidates" / "batch1" / "tgp_delta_7d.py"
    out_dir = default_out_dir(candidate_path)
    out_dir.mkdir(parents=True)
    if run_log is not None:
        (out_dir / RUN_LOG_FILENAME).write_text(run_log)
    if results is not None:
        (out_dir / RESULTS_FILENAME).write_text(
            results if isinstance(results, str) else json.dumps(results)
        )
    return _body_for("experiments/batches/batch1", "experiments/candidates/batch1/tgp_delta_7d.py")


_TRACEBACK = "some output\nTraceback (most recent call last):\n  File \"x.py\", line 1\nValueError: boom\n"


def _stamp(delta: timedelta) -> str:
    return (datetime.now(timezone.utc) - delta).isoformat().replace("+00:00", "Z")


_OLD = STALE_AFTER + timedelta(hours=1)
_RECENT = timedelta(minutes=5)


# ── find_stale_claims (fault injection) ──────────────────────────────────────

def test_find_stale_claims_flags_old_traceback_with_no_results(tmp_path, monkeypatch):
    body = _stale_fixture(tmp_path, monkeypatch, run_log=_TRACEBACK)
    monkeypatch.setattr(subprocess, "run", _listing(_issue(body, _stamp(_OLD))))

    stale = find_stale_claims()
    assert len(stale) == 1
    assert stale[0]["issue"]["number"] == 1
    assert "ValueError: boom" in stale[0]["traceback_tail"]
    assert stale[0]["action"] == "release"
    assert stale[0]["retry_count"] == 1


def test_find_stale_claims_blocks_crashed_run_once_retry_budget_exhausted(tmp_path, monkeypatch):
    """fps-rtd PR #304 review finding #1: the crashed-mid-run (traceback) shape
    must be budgeted too, not just the RETRYABLE_STATUSES shape.

    STALE_AFTER only delays the FIRST release (it re-satisfies itself between
    any two nightly runs), so without this a candidate that crashes the same
    deterministic way every attempt re-wins the oldest-first claim query forever
    -- the exact starvation fps-rtd was filed about, just reached through a
    traceback instead of a RETRYABLE_STATUSES result.
    """
    body = _stale_fixture(tmp_path, monkeypatch, run_log=_TRACEBACK)
    monkeypatch.setattr(subprocess, "run", _listing(_issue(body, _stamp(_OLD), retried=True)))

    stale = find_stale_claims()
    assert len(stale) == 1
    assert stale[0]["action"] == "block"
    assert "retry_count" not in stale[0]


def test_find_stale_claims_ignores_recent_claim(tmp_path, monkeypatch):
    body = _stale_fixture(tmp_path, monkeypatch, run_log=_TRACEBACK)
    monkeypatch.setattr(subprocess, "run", _listing(_issue(body, _stamp(timedelta(hours=1)))))

    assert find_stale_claims() == []


@pytest.mark.parametrize("status", sorted(RETRYABLE_STATUSES))
def test_find_stale_claims_reclaims_retryable_abort(tmp_path, monkeypatch, status):
    """fps-g31: an aborted run leaves a results.json, which used to read as "done".

    fps-32p aborted with aborted_candidate at 22:07 and stayed claimed
    indefinitely: results.json existing short-circuited the sweep, and there was
    no traceback for the fallback path to find either. A retryable status must
    release the claim, and must do so with no age gate -- results.json existing
    is proof the run is over.
    """
    body = _stale_fixture(tmp_path, monkeypatch, results={"status": status, "error": "bad config"})
    monkeypatch.setattr(subprocess, "run", _listing(_issue(body, _stamp(_RECENT))))

    stale = find_stale_claims()
    assert len(stale) == 1
    assert status in stale[0]["traceback_tail"]
    assert stale[0]["action"] == "release"
    assert stale[0]["retry_count"] == 1


def test_find_stale_claims_skips_unassigned_issue(tmp_path, monkeypatch):
    """Assignee IS the claim (see module docstring). bd's `--status in_progress`
    filtered the sweep for free; on GitHub an unassigned issue is a queued
    candidate nobody has started, and touching it would release a claim that was
    never made -- and, with a retryable results.json from a PREVIOUS cycle still
    on disk, would burn its retry budget on the spot.
    """
    body = _stale_fixture(tmp_path, monkeypatch, results={"status": "aborted_pipeline"})
    monkeypatch.setattr(subprocess, "run", _listing(_issue(body, _stamp(_RECENT), assigned=False)))

    assert find_stale_claims() == []


def test_find_stale_claims_skips_blocked_issue(tmp_path, monkeypatch):
    """`blocked` was an exclusive bd STATUS and is now a non-exclusive GitHub LABEL.

    A blocked issue stays assigned on purpose -- the assignment is what keeps it out
    of the claim queue. So without this filter it looks exactly like a live claim
    with a retryable results.json, and the sweep would re-examine (and re-comment on)
    it every single night forever.
    """
    body = _stale_fixture(tmp_path, monkeypatch, results={"status": "aborted_pipeline"})
    monkeypatch.setattr(
        subprocess, "run", _listing(_issue(body, _stamp(_RECENT), blocked=True, retried=True)),
    )

    assert find_stale_claims() == []


def test_retry_count_reads_the_retried_label(tmp_path, monkeypatch):
    """The retry counter is the `retried` label, present or absent.

    Under bd it was a free-form metadata integer, and _retry_count had to defend
    against a non-dict `metadata`, a non-numeric string, `None`, a numeric string
    (honoured -- PR #304 review finding #5), and a hand-edited negative value that
    would have offset the `+ 1` in _decide_release_or_block. A label has no such
    states: every one of those cases is now unrepresentable rather than handled, and
    the tests that covered them are gone with the failure modes. What remains worth
    asserting is that the label is the counter, in both directions.
    """
    body = _stale_fixture(tmp_path, monkeypatch, results={"status": "aborted_pipeline"})

    monkeypatch.setattr(subprocess, "run", _listing(_issue(body, _stamp(_RECENT), retried=False)))
    assert find_stale_claims()[0]["action"] == "release"

    monkeypatch.setattr(subprocess, "run", _listing(_issue(body, _stamp(_RECENT), retried=True)))
    assert find_stale_claims()[0]["action"] == "block"


def test_find_stale_claims_blocks_once_retry_budget_exhausted(tmp_path, monkeypatch):
    """fps-rtd: a second retryable abort of the same claim must block, not release.

    A released-but-unassigned issue keeps its original creation date, so an
    unbounded release always wins the oldest-first claim query and starves the rest
    of the queue forever. MAX_RETRIES bounds this.
    """
    body = _stale_fixture(tmp_path, monkeypatch, results={"status": "aborted_pipeline", "error": "bad config"})
    monkeypatch.setattr(subprocess, "run", _listing(_issue(body, _stamp(_RECENT), retried=True)))

    stale = find_stale_claims()
    assert len(stale) == 1
    assert stale[0]["action"] == "block"
    assert "budget" in stale[0]["traceback_tail"]
    assert "retry_count" not in stale[0]


@pytest.mark.parametrize("status", ["graded", "disqualified", "aborted_candidate"])
def test_find_stale_claims_leaves_terminal_status_alone(tmp_path, monkeypatch, status):
    """A verdict legitimately consumes the claim -- only retryable aborts come back."""
    body = _stale_fixture(tmp_path, monkeypatch, results={"status": status})
    monkeypatch.setattr(subprocess, "run", _listing(_issue(body, _stamp(_OLD))))

    assert find_stale_claims() == []


@pytest.mark.parametrize("status", ["graded", "disqualified", "aborted_candidate"])
def test_find_stale_claims_resets_retry_label_on_terminal_verdict_with_spent_budget(
    tmp_path, monkeypatch, status,
):
    """fps-rtd PR #304 review finding #3: a spent retry budget must not outlive
    its cycle. Without this, a candidate that spent its one retry and then
    eventually succeeded would be born already at budget the next time a
    human manually re-queues this SAME issue (e.g. against a re-frozen batch)
    -- its first retryable abort in that later, unrelated cycle would block
    immediately instead of getting the one retry the docstring promises.
    """
    body = _stale_fixture(tmp_path, monkeypatch, results={"status": status})
    monkeypatch.setattr(subprocess, "run", _listing(_issue(body, _stamp(_OLD), retried=True)))

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
    body = _stale_fixture(tmp_path, monkeypatch, results=written)
    monkeypatch.setattr(subprocess, "run", _listing(_issue(body, _stamp(_OLD))))

    assert find_stale_claims() == []


def test_find_stale_claims_ignores_completed_run(tmp_path, monkeypatch):
    body = _stale_fixture(tmp_path, monkeypatch, results="{}", run_log=_TRACEBACK)
    monkeypatch.setattr(subprocess, "run", _listing(_issue(body, _stamp(_OLD))))

    assert find_stale_claims() == []


def test_find_stale_claims_ignores_non_traceback_log(tmp_path, monkeypatch):
    body = _stale_fixture(tmp_path, monkeypatch, run_log="still fitting fold 7...\n")
    monkeypatch.setattr(subprocess, "run", _listing(_issue(body, _stamp(_OLD))))

    assert find_stale_claims() == []


def test_find_stale_claims_queries_open_experiment_issues(tmp_path, monkeypatch):
    """The one query both the sweep and the claim share, and it must stay the flag form.

    Measured 2026-09-08 (see ISSUE_LIST_LIMIT): the `--search` index reflects an
    assignee change 2-3s late, where the label listing has it at the first read. main()
    runs the sweep and the claim about half a second apart, so on the search path an
    issue this run just blocked could still read as unassigned and be re-claimed.
    """
    body = _stale_fixture(tmp_path, monkeypatch, run_log=_TRACEBACK)
    fake_run, calls = _recording(_issue(body, _stamp(_OLD)))
    monkeypatch.setattr(subprocess, "run", fake_run)

    find_stale_claims()

    listed = [cmd for cmd in calls if cmd[:3] == ["gh", "issue", "list"]]
    assert len(listed) == 1
    assert "--label" in listed[0] and EXPERIMENT_LABEL in listed[0]
    assert listed[0][listed[0].index("--state") + 1] == "open"
    assert "--search" not in listed[0]


# ── release_stale_claim ───────────────────────────────────────────────────────

def test_release_stale_claim_comments_and_unassigns(monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return _completed_process("")

    monkeypatch.setattr(subprocess, "run", fake_run)
    release_stale_claim({"number": 1}, "Traceback (most recent call last):\nValueError: boom")

    commands = [c[0] for c in calls]
    assert commands[0] == ["gh", "issue", "comment", "1", "--body-file", "-"]
    assert "boom" in calls[0][1]["input"]
    assert commands[1] == ["gh", "issue", "edit", "1", "--remove-assignee", "@me"]
    assert len(commands) == 2  # no --add-label call when retry_count isn't given


def test_release_stale_claim_records_the_retried_label(monkeypatch):
    fake_run, calls = _recording()
    monkeypatch.setattr(subprocess, "run", fake_run)
    release_stale_claim({"number": 1}, "retryable abort", retry_count=1)

    assert calls == [
        ["gh", "issue", "comment", "1", "--body-file", "-"],
        ["gh", "issue", "edit", "1", "--add-label", RETRIED_LABEL],
        ["gh", "issue", "edit", "1", "--remove-assignee", "@me"],
    ]


def test_release_stale_claim_labels_before_unassigning(monkeypatch):
    """fps-rtd PR #304 review finding #2, ported.

    If a call partway through this sequence fails, the safer stuck state is "looks
    already-retried" (blocks one cycle early) rather than "counter never advanced"
    (the same fault gets released and retried forever). Under bd that meant
    --set-metadata before --status open, because --status open was what made the
    issue claimable again. On GitHub it is the UNASSIGN that makes it claimable, so
    that is what the label now has to precede -- same invariant, different call.
    """
    fake_run, calls = _recording()
    monkeypatch.setattr(subprocess, "run", fake_run)
    release_stale_claim({"number": 1}, "retryable abort", retry_count=1)

    label_idx = calls.index(["gh", "issue", "edit", "1", "--add-label", RETRIED_LABEL])
    unassign_idx = calls.index(["gh", "issue", "edit", "1", "--remove-assignee", "@me"])
    assert label_idx < unassign_idx


def test_block_exhausted_claim_labels_and_assigns_in_one_edit(monkeypatch):
    """Blocking is two things at once and GitHub needs both written: the assignment
    drops the issue out of the unassigned claim query, the label makes it legible and
    is what claim_next_candidate filters on. One `gh issue edit` so a partial failure
    can't leave it labelled-but-claimable or claimed-but-unlabelled.
    """
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return _completed_process("")

    monkeypatch.setattr(subprocess, "run", fake_run)
    launch_module.block_exhausted_claim({"number": 1}, "retry budget exhausted")

    commands = [c[0] for c in calls]
    assert commands[0] == ["gh", "issue", "comment", "1", "--body-file", "-"]
    assert "exhausted" in calls[0][1]["input"]
    assert commands[1] == [
        "gh", "issue", "edit", "1", "--add-label", BLOCKED_LABEL, "--add-assignee", "@me",
    ]
    assert len(commands) == 2


def test_clear_retry_metadata_removes_the_label_without_touching_the_claim(monkeypatch):
    """fps-rtd PR #304 review finding #3."""
    fake_run, calls = _recording()
    monkeypatch.setattr(subprocess, "run", fake_run)
    launch_module.clear_retry_metadata({"number": 1}, "terminal verdict, spent retry")

    assert calls == [
        ["gh", "issue", "comment", "1", "--body-file", "-"],
        ["gh", "issue", "edit", "1", "--remove-label", RETRIED_LABEL],
    ]
    assert not any("assignee" in arg for cmd in calls for arg in cmd)


# ── _abort_claim: pre-launch validation failures share the budget (fps-rtd PR #304 review finding #4) ──

def test_abort_claim_releases_with_retry_label_on_first_failure(monkeypatch):
    fake_run, calls = _recording()
    monkeypatch.setattr(subprocess, "run", fake_run)
    launch_module._abort_claim({"number": 9}, "malformed candidate reference: boom")

    assert ["gh", "issue", "edit", "9", "--add-label", RETRIED_LABEL] in calls
    assert ["gh", "issue", "edit", "9", "--remove-assignee", "@me"] in calls
    assert not any(BLOCKED_LABEL in cmd for cmd in calls)


def test_abort_claim_blocks_once_retry_budget_exhausted(monkeypatch):
    """A candidate whose body/module is broken in a way that will never
    self-correct must not burn the nightly slot forever either -- same
    starvation shape as the runtime abort paths, just reached through
    pre-launch validation.
    """
    fake_run, calls = _recording()
    monkeypatch.setattr(subprocess, "run", fake_run)
    issue = {"number": 9, "labels": [{"name": RETRIED_LABEL}]}
    launch_module._abort_claim(issue, "malformed candidate reference: boom")

    assert ["gh", "issue", "edit", "9", "--remove-assignee", "@me"] not in calls
    assert [
        "gh", "issue", "edit", "9", "--add-label", BLOCKED_LABEL, "--add-assignee", "@me",
    ] in calls


# ── recover_stale_claims: retry budget end-to-end (fps-rtd) ──────────────────

def test_recover_stale_claims_second_consecutive_retryable_completion_does_not_release(
    tmp_path, monkeypatch,
):
    """Acceptance criterion: two consecutive retryable completions of the same
    claim; the second does not release.
    """
    body = _stale_fixture(tmp_path, monkeypatch, results={"status": "aborted_pipeline", "error": "bad config"})

    # First completion: fresh claim, no prior retries.
    fake_run, calls = _recording(_issue(body, _stamp(_RECENT)))
    monkeypatch.setattr(subprocess, "run", fake_run)
    recover_stale_claims()
    assert ["gh", "issue", "edit", "1", "--add-label", RETRIED_LABEL] in calls
    assert ["gh", "issue", "edit", "1", "--remove-assignee", "@me"] in calls

    # Second completion: the claim now carries the `retried` label (budget spent).
    fake_run, calls = _recording(_issue(body, _stamp(_RECENT), retried=True))
    monkeypatch.setattr(subprocess, "run", fake_run)
    recover_stale_claims()

    assert ["gh", "issue", "edit", "1", "--remove-assignee", "@me"] not in calls
    assert [
        "gh", "issue", "edit", "1", "--add-label", BLOCKED_LABEL, "--add-assignee", "@me",
    ] in calls


def test_exhausted_candidate_does_not_prevent_a_second_candidate_from_being_claimed(
    tmp_path, monkeypatch,
):
    """Acceptance criterion: a retryably-failing candidate does not prevent a
    second queued candidate from being claimed.

    Once #1's retry budget is spent, recover_stale_claims blocks it (assigned +
    `blocked`) rather than unassigning it -- so it never goes back in front of #2 in
    the oldest-first ordering, even though it is much the older issue.
    claim_next_candidate then skips it on BOTH grounds and claims #2.
    """
    body = _stale_fixture(tmp_path, monkeypatch, results={"status": "aborted_environment", "error": "disk full"})
    exhausted = _issue(body, _stamp(_RECENT), number=1, retried=True, created_at="2026-01-01T00:00:00Z")
    queued = _issue("Batch: b\nModule: m", _stamp(_RECENT), number=2, assigned=False,
                    created_at="2026-06-01T00:00:00Z")

    calls: list[list[str]] = []
    state = {"issues": [exhausted, queued]}

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[:3] == ["gh", "issue", "list"]:
            return _completed_process(json.dumps(state["issues"]))
        return _completed_process("")

    monkeypatch.setattr(subprocess, "run", fake_run)

    recover_stale_claims()
    assert ["gh", "issue", "edit", "1", "--remove-assignee", "@me"] not in calls
    assert [
        "gh", "issue", "edit", "1", "--add-label", BLOCKED_LABEL, "--add-assignee", "@me",
    ] in calls

    # The block landed on GitHub; reflect it in what the next list returns.
    state["issues"] = [
        _issue(body, _stamp(_RECENT), number=1, retried=True, blocked=True,
               created_at="2026-01-01T00:00:00Z"),
        queued,
    ]
    claimed = claim_next_candidate()
    assert claimed["number"] == 2
    assert ["gh", "issue", "edit", "2", "--add-assignee", "@me"] in calls


def test_recover_stale_claims_dispatches_reset_retry_action(tmp_path, monkeypatch):
    """fps-rtd PR #304 review finding #3, end to end through recover_stale_claims."""
    body = _stale_fixture(tmp_path, monkeypatch, results={"status": "graded"})
    fake_run, calls = _recording(_issue(body, _stamp(_OLD), retried=True))
    monkeypatch.setattr(subprocess, "run", fake_run)

    recover_stale_claims()
    assert ["gh", "issue", "edit", "1", "--remove-label", RETRIED_LABEL] in calls
    edits = [cmd for cmd in calls if cmd[:3] == ["gh", "issue", "edit"]]
    assert not any("assignee" in arg for cmd in edits for arg in cmd)


def test_no_bd_command_survives_anywhere_in_the_recovery_path(tmp_path, monkeypatch):
    """The cutover's blunt postcondition: this module shells out to `gh` and nothing
    else. In particular there is no `bd dolt pull`/`push` any more -- GitHub is the
    store, so there is no second sync protocol to keep in step (that sync was the
    whole of fps-sk0).
    """
    body = _stale_fixture(tmp_path, monkeypatch, results={"status": "aborted_pipeline"})
    fake_run, calls = _recording(_issue(body, _stamp(_RECENT)))
    monkeypatch.setattr(subprocess, "run", fake_run)

    recover_stale_claims()
    claim_next_candidate()

    assert calls, "expected some subprocess activity"
    assert all(cmd[0] == "gh" for cmd in calls)
    assert not hasattr(launch_module, "sync_push")
    assert not hasattr(launch_module, "sync_pull")


# ── claim_next_candidate ──────────────────────────────────────────────────────

def test_claim_next_candidate_returns_none_when_queue_empty(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _listing())
    assert claim_next_candidate() is None


def test_claim_next_candidate_claims_the_oldest_unassigned_issue(monkeypatch):
    """`gh issue list` returns newest-first and the REST path has no ascending-sort
    flag, so oldest-first -- the ordering the fps-rtd retry budget is defined against
    -- is done locally. Hand it the list in the order gh really returns it.
    """
    newest = _issue("b", "2026-09-03T00:00:00Z", number=3, assigned=False, created_at="2026-09-03T00:00:00Z")
    oldest = _issue("b", "2026-09-01T00:00:00Z", number=1, assigned=False, created_at="2026-09-01T00:00:00Z")
    fake_run, calls = _recording(newest, oldest)
    monkeypatch.setattr(subprocess, "run", fake_run)

    issue = claim_next_candidate()

    assert issue["number"] == 1
    assert ["gh", "issue", "edit", "1", "--add-assignee", "@me"] in calls


def test_claim_next_candidate_skips_claimed_and_blocked_issues(monkeypatch):
    """Assignee is the claim, and `blocked` is a label that excludes nothing on its
    own -- so the claim query has to state both exclusions itself. The older two
    issues here are both ineligible; only #3 may be claimed.
    """
    claimed = _issue("b", "x", number=1, assigned=True, created_at="2026-01-01T00:00:00Z")
    blocked = _issue("b", "x", number=2, assigned=False, blocked=True, created_at="2026-02-01T00:00:00Z")
    free = _issue("b", "x", number=3, assigned=False, created_at="2026-03-01T00:00:00Z")
    fake_run, calls = _recording(claimed, blocked, free)
    monkeypatch.setattr(subprocess, "run", fake_run)

    issue = claim_next_candidate()

    assert issue["number"] == 3
    assert ["gh", "issue", "edit", "3", "--add-assignee", "@me"] in calls


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
    monkeypatch.setattr(launch_module, "recover_stale_claims", lambda: [])
    monkeypatch.setattr(launch_module, "claim_next_candidate", lambda: None)

    result = CliRunner().invoke(main, [])

    assert result.exit_code == 0
    assert "no experiment work ready" in result.output


def test_main_aborts_claim_on_malformed_candidate_reference(monkeypatch):
    fake_run, calls = _recording()
    monkeypatch.setattr(launch_module, "recover_stale_claims", lambda: [])
    monkeypatch.setattr(
        launch_module, "claim_next_candidate",
        lambda: {"number": 9, "body": "no structured fields here"},
    )
    monkeypatch.setattr(subprocess, "run", fake_run)

    result = CliRunner().invoke(main, [])

    assert result.exit_code == 0
    assert "aborted" in result.output
    assert ["gh", "issue", "edit", "9", "--add-label", RETRIED_LABEL] in calls
    assert ["gh", "issue", "edit", "9", "--remove-assignee", "@me"] in calls


def test_main_comments_the_launch_on_the_issue(tmp_path, monkeypatch):
    """The launch confirmation is the only record that the detached child was ever
    started; MAX_RETRIES aside, nothing else writes to the issue on the happy path.
    """
    body = _stale_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(launch_module, "recover_stale_claims", lambda: [])
    monkeypatch.setattr(launch_module, "claim_next_candidate", lambda: _issue(body, "x", number=7))
    monkeypatch.setattr(launch_module, "load_candidate_module", lambda path: object())
    monkeypatch.setattr(launch_module, "load_features", lambda path: object())
    monkeypatch.setattr(launch_module, "validate_candidate", lambda candidate, frame: None)
    monkeypatch.setattr(launch_module, "launch_detached", lambda cmd, out_dir: 4242)
    fake_run, calls = _recording()
    monkeypatch.setattr(subprocess, "run", fake_run)

    result = CliRunner().invoke(main, [])

    assert result.exit_code == 0, result.output
    assert "#7: launched detached pid=4242" in result.output
    assert ["gh", "issue", "comment", "7", "--body-file", "-"] in calls


# ── the label encoding's precondition ─────────────────────────────────────────

def test_retried_label_can_still_represent_the_whole_retry_budget():
    """The `retried` label is a BOOLEAN, so it can only encode a budget of one.

    bd held retry_count in a free-form metadata integer, which would have carried any
    MAX_RETRIES. The GitHub port traded that for a label because MAX_RETRIES is 1. If
    someone raises it, _retry_count can no longer tell "retried once" from "retried
    twice" and will silently grant unlimited retries -- the exact starvation fps-rtd
    was filed about. Fail here rather than there.
    """
    assert MAX_RETRIES == 1, (
        "MAX_RETRIES > 1 cannot be represented by the boolean `retried` label — "
        "see _retry_count; the counter needs a real store again before this can rise."
    )
