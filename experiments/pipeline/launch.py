"""Launch routine (fps-3jj.5) — the nightly, ~10-minute, Claude-active piece of the
AI-sourced feature pipeline: claim the next `experiment`-labelled bd issue, validate
its candidate module, launch the (hours-long, Claude-free) runner detached, and exit.

Detachment: `subprocess.Popen(..., start_new_session=True)`, NOT a shelled-out
`setsid nohup ... &`. The parent design (fps-3jj) describes the shell recipe as
prose, not a literal requirement, and `setsid` the CLI tool is a Linux util-linux
binary not guaranteed present on macOS — where this local routine actually runs.
`start_new_session=True` calls `os.setsid()` in the forked child before exec, which
gives the same detachment semantics (new session, no controlling terminal, immune to
the parent's SIGHUP) via the standard library, portably.

Candidate-bead convention (this module is the consumer; fps-3jj.7's generator session
is the producer and must follow it): an `experiment`-labelled bd issue's description
must contain two lines,

    Batch: experiments/batches/<batch-name>
    Module: experiments/candidates/<batch-name>/<candidate-name>.py

parsed by parse_candidate_ref(). The runner's out_dir is
default_out_dir(candidate_path) — candidate_path with its .py suffix stripped
(see experiments/pipeline/runner.py) — so run.log and results.json land in a
per-candidate subdirectory, not the shared batch directory (fps-icv). That's
also where stale-claim recovery looks.

Stale-claim recovery mirrors CLAUDE.md's worker-routine pickup rule 4, adapted to the
experiment queue: dir exists, run.log ends in a traceback, no results.json, claimed
>12h ago -> post the traceback to the bead and release the claim.

Retry budget (fps-rtd): releasing a claim so the next sweep can re-claim it sounds
harmless, but a released bead keeps its original creation date -- so an unbounded
release is always the oldest unassigned issue and starves everything else in the
queue forever. MAX_RETRIES caps this at one retry: the retry count lives in bd
metadata (`retry_count`), the one place claim state already lives, rather than in
results.json (deleted at the start of every run_candidate() call, so it can't
survive across attempts). Once the budget is spent, the claim is blocked instead of
released -- blocked issues drop out of `bd ready` entirely, so this is what actually
stops the starvation, not just slows it. A human clears it by fixing the underlying
fault, then `bd update <id> --status open --unset-metadata retry_count`.

The budget applies uniformly to every path that gives up on a claim -- a
RETRYABLE_STATUSES abort, a crashed-mid-run traceback, AND a pre-launch validation
failure (fps-rtd PR #304 review findings #1/#4) -- sharing one counter per claim, not
one per failure shape, via `_decide_release_or_block` and `release_stale_claim` /
`block_exhausted_claim`. The counter also resets once a claim reaches a genuine
TERMINAL_STATUSES verdict (review finding #3), via `clear_retry_metadata`: otherwise a
candidate that spent its one retry, then eventually succeeded, would be born already
at budget the next time a human manually re-queues that SAME bead (e.g. against a
re-frozen batch) -- its first retryable abort in that later, unrelated cycle would
block immediately instead of getting the one retry the docstring promises.
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
from datetime import datetime, timedelta, timezone

import click

from experiments.pipeline.runner import (
    RETRYABLE_STATUSES,
    TERMINAL_STATUSES,
    default_out_dir,
    read_run_status,
)
from experiments.pipeline.validate import (
    CandidateImportError,
    load_candidate_module,
    validate_candidate,
)
from fuel_signal.features import load_features

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = REPO_ROOT / "experiments"
EXPERIMENT_LABEL = "experiment"
STALE_AFTER = timedelta(hours=12)
RUN_LOG_FILENAME = "run.log"
RESULTS_FILENAME = "results.json"

# fps-rtd: one automatic retry per claim (see module docstring), tracked in bd
# metadata since results.json can't survive across run_candidate() attempts.
RETRY_COUNT_METADATA_KEY = "retry_count"
MAX_RETRIES = 1

_BATCH_RE = re.compile(r"^Batch:\s*(\S+)\s*$", re.MULTILINE)
_MODULE_RE = re.compile(r"^Module:\s*(\S+)\s*$", re.MULTILINE)


class CandidateRefError(ValueError):
    """A bd issue's description didn't carry a well-formed Batch:/Module: pair."""


def parse_candidate_ref(description: str) -> tuple[pathlib.Path, pathlib.Path]:
    """Extract (batch_dir, candidate_path) from an experiment bead's description.

    Paths are repo-root-relative in the bead text; returned as resolved absolute
    paths. Both must resolve inside EXPERIMENTS_ROOT — an unattended nightly
    routine that `exec_module`s whatever `Module:` points at (validate.py's
    load_candidate_module) must not follow a `..` traversal or an absolute path
    out of experiments/, however that string ended up in a bead description.
    """
    batch_match = _BATCH_RE.search(description or "")
    module_match = _MODULE_RE.search(description or "")
    if not batch_match or not module_match:
        raise CandidateRefError(
            "description must contain a 'Batch: <path>' line and a 'Module: <path>' line"
        )
    batch_dir = (REPO_ROOT / batch_match.group(1)).resolve()
    candidate_path = (REPO_ROOT / module_match.group(1)).resolve()
    for path, label in ((batch_dir, "Batch"), (candidate_path, "Module")):
        if not path.is_relative_to(EXPERIMENTS_ROOT):
            raise CandidateRefError(f"{label} path resolves outside experiments/: {path}")
    return batch_dir, candidate_path


def sync_pull() -> None:
    subprocess.run(["bd", "dolt", "pull"], check=True)


def sync_push() -> None:
    subprocess.run(["bd", "dolt", "push"], check=True)


def _bd_json(*args: str) -> list[dict]:
    result = subprocess.run(["bd", *args, "--json"], check=True, capture_output=True, text=True)
    return json.loads(result.stdout) if result.stdout.strip() else []


def _looks_like_traceback_tail(log_path: pathlib.Path, tail_lines: int = 40) -> str | None:
    if not log_path.exists():
        return None
    lines = log_path.read_text(errors="replace").splitlines()
    tail = "\n".join(lines[-tail_lines:])
    return tail if "Traceback (most recent call last):" in tail else None


def _retryable_status(out_dir: pathlib.Path) -> str | None:
    """The run's status if it finished in a RETRYABLE_STATUSES state, else None.

    A malformed/unreadable results.json reads as "not retryable" (read_run_status
    returns None) -- releasing a claim on the strength of a file we couldn't
    parse is the more dangerous guess of the two.
    """
    status = read_run_status(out_dir)
    return status if status in RETRYABLE_STATUSES else None


def _retry_count(issue: dict) -> int:
    """Retries already spent on this claim (fps-rtd), from bd metadata.

    bd metadata survives the assign/status changes release_stale_claim makes
    (results.json does not -- it's deleted at the start of every
    run_candidate() call), so it's where a counter that must outlive one
    attempt has to live. Missing metadata, a non-dict `metadata`, or a value
    `int()` can't parse (a non-numeric string, `None`) reads as 0 -- the safer
    misread here is "budget not yet spent", not an early block on a candidate
    that never actually retried. A NEGATIVE value is clamped to 0 rather than
    honoured, so a hand-edited `retry_count: -5` can't offset the `+ 1` in
    _decide_release_or_block to extend the budget. A numeric STRING (`"2"`,
    as opposed to a non-numeric one) IS honoured as its int value -- `bd
    update --set-metadata` always writes real ints, never strings, but the
    JSON form (`bd update --metadata '{"retry_count":"2"}'`) could, and a
    string that parses cleanly is exactly the kind of value this function's
    "read what's actually meant" contract should not zero out
    (fps-rtd PR #304 review finding #5).
    """
    metadata = issue.get("metadata")
    if not isinstance(metadata, dict):
        return 0
    value = metadata.get(RETRY_COUNT_METADATA_KEY, 0)
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _decide_release_or_block(issue: dict) -> tuple[str, int | None]:
    """Whether a claim that's giving up should be released for a retry or
    blocked outright (fps-rtd).

    Returns ("block", None) once MAX_RETRIES is already spent on this claim,
    else ("release", retry_count + 1) -- the count to record if released.
    Pure and side-effect-free: shared by every path that gives up on a claim
    (a RETRYABLE_STATUSES abort, a crashed-mid-run traceback, and a pre-launch
    validation failure -- fps-rtd PR #304 review findings #1/#4) so all three
    draw from the SAME counter rather than each getting its own private
    budget a persistently-broken candidate could exhaust independently.
    """
    retry_count = _retry_count(issue)
    if retry_count >= MAX_RETRIES:
        return "block", None
    return "release", retry_count + 1


def find_stale_claims(now: datetime | None = None) -> list[dict]:
    """Read-only: which in_progress experiment issues need their claim released,
    blocked, or have a spent retry counter reset.

    Three shapes:

    1. Crashed mid-run: no results.json, run.log tail looks like a Python
       traceback, claimed more than STALE_AFTER ago. The age gate matters here
       because a run with no results.json may still be in flight. Bounded by
       MAX_RETRIES same as shape 2 (fps-rtd PR #304 review finding #1) -- a
       candidate that crashes the same deterministic way every attempt must
       not re-win `bd ready` forever just because its failure mode happens to
       be a traceback rather than a RETRYABLE_STATUSES result.

    2. Finished with a RETRYABLE_STATUSES status (aborted_pipeline /
       aborted_environment): the candidate never got a fair hearing, so its
       claim must go back on the queue. No age gate -- results.json existing is
       proof the run is over, so there is nothing to wait for. Bounded by
       MAX_RETRIES (fps-rtd): once a claim has already burned its retry and
       aborts retryably again, action is "block" instead of "release" -- see
       module docstring for why an unbounded release starves the whole queue.

    3. Finished with a TERMINAL_STATUSES verdict (rejected / disqualified /
       aborted_candidate) AND this claim's retry_count metadata is still
       nonzero from an earlier abort in the SAME cycle: action "reset_retry"
       clears the counter (fps-rtd PR #304 review finding #3) so a LATER,
       unrelated manual re-run of this same bead isn't born already at
       budget. Doesn't touch status/assignee -- the claim is still
       legitimately consumed and still needs a human or the dossier routine
       to close it, same as any other terminal verdict.

    A completed run with a fresh (zero) retry_count, or an unrecognised/
    unparseable results.json status, is left alone. An issue whose description
    doesn't parse, or that has no run.log yet (still validating, or launch
    crashed before ever writing one), is left alone too -- not this function's
    job to guess at those.

    Each returned entry carries an "action" ("release", "block", or
    "reset_retry"); "release" entries also carry the "retry_count" to record.
    """
    now = now or datetime.now(timezone.utc)
    stale: list[dict] = []
    for issue in _bd_json("list", "--status", "in_progress", "--label", EXPERIMENT_LABEL):
        try:
            _, candidate_path = parse_candidate_ref(issue.get("description", ""))
        except CandidateRefError:
            continue
        out_dir = default_out_dir(candidate_path)

        retryable = _retryable_status(out_dir)
        if retryable is not None:
            action, next_count = _decide_release_or_block(issue)
            if action == "block":
                stale.append({
                    "issue": issue,
                    "action": "block",
                    "traceback_tail": (
                        f"run finished with retryable status {retryable!r} again -- retry "
                        f"budget (MAX_RETRIES={MAX_RETRIES}) exhausted. Blocking instead of "
                        f"releasing so this stops starving the rest of the queue (fps-rtd)."
                    ),
                })
            else:
                stale.append({
                    "issue": issue,
                    "action": "release",
                    "retry_count": next_count,
                    "traceback_tail": (
                        f"run finished with retryable status {retryable!r} -- a pipeline/"
                        f"environment fault, not a verdict on the candidate. Releasing for "
                        f"re-run ({next_count}/{MAX_RETRIES})."
                    ),
                })
            continue

        if (out_dir / RESULTS_FILENAME).exists():
            status = read_run_status(out_dir)
            if status in TERMINAL_STATUSES and _retry_count(issue) > 0:
                stale.append({
                    "issue": issue,
                    "action": "reset_retry",
                    "traceback_tail": (
                        f"run finished with terminal status {status!r} after this claim had "
                        f"already spent a retry -- clearing the spent retry_count so a future, "
                        f"unrelated re-run of this SAME bead gets the full budget again "
                        f"(fps-rtd PR #304 review finding #3)."
                    ),
                })
            continue

        traceback_tail = _looks_like_traceback_tail(out_dir / RUN_LOG_FILENAME)
        if traceback_tail is None:
            continue
        claimed_at_raw = issue.get("started_at") or issue.get("updated_at")
        if not claimed_at_raw:
            continue
        claimed_at = datetime.fromisoformat(claimed_at_raw.replace("Z", "+00:00"))
        if now - claimed_at < STALE_AFTER:
            continue

        action, next_count = _decide_release_or_block(issue)
        if action == "block":
            stale.append({
                "issue": issue,
                "action": "block",
                "traceback_tail": (
                    f"{traceback_tail}\n\nretry budget (MAX_RETRIES={MAX_RETRIES}) exhausted "
                    f"after a prior release -- blocking instead of releasing again (fps-rtd)."
                ),
            })
        else:
            stale.append({
                "issue": issue, "action": "release", "retry_count": next_count,
                "traceback_tail": traceback_tail,
            })
    return stale


def release_stale_claim(issue: dict, traceback_tail: str, *, retry_count: int | None = None) -> None:
    """Post the reason, unassign, and reopen one stale-claimed experiment issue.

    `traceback_tail` carries whichever evidence the caller found -- a
    traceback tail for a crashed run, or a one-line explanation for a run that
    finished in a retryable status -- so the bead records WHY it was released.

    `retry_count`, when given, is recorded on the bead as the retries already
    spent on this claim -- see MAX_RETRIES / fps-rtd. The `--set-metadata`
    write happens BEFORE `--status open` (fps-rtd PR #304 review finding #2):
    if a `bd` call partway through this sequence fails (dolt lock contention,
    a killed process), the safer stuck state is "looks already-retried"
    (blocks one cycle early) rather than "counter never advanced" (the same
    fault gets released and retried forever).
    """
    issue_id = issue["id"]
    subprocess.run(
        ["bd", "comment", issue_id, "--stdin"],
        input=f"[launch] claim released for re-run.\n\n{traceback_tail}",
        text=True,
        check=True,
    )
    subprocess.run(["bd", "assign", issue_id, ""], check=True)
    if retry_count is not None:
        subprocess.run(
            ["bd", "update", issue_id, "--set-metadata", f"{RETRY_COUNT_METADATA_KEY}={retry_count}"],
            check=True,
        )
    subprocess.run(["bd", "update", issue_id, "--status", "open"], check=True)


def block_exhausted_claim(issue: dict, reason: str) -> None:
    """Retry budget spent (fps-rtd): block the claim instead of releasing it.

    A released-but-unassigned issue keeps its original creation date, so it's
    always the oldest `bd ready` result and would be re-claimed ahead of
    everything else, forever -- the exact starvation this issue is about.
    `blocked` status drops it out of `bd ready` entirely while keeping it
    visible for triage (not silently dropped -- module docstring covers the
    human recovery step).
    """
    issue_id = issue["id"]
    subprocess.run(
        ["bd", "comment", issue_id, "--stdin"],
        input=f"[launch] retry budget exhausted -- blocking, not releasing.\n\n{reason}",
        text=True,
        check=True,
    )
    subprocess.run(["bd", "assign", issue_id, ""], check=True)
    subprocess.run(["bd", "update", issue_id, "--status", "blocked"], check=True)


def clear_retry_metadata(issue: dict, reason: str) -> None:
    """Reset a claim's spent retry budget once it reaches a real verdict
    (fps-rtd PR #304 review finding #3).

    Doesn't touch status or assignee -- the claim is still `in_progress` and
    still needs a human or the dossier routine to close it, same as any other
    TERMINAL_STATUSES verdict (see find_stale_claims). This only prevents a
    stale counter from an earlier failure cycle silently costing a later,
    unrelated cycle its retry.
    """
    issue_id = issue["id"]
    subprocess.run(
        ["bd", "comment", issue_id, "--stdin"],
        input=f"[launch] clearing spent retry budget on this now-terminal claim.\n\n{reason}",
        text=True,
        check=True,
    )
    subprocess.run(["bd", "update", issue_id, "--unset-metadata", RETRY_COUNT_METADATA_KEY], check=True)


def recover_stale_claims(now: datetime | None = None) -> list[dict]:
    stale = find_stale_claims(now=now)
    for entry in stale:
        action = entry.get("action")
        if action == "block":
            block_exhausted_claim(entry["issue"], entry["traceback_tail"])
        elif action == "reset_retry":
            clear_retry_metadata(entry["issue"], entry["traceback_tail"])
        else:
            release_stale_claim(
                entry["issue"], entry["traceback_tail"], retry_count=entry.get("retry_count")
            )
    if stale:
        sync_push()
    return stale


def claim_next_candidate() -> dict | None:
    """Atomically claim the oldest ready experiment issue, or None if the queue is empty."""
    claimed = _bd_json(
        "ready", "--label", EXPERIMENT_LABEL, "--unassigned", "--sort", "oldest", "-n", "1", "--claim",
    )
    if not claimed:
        return None
    sync_push()
    return claimed[0]


def build_runner_cmd(batch_dir: pathlib.Path, candidate_path: pathlib.Path, bead_id: str) -> list[str]:
    return [
        "uv", "run", "python", "-m", "experiments.pipeline.runner",
        "--batch-dir", str(batch_dir),
        "--candidate", str(candidate_path),
        "--bead-id", bead_id,
    ]


def launch_detached(
    cmd: list[str], out_dir: pathlib.Path, *, log_name: str = RUN_LOG_FILENAME,
) -> int:
    """Launch `cmd` fully detached from this process; returns its pid.

    stdin is /dev/null and stdout+stderr go to out_dir/log_name so the child never
    blocks on or inherits this process's terminal. start_new_session=True is what
    lets it outlive this process and the shell that invoked it -- see module
    docstring for why this replaces the design doc's `setsid nohup` shell recipe.
    """
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / log_name
    with open(log_path, "ab") as logfile, open(os.devnull) as devnull:
        process = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            stdin=devnull,
            stdout=logfile,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return process.pid


@click.command("launch")
def main() -> None:
    """Nightly entry point: sync, recover stale claims, claim + validate + launch one candidate."""
    sync_pull()

    recovered = recover_stale_claims()
    verbs = {"block": "blocked", "reset_retry": "reset retry budget on"}
    for entry in recovered:
        verb = verbs.get(entry.get("action"), "recovered")
        click.echo(f"[launch] {verb} stale claim {entry['issue']['id']}")

    issue = claim_next_candidate()
    if issue is None:
        click.echo("[launch] no experiment work ready")
        return

    issue_id = issue["id"]
    try:
        batch_dir, candidate_path = parse_candidate_ref(issue.get("description", ""))
    except CandidateRefError as exc:
        _abort_claim(issue, f"malformed candidate reference: {exc}")
        return

    try:
        candidate = load_candidate_module(candidate_path)
        frame = load_features(batch_dir / "features.csv")
        validate_candidate(candidate, frame)
    except CandidateImportError as exc:
        _abort_claim(issue, f"candidate module failed to import: {exc}")
        return
    except Exception as exc:  # noqa: BLE001 — any validation failure aborts the claim, not the routine
        _abort_claim(issue, f"validation failed: {exc!r}")
        return

    cmd = build_runner_cmd(batch_dir, candidate_path, issue_id)
    out_dir = default_out_dir(candidate_path)
    try:
        pid = launch_detached(cmd, out_dir)
    except OSError as exc:
        # A validated candidate that fails to actually launch (missing `uv`,
        # permission error creating out_dir, etc.) must not leave the bead
        # claimed forever -- same "release rather than strand" rule as a
        # validation failure above.
        _abort_claim(issue, f"failed to launch detached runner: {exc!r}")
        return
    log_path = out_dir / RUN_LOG_FILENAME
    subprocess.run(
        ["bd", "comment", issue_id, f"[launch] validated, launched detached pid={pid}, log={log_path}"],
        check=True,
    )
    sync_push()
    click.echo(f"[launch] {issue_id}: launched detached pid={pid} log={log_path}")


def _abort_claim(issue: dict, reason: str) -> None:
    """Validation failed before launch: apply the same retry budget as any
    other give-up path instead of releasing unconditionally (fps-rtd PR #304
    review finding #4) -- a candidate whose description or module is broken
    in a way that will never self-correct would otherwise re-win `bd ready`
    and burn the nightly slot forever, exactly the starvation shape this bead
    was filed about, just reached through pre-launch validation instead of a
    runtime abort. Reuses release_stale_claim / block_exhausted_claim so this
    path shares their (already-tested) bd-call sequence rather than a third
    near-duplicate of it.
    """
    issue_id = issue["id"]
    action, next_count = _decide_release_or_block(issue)
    if action == "block":
        block_exhausted_claim(issue, f"aborted before launch — {reason}")
    else:
        release_stale_claim(issue, f"aborted before launch — {reason}", retry_count=next_count)
    sync_push()
    click.echo(f"[launch] {issue_id}: aborted before launch ({action}) — {reason}")


if __name__ == "__main__":
    main()
