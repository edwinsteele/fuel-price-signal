"""Launch routine (fps-3jj.5) — the nightly, ~10-minute, Claude-active piece of the
AI-sourced feature pipeline: claim the next `experiment`-labelled GitHub issue, validate
its candidate module, launch the (hours-long, Claude-free) runner detached, and exit.

Tracker: **GitHub Issues**, via the `gh` CLI. This module was originally written against
`bd` (Beads); the 2026-09 cutover replaced every `bd` call with a `gh` one. The state
model changed shape in exactly three places, and those three are where the bugs would be:

  * **Assignee is the claim.** bd had an explicit `in_progress` status; GitHub does not,
    and does not need one. `bd ready --unassigned` becomes "open, `experiment`-labelled,
    `assignees` empty"; `bd update --claim` becomes `--add-assignee "@me"`;
    `bd assign <id> ""` becomes `--remove-assignee "@me"`.
  * **`blocked` is a label, not a status.** In bd, `blocked` was an exclusive status that
    dropped the issue out of `bd ready` on its own. A GitHub label does not exclude
    anything by itself, so blocking now does BOTH: it assigns the issue to the owner
    (which is what actually drops it out of the unassigned queue and stops the
    starvation) and adds the `blocked` label (which is what makes it legible in the UI
    and is what the claim query filters on explicitly).
  * **`retry_count` is a label, not free-form metadata.** GitHub issues have no metadata
    bag. Since MAX_RETRIES is 1, the counter only ever holds 0 or 1 — so it is the
    `retried` label, present or absent. See `_retry_count`.

There is no `bd dolt pull` / `bd dolt push` equivalent and none is needed: GitHub is the
store, so there is no second sync protocol to keep in step. (That sync was the entire
content of fps-sk0, the bug that disabled the chore/polish worker Routine.)

Detachment: `subprocess.Popen(..., start_new_session=True)`, NOT a shelled-out
`setsid nohup ... &`. The parent design (fps-3jj) describes the shell recipe as
prose, not a literal requirement, and `setsid` the CLI tool is a Linux util-linux
binary not guaranteed present on macOS — where this local routine actually runs.
`start_new_session=True` calls `os.setsid()` in the forked child before exec, which
gives the same detachment semantics (new session, no controlling terminal, immune to
the parent's SIGHUP) via the standard library, portably.

Candidate-issue convention (this module is the consumer; fps-3jj.7's generator session
is the producer and must follow it): an `experiment`-labelled issue's body must contain
two lines,

    Batch: experiments/batches/<batch-name>
    Module: experiments/candidates/<batch-name>/<candidate-name>.py

parsed by parse_candidate_ref(). The runner's out_dir is
default_out_dir(candidate_path) — candidate_path with its .py suffix stripped
(see experiments/pipeline/runner.py) — so run.log and results.json land in a
per-candidate subdirectory, not the shared batch directory (fps-icv). That's
also where stale-claim recovery looks.

Stale-claim recovery mirrors CLAUDE.md's worker-routine pickup rule 4, adapted to the
experiment queue: dir exists, run.log ends in a traceback, no results.json, claimed
>12h ago -> post the traceback to the issue and release the claim.

Retry budget (fps-rtd): releasing a claim so the next sweep can re-claim it sounds
harmless, but a released issue keeps its original creation date -- so an unbounded
release is always the oldest unassigned issue and starves everything else in the
queue forever. MAX_RETRIES caps this at one retry: the retry count lives on the issue
itself (the `retried` label), the one place claim state already lives, rather than in
results.json (deleted at the start of every run_candidate() call, so it can't
survive across attempts). Once the budget is spent, the claim is blocked instead of
released -- a blocked issue is assigned and `blocked`-labelled, so it drops out of the
claim query entirely, which is what actually stops the starvation rather than just
slowing it. A human clears it by fixing the underlying fault, then
`gh issue edit <N> --remove-label blocked --remove-label retried --remove-assignee <owner>`.

The budget applies uniformly to every path that gives up on a claim -- a
RETRYABLE_STATUSES abort, a crashed-mid-run traceback, AND a pre-launch validation
failure (fps-rtd PR #304 review findings #1/#4) -- sharing one counter per claim, not
one per failure shape, via `_decide_release_or_block` and `release_stale_claim` /
`block_exhausted_claim`. The counter also resets once a claim reaches a genuine
TERMINAL_STATUSES verdict (review finding #3), via `clear_retry_metadata`: otherwise a
candidate that spent its one retry, then eventually succeeded, would be born already
at budget the next time a human manually re-queues that SAME issue (e.g. against a
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
BLOCKED_LABEL = "blocked"
STALE_AFTER = timedelta(hours=12)
RUN_LOG_FILENAME = "run.log"
RESULTS_FILENAME = "results.json"

# fps-rtd: one automatic retry per claim (see module docstring). A GitHub issue has no
# metadata bag, but with MAX_RETRIES == 1 the counter is a boolean, so it is a label.
RETRIED_LABEL = "retried"
MAX_RETRIES = 1

# Fields every issue-shaped dict in this module carries. `body` is bd's `description`;
# `number` is bd's `id`; `updatedAt` is the only timestamp GitHub exposes on the list
# endpoint (bd had a dedicated `started_at`) -- see find_stale_claims for what that
# costs.
ISSUE_JSON_FIELDS = "number,title,body,labels,assignees,createdAt,updatedAt"

# `gh issue list --label ...` returns newest-first and has no ascending-sort flag, so the
# oldest-first ordering the retry budget depends on is done locally, over a window this
# big. The queue has never held more than a handful of open candidates;
# _open_experiment_issues warns if it ever reaches this limit, because past it the local
# sort is no longer sorting the whole queue.
#
# Deliberately NOT `--search "label:experiment no:assignee sort:created-asc"`, which
# would give the ordering for free. Measured 2026-09-08, this repo, polling at 0.5s:
#
#   query form   issue created    assignee added    assignee removed
#   --label      ~6.8s            first poll        first poll
#   --search     ~7.0s            1.96s             3.07s
#
# Both forms lag CREATION by ~7s (see docs/routines/launch.md — a candidate filed
# seconds before the routine fires waits a night). The difference that matters is the
# MUTATION column: main() runs recover_stale_claims() and then claim_next_candidate()
# about half a second apart, so on the search path an issue this run just blocked can
# still read as unassigned and be re-claimed on the spot — the starvation fps-rtd is
# about, reached through a stale index. On the label path the block was already visible
# at the first read. The margin is thin either way; if a run ever does re-claim
# something it just blocked, the cost is one wasted nightly slot and it self-corrects
# the next night (the issue is blocked by then), which is why this is a measured
# preference rather than a lock.
ISSUE_LIST_LIMIT = 200

_BATCH_RE = re.compile(r"^Batch:\s*(\S+)\s*$", re.MULTILINE)
_MODULE_RE = re.compile(r"^Module:\s*(\S+)\s*$", re.MULTILINE)


class CandidateRefError(ValueError):
    """An issue's body didn't carry a well-formed Batch:/Module: pair."""


def parse_candidate_ref(body: str) -> tuple[pathlib.Path, pathlib.Path]:
    """Extract (batch_dir, candidate_path) from an experiment issue's body.

    Paths are repo-root-relative in the issue text; returned as resolved absolute
    paths. Both must resolve inside EXPERIMENTS_ROOT — an unattended nightly
    routine that `exec_module`s whatever `Module:` points at (validate.py's
    load_candidate_module) must not follow a `..` traversal or an absolute path
    out of experiments/, however that string ended up in an issue body.
    """
    batch_match = _BATCH_RE.search(body or "")
    module_match = _MODULE_RE.search(body or "")
    if not batch_match or not module_match:
        raise CandidateRefError(
            "body must contain a 'Batch: <path>' line and a 'Module: <path>' line"
        )
    batch_dir = (REPO_ROOT / batch_match.group(1)).resolve()
    candidate_path = (REPO_ROOT / module_match.group(1)).resolve()
    for path, label in ((batch_dir, "Batch"), (candidate_path, "Module")):
        if not path.is_relative_to(EXPERIMENTS_ROOT):
            raise CandidateRefError(f"{label} path resolves outside experiments/: {path}")
    return batch_dir, candidate_path


def _gh_json(*args: str) -> list[dict]:
    result = subprocess.run(["gh", *args], check=True, capture_output=True, text=True)
    return json.loads(result.stdout) if result.stdout.strip() else []


def issue_ref(issue: dict) -> str:
    """The `gh issue <verb>` argument for this issue — its number, as a string."""
    return str(issue["number"])


def _label_names(issue: dict) -> set[str]:
    labels = issue.get("labels") or []
    return {label.get("name") for label in labels if isinstance(label, dict)}


def _is_claimed(issue: dict) -> bool:
    """Assignee IS the claim — see module docstring. bd's `in_progress` status has no
    GitHub counterpart and needs none."""
    return bool(issue.get("assignees"))


def _gh_comment(issue: dict, body: str) -> None:
    subprocess.run(
        ["gh", "issue", "comment", issue_ref(issue), "--body-file", "-"],
        input=body,
        text=True,
        check=True,
    )


def _gh_edit(issue: dict, *args: str) -> None:
    subprocess.run(["gh", "issue", "edit", issue_ref(issue), *args], check=True)


def _open_experiment_issues() -> list[dict]:
    """Every open `experiment`-labelled issue, newest-first (gh's REST ordering).

    Callers filter and sort; this is the one query both the stale sweep and the claim
    share. See ISSUE_LIST_LIMIT for the measurement behind the flag form rather than
    `--search`, and for what this listing is and isn't consistent about.
    """
    issues = _gh_json(
        "issue", "list",
        "--label", EXPERIMENT_LABEL,
        "--state", "open",
        "--limit", str(ISSUE_LIST_LIMIT),
        "--json", ISSUE_JSON_FIELDS,
    )
    if len(issues) >= ISSUE_LIST_LIMIT:
        click.echo(
            f"[launch] WARNING: hit the {ISSUE_LIST_LIMIT}-issue list limit; oldest-first "
            f"claim ordering is no longer over the whole queue. Raise ISSUE_LIST_LIMIT.",
            err=True,
        )
    return issues


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
    """Retries already spent on this claim (fps-rtd): 1 if the `retried` label is on
    the issue, else 0.

    The label survives the assignee change release_stale_claim makes (results.json
    does not -- it's deleted at the start of every run_candidate() call), so it's
    where a counter that must outlive one attempt has to live.

    This used to read a bd metadata integer, which brought a family of misread cases
    with it -- a non-numeric string, a numeric string, `None`, a hand-edited negative
    value offsetting the `+ 1` in _decide_release_or_block (fps-rtd PR #304 review
    finding #5). A label is present or absent, so every one of those is now
    structurally impossible rather than defended against. The return type stays `int`
    because MAX_RETRIES arithmetic is the contract this feeds; if MAX_RETRIES ever
    rises above 1, a boolean label stops being able to represent the counter and this
    is the function that has to change first.
    """
    return 1 if RETRIED_LABEL in _label_names(issue) else 0


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
    """Read-only: which claimed experiment issues need their claim released,
    blocked, or have a spent retry counter reset.

    "Claimed" is "open, `experiment`-labelled, has an assignee, and not already
    `blocked`" — the GitHub spelling of bd's `--status in_progress --label experiment`.
    The `blocked` exclusion is load-bearing, not cosmetic: in bd, `blocked` was an
    exclusive *status*, so a blocked issue could not also be in_progress and this
    query skipped it for free. A label excludes nothing on its own, and a blocked
    issue is deliberately left assigned (that is what keeps it out of the claim
    queue), so without this filter every blocked issue would look like a live claim
    and be re-examined forever.

    Three shapes:

    1. Crashed mid-run: no results.json, run.log tail looks like a Python
       traceback, claimed more than STALE_AFTER ago. The age gate matters here
       because a run with no results.json may still be in flight. Bounded by
       MAX_RETRIES same as shape 2 (fps-rtd PR #304 review finding #1) -- a
       candidate that crashes the same deterministic way every attempt must
       not re-win the claim query forever just because its failure mode happens
       to be a traceback rather than a RETRYABLE_STATUSES result.

    2. Finished with a RETRYABLE_STATUSES status (aborted_pipeline /
       aborted_environment): the candidate never got a fair hearing, so its
       claim must go back on the queue. No age gate -- results.json existing is
       proof the run is over, so there is nothing to wait for. Bounded by
       MAX_RETRIES (fps-rtd): once a claim has already burned its retry and
       aborts retryably again, action is "block" instead of "release" -- see
       module docstring for why an unbounded release starves the whole queue.

    3. Finished with a TERMINAL_STATUSES verdict (graded / disqualified /
       aborted_candidate) AND this claim still carries the `retried` label from
       an earlier abort in the SAME cycle: action "reset_retry" clears the
       label (fps-rtd PR #304 review finding #3) so a LATER, unrelated manual
       re-run of this same issue isn't born already at budget. Doesn't touch
       assignee -- the claim is still legitimately consumed and still needs a
       human or the dossier routine to close it, same as any other terminal
       verdict.

    A completed run with a fresh (unlabelled) retry budget, or an unrecognised/
    unparseable results.json status, is left alone. An issue whose body
    doesn't parse, or that has no run.log yet (still validating, or launch
    crashed before ever writing one), is left alone too -- not this function's
    job to guess at those.

    The age gate in shape 1 reads `updatedAt`, the only timestamp GitHub's list
    endpoint carries; bd had a dedicated `started_at` that a comment did not move.
    `updatedAt` therefore restarts the 12h clock every time this routine (or a human)
    comments on a live claim. That is conservative in the safe direction — it delays a
    release, never causes an early one — and a claim nobody is touching still ages out
    normally.

    Each returned entry carries an "action" ("release", "block", or
    "reset_retry"); "release" entries also carry the "retry_count" to record.
    """
    now = now or datetime.now(timezone.utc)
    stale: list[dict] = []
    for issue in _open_experiment_issues():
        if not _is_claimed(issue) or BLOCKED_LABEL in _label_names(issue):
            continue
        try:
            _, candidate_path = parse_candidate_ref(issue.get("body", ""))
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
                        f"already spent a retry -- clearing the spent `{RETRIED_LABEL}` label "
                        f"so a future, unrelated re-run of this SAME issue gets the full "
                        f"budget again (fps-rtd PR #304 review finding #3)."
                    ),
                })
            continue

        traceback_tail = _looks_like_traceback_tail(out_dir / RUN_LOG_FILENAME)
        if traceback_tail is None:
            continue
        claimed_at_raw = issue.get("updatedAt")
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
    """Post the reason and unassign one stale-claimed experiment issue.

    `traceback_tail` carries whichever evidence the caller found -- a
    traceback tail for a crashed run, or a one-line explanation for a run that
    finished in a retryable status -- so the issue records WHY it was released.

    `retry_count`, when given, is recorded as the `retried` label. **The label is
    added BEFORE the assignee is removed** (fps-rtd PR #304 review finding #2, ported):
    if a `gh` call partway through this sequence fails (network, rate limit, a killed
    process), the safer stuck state is "looks already-retried" (blocks one cycle early)
    rather than "counter never advanced" (the same fault gets released and retried
    forever). Under bd the equivalent ordering was `--set-metadata` before
    `--status open`, because `--status open` was what made the issue claimable again.
    Here it is the *unassign* that makes it claimable, so that is what the label has to
    precede -- the same invariant, attached to a different call.

    With MAX_RETRIES == 1 `retry_count` is only ever 1 when given; the boolean label
    can represent it exactly. See _retry_count for what has to change if that rises.
    """
    _gh_comment(issue, f"[launch] claim released for re-run.\n\n{traceback_tail}")
    if retry_count is not None:
        _gh_edit(issue, "--add-label", RETRIED_LABEL)
    _gh_edit(issue, "--remove-assignee", "@me")


def block_exhausted_claim(issue: dict, reason: str) -> None:
    """Retry budget spent (fps-rtd): block the claim instead of releasing it.

    A released-but-unassigned issue keeps its original creation date, so it's
    always the oldest result of the claim query and would be re-claimed ahead of
    everything else, forever -- the exact starvation this issue is about.

    Blocking is two things at once, and only one of them is mechanical: **the
    assignment** is what drops the issue out of the unassigned claim query, and
    **the `blocked` label** is what makes it visible for triage in the UI and is what
    claim_next_candidate filters on explicitly (not silently dropped -- module
    docstring covers the human recovery step). bd got both from one exclusive
    `blocked` status; GitHub needs both written. They go in one `gh issue edit` call so
    a partial failure can't leave the issue labelled-but-claimable or
    claimed-but-unlabelled.

    Assigning to `@me` rather than a hardcoded login: this routine runs locally as the
    owner (docs/routines/launch.md), so `@me` IS the owner, and the issue it is being
    handed back to is theirs to triage.
    """
    _gh_comment(issue, f"[launch] retry budget exhausted -- blocking, not releasing.\n\n{reason}")
    _gh_edit(issue, "--add-label", BLOCKED_LABEL, "--add-assignee", "@me")


def clear_retry_metadata(issue: dict, reason: str) -> None:
    """Reset a claim's spent retry budget once it reaches a real verdict
    (fps-rtd PR #304 review finding #3).

    Doesn't touch the assignee -- the claim is still held and still needs a
    human or the dossier routine to close it, same as any other
    TERMINAL_STATUSES verdict (see find_stale_claims). This only prevents a
    stale counter from an earlier failure cycle silently costing a later,
    unrelated cycle its retry.
    """
    _gh_comment(issue, f"[launch] clearing spent retry budget on this now-terminal claim.\n\n{reason}")
    _gh_edit(issue, "--remove-label", RETRIED_LABEL)


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
    return stale


def claim_next_candidate() -> dict | None:
    """Claim the oldest ready experiment issue, or None if the queue is empty.

    Ready is "open, `experiment`-labelled, unassigned, not `blocked`", oldest by
    creation date -- exactly what `bd ready --label experiment --unassigned
    --sort oldest -n 1` returned. The oldest-first ordering is what the fps-rtd retry
    budget is defined against; don't change it without re-reading that argument.

    **This is no longer atomic.** bd claimed in the same call that queried
    (`--claim`); GitHub needs a list then an edit. Accepted deliberately: there is one
    nightly process, and contention was measured at ~1 in-progress out of 28. If it
    ever matters, the genuinely atomic primitive already in use here is free --
    `git push origin HEAD:refs/heads/<branch>` fails if the ref exists, so branch
    existence can be the claim. Don't build a heavier protocol before there is
    contention to justify it.
    """
    ready = [
        issue for issue in _open_experiment_issues()
        if not _is_claimed(issue) and BLOCKED_LABEL not in _label_names(issue)
    ]
    if not ready:
        return None
    issue = min(ready, key=lambda i: i.get("createdAt") or "")
    _gh_edit(issue, "--add-assignee", "@me")
    return issue


def build_runner_cmd(batch_dir: pathlib.Path, candidate_path: pathlib.Path, bead_id: str) -> list[str]:
    """The detached runner invocation.

    `bead_id` now carries a GitHub issue NUMBER, not a bd id. The name is kept
    deliberately: it is also the `--bead-id` flag on runner.py and the `bead_id` field
    written into every results.json / facts.json, including the ones already committed
    under experiments/candidates/. Renaming it would change an artifact schema to buy
    nothing.
    """
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
    """Nightly entry point: recover stale claims, claim + validate + launch one candidate."""
    recovered = recover_stale_claims()
    verbs = {"block": "blocked", "reset_retry": "reset retry budget on"}
    for entry in recovered:
        verb = verbs.get(entry.get("action"), "recovered")
        click.echo(f"[launch] {verb} stale claim #{issue_ref(entry['issue'])}")

    issue = claim_next_candidate()
    if issue is None:
        click.echo("[launch] no experiment work ready")
        return

    ref = issue_ref(issue)
    try:
        batch_dir, candidate_path = parse_candidate_ref(issue.get("body", ""))
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

    cmd = build_runner_cmd(batch_dir, candidate_path, ref)
    out_dir = default_out_dir(candidate_path)
    try:
        pid = launch_detached(cmd, out_dir)
    except OSError as exc:
        # A validated candidate that fails to actually launch (missing `uv`,
        # permission error creating out_dir, etc.) must not leave the issue
        # claimed forever -- same "release rather than strand" rule as a
        # validation failure above.
        _abort_claim(issue, f"failed to launch detached runner: {exc!r}")
        return
    log_path = out_dir / RUN_LOG_FILENAME
    _gh_comment(issue, f"[launch] validated, launched detached pid={pid}, log={log_path}")
    click.echo(f"[launch] #{ref}: launched detached pid={pid} log={log_path}")


def _abort_claim(issue: dict, reason: str) -> None:
    """Validation failed before launch: apply the same retry budget as any
    other give-up path instead of releasing unconditionally (fps-rtd PR #304
    review finding #4) -- a candidate whose body or module is broken
    in a way that will never self-correct would otherwise re-win the claim query
    and burn the nightly slot forever, exactly the starvation shape this issue
    was filed about, just reached through pre-launch validation instead of a
    runtime abort. Reuses release_stale_claim / block_exhausted_claim so this
    path shares their (already-tested) call sequence rather than a third
    near-duplicate of it.
    """
    ref = issue_ref(issue)
    action, next_count = _decide_release_or_block(issue)
    if action == "block":
        block_exhausted_claim(issue, f"aborted before launch — {reason}")
    else:
        release_stale_claim(issue, f"aborted before launch — {reason}", retry_count=next_count)
    click.echo(f"[launch] #{ref}: aborted before launch ({action}) — {reason}")


if __name__ == "__main__":
    main()
