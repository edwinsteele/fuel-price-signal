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
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
from datetime import datetime, timedelta, timezone

import click

from experiments.pipeline.runner import default_out_dir
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


def find_stale_claims(now: datetime | None = None) -> list[dict]:
    """Read-only: which in_progress experiment issues look abandoned mid-run.

    Stale iff: no results.json (run never completed), run.log exists and its tail
    looks like a Python traceback, and the issue was claimed more than STALE_AFTER
    ago. An issue whose description doesn't parse, or that has no run.log yet (still
    validating, or launch crashed before ever writing one), is left alone here --
    not this function's job to guess at those.
    """
    now = now or datetime.now(timezone.utc)
    stale: list[dict] = []
    for issue in _bd_json("list", "--status", "in_progress", "--label", EXPERIMENT_LABEL):
        try:
            _, candidate_path = parse_candidate_ref(issue.get("description", ""))
        except CandidateRefError:
            continue
        out_dir = default_out_dir(candidate_path)
        if (out_dir / RESULTS_FILENAME).exists():
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
        stale.append({"issue": issue, "traceback_tail": traceback_tail})
    return stale


def release_stale_claim(issue: dict, traceback_tail: str) -> None:
    """Post the traceback, unassign, and reopen one stale-claimed experiment issue."""
    issue_id = issue["id"]
    subprocess.run(
        ["bd", "comment", issue_id, "--stdin"],
        input=(
            f"[launch] stale claim recovered — run.log ended in a traceback with no "
            f"results.json, claimed more than {STALE_AFTER} ago. Releasing.\n\n{traceback_tail}"
        ),
        text=True,
        check=True,
    )
    subprocess.run(["bd", "assign", issue_id, ""], check=True)
    subprocess.run(["bd", "update", issue_id, "--status", "open"], check=True)


def recover_stale_claims(now: datetime | None = None) -> list[dict]:
    stale = find_stale_claims(now=now)
    for entry in stale:
        release_stale_claim(entry["issue"], entry["traceback_tail"])
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
    for entry in recovered:
        click.echo(f"[launch] recovered stale claim {entry['issue']['id']}")

    issue = claim_next_candidate()
    if issue is None:
        click.echo("[launch] no experiment work ready")
        return

    issue_id = issue["id"]
    try:
        batch_dir, candidate_path = parse_candidate_ref(issue.get("description", ""))
    except CandidateRefError as exc:
        _abort_claim(issue_id, f"malformed candidate reference: {exc}")
        return

    try:
        candidate = load_candidate_module(candidate_path)
        frame = load_features(batch_dir / "features.csv")
        validate_candidate(candidate, frame)
    except CandidateImportError as exc:
        _abort_claim(issue_id, f"candidate module failed to import: {exc}")
        return
    except Exception as exc:  # noqa: BLE001 — any validation failure aborts the claim, not the routine
        _abort_claim(issue_id, f"validation failed: {exc!r}")
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
        _abort_claim(issue_id, f"failed to launch detached runner: {exc!r}")
        return
    log_path = out_dir / RUN_LOG_FILENAME
    subprocess.run(
        ["bd", "comment", issue_id, f"[launch] validated, launched detached pid={pid}, log={log_path}"],
        check=True,
    )
    sync_push()
    click.echo(f"[launch] {issue_id}: launched detached pid={pid} log={log_path}")


def _abort_claim(issue_id: str, reason: str) -> None:
    """Validation failed before launch: release the claim rather than launch a doomed run."""
    subprocess.run(["bd", "comment", issue_id, f"[launch] aborted before launch — {reason}"], check=True)
    subprocess.run(["bd", "assign", issue_id, ""], check=True)
    subprocess.run(["bd", "update", issue_id, "--status", "open"], check=True)
    sync_push()
    click.echo(f"[launch] {issue_id}: aborted — {reason}")


if __name__ == "__main__":
    main()
