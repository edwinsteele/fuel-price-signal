# Automation workflow

> **Status (2026-08-15): the Cloud Routine is disabled.** `bd dolt push` from inside a Routine sandbox hits an HTTP 403 pushing to Dolt's git-ref namespace — a Claude Code Routines platform limitation (sandboxed git credentials can't push non-standard refs), not something fixable from this repo. Root cause, live reproduction, and an upstream report are documented in bd issue `fps-sk0` (status `blocked`, P3). The mechanics below remain accurate and this doc doesn't need rewriting if the Routine is re-enabled later — check `fps-sk0` first to see if the upstream blocker has moved.

This document describes how `chore` and `polish` issues flow from filing to merged PR with minimal owner involvement, while `design` issues stay manual.

Issues referenced below live in Beads (`bd`), not GitHub Issues — see [AGENTS.md § Beads](../AGENTS.md#beads). PRs, CI, and review threads are still on GitHub; only issue tracking moved.

## Routine prompts live in `docs/routines/`, not in the scheduler

Each *scheduled* routine's stored prompt (a Claude Code Routine's `job_config`, or a local scheduled-task `SKILL.md`) is a three-line shim — who the routine is, where the repo checkout is, and which repo doc to follow. The actual rules live in a tracked `docs/routines/<name>.md` (or, for this worker, directly in CLAUDE.md's ["If you are the scheduled worker routine"](../CLAUDE.md#if-you-are-the-scheduled-worker-routine) section, which `docs/routines/worker.md` points to). This is deliberate: instructions duplicated into the scheduler are instructions nobody remembers to update when the workflow changes — see `docs/routines/worker.md` for the incident that motivated it.

Not every file under `docs/routines/` has a scheduler entry, though — `docs/routines/generator.md` (fps-3jj.7, the AI-sourced feature pipeline's candidate generator) is invoked interactively by the owner at batch setup, not on a schedule, so there is no shim for it anywhere. It's still tracked here rather than typed fresh each time, for the same reason as the shimmed ones.

## Issue labels

| Label | Who files | Who works it | Merge path |
|-------|-----------|--------------|------------|
| `chore` | Owner or worker (via spawn_task redirect) | Worker routine | Auto-merged once CI green + `auto-merge-ok` applied (see below) |
| `polish` | Owner or worker | Worker routine | Owner review required |
| `design` | Owner | Owner (interactive) | Normal PR review |
| `claude-authored` | Applied by worker automatically | — | Identifies worker-opened PRs |
| `auto-merge-ok` | Applied by worker to `chore` PRs on open | — | Triggers `.github/workflows/auto-merge.yml` |

`experiment` doesn't fit the table above — it's a bd-issue label only, with no PR path (the
runner it triggers never opens a PR). It marks a candidate-feature bead for the separate local
launch routine (fps-3jj.5, [docs/routines/launch.md](routines/launch.md)) and exists so the
chore/polish worker's `bd ready --label chore`/`--label polish` queries never claim one and try to
implement it in a cloud container with no data.

## State machine

```
Issue filed (chore/polish)
        │
        ▼
Worker picks up (next scheduled run, no open claude-authored PRs)
        │
        ├─ Implements minimal change
        ├─ Runs ruff + pytest locally
        └─ Opens PR ready-for-review (claude-authored + chore|polish;
           chore PRs also get auto-merge-ok); 3–5 bullet plan in the PR body
                │
                ▼
        CI runs (lint, test, signal-regression)
                │
          ┌─────┴──────┐
        fail           pass
          │               │
        Worker    ┌───────┴────────┐
        fixes &  chore            polish
        pushes     │                │
             auto-merge.yml    Owner reviews
             sweep merges           │
             once PR age  ┌─────────┴──────────┐
             ≥900s   Comments left         No comments
                           │                    │
                   Worker addresses        Owner merges
                   on next scheduled run
                           │
                   [worker] Done / Needs owner input
                   reply per thread + push
                           │
                   Owner resolves threads + merges
```

For `polish` issues that turn out to need design work:

```
Worker discovers design work needed
        │
        ├─ Relabels issue: polish → design
        ├─ Posts comment: why it needs design + what the question is
        └─ Stops (no code written), moves to next issue in batch
```

## Review response

On each scheduled run the worker checks open `claude-authored` PRs for unresolved review threads before looking for new issues. A thread needs a response if it is unresolved and has no comment starting with `[worker]`.

The worker reads all actionable threads together, makes the changes in one pass, pushes, then replies to each thread:
- `[worker] Done — <one sentence>` for addressed threads
- `[worker] Needs owner input — <question>` for anything ambiguous or requiring a design decision

You resolve the threads and merge when satisfied. The `[worker]` prefix is how the worker avoids re-processing threads it has already replied to.

## WIP cap

The worker keeps at most **one** open `claude-authored` PR at a time. Before picking up issues, it checks `gh pr list --label claude-authored --state open`. If any open PR exists, it exits without doing anything (one PR at a time, ready-for-review — no draft/batch flow).

This means: review (or merge) the open PR before the worker will pick up anything new.

## Pausing the worker

The worker is a scheduled remote Claude Code routine. To pause it:
1. Go to the Claude Code scheduled tasks and disable the routine, **or**
2. Open a PR manually with the `claude-authored` label — the WIP cap will stop the worker from picking up anything.

(Currently disabled via option 1 — see the status note at the top of this doc.)

## Spend monitoring

Each worker run (twice daily — `0 9,20 * * *` UTC) uses Sonnet. The WIP cap of 3 issues per batch limits spend. Each run that finds no work exits in seconds (cheap). Implement sessions with actual work are estimated at ~$0.10–0.30 per issue depending on complexity.

Monitor spend in the Anthropic console. If costs are unexpectedly high, check whether the worker is getting stuck in retry loops (visible in the routine's run history).

## Override / emergency stop

If a worker PR is causing problems and you need to stop everything immediately:
1. Close all open `claude-authored` PRs.
2. Disable the scheduled routine in Claude Code settings.
3. File a `design` issue describing what went wrong so there's a record.

## Bootstrap sequence

The workflow was set up in this order (so future readers understand the dependency):

1. **Labels + CLAUDE.md + issue templates + PR template + this doc** — landed first, so the worker has conventions to follow from its first run.
2. **CI enrichment** (signal-regression check) — landed second, so CI is informative before the worker starts opening PRs.
3. **Worker routine** — activated last, once the full infrastructure was in place.
