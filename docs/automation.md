# Automation workflow

This document describes how `chore` and `polish` issues flow from filing to merged PR with minimal owner involvement, while `design` issues stay manual.

## Issue labels

| Label | Who files | Who works it | Merge path |
|-------|-----------|--------------|------------|
| `chore` | Owner or worker (via spawn_task redirect) | Worker routine | Owner review required |
| `polish` | Owner or worker | Worker routine | Owner review required |
| `design` | Owner | Owner (interactive) | Normal PR review |
| `claude-authored` | Applied by worker automatically | — | Identifies worker-opened PRs |

## State machine

```
Issue filed (chore/polish)
        │
        ▼
Worker picks up (next hourly run, no open claude-authored PRs)
        │
        ├─ Posts plan comment (3–5 bullets)
        ├─ Implements minimal change
        ├─ Runs ruff + pytest locally
        └─ Opens draft PR (claude-authored + chore|polish)
                │
                ▼
        CI runs (lint, test, signal-regression)
                │
          ┌─────┴──────┐
        fail           pass
          │               │
        Worker         Mark ready-for-review
        fixes &            │
        pushes         Owner reviews and merges
```

For `polish` issues that turn out to need design work:

```
Worker discovers design work needed
        │
        ├─ Relabels issue: polish → design
        ├─ Posts comment: why it needs design + what the question is
        └─ Stops (no code written), moves to next issue in batch
```

## WIP cap

The worker keeps at most **one batch of 3 draft PRs** open at a time. Before picking up issues, it checks `gh pr list --label claude-authored --state open`. If any open PR exists, it exits without doing anything.

This means: review your open PRs before filing more issues if you want the queue to move.

## Pausing the worker

The worker is a scheduled remote Claude Code routine. To pause it:
1. Go to the Claude Code scheduled tasks and disable the routine, **or**
2. Open a draft PR manually with the `claude-authored` label — the WIP cap will stop the worker from picking up anything.

## Spend monitoring

Each worker run (hourly) uses Sonnet. The WIP cap of 3 issues per batch limits spend. Each run that finds no work exits in seconds (cheap). Implement sessions with actual work are estimated at ~$0.10–0.30 per issue depending on complexity.

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
