# Worker routine — prompt source

Canonical, tracked home for the prompt used by the "Fuel Price Signal Chore and Polish Worker" scheduled Routine (`trig_01Mhkd4YLBpuGLhcXLMHaZVQ`, cron `0 9,20 * * *` UTC, environment `env_013cCwqEo6PFqNSNxsYgny1k` — an Anthropic-managed cloud environment that gets a fresh checkout every run).

## Why this file exists

Before this file, the actual pickup/PR rules were duplicated into two places outside the repo:

1. The Routine's stored `job_config` message (edited only through the Routines UI/API).
2. A dormant local `~/.claude/scheduled-tasks/fuel-price-signal-worker/SKILL.md` on the owner's Mac.

Both drifted out of step with CLAUDE.md, in opposite directions across a year: they were written against GitHub Issues, went stale when the 2026-08-06 Beads migration moved the tracker to `bd`, and are stale again now the 2026-09-07 cutover moved it back. Two untracked copies of the same instructions are two places to forget to update — twice over, here.

The fix is to keep exactly **one tracked copy of the shim text**, here. The scheduler's stored prompt — remote `job_config` or local `SKILL.md` — should hold nothing but a pointer to CLAUDE.md, never the rules themselves. When the pickup/PR process changes, only CLAUDE.md's ["If you are the scheduled worker routine"](../../CLAUDE.md#if-you-are-the-scheduled-worker-routine) section needs editing; a three-line shim has nothing substantive left to go stale.

This is the intended pattern for **every** scheduled routine in this project, not just this one — add `docs/routines/<name>.md` for each new routine (see PLAN_ml_signal.md's feature-pipeline routines) rather than writing instructions straight into the scheduler.

## The shim

This is the exact text that should be the Routine's stored prompt (`job_config.ccr.events[0].data.message.content` for the remote trigger; the `SKILL.md` body for a local scheduled task):

```
You are the fuel-price-signal worker routine.
Your working directory already has a fresh checkout of `edwinsteele/fuel-price-signal`.
Follow CLAUDE.md's "If you are the scheduled worker routine" section exactly.
```

Three lines, nothing else: who you are, where the repo is, where the rules live. No embedded steps, no `gh` commands, no reminders about what CLAUDE.md's section says — CLAUDE.md is the only place that gets to say what the rules are, so the shim doesn't paraphrase any of it.

## Applying it

- **Remote Routine** (`trig_01Mhkd4YLBpuGLhcXLMHaZVQ`): needs to be updated via the Routines UI (or `update_trigger`) with `prompt` set to the shim above. **Not done as part of this change** — the trigger was created via the Routines UI directly (`created_via: "http_api"`), and `update_trigger` only permits an agent session to modify a trigger it created itself via `create_trigger`. This is an owner action.
- **Local scheduled task** (`~/.claude/scheduled-tasks/fuel-price-signal-worker/SKILL.md` on the owner's Mac): not reachable from a repo PR or a cloud session — needs the owner to delete it (it's currently orphaned/non-firing) or replace its body with the shim above.
- Once both are updated, trigger one live run (`fire_trigger` or wait for the next scheduled fire) and confirm the transcript follows CLAUDE.md's pickup rules and nothing else. This is phase 7 of [#398](https://github.com/edwinsteele/fuel-price-signal/issues/398). **Both owner actions above were already done** by the time of the first fire (2026-09-08), and that fire found a real blocker: the Routine's environment token only serves a pinned set of PR-review GraphQL operations, and every `gh issue/pr list/view --json`/`gh issue/pr edit` call the pickup rules used resolves through GraphQL, so they all 403'd. CLAUDE.md's pickup rules are now rewritten to use `gh api` (REST) instead — untested against the real restricted token. A second live fire is what's left to confirm phase 7.
