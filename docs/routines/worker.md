# Worker routine — prompt source

Canonical, tracked home for the prompt used by the "Fuel Price Signal Chore and Polish Worker" scheduled Routine (`trig_01Mhkd4YLBpuGLhcXLMHaZVQ`, cron `0 9,20 * * *` UTC, environment `env_013cCwqEo6PFqNSNxsYgny1k` — an Anthropic-managed cloud environment that gets a fresh checkout every run).

## Why this file exists

Before this file, the actual pickup/PR rules were duplicated into two places outside the repo:

1. The Routine's stored `job_config` message (edited only through the Routines UI/API).
2. A dormant local `~/.claude/scheduled-tasks/fuel-price-signal-worker/SKILL.md` on the owner's Mac.

Both copies predated the Beads migration (2026-08-06, PR #278) and both drifted: they told the worker to use `gh issue list`, `gh issue edit --add-label design`, and `closes #N` commit messages — none of which match CLAUDE.md's current `bd`/Dolt-based pickup rules. Two untracked copies of the same instructions are two places to forget to update.

The fix is to keep exactly **one tracked copy of the shim text**, here. The scheduler's stored prompt — remote `job_config` or local `SKILL.md` — should hold nothing but a pointer to CLAUDE.md, never the rules themselves. When the pickup/PR process changes, only CLAUDE.md's ["If you are the scheduled worker routine"](../../CLAUDE.md#if-you-are-the-scheduled-worker-routine) section needs editing; a three-line shim has nothing substantive left to go stale.

This is the intended pattern for **every** scheduled routine in this project, not just this one — add `docs/routines/<name>.md` for each new routine (see PLAN_ml_signal.md's feature-pipeline routines) rather than writing instructions straight into the scheduler.

## The shim

This is the exact text that should be the Routine's stored prompt (`job_config.ccr.events[0].data.message.content` for the remote trigger; the `SKILL.md` body for a local scheduled task):

```
You are the fuel-price-signal worker routine. Your working directory already has a fresh checkout of `edwinsteele/fuel-price-signal`. Follow CLAUDE.md's "If you are the scheduled worker routine" section exactly — all issue tracking is in `bd`/Dolt, not GitHub Issues; do not use `gh issue` commands.
```

Three lines' worth of content: who you are, where the repo is, where the rules live. Nothing else — no embedded steps, no `gh` commands.

## Applying it

- **Remote Routine** (`trig_01Mhkd4YLBpuGLhcXLMHaZVQ`): needs to be updated via the Routines UI (or `update_trigger`) with `prompt` set to the shim above. **Not done as part of this change** — the trigger was created via the Routines UI directly (`created_via: "http_api"`), and `update_trigger` only permits an agent session to modify a trigger it created itself via `create_trigger`. This is an owner action.
- **Local scheduled task** (`~/.claude/scheduled-tasks/fuel-price-signal-worker/SKILL.md` on the owner's Mac): not reachable from a repo PR or a cloud session — needs the owner to delete it (it's currently orphaned/non-firing) or replace its body with the shim above.
- Once both are updated, trigger one live run (`fire_trigger` or wait for the next scheduled fire) and confirm the transcript follows only CLAUDE.md's `bd`-based pickup rules, with no `gh issue` calls.
