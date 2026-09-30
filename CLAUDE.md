# fuel-price-signal — Claude instructions

For architecture, module layout, data strategy, signal logic, code review and
agent workflow conventions, see [AGENTS.md](AGENTS.md). It is the shared
instructions file for every agent working in this repo — Claude Code and
Codex both — and this file is deliberately just a pointer to it plus the
handful of things that are Claude-Code-specific tooling rather than repo
convention. How-we-work rules are in [docs/CONVENTIONS.md](docs/CONVENTIONS.md);
current build state is [docs/STATUS.md](docs/STATUS.md).

**Read AGENTS.md before doing anything else in this repo.**

## Automated worker vs interactive session

### If you are the scheduled worker routine

Follow [docs/routines/worker.md § Rules](docs/routines/worker.md#rules) exactly —
pickup, PR and maintenance rules all live there.

### If you are an interactive session

Follow [docs/CONVENTIONS.md § Interactive sessions](docs/CONVENTIONS.md#interactive-sessions),
including its post-merge checklist, and § PR feedback loop after every PR.

## Claude-Code-specific tooling

- **Model/effort:** Sonnet for implementation (downloader, transformer, DB
  layer, tests); Opus for analytically hard design (cycle detection math,
  backtest engine architecture, leading indicator analysis).
- **`codex-pr-review` skill** — `~/.claude/skills/codex-pr-review/SKILL.md`,
  user-level. Use it for every PR review-status check; the raw facts are in
  AGENTS.md § Code review if the skill is unavailable.
- **`api-contract` skill** — the workflow for changing `docs/api-contract.md`
  with the iOS app's session; AGENTS.md and CONVENTIONS § API contract carry
  the rules.
- **`ScheduleWakeup(delaySeconds=270)`** right after `gh pr create` — the
  review wait in CONVENTIONS § PR feedback loop. Not available to the worker
  routine, which uses `sleep 270`.
- **`spawn_task` → propose a GitHub issue instead.** When `spawn_task` would
  normally be the right call, don't spawn a session: ask the owner, then
  `gh issue create` per CONVENTIONS § Filing and finding issues.
- Claude's own private per-project memory (this session's `MEMORY.md`) is for
  facts about the **owner** — how Edwin likes to work, cross-project
  preferences — not for facts about this repo's code, data or tools. Those go
  in [`docs/memory/`](docs/memory/INDEX.md) per AGENTS.md, so Codex sees them
  too.
