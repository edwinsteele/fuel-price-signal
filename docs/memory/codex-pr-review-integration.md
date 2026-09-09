---
name: codex-pr-review-integration
description: "Codex PR review posts in ~254s, well inside auto-merge's 900s window; it needs a cloud environment created first, and its rules live in AGENTS.md § Code Review Rules under a hard 32 KiB budget."
metadata:
  type: reference
---

The ChatGPT Codex connector reviews PRs here. Enabled 2026-09-09; first live review was
PR #408.

**It needs a cloud environment for the repo before it will do anything.** Without one it
still comments on every new PR, but only to say "To use Codex here, create an environment
for this repo". That comment is not a review and does not mean the integration is broken.

**Triggers:** PR opened for review, draft marked ready, or a `@codex review` comment.
`@codex` alone also worked. It answers questions and can push fixes with
`@codex address that feedback` when it has push permission.

**Measured latency: 254s** (PR #408, `@codex` at 21:47:34Z → review posted at 21:51:48Z).
That matters because `.github/workflows/auto-merge.yml` merges a green worker `chore` PR
at `MIN_AGE_SECONDS=900`, and a reviewer slower than that would land its findings on an
already-merged, already-deleted branch — the PR #201 failure the window exists to prevent.
At 254s there is 3.5x margin and **no change to `MIN_AGE_SECONDS` is needed**. Caveat: one
sample, on a docs-only diff. Re-measure before trusting it for a large code diff.

**Its rules live in [AGENTS.md](../../AGENTS.md) § Code Review Rules**, the section name the
GitHub integration looks for. That file is capped at 32 KiB (see the budget note in its
header and `tests/test_agents_md_budget.py`), so review rules compete for budget with
architecture — keep the section a compressed pointer into
[CONVENTIONS.md § Code review caution](../CONVENTIONS.md#code-review-caution) and
[INDEX.md](INDEX.md), never a copy of them.

**Sourcery's budget is 250,000 diff characters per 7 days**, shared with the owner's own
manual reviews, and it refuses with a `COMMENTED` review saying so. Same handling as a
CodeRabbit rate-limit: skip and move on, do not reschedule. Related:
[[gh-issue-list-consistency]].
