---
name: codex-pr-review-integration
description: "Codex findings live in INLINE review comments (pulls/N/comments), not review bodies — gh pr view shows an empty-looking review. Sourcery FAILURE can be blocking security findings, not just the rate limit."
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

## Where the findings actually are — read this before saying "no review yet"

**Codex puts its findings in INLINE REVIEW COMMENTS, not in the review body.** The
review body is boilerplate ("Here are some automated review suggestions… Reviewed
commit: `<sha>`") and carries no findings at all. So:

```bash
# WRONG — shows only the boilerplate wrapper, looks like an empty review
gh pr view <N> --json reviews,comments

# RIGHT — the actual findings
gh api repos/{owner}/{repo}/pulls/<N>/comments --paginate
```

`gh pr view --json comments` returns ISSUE comments (the PR conversation);
`--json reviews` returns review bodies. Neither includes inline review comments.

**A CLEAN result arrives on a different channel from findings.** When Codex has
findings it posts a *review* plus inline comments; when it has none it posts an
**issue comment** — "Codex Review: Didn't find any major issues. :rocket:" with its
own `Reviewed commit:` line — and no review at all. So polling `pulls/N/reviews`
for the sha to advance waits forever on a clean pass. Check both, or just check the
issue comments for the newest `Reviewed commit:` line:

Confirmed again on PR #411 (`@codex review` → "Codex Review: Didn't find any major
issues. You're on a roll." as an issue comment, no review). The flavor text after
"Didn't find any major issues." varies between runs (":rocket:" on #410, "You're on
a roll." on #411) — match on the "Codex Review: Didn't find any major issues"
prefix and the `Reviewed commit:` line, not the exact sentence.

```bash
gh api repos/{owner}/{repo}/issues/<N>/comments --paginate \
  --jq '.[] | select(.user.login|startswith("chatgpt")) | .body' | grep -o 'Reviewed commit:.*`'
```
Missing this on PR #410 meant reporting "no substantive review yet" while six P1
findings sat on the diff. Also check `.[].line` — some findings come back with
`line: null` (outdated/file-level) and are easy to skip when eyeballing.

**Codex re-reviews pushes, but NOT indefinitely — do not assume it will catch up.**
On PR #410 it ran six automatic passes (~4-6 min behind each push) and then stopped,
leaving the last three commits unreviewed with no notice. Whether that is a
re-review cap like Sourcery's five, or a rate limit, is not established; what is
established is that "it will get to it" is not safe. `@codex review` re-triggers it
and is the documented way to cover commits it skipped.

A pass whose "Reviewed commit" is not HEAD has not seen your latest fix — absence of
new findings there means "hasn't looked", not "clean". **Check the reviewed sha
against HEAD before concluding anything, and say which of the two you mean:**

```bash
gh api repos/{owner}/{repo}/pulls/<N>/reviews --paginate \
  --jq '.[] | select(.user.login|startswith("chatgpt")) | .body' \
  | grep -o 'Reviewed commit:.*`'
```

**A Codex finding's qualifiers are load-bearing.** On #410 it said to derive the
routing date in Sydney "(separately from any freshness reference date)"; the
parenthetical was dismissed as pedantry and the two collapsed into one date, which
it then re-reported a round later as a reproducibility bug. If you are about to
drop part of a finding, work out what it was for first.

## Sourcery fails the check in two different ways — do not conflate them

A `Sourcery review` check reading FAILURE is **not** necessarily the rate limit.
It also posts **blocking security findings** (opengrep rules) as inline comments,
and those turn the check red. Read `output.title` / `output.summary` before
attributing it:

```bash
gh api repos/{owner}/{repo}/commits/<sha>/check-runs \
  --jq '.check_runs[] | select(.name|test("Sourcery")) | {conclusion, title: .output.title}'
```

On #410 that distinction was missed twice in a row, reporting a red blocking check
as a harmless rate limit. The rate limit arrives as a `COMMENTED` **review**;
security findings arrive as a **failed check** plus inline comments.

Known live rule: `python.sqlalchemy.security.sqlalchemy-execute-raw-query` fires on
any SQL built by concatenation, even when every concatenated piece is a hard-coded
constant and the values are bound. Prefer a single constant query with a bind
parameter that is always present (e.g. an integer floor of `0` for "unbounded")
over a conditionally-appended clause — it satisfies the rule honestly rather than
by suppression, and removes the shape that invites a later edit to interpolate.

**Sourcery's budget is 250,000 diff characters per 7 days**, shared with the owner's own
manual reviews, and it refuses with a `COMMENTED` review saying so. Same handling as a
CodeRabbit rate-limit: skip and move on, do not reschedule. Related:
[[gh-issue-list-consistency]].
