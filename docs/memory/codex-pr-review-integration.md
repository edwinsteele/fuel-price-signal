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

**UPDATE 2026-09-12: the auto-trigger unreliability below was a setup-period problem, now
resolved — do not manually post `@codex review` any more.** The #409/#411 silence dated to
right after the connector was enabled (2026-09-09); by PR #423 (2026-09-12) the auto-trigger
fired automatically and reliably twice in one PR — once on PR open (review posted 5m27s
later, unprompted) and once on a fixup push (review posted ~4m after the push, unprompted,
correctly reviewing the new commit). No manual `@codex review` was needed either time. Still
check for Codex activity the way you'd check for Sourcery's (see the channels below — findings
land as inline comments, a clean pass can be a bare reaction), but **absence after the wait
means "hasn't reached this PR yet", not "never triggered"** — do not nudge it with a manual
`@codex review` comment. The paragraph immediately below is kept as the historical record of
the setup-period failure mode, not current guidance.

**[Historical, setup period only] The PR-open auto-trigger was NOT reliable — never assume it
already ran.** Confirmed firing automatically on #410 (review posted 4m44s after PR open, no
comment asked for it). Confirmed NOT firing on #409 and #411 — both sat completely silent (no
comment, no review, no reaction, no check-run) until a manual `@codex review`, and in both
cases the repo's Codex environment was already configured (no "create an environment" nag
either, which is the tell for that separate failure mode). This was the state of the
integration in its first three days (2026-09-09 to 2026-09-11); see the 2026-09-12 update
above for the current, resolved behaviour.

**Standard practice: announce Codex's status the same way you already announce
Sourcery's.** When you see Codex has posted (a review with findings, or the "Codex
Review: Didn't find any major issues" comment), say so in your own PR-status narration —
"Codex has started a review" / "Codex hasn't found any issues" / "Codex found N issues,
addressing them" — exactly the same register as "checking for Sourcery's review". This
was requested explicitly (2026-09-10, PR #411 aftermath) after a review where Codex's
clean pass wasn't surfaced to the user at all.

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

**[Historical, setup period] Codex re-reviews pushes, but on PR #410 (2026-09-09-10, the same
setup period as the trigger issue above) it ran six automatic passes then stopped, leaving the
last three commits unreviewed with no notice.** Whether that was a re-review cap or a rate
limit tied to the setup-period problems above is not established, and it has not recurred since
(PR #423, 2026-09-12, re-reviewed a fixup push normally). Per the 2026-09-12 update at the top
of this file, **do not manually post `@codex review`** to work around a suspected stall —
if a later PR with many pushes does show this again, note it fresh rather than reaching for the
old remedy.

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

## A third clean-pass channel: a bare 👍 reaction, no comment or review at all

PR #412 (worker run, 2026-09-10, owner confirmed this was an **"Exhaustive review"**
depth setting on Codex's side) added a THIRD observed clean-pass signal, distinct from
both channels above:

- Findings → review + inline comments (documented above).
- Clean pass, "normal" depth → an issue comment: "Codex Review: Didn't find any major
  issues."
- Clean pass, **this run** → **no review, no inline comments, no issue comment at
  all** — just a `+1` reaction from `chatgpt-codex-connector[bot]` left directly on
  the PR body. No dedicated Codex check-run showed up either (only `signal-diff`,
  `test`, and `Sourcery review` on this commit) — the reaction was the *only* trace
  Codex had looked at the PR.

Check for it explicitly; a status check that only reads comments/reviews/check-runs
will report "no Codex activity" on a PR Codex has actually cleared:

```bash
gh api repos/{owner}/{repo}/issues/<N>/reactions --paginate \
  --jq '.[] | select(.user.login == "chatgpt-codex-connector[bot]")'
```

**Timing on #412:** the reaction landed 2m17s after the last commit was pushed
(`a811ccf` authored 21:19:08Z → reaction at 21:21:25Z, ~10s after the `test` check
finished at 21:21:15Z) — i.e. it waited for CI before reacting, or its own pass just
happened to land right after. Measuring from the *original* PR-open time (21:10:53Z)
instead gives 10m32s, but that span includes an intervening Sourcery finding and a
fixup push, so it overstates Codex's own latency — **use the latest pushed commit's
timestamp as the baseline, not PR-open, whenever a PR had a mid-review fixup.** One
sample; re-measure before trusting either figure generally, and note whether
"Exhaustive" vs. default review depth is configured when comparing across PRs.

**Second confirmation, PR #414 (2026-09-11):** the silent-`+1`-reaction clean pass is not
a one-off. Single-commit `polish` PR, no fixup push: commit `7a602c6` authored
20:05:57Z → `chatgpt-codex-connector[bot]` `+1` reaction on the PR body at 20:08:36Z,
~2m39s later — in the same ~2-3 min band as #412. Sourcery was simultaneously
rate-limited on this same PR (`COMMENTED` review, budget exhausted) — the two bots'
statuses are independent and must both be checked and both narrated; one being silent
or unavailable says nothing about the other. Reported to the owner as: "Sourcery:
rate-limited, skipped. Codex: reviewed, clean (silent `+1`)." — **that paired framing
(what each of Sourcery and Codex did, even when one did nothing) is the expected
standing format for narrating PR review status, not just for Codex in isolation.**
