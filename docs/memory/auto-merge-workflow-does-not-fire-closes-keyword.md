---
name: auto-merge-workflow-does-not-fire-closes-keyword
description: A PR merged by auto-merge.yml (GITHUB_TOKEN, github-actions[bot]) does NOT auto-close its Closes #N issue. Human/API merges do. This silently breaks worker rule 1 for exactly the chore PRs the worker creates.
metadata:
  type: reference
---

`Closes #<N>` in a PR body is the *entire* mechanism that closes an issue on merge in this repo
post-Beads (CLAUDE.md "For each PR"). **It does not fire when `auto-merge.yml` does the merging.**

`.github/workflows/auto-merge.yml` runs `gh pr merge` with
`GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}`. GitHub deliberately suppresses downstream event
triggering for actions taken with the default `GITHUB_TOKEN`, and the linked-issue auto-close is
one of the things suppressed. The merge succeeds; the issue just stays open.

Observed 2026-09-10, four PRs merged within ~24h, clean contrast:

| PR | merged by | body | issue | auto-closed? |
|---|---|---|---|---|
| #405 | `edwinsteele` | `Closes #376` | #376 | **yes** |
| #402 | `edwinsteele` | `Closes #367` | #367 | **yes** |
| #407 | `edwinsteele` (API merge) | `Closes #403` | #403 | **yes** |
| #406 | `github-actions[bot]` | `Closes #377` | #377 | **NO** |

#377's timeline shows only a `referenced` event from `github-actions[bot]` one second after the
merge — never a `closed` event. It sat open until the owner closed it by hand ~31 minutes later.
One observation of the bot path, but the mechanism is documented GitHub behaviour and the
three-way contrast against human merges is clean.

**Why this is load-bearing, not trivia.** The two facts compose badly:

- A `chore` PR from the worker gets the `auto-merge-ok` label specifically so `auto-merge.yml`
  merges it — so **the bot path is the normal path for worker PRs**, not an edge case.
- Worker pickup **rule 1 says "There is nothing to close out. A merged PR whose body carries
  `Closes #<N>` closes its issue by itself, so a run no longer starts by reconciling merged PRs
  against the tracker."** That premise is false on exactly the PRs the worker produces.

So every auto-merged worker `chore` issue can be left open indefinitely, and rule 1 explicitly
removed the reconciliation pass that used to catch it. The issue also stays *assigned*, which
means the stale-claim recovery in rule 4 will not touch it either (it has a merged PR, so it
never looks orphaned).

Note this is the mirror image of [[github-closes-keyword-substring-match]]: that one is the
parser firing when you did not want it to, this one is the merge path not firing it when you did.
Both mean **the post-merge issue-state check is not optional** — CLAUDE.md's interactive
post-merge checklist step 1 (`gh issue view <N> --json state`, close by hand if still open) is the
only thing standing between this and a silently-drifting tracker.
