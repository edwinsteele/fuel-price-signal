---
name: gh-auth-status-false-negative-restricted-token
description: gh auth status reports the worker Routine's token as invalid; it is not, it's just GraphQL-restricted. Use gh api user to actually verify.
metadata:
  type: reference
---

`gh auth status` failed outright in the worker Routine's environment on the 2026-09-08 live fire
(phase 7 of #398, second attempt):

```
X Failed to log in to github.com using token (GH_TOKEN)
- The token in GH_TOKEN is invalid.
```

The token is not invalid. `gh api user` (plain REST) succeeded immediately with the same
`GH_TOKEN`/`GITHUB_TOKEN`, returning the correct account. So did every REST issues/pulls call the
rewritten pickup rules use (see [[gh-issue-list-consistency]]'s addendum for the GraphQL-vs-REST
background). `gh auth status` apparently validates the token via a GraphQL call under the hood,
which 403s under this token's PR-review-only scope restriction exactly like every other GraphQL
call does — it's the same blocker CLAUDE.md's pickup rules were already rewritten around, just
hitting a command the rewrite didn't touch.

This matters because CLAUDE.md's rule 0 for the scheduled worker literally runs `gh auth status`
as its "is gh usable" check and says: *"If it fails, fail hard and say so; do not proceed as though
there were no work."* Taken literally, that stops every run in this environment before it starts,
even though the token works fine for everything the rest of the routine actually does.

**What actually confirms gh works here:** `gh api user` returning a real login, not `gh auth
status` exiting 0. Rule 0 should be read (or rewritten) that way until it's updated to swap the
check. Filed as a follow-up: see the issue this memory's commit references.
