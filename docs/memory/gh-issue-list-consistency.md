---
name: gh-issue-list-consistency
description: gh issue list --search and --assignee are search-index backed and lag a mutation 2-4s; plain --label does not. Every form lags creation ~7s.
metadata:
  type: reference
---

`gh issue list` has two consistency classes, and automation that mutates an issue then re-reads a
list in the same run must know which it is using.

- `--search "..."` **and** `--assignee` go through GitHub's search index: a mutation this run just
  made is invisible for **2-4s** after it lands.
- Plain `--label` goes through the REST listing: the same mutation is visible on the **first**
  read (measured 0.65-0.71s).
- **Every** form lags issue *creation* by about **7s**, `--label` included. That only matters if
  you file an issue and immediately list for it in the same run.

So: list by `--label`, filter client-side for assignee, `blocked`, and ordering. `--label` returns
newest-first with no ascending-sort flag, hence the client-side oldest-by-`createdAt` pick in
`experiments/pipeline/launch.py` (see its `ISSUE_LIST_LIMIT` comment for the measurements).

Measured 2026-09-08 during the Beads to GitHub cutover. It bit twice in one port: a claim query
that could not see the issue it had just blocked, and a stale sweep that could not see a release.
Mocked tests were green through both -- 30/30 on the mocked-equivalent paths, then 9/30 on the
first live run. See [[worktree-missing-gitignored-batch-data]] for the other flavour of "the test
environment is not the real one".
