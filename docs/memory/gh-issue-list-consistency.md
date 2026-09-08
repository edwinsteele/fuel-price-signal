---
name: gh-issue-list-consistency
description: gh issue list --search and --assignee are search-index backed and lag a mutation 2-4s; plain --label does not. Every form lags creation ~7s.
metadata:
  type: reference
---

`gh issue list` has two consistency classes, and automation that mutates an issue then re-reads a
list in the same run must know which it is using.

Measured on this repo 2026-09-08, `gh` 2.98.0, polling at 0.5s:

| query | issue CREATED appears | assignee change reflected |
|---|---|---|
| `gh issue list --label X --state open` | ~6.8-7.3s | **0.65-0.71s** (first read) |
| `gh issue list --label X --assignee "@me"` | ~7s | **3.74s** |
| `gh issue list --search "label:X no:assignee sort:created-asc"` | ~7.0s | 1.96s assign / 3.07s unassign |

Two separate facts:

1. **Every** form lags issue *creation* by ~7s, `--label` included. An issue filed seconds
   before an automated sweep runs is invisible to it.
2. **`--assignee` is search-index backed, same as `--search`.** Its 3.74s matches the `--search`
   column, not the plain-listing column. Only the bare `--label` listing reflects a mutation on
   the first read.

So a filter flag is not free: reaching for `--assignee` to narrow a query silently moves it onto
the eventually-consistent index.

So: list by `--label`, filter client-side for assignee, `blocked`, and ordering. `--label` returns
newest-first with no ascending-sort flag, hence the client-side oldest-by-`createdAt` pick in
`experiments/pipeline/launch.py` (see its `ISSUE_LIST_LIMIT` comment for the measurements).

Measured 2026-09-08 during the Beads to GitHub cutover. It bit twice in one port: a claim query
that could not see the issue it had just blocked, and a stale sweep that could not see a release.
Mocked tests were green through both -- 30/30 on the mocked-equivalent paths, then 9/30 on the
first live run. See [[worktree-missing-gitignored-batch-data]] for the other flavour of "the test
environment is not the real one".
