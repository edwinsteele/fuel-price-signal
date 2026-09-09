---
name: gh-pr-merge-delete-branch-fails-in-worktree
description: "`gh pr merge --delete-branch` from a worktree dies with \"fatal: 'main' is already used by worktree\" AFTER the merge has already succeeded. Verify state; never retry."
metadata:
  type: reference
---

Running `gh pr merge <N> --squash --delete-branch` from inside a worktree fails with:

```
failed to run git: fatal: 'main' is already used by worktree at '/Users/esteele/Code/fuel-price-signal'
```

**The merge already happened.** `gh` merges through the API first, then tries local
cleanup — switching the current checkout to the base branch before deleting the head
branch. In this repo `main` is permanently checked out in the primary worktree, so git
refuses the switch and `gh` reports a failure for work that is done. The remote head
branch is deleted too (repo auto-deletes on merge), so the only thing that did not happen
is the local checkout dance.

**Never retry the merge on this error** — a retry finds the PR already closed and produces
a second, genuinely confusing failure. Verify instead:

```bash
gh api repos/{owner}/{repo}/pulls/<N> --jq '{state, merged, merged_at, merge_commit_sha}'
```

Then resync the worktree by hand: `git fetch origin`, confirm the branch's content landed
(`git diff HEAD origin/main -- $(git diff --name-only origin/main...HEAD)` empty — a squash
merge never reads as an ancestor), and `git reset --hard origin/main` on a clean tree.
Hit live 2026-09-09 on PR #408. Related: [[worktree-missing-gitignored-batch-data]].
