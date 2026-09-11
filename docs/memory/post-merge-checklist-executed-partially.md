---
name: post-merge-checklist-executed-partially
description: The interactive-session post-merge checklist (CLAUDE.md) was run for the merger's own branch/issue but silently skipped the "sync other clean, behind worktrees" step — the owner had to point it out.
metadata:
  type: feedback
---

CLAUDE.md's post-merge checklist has four items: confirm the issue closed, delete
the now-merged local branch (expected to fail from inside its own worktree —
don't force it), fast-forward *other* worktrees that are clean and now behind
`main`, and never sweep (remove) another worktree. After merging #421 (#415),
steps 1 and 2 were done correctly, but step 3 was skipped entirely — not
attempted and not mentioned as skipped. The primary worktree sat two commits
behind `origin/main` (clean, so purely a missed `git pull --ff-only`) until the
owner asked "why are there uncommitted changes here?" and it had to be found and
fixed reactively.

**Why this step is the one that gets dropped:** steps 1–2 are about the branch
you were just working on — they're front of mind because they're literally what
you just did. Step 3 requires deliberately turning attention to worktrees you
were *not* just working in, which is easy to treat as "someone else's problem"
or forget exists at all once your own branch is tidied up. The checklist reads
as one list, but psychologically it's two different attentional moves.

**How to apply:** after any merge (via `gh pr merge` or the API), before
declaring the close-out done, explicitly run `git worktree list` and check each
listed worktree's status — not just the one you were working in. A worktree
qualifies for `git pull --ff-only` only if BOTH: (a) `git status --short` is
empty, and (b) it's now behind `main` (a fast-forward is possible). Do this as
its own explicit step with its own tool call, not as something you'll "remember
to check" while focused on your own branch — this is exactly the failure mode
that dropped it last time. Still never remove or otherwise mutate a worktree
that isn't clean-and-behind, and never touch another worktree's checked-out
branch, in-progress edits, or untracked files (that part of the rule held).
