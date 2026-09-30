---
name: run-outputs-live-where-the-run-ran
description: Long runs execute in the primary checkout while analysis happens in a worktree — read the primary by absolute path, edit only in the worktree, and never trust a gitignored *.log without checking which checkout's copy it is.
metadata:
  type: reference
---

Long pipeline runs are launched from the primary checkout (`~/Code/fuel-price-signal`, on
`main`); the session analysing them is usually in a `.claude/worktrees/<slug>` worktree. Two
traps, both hit in one session (2026-09-07, batch1 410 re-runs):

**Editing where you read.** `cd`-ing into the primary to inspect results, then editing there out
of momentum, lands edits in the wrong checkout — twice in that session, each needing a copy
across and a `git -C <primary> restore`. Read the primary with **absolute paths** (never `cd`),
edit and commit only in the worktree. To track outputs, copy just the tracked files across
(`results.json`; parquets and logs are gitignored), verify with `md5 -q` on both sides, then
delete the untracked copies in the primary — an untracked file colliding with an incoming
tracked one blocks `git pull` there.

**Gitignored outputs don't travel and diverge silently.** Verifying a write-up against
`experiments/.../analyse.log` returned a screen of false MISSING results: the primary held the
*previous* session's log, the worktree the new one. Same path, same name, no git signal. Check
the mtime (`ls -la`) of the copy you are reading, or better, verify against the tracked
artifact or a fresh re-run.

Related: [[worktree-missing-gitignored-batch-data]].
