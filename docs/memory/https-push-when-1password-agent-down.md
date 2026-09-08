---
name: https-push-when-1password-agent-down
description: HTTPS push works while the 1Password agent is down, but silently staleness origin/main.
metadata:
  type: reference
---

When the 1Password SSH agent is down ([[1password-ssh-push]] — git push -> 'Permission denied (publickey)'), you can still push and fetch over HTTPS without touching the user's remote config, because 'gh auth setup-git' has a credential helper installed: 'git push https://github.com/edwinsteele/fuel-price-signal.git <branch>:<branch>'. Use this when the owner is away and cannot open 1Password; still tell them SSH is broken.

GOTCHA THAT BITES NEXT: pushing via an explicit URL does NOT update refs/remotes/origin/main, and 'git fetch origin' fails for the same SSH reason — so origin/main silently goes stale. 'git worktree add -b <branch> origin/main' then branches from an OLD commit, which is how a fix worktree ended up missing an already-merged change (2026-08-19). Refresh the tracking ref the same way: 'git fetch https://github.com/edwinsteele/fuel-price-signal.git main:refs/remotes/origin/main'. Or just branch from local 'main', which stays correct — but note that conflicts with the standing rule to branch PRs off `origin/main` (docs/CONVENTIONS.md § Git workflow), so refresh the tracking ref instead when you are about to cut a PR branch. Do NOT run ssh-add — the owner has ruled that out.
