---
name: 1password-ssh-push
description: 'Permission denied (publickey)' on push means the 1Password SSH agent is down — never ssh-add.
metadata:
  type: reference
---

git push failing with 'Permission denied (publickey)' on this machine means the 1Password SSH agent isn't running (it holds the GitHub key, not a plain ~/.ssh file). Fix: ask the user to open/check 1Password, then retry. If the owner is away and cannot open it, see [[https-push-when-1password-agent-down]] for the HTTPS fallback. Do NOT run ssh-add — the user has explicitly ruled that out as the wrong path.
