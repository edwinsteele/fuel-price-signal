---
name: zsh-equals-word-aborts-command-list
description: The agent shell is zsh; a word starting with `=` (e.g. `echo ===` as a separator) is =-expansion, fails with "== not found", and aborts the rest of the `;`-joined command line — later commands silently never run.
metadata:
  type: project
---

The Bash tool here runs **zsh**, not bash. In zsh a word beginning with `=` is
`=`-expansion (`=ls` → `/bin/ls`), so `===`, `====` or `=====` as a visual
separator is a command lookup for `==`, which fails:

```
(eval):1: == not found
```

It is an expansion error, so zsh **abandons the whole command list** — every
command after it on a `;`-joined line is skipped, and the tool reports exit 1.
Hit three times in one session (2026-10-01); one of those silently skipped a
second `codex-wait.sh` call, so the second PR's review status was never
checked and the failure read as "the wait failed".

**How to apply:** use `echo '==='`, `echo ---`, or `printf '%s\n' ===` for
separators, and quote any other argument that starts with `=`. For a script that
must be bash, run it as `bash -c '…'` or a file with a bash shebang — a
`#!/usr/bin/env bash` script is unaffected; only the interactive command line is
zsh. Another zsh difference that bites in the same way: `$?` after a pipeline
is the last stage's; the first stage's is `${pipestatus[1]}` (1-indexed), not
bash's `${PIPESTATUS[0]}`.
