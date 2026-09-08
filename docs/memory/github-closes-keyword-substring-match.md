---
name: github-closes-keyword-substring-match
description: GitHub's auto-close parser matches "Closes #N" as a literal substring, even inside a sentence explaining that it's deliberately NOT a closing reference — it has no concept of negation.
metadata:
  type: reference
---

Writing `Closes #398 is deliberately **not** included — this PR only fixes X` in a PR body still
closes #398 on merge. GitHub's keyword parser (`close(s)`/`fix(es)`/`resolve(s)` + `#N`) is a
plain substring match against the rendered body text; it does not parse the surrounding English,
so a sentence explaining that a closing reference was deliberately omitted *is itself* a closing
reference.

Hit live 2026-09-08: PR #401's body said "`Closes #398` is deliberately not included — phase 7
isn't done until a live fire confirms it" (correct english, wrong assumption about how the parser
works). #398 closed on merge anyway. Caught via the post-merge checklist's issue-state check,
reopened with an explanatory comment, no other damage — but it could as easily have gone
unnoticed on an issue nobody was actively watching.

**The only safe way to write about an issue number you are NOT closing:** don't put the word
"Closes"/"Fixes"/"Resolves" directly before its `#N` anywhere in the body, including inside a
negated clause. Say "issue #398" or "phase 7 (`#398`)" instead, and reserve the actual keyword
line for the one issue the PR is genuinely meant to close.

This is a general GitHub behavior, not specific to this repo's automation, but it interacts badly
with this repo's own `Closes #<N>` convention (CLAUDE.md's "For each PR" section, and the
worker-routine pickup rules): `Closes #<N>` is the *entire* mechanism that closes an issue on
merge post-Beads, checked mechanically by GitHub's parser, not by intent — so a body that merely
*discusses* an issue number defensively needs to avoid the keyword entirely, not just phrase it
carefully.
