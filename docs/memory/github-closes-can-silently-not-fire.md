---
name: github-closes-can-silently-not-fire
description: A merged PR's Closes #N can fail to close the issue even when GitHub shows the closing reference registered — check the issue's state, never the link.
metadata:
  type: reference
---

Observed live 2026-09-10: PR #406 merged into `main` (the default branch) with a correctly
formed `Closes #377` in its body. GitHub *registered* the link — `closingIssuesReferences`
on the PR listed #377 — but the issue was still `open`, `state_reason: null`, more than
twenty minutes after the merge. It only closed later, when an unrelated direct commit to
`main` carried its own `Closes #377`.

**Cause unknown.** No conflicting reopen, no non-default base branch, no malformed keyword —
all the usual explanations were checked and ruled out. A sweep of every other recent merged
PR (#400–#407) found all their closing references had fired normally, so this is rare rather
than systemic. Don't build a theory on one occurrence; build the check.

**The rule: `closingIssuesReferences` being present is NOT evidence the issue closed.** They
are two different pieces of state and this is the case that proves they can disagree. The
post-merge checklist in CLAUDE.md already says to confirm closure — do it by reading the
issue's own state, which is the only thing that settles it:

```bash
gh api repos/{owner}/{repo}/issues/<N> --jq '{state, state_reason}'
```

and `gh issue close <N>` if it's still open.

This is the false-*negative* direction of the same untrustworthy mechanism whose false-
*positive* direction is [[github-closes-keyword-substring-match]] (a `Closes #N` inside a
negated sentence still closes). Both land on the same conclusion: `Closes #<N>` is the whole
post-Beads closure mechanism, it is checked mechanically rather than by intent, and it can
fail in either direction — so the post-merge state check is not optional bookkeeping.
