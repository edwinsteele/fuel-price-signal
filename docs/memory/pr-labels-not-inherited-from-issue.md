---
name: pr-labels-not-inherited-from-issue
description: Opening a PR for a design-labelled issue doesn't carry that label onto the PR — signal-regression.yml gates on the PR's own labels and silently `skipped` on PR #424 until the label was added by hand.
metadata:
  type: reference
---

`gh pr create` does not copy the source issue's labels onto the PR. On
[PR #424](https://github.com/edwinsteele/fuel-price-signal/pull/424) (issue
#418, `design`-labelled), the PR was opened with no `--label`, and
`.github/workflows/signal-regression.yml`'s `signal-diff` job — which only
runs `if: contains(github.event.pull_request.labels.*.name, 'design')` —
came back `"conclusion": "skipped"` on the check-run list. There was no error,
no red X, nothing in the PR body about it; the job simply never ran, and the
live-DB regression check the repo relies on for `design`-labelled changes
(diffing signal output for fixed `--as-of` dates between base and PR) was
silently absent.

The workflow re-triggers on the `labeled` event (`on.pull_request.types`
includes `labeled`), so `gh pr edit <N> --add-label design` after the fact is
enough to make it run — but only if someone notices the skip in the first
place. `mergeable_state` and the other check-runs (`test`, `Sourcery review`)
all showed green the whole time; only reading `check_runs[].conclusion` for
every entry (not just skimming for failures) surfaces a `skipped` job.

**Consequence:** when opening a PR for a `design` (or otherwise
label-gated-workflow) issue, pass the label at `gh pr create` time
(`--label design`), or check `commits/<sha>/check-runs` afterward for any
`"conclusion": "skipped"` entries before treating a green PR as fully
checked.
