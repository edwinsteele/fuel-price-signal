# Technical memory index

Atomic, load-bearing gotchas about this repo — the kind that are cheap to read and
expensive to rediscover. One fact per file, with `name` / `description` / `type`
frontmatter and wiki-style `[[name]]` links between related facts.

**Where things live.** Rules for *how we work* (git, PRs, review, experiment
scripts) are in [CONVENTIONS.md](../CONVENTIONS.md). Architecture is in
[AGENTS.md](../../AGENTS.md). Current model state is in [STATUS.md](../STATUS.md).
This directory holds the residue that fits none of those: traps in specific code
paths, specific data, and specific tools.

**Reading it.** `grep -l <term> docs/memory/*.md`, or scan the hooks below and open
the one file you need. Each hook is written to be enough to tell you whether you
need the file, not enough to substitute for it.

**Adding one.** See [AGENTS.md § Technical memories](../../AGENTS.md#technical-memories)
for the write protocol — classify, check for a duplicate, one fact per file, add a hook
here. Straight to `main`, no PR.

Provenance: these were `bd remember` entries until 2026-09-08; see
[MIGRATION.md](MIGRATION.md) for the full accounting of all 53.

## Before you compare two runs

- [baseline-fingerprint-before-comparing-runs](baseline-fingerprint-before-comparing-runs.md) — different fingerprint = not commensurable, whatever the deltas say. Current lock `54:1a6ec2d84a69`; a missing fingerprint is itself the signal.
- [screen-and-arbiter-share-no-population](screen-and-arbiter-share-no-population.md) — the WFCV screen and the realised arbiter share **0.71%** of their rows. Per-fold sign disagreement is expected, not a signal.
- [aggregate-py-unpaired-seed-median](aggregate-py-unpaired-seed-median.md) — published per-fold deltas are a difference of medians over *paired* seeds. Deliberately unfixed; compute the paired version yourself if you read signs.
- [tau-is-calibrated-not-raw](tau-is-calibrated-not-raw.md) — τ=0.25 is calibrated, `rowpreds.proba` is raw. Equivalent raw threshold ≈0.11–0.16; comparing them is a category error worth ~0.10 of probability.
- [decision-flip-substrate-is-fills-not-rowpreds](decision-flip-substrate-is-fills-not-rowpreds.md) — rowpreds comes from a *different model* than the arbiter. Diff `fills.parquet`.

## The locked feature set

- [graduating-a-column-is-two-edits](graduating-a-column-is-two-edits.md) — append to `LOCKED_FEATURE_COLUMNS` **and** delete from `NON_MODEL_COLUMNS`. Order is part of the contract; the contract test skips in CI.
- [five-all-nan-lga-trough-columns](five-all-nan-lga-trough-columns.md) — five locked columns are permanently 100% NaN, so a complete-case mask over the locked set selects **zero rows**, silently.
- [decide-feature-parity-gap](decide-feature-parity-gap.md) — `backtest.py`'s `decide()` recomputes features independently of `features.py`. A new feature family needs wiring in both or the realised backtest aborts.

## Cadence, tau and the tank

- [brim-bridge-threshold-unvalidated](brim-bridge-threshold-unvalidated.md) — `FALLING_CENTS_PER_DAY = -0.5` was reasoned, not measured, and lands on the **median** of the drift distribution (p50 = -0.47), splitting days 49/51. Least stable cut available.
- [cadence-not-a-free-knob](cadence-not-a-free-knob.md) — oracle-vs-model headroom is only meaningful at **1, 2 and 7 days**; 3–6 and 8–14 are invalid (run-dry paths diverge between the two engines).
- [tau-selector-is-cadence-blind](tau-selector-is-cadence-blind.md) — the selector never sees the tank, so an unmoved τ after a cadence re-lock proves nothing. Harmless by measurement, not by construction.
- [pbuy-is-station-relative](pbuy-is-station-relative.md) — `P(BUY)` is measured against each station's **own** trailing percentile, so it cannot rank stations. Sorting by it points at the dearest pump.
- [noise-floor-force-vs-cadence-relock](noise-floor-force-vs-cadence-relock.md) — `--force` recovers a floor after a *column* re-lock, but is refused after a *cadence* re-lock. Freeze a new batch instead.

## Pipeline layout and plumbing

- [batch-dir-vs-candidates-dir](batch-dir-vs-candidates-dir.md) — `experiments/batches/<b>/` vs `experiments/candidates/<b>/` are different dirs with near-identical names; the wrong one yields a plausible-but-false "never written" reason.
- [experiments-pipeline-import-cycle](experiments-pipeline-import-cycle.md) — `dossier_tables → runner → batch_freeze` is load-bearing; import `noise_floor` inside function bodies, never at module level.
- [status-rejected-means-graded-not-failed](status-rejected-means-graded-not-failed.md) — `STATUS_REJECTED` means "finished grading", win or lose. Not a verdict.
- [candidate-output-align-by-index-label](candidate-output-align-by-index-label.md) — align by index label; a positional `.to_numpy()` pairs rows with the wrong station and the run still passes validation.
- [batch-freeze-stale-features](batch-freeze-stale-features.md) — the old "run features before freezing" workaround is obsolete; `refresh_db()` hard-gates both now.
- [handover-before-results-csv-write](handover-before-results-csv-write.md) — hand over to the user *before* any step that writes `experiments/results.csv`. Running it is the violation.

## Numerical and test traps

- [np-std-float-degeneracy](np-std-float-degeneracy.md) — `np.std` of identical floats is 1.78e-18, not 0.0. A `std > 0` guard reads a degenerate band as a confident **reject**. Pair with `np.ptp`. The artefact is length-dependent (n=20 yes, n=50 no), so a test at the wrong n is vacuous.
- [pandas-assert-equal-default-tolerance-hides-leaks](pandas-assert-equal-default-tolerance-hides-leaks.md) — default `rtol=1e-5` silently passes a real leak at YYYYMMDD magnitudes. Leak tests need `check_exact=True`.
- [float-reformulation-not-strictly-more-accurate](float-reformulation-not-strictly-more-accurate.md) — a "more exact" path trades rounding failure modes rather than removing them. Check against exact rational arithmetic before claiming equivalence.
- [default-flip-breaks-contrast-tests](default-flip-breaks-contrast-tests.md) — flipping a default silently disarms every test that used the new value as its contrast arm. Five of six kept passing while comparing a value to itself.
- [noise-floor-fixture-needs-n-placebo-columns](noise-floor-fixture-needs-n-placebo-columns.md) — a 2+ column fixture without `n_placebo_columns` trips the arity refusal branch instead of what the test meant to exercise.
- [backfill-perf-work-needs-synthetic-benchmark](backfill-perf-work-needs-synthetic-benchmark.md) — the 492MB profiling DB is gitignored and usually absent; write a synthetic parity + direction check and cite the experiment for the production-scale number.

## Writing claims that nothing executes

- [prose-justifications-fail-silently](prose-justifications-fail-silently.md) — "X is this way BECAUSE Y" is the claim nothing tests. Ask what would fail if Y were false; if nothing, delete the sentence or attach the measurement.
- [test-discrimination-claims-need-mutation](test-discrimination-claims-need-mutation.md) — "this test catches X" needs the mutation *run*, and needs you to read the line that answers the question. `tail -n` truncates exactly the count you were asking for.

## Environment and tooling traps

- [venv-corruption-concurrent-uv](venv-corruption-concurrent-uv.md) — `ModuleNotFoundError` for `six`/`py` means two concurrent `uv run`s half-synced `.venv`. Only fix: `rm -rf .venv && uv sync`. Run pipeline stages sequentially.
- [worktree-missing-gitignored-batch-data](worktree-missing-gitignored-batch-data.md) — gitignored batch/candidate artifacts don't exist in a fresh worktree; copy them in first or the run dies with `FileNotFoundError`.
- [webfetch-403-data-nsw-gov-au](webfetch-403-data-nsw-gov-au.md) — WebFetch gets 403 on `data.nsw.gov.au`; `requests` gets a clean 200. Use `history.py`'s discovery or the Browser tool.
- [1password-ssh-push](1password-ssh-push.md) — `Permission denied (publickey)` means the 1Password SSH agent is down. **Never** `ssh-add`; the owner has ruled that out.
- [https-push-when-1password-agent-down](https-push-when-1password-agent-down.md) — the HTTPS fallback works, but silently leaves `origin/main` stale, which is how a worktree branched off an old commit.
- [codex-pr-review-integration](codex-pr-review-integration.md) — findings are INLINE comments (`pulls/N/comments`), not review bodies; `gh pr view` shows an empty-looking review. A Sourcery FAILURE may be blocking security findings, not the rate limit.
- [gh-pr-merge-delete-branch-fails-in-worktree](gh-pr-merge-delete-branch-fails-in-worktree.md) — "fatal: 'main' is already used by worktree" comes AFTER a successful merge. Verify, never retry.
- [gh-issue-list-consistency](gh-issue-list-consistency.md) — `gh issue list --search`/`--assignee` are search-index backed and lag a mutation 2–4s; plain `--label` does not. Every form lags *creation* ~7s.
- [gh-auth-status-false-negative-restricted-token](gh-auth-status-false-negative-restricted-token.md) — `gh auth status` reports the worker Routine's restricted token as invalid (it isn't — that check is GraphQL too). Verify with `gh api user` instead.
- [github-closes-keyword-substring-match](github-closes-keyword-substring-match.md) — `Closes #N` closes the issue even inside a sentence saying it's deliberately *not* a closing reference. Never put the keyword directly before a `#N` you don't mean to close.
- [auto-merge-workflow-does-not-fire-closes-keyword](auto-merge-workflow-does-not-fire-closes-keyword.md) — the mirror image: a PR merged by `auto-merge.yml` (GITHUB_TOKEN) does NOT close its `Closes #N` issue. That is the normal path for worker `chore` PRs, and worker rule 1 assumes the opposite.
- [github-closes-can-silently-not-fire](github-closes-can-silently-not-fire.md) — the converse: a merged PR's `Closes #N` can fail to close the issue while GitHub still shows the reference registered. Check the issue's `state`, not the link.
- [icloud-conflict-copies](icloud-conflict-copies.md) — **obsolete** since the repo moved out of `~/Documents`. Kept only so nobody re-derives the detection recipe.
