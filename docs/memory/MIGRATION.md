# `bd remember` → repo, the accounting

The Beads → GitHub cutover's phase 5 (issue
[#398](https://github.com/edwinsteele/fuel-price-signal/issues/398)). The 53 `bd
remember` entries were frozen at `docs/bd-archive/memories.json` in phase 1 and
sorted here on 2026-09-08. **No `bd` command was run** — bd no longer reflects
reality and is not to be invoked, read or write.

Every one of the 53 is listed below with where it went. Three destinations:

- **`docs/memory/`** (31) — atomic gotchas about specific code paths, data and tools.
- **`docs/CONVENTIONS.md`** (14) — git, worktree and GitHub workflow, the layer that
  file already owns. Prose rewritten to fit its **rule + Why** shape rather than
  pasted; a few were merged into one bullet where they were halves of one rule.
- **Deleted** (8) — 7 describe `bd`/Dolt mechanics that no longer exist, and one
  had *inverted* rather than gone stale.

The one deletion that is not merely moot:

> **`github-issue-state-meaningless-post-migration`** said *"never use `gh issue`
> state to reconcile — the bead is the only live record."* That was true from
> 2026-08-06, when the migration bulk-closed every GitHub Issue. After cutover
> phase 4 it is **actively dangerous**: `launch.py` now reads issue open/closed
> state and assignees to decide what is claimable. Deleted and replaced by
> *"GitHub Issue state is authoritative again, and code depends on it"* in
> CONVENTIONS.md § Filing and finding issues, which keeps the 2026-08-06
> bulk-close as the caveat a reader still needs.

## All 53

| # | `bd remember` key | Destination |
|---|---|---|
| 1 | `1password-ssh-push` | `docs/memory/` |
| 2 | `absolute-path-edit-wrong-worktree` | CONVENTIONS § Worktrees |
| 3 | `aggregate-py-unpaired-seed-median` | `docs/memory/` |
| 4 | `backfill-perf-work-needs-synthetic-benchmark` | `docs/memory/` |
| 5 | `baseline-fingerprint-before-comparing-runs` | `docs/memory/` |
| 6 | `batch-dir-vs-candidates-dir` | `docs/memory/` |
| 7 | `batch-freeze-stale-features` | `docs/memory/` |
| 8 | `bd-dolt-remote-add-auto-commits-config` | **deleted** — Dolt-only; no Dolt remote exists after phase 6 |
| 9 | `bd-field-wipe-recovery` | **deleted** — recovery recipe for a DB that is no longer written |
| 10 | `bd-handoff-goes-in-description` | **deleted** — `bd show`/`bd comment` mechanics; the transferable half (handoff pointers at the top of the body, comments are not handoff) promoted to CONVENTIONS § Filing and finding issues |
| 11 | `bd-search-limits` | **deleted** — `bd search` substring semantics; the corollary that durable instructions belong in pushed channels is already CONVENTIONS § Decisions land in repo docs |
| 12 | `beads-commit-tracked-state` | **deleted** — `.beads/` is removed in phase 6 |
| 13 | `beads-writes-through-to-primary-worktree` | **deleted** — described bd's DB auto-discovery |
| 14 | `branch-before-first-commit` | CONVENTIONS § Git workflow |
| 15 | `branch-prs-from-origin-main` | CONVENTIONS § Git workflow |
| 16 | `cadence-not-a-free-knob` | `docs/memory/` |
| 17 | `candidate-output-align-by-index-label` | `docs/memory/` |
| 18 | `concurrent-worktree-commit-by-another-session` | CONVENTIONS § Worktrees |
| 19 | `decide-feature-parity-gap` | `docs/memory/` |
| 20 | `decision-flip-substrate-is-fills-not-rowpreds` | `docs/memory/` |
| 21 | `default-flip-breaks-contrast-tests` | `docs/memory/` |
| 22 | `dolt-git-remote-push-needs-embedded-token` | **deleted** — the `fps-sk0` workaround; phase 7 retires the whole token-embedding dance |
| 23 | `experiments-pipeline-import-cycle` | `docs/memory/` |
| 24 | `five-all-nan-lga-trough-columns` | `docs/memory/` |
| 25 | `float-reformulation-not-strictly-more-accurate` | `docs/memory/` |
| 26 | `gh-pr-merge-worktree-conflict` | CONVENTIONS § Git workflow |
| 27 | `gh-pr-view-json-fields` | CONVENTIONS § Git workflow |
| 28 | `git-pull-diverge-daily-snapshot` | CONVENTIONS § Git workflow |
| 29 | `github-auto-deletes-merged-branch` | CONVENTIONS § Git workflow |
| 30 | `github-issue-state-meaningless-post-migration` | **deleted and replaced** — see above |
| 31 | `graduating-a-column-is-two-edits` | `docs/memory/` |
| 32 | `handover-before-results-csv-write` | `docs/memory/` |
| 33 | `https-push-when-1password-agent-down` | `docs/memory/` |
| 34 | `icloud-conflict-copies` | `docs/memory/` (already self-marked obsolete; kept so it isn't re-derived) |
| 35 | `no-db-writes-before-pr-merged` | CONVENTIONS § Git workflow |
| 36 | `noise-floor-fixture-needs-n-placebo-columns` | `docs/memory/` |
| 37 | `noise-floor-force-vs-cadence-relock` | `docs/memory/` |
| 38 | `non-primary-worktree-branch-stale-after-pr-merge` | CONVENTIONS § Worktrees |
| 39 | `np-std-float-degeneracy` | `docs/memory/` |
| 40 | `pandas-assert-equal-default-tolerance-hides-leaks` | `docs/memory/` |
| 41 | `primary-worktree-stray-branch-postmerge` | CONVENTIONS § Worktrees (the `.beads/interactions.jsonl` half dropped — that file stops existing in phase 6; the stranded-branch half is the durable part) |
| 42 | `prose-justifications-fail-silently` | `docs/memory/` |
| 43 | `results-csv-main-only` | CONVENTIONS § Git workflow |
| 44 | `screen-and-arbiter-share-no-population` | `docs/memory/` |
| 45 | `search-artifact-noun-before-filing` | CONVENTIONS § Filing and finding issues (rewritten for `gh`; the `bd search` substring mechanics dropped with #11) |
| 46 | `squash-merge-breaks-branch-ancestry` | CONVENTIONS § Worktrees |
| 47 | `status-rejected-means-graded-not-failed` | `docs/memory/` |
| 48 | `tau-is-calibrated-not-raw` | `docs/memory/` |
| 49 | `tau-selector-is-cadence-blind` | `docs/memory/` |
| 50 | `test-discrimination-claims-need-mutation` | `docs/memory/` |
| 51 | `venv-corruption-concurrent-uv` | `docs/memory/` |
| 52 | `webfetch-403-data-nsw-gov-au` | `docs/memory/` |
| 53 | `worktree-missing-gitignored-batch-data` | `docs/memory/` |

## What this migration does not cover

Claude's **private** memory dir (`~/.claude/projects/…/memory/`, ~157 files) is
untouched. It holds a large amount of material that is about *this repo* rather
than about the owner, and which Codex therefore cannot see — the same argument
that put these 53 in-repo applies to it. That is a separate, larger migration,
along with the write-routing mechanism (a CLAUDE.md rule and a redirect at the top
of the private `MEMORY.md`) needed to stop new project-technical memories being
written to the private dir by default.

The `fps-*` citations throughout this repo (1,439 across 119 files) are
deliberately **not** rewritten. They resolve by lookup in [../bd-id-map.md](../bd-id-map.md) and the frozen
corpus at `docs/bd-archive/`;
rewriting lab-book entries would falsify a record. Settled 2026-09-07.
