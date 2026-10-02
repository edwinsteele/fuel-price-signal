# Weekly cleanup — project rules

Read by the owner's user-level `weekly-cleanup` skill when the
`fuel-price-signal-weekly-cleanup` local scheduled task runs (Tuesdays). The
skill holds the general procedure — worktrees and branches, memory, docs
review, wrap-up; this file holds only what is specific to this repo. Where they
disagree, this file wins.

The scheduled task's stored prompt is the three-line shim, nothing else (same
pattern as [worker.md § The shim](worker.md#the-shim)):

```text
You are the fuel-price-signal weekly cleanup routine.
The primary checkout is ~/Code/fuel-price-signal.
Run the weekly-cleanup skill; project rules are in docs/routines/weekly-cleanup.md.
```

## Issue filing

Propose only — never run `gh issue create`. The owner is asked before any issue
is filed ([CONVENTIONS § Filing and finding issues](../CONVENTIONS.md#filing-and-finding-issues)),
and this runs unattended. Shape each proposal the way that section describes
(artifact noun in the title; one routing label `chore`/`polish`/`design` plus a
topic label). Anything needing a design decision is a proposed `design` issue.

## Doc changes

Doc-only changes (`AGENTS.md`, `CLAUDE.md`, `docs/**`, including
`docs/memory/**`) commit and push straight to `main` —
[CONVENTIONS § Git workflow](../CONVENTIONS.md#git-workflow). A change touching
code, tests or `.github/**` is not doc-only; propose it instead.

## Leave alone

- `docs/api-contract.md` — canonical contract for the fps-app client. Changes
  go through the `api-contract` workflow with the app's session, never a
  cleanup edit, even for an apparent typo.
- `experiments/**` — the lab book. Each dir is a record of what was run.
- `docs/bd-archive/**` and `docs/bd-id-map.md` — frozen Beads corpus and its
  lookup table.

## Link-check exclusions

- `experiments/**` — lab-book records; a link that was right when written stays.
- `docs/bd-archive/**` — frozen corpus.

## Not dead links

Historical `fps-*` (Beads) ids in prose and lab-book entries are intentional
citations that resolve via [bd-id-map.md](../bd-id-map.md) /
[bd-archive/](../bd-archive/). Never rewrite them and never flag them. `bd` has
been removed from this repo — don't run it or recommend it.

## Single sources of truth

- [docs/STATUS.md](../STATUS.md) — current model state: feature count,
  calibration, τ, phase, what's shipped. Other docs link to it rather than
  restating those numbers.

## Private memory

Claude's private memory here is mid-migration: most notes are repo facts that
belong in `docs/memory/` (claude-codex-setup § 5). The migration is done in
reviewed batches, so a cleanup run never *creates* a `docs/memory/` note from a
private one. Everything else in the skill's Part 2 still applies to every
private note, repo fact or not:
- retire notes for finished work and notes contradicted by the repo;
- merge duplicates;
- delete a private note whose fact a `docs/memory/` note, `AGENTS.md` or
  `docs/CONVENTIONS.md` already carries (the owner's own `MEMORY.md` policy).

Report how many private notes still look like repo facts.

## Extra checks

None yet.
