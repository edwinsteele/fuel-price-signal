# fuel-price-signal — Claude instructions

For project architecture, CLI patterns, data strategy, signal logic, and automation conventions, see [AGENTS.md](AGENTS.md).

## Orientation (read these when picking up cold)

- [AGENTS.md](AGENTS.md) — architecture, CLI pattern, data strategy, signal logic
- [docs/STATUS.md](docs/STATUS.md) — current build state; what's shipped vs pending
- [docs/CONVENTIONS.md](docs/CONVENTIONS.md) — code & workflow rules (the changeable how-we-do-things layer)
- [docs/ML_PIPELINE.md](docs/ML_PIPELINE.md) — CLI reference for training/evaluating/diagnosing the ML model (dev reference, not day-to-day usage)
- [docs/ML_SIGNAL.md](docs/ML_SIGNAL.md) — ML model design decisions; [docs/feature-pipeline.md](docs/feature-pipeline.md) — the AI-sourced candidate-feature pipeline's machinery
- `PLAN_ml_signal.md` — active ML-signal plan. **Lives at repo root and is gitignored** (despite some docs saying `docs/PLAN_ml_signal.md` — that path is wrong).
- **Work items live in GitHub Issues** (`gh issue list`); the current backlog opened at #365–#388
  on 2026-09-07. `fps-*` ids in older prose are Beads issue ids from the 2026-08/09 experiment and
  resolve by lookup, not by any live command: [docs/bd-id-map.md](docs/bd-id-map.md) for the 24
  that migrated, [docs/bd-archive/](docs/bd-archive/) for the rest. `bd` is gone — do not try to
  run it.
- [docs/memory/INDEX.md](docs/memory/INDEX.md) — this repo's atomic technical gotchas (pipeline layout, numerical traps, environment traps). Short, load-bearing, and cheap to read; several are rules you will otherwise break before noticing. Git/worktree/GitHub discipline moved to [docs/CONVENTIONS.md](docs/CONVENTIONS.md) instead.

## Model/effort guidance

- Sonnet for implementation (downloader, transformer, DB layer, tests)
- Opus for analytically hard design: cycle detection math, backtest engine architecture, leading indicator analysis

## Reuse from old projects

The port from the original repos is done. If you ever need to trace original logic, the source lives in `~/Code/ff-aws-backend` (primary: `recommendations.py`, `purchasing_strategy.py`, FuelCheck OAuth task, Click `cli.py`) and `~/Code/petrol_prices` (secondary: transformer/downloader/gap-fill commands, postcode→LGA map). Do not carry over the AWS/Django infrastructure (DynamoDB, S3, SQS, SNS, Serverless, Django ORM) or `jsonpickle` / `msrest`.

## Automated worker vs interactive session

Work items live in **GitHub Issues** — see [AGENTS.md § Issue tracking](AGENTS.md#issue-tracking)
for the general model.

### If you are the scheduled worker routine

> **Status: disabled since 2026-08-15, and the reason is now gone.** It was disabled solely
> because `bd dolt push` could not authenticate from a Routine sandbox (`fps-sk0`,
> [#384](https://github.com/edwinsteele/fuel-price-signal/issues/384), closed as moot). The
> Beads→GitHub cutover removed `bd` entirely, and `gh` already authenticates in that sandbox,
> so the blocker no longer exists. Re-enabling is phase 7 of
> [#398](https://github.com/edwinsteele/fuel-price-signal/issues/398) and needs a live Routine
> fire to confirm. Until that lands, don't assume the twice-daily schedule below is firing.

You are a Sonnet worker running as a **Claude Code Routine** (see [docs/automation.md](docs/automation.md))
on a **twice-daily** schedule (`0 9,20 * * *` UTC), and you get a fresh checkout each run rather
than a persistent interactive session's disk. Everything you need lives in GitHub and in git;
there is no second store to sync.

Your job is to pick up `chore`- and `polish`-labelled issues and open PRs.

**Pickup rules:**
0. Ensure `gh` is present and authenticated — this environment does not always have it
   pre-installed:
   ```bash
   command -v gh >/dev/null 2>&1 || {
     GH_VERSION="2.97.0"
     curl -fsSL -o /tmp/gh.tar.gz "https://github.com/cli/cli/releases/download/v${GH_VERSION}/gh_${GH_VERSION}_linux_amd64.tar.gz" \
       && tar -xzf /tmp/gh.tar.gz -C /tmp \
       && cp "/tmp/gh_${GH_VERSION}_linux_amd64/bin/gh" /usr/local/bin/gh \
       && chmod +x /usr/local/bin/gh
   }
   gh auth status || { echo "gh cannot authenticate — stopping." >&2; exit 1; }
   ```
   Pin `GH_VERSION`, and bump it by hand once the owner confirms a newer `gh` works — never
   chase `@latest`. **Deliberately do not `curl` `api.github.com/.../releases/latest` to
   discover the version:** that host has returned 403 in this sandbox, and the failure is
   silent — it feeds an empty version into the download URL and 404s the download itself.

   Verify with a real invocation (`gh auth status`), not just `command -v` — a shim on `PATH`
   is not a working binary with working credentials. If it fails, fail hard and say so; do not
   proceed as though there were no work.
1. **There is nothing to close out.** A merged PR whose body carries `Closes #<N>` closes its
   issue by itself, so a run no longer starts by reconciling merged PRs against the tracker.
2. Check for open `claude-authored` PRs that need maintenance. Get all open PR numbers:
   ```bash
   gh pr list --label claude-authored --state open --json number | jq -r '.[].number'
   ```
   For each number N, a PR qualifies if either:
   - `gh pr view N --json mergeable | jq -r '.mergeable'` returns `CONFLICTING`, **or**
   - `gh pr view N --json reviews | jq '[.reviews[] | select(.body | length > 20)] | length'` is >0 **and** `gh pr view N --json comments | jq '[.comments[] | select(.body | startswith("[worker]"))] | length'` is 0 (reviews exist but worker hasn't replied yet).

   If any PR qualifies, perform maintenance (see **PR maintenance** below), then exit.
3. Check for open `claude-authored` PRs (any). If any exist, **exit immediately** — one at a time.
4. **Recover stale claims.** A prior run can crash between claiming an issue (rule 6) and
   opening its PR (rule 3 of "For each PR"), leaving it assigned forever and invisible to the
   claim query. This rule runs sequentially, before any new claim is made in *this* run, so
   there is no race with rule 6 — a "fresh" claim is always at least one full rule-4 pass old
   by the time rule 6 runs again next run.
   1. List candidates with the plain label listing and filter client-side (never `--search` or
      `--assignee` — see the warning under rule 5):
      ```bash
      gh issue list --label chore  --state open --limit 200 --json number,title,labels,assignees,createdAt,updatedAt
      gh issue list --label polish --state open --limit 200 --json number,title,labels,assignees,createdAt,updatedAt
      ```
      Keep the issues assigned to **your own login** (`gh api user --jq .login`) and not
      carrying the `blocked` label. **"Has an assignee" is not the test** — the assignee is an
      identity, not a status, so that wider filter sweeps the owner's own parked work as if it
      were a crashed claim. Compare logins.
   2. For each, check whether it has a live branch (`git ls-remote --heads origin 'worker/<N>-*'`)
      or an open PR referencing it (`gh pr list --label claude-authored --state open --json number,body`,
      grepping bodies for `Closes #<N>`).
   3. If neither exists **and** `updatedAt` is more than 90 minutes old (long enough to cover a
      normal claim→PR cycle within one run, short enough that a crash isn't lost for days), the
      claim is orphaned.
   4. Release each orphaned issue: `gh issue edit <N> --remove-assignee "@me"`.
5. Claim the next issue. Using the same plain-`--label` listing as rule 4, take `chore` first
   and fall back to `polish` if `chore` yields nothing; keep the issues with **no** assignee and
   without the `blocked` label; take the oldest by `createdAt`.

   ⚠️ **List by `--label`, not `--search` or `--assignee`.** Those two are search-index backed
   and lag a mutation by 2–4s, so a claim or a release this run just made can be invisible to
   the very next read — which is exactly the shape of rules 4 and 5. The plain `--label`
   listing reflects a mutation on the first read. Details and measurements:
   [docs/memory/gh-issue-list-consistency.md](docs/memory/gh-issue-list-consistency.md).
6. Claim it: `gh issue edit <N> --add-assignee "@me"`. **The assignee is the claim** — there is
   no separate status to set, and none to forget to clear.
7. Create a branch `worker/<N>-<slug>` for the issue.

**For each PR:**
1. Implement the minimal change — do not scope-creep.
2. Run `uv run ruff check . && uv run pytest -q` locally before pushing. Fix any failures.
3. Open PR titled `fix: <issue title> (#<N>)` for a `chore` issue, `feat: <issue title> (#<N>)`
   for a `polish` issue — targeting `main` (`--base main`) with labels `claude-authored` + the
   issue's original label. For a `chore` issue, also add `auto-merge-ok` — this is what makes
   the auto-merge workflow (see below) actually fire; without it the PR sits green forever
   waiting for a manual merge (`fps-hg7`). PR body must include a 3–5 bullet plan (what changed,
   what didn't, what test was added) **and a `Closes #<N>` line** — that line is what closes the
   issue when the PR merges.
4. After opening the PR, do other useful sequenced work (write a memory to `docs/memory/`, file
   any follow-up issues with `gh issue create`). Once ≈270s of real elapsed time has passed, run
   `gh pr view N --json comments,reviews,mergeable,statusCheckRollup` to check for reviews. If
   there is no other useful work, run `sleep 270` then check. (`ScheduleWakeup` is only
   available in `/loop` mode — do not attempt it here.) Act on any actionable comments found in
   `reviews[].body`. If CodeRabbit is rate-limited or absent, skip and move on — do not
   reschedule. Implement comments, run `uv run ruff check . && uv run pytest -q`, push. Repeat
   until no actionable comments remain.

Note: this run does not wait for the merge — that is gated by the separate `auto-merge.yml`
workflow (≥900s age + green checks). Nothing is left over for a later run to finish: the
`Closes #<N>` line closes the issue when the merge lands.

**PR maintenance:**
When pickup rule 2 triggers, for each qualifying PR:

*Merge conflicts:*
1. Check out the branch locally.
2. `git fetch origin && git rebase origin/main`. Resolve any conflicts — prefer the incoming (`main`) change unless the branch change is clearly intentional, in which case keep both.
3. Run `uv run ruff check . && uv run pytest -q`. Fix any failures.
4. `git push --force-with-lease`.

*Unresolved review threads:*
1. Run `gh pr view N --json comments,reviews,mergeable,statusCheckRollup` and inspect each review body for actionable inline comments not yet addressed (i.e. no `[worker]` reply in `comments`).
2. Read all such threads together to understand the full set of requested changes.
3. For any thread that is ambiguous or requires a design decision: reply `[worker] Needs owner input — <question>` and skip it. Do not make changes for that thread.
4. Make the minimal changes to address the remaining threads.
5. Run `uv run ruff check . && uv run pytest -q`. Fix any failures.
6. Push.
7. Reply to each addressed thread: `[worker] Done — <one sentence describing what changed>`.

Handle conflicts first, then review threads, in a single pass per PR.

### If you are an interactive session

- **Do not pick up `chore` or `polish` issues yourself.** File one instead (see below). If the
  user explicitly directs you to work one anyway, it's yours to finish — including closing it —
  don't leave it for the worker.
- **`design` issues are fair game** for interactive work. `gh issue edit <N> --add-assignee "@me"`
  when you start; the `Closes #<N>` line in the PR body closes it on merge.
- **Post-merge checklist — the instant you have direct merge confirmation, run all of this in the same turn, unprompted.** "Direct confirmation" means you ran `gh pr merge` yourself, or the user just told you it merged. Don't wait to be asked for any of these, and don't split them across turns:
  1. **Confirm the issue actually closed** (`gh issue view <N> --json state`). A `Closes #<N>`
     line in the PR body closes it automatically on merge, so this is a check, not a step —
     but only that exact syntax works, and a PR body that referenced the issue any other way
     leaves it open. Close it by hand if so: `gh issue close <N>`.
  2. `git branch -D <branch>` for the now-local-only branch. Squash-merge means git won't recognize it as an ordinary merge (`branch -d` refuses), but the content is already in the squash commit on `main`, so force-deleting the local pointer loses nothing. The remote copy is usually already gone — this repo auto-deletes head branches on merge, so don't treat a failed manual delete ("remote ref does not exist") as an error.
     - **If the branch is checked out in the worktree this session is running from** (rather than a different, already-idle worktree), `branch -D` fails — git refuses to delete a branch checked out anywhere, including from another worktree's shell. This isn't rare: it's the normal case for a per-issue worktree session finishing its own PR. Don't force past it, and **don't ask the user how to proceed** — the owner's standing answer is always "leave it for a later cleanup session" (asked and answered 2026-08-23; removing your own worktree mid-session is disruptive and was never actually wanted). Just note in your final summary that the branch/worktree is stale and merged, and move on — no question needed.
  3. `git pull --ff-only` in any other worktree (including the primary one) that's now behind `main` and has a clean `git status --short` — a bare fast-forward on a clean tree can't lose anything.
  4. **Sweep other idle worktrees you notice, not just your own.** Worktree directory names are stale by design (see [docs/CONVENTIONS.md § Worktrees](docs/CONVENTIONS.md#worktrees)) — the harness reuses a slot's directory name across unrelated later branches, so a name never tells you what's actually checked out there or whether it's still live. Run `git worktree list`, then for every worktree that ISN'T the one this session is in: check its actual branch (`git -C <dir> branch --show-current`), confirm that branch's work has actually landed, and confirm `git -C <dir> status --short` is empty. **Confirm "landed" by content, not by ancestry** — a squash-merged branch never reads as merged to `git branch --merged main`, so use `git diff HEAD origin/main -- $(git -C <dir> diff --name-only origin/main...HEAD)` and require it empty. Then check there's no open PR for it. If all of that holds, `git worktree remove --force <dir>` and `git branch -D <branch>`. **If a worktree has any uncommitted changes, leave it alone** — note it in your summary instead of discarding someone else's in-progress work.

  This checklist is scoped to branches/worktrees confirmed merged with no uncommitted changes — never force-remove a worktree that still has staged/unstaged changes without flagging it first, and it is not a general license for `branch -D`/force operations elsewhere.
- Do not open PRs with `claude-authored` label — that label is exclusively for the worker.
- After each commit + push, open a PR immediately without asking.
- After submitting a PR, wait 270s (4.5 min), then check for review comments (`gh pr view N --json comments,reviews,mergeable,statusCheckRollup`). Act on any actionable comments present. If CodeRabbit is rate-limited or absent, **skip it and move on — do not reschedule to wait for it**. Implement appropriate comments, push, repeat until no actionable comments remain.
- **`experiments/**` and `docs/memory/**` are exempt from the PR rule.** Lab book entries (per-experiment `README.md`, scripts, CSV outputs), `experiments/INDEX.md`, and technical memory files may be committed **and pushed** directly to `main` without a PR. Those are the only paths that bypass review; everything else still requires one, and a commit touching an exempt path *and* code is not exempt — split it. **Direct-to-`main` includes the push** — a commit left on the local `main` is not landed, and unattended routines are exactly where that goes unnoticed (see [docs/CONVENTIONS.md](docs/CONVENTIONS.md) § Git workflow for the 2026-08-26 incident). End any session that writes to `main` with `git status --short` empty and `git log --oneline origin/main..main` empty.

## spawn_task → `gh issue create` redirect

When `mcp__ccd_session__spawn_task` would normally be the right call (you noticed an out-of-scope
issue while working), **do not spawn a session**. File an issue instead:

```bash
gh issue create \
  --title "Short imperative title" \
  --label "chore" \
  --body "$(cat <<'EOF'
## What
<what needs doing>

## Why I noticed this
<file paths + context>

## Files likely affected
- fuel_signal/foo.py

## Acceptance criteria
- [ ] ...
EOF
)"
```

Use `--label "polish"` or `--label "design"` in place of `"chore"` as appropriate, and add the
topic label (`pipeline`, `research`, `data`, `product`, `infra`) alongside it — see
[AGENTS.md § Issue label taxonomy](AGENTS.md#issue-label-taxonomy).

**Ask the owner before filing.** The backlog needs active triage, so a new issue is a decision,
not a side effect — propose it and let them say yes.
