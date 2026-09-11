# fuel-price-signal — Claude instructions

For project architecture, CLI patterns, data strategy, signal logic, and automation conventions, see [AGENTS.md](AGENTS.md).

## Orientation (read these when picking up cold)

- [AGENTS.md](AGENTS.md) — architecture, CLI pattern, data strategy, signal logic
- [docs/STATUS.md](docs/STATUS.md) — current build state; what's shipped vs pending
- [docs/CONVENTIONS.md](docs/CONVENTIONS.md) — code & workflow rules (the changeable how-we-do-things layer)
- [docs/data-semantics.md](docs/data-semantics.md) — station classification, price-data shape, class-filtering traps
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

> **Status: enabled and confirmed working.** Phase 7 of
> [#398](https://github.com/edwinsteele/fuel-price-signal/issues/398) is done. The `fps-sk0`
> blocker (`bd dolt push` couldn't authenticate) is gone with `bd` deleted; a first live fire
> after re-enabling (2026-09-08) then hit a *different* blocker — this Routine's environment
> token serves only a pinned set of PR-review GraphQL operations and 403s on everything else,
> which is what `gh issue/pr list/view --json` and `gh issue/pr edit` resolve through — fixed by
> rewriting the pickup rules below to use REST (`gh api`) throughout (PR #401). A second live
> fire the same day (session `cse_01T65ABEHNfEAiM2sUdibfjo`) confirmed the rewrite end to end
> under the real restricted token: claimed #367, implemented it, opened PR #402 via REST, checked
> for reviews via REST, addressed Sourcery's findings, and the PR merged clean.
>
> One residual gap found on that run, **fixed in #403**: rule 0's own liveness check was
> `gh auth status`, which is GraphQL too — it 403s under this token and reports "invalid"
> even though the token works fine for everything else. The run treated that correctly as a
> false negative rather than failing hard; rule 0 below now checks `gh api user` instead. See
> [docs/memory/gh-auth-status-false-negative-restricted-token.md](docs/memory/gh-auth-status-false-negative-restricted-token.md).

You are a Sonnet worker running as a **Claude Code Routine** (see [docs/automation.md](docs/automation.md))
on a **twice-daily** schedule (`0 9,20 * * *` UTC), and you get a fresh checkout each run rather
than a persistent interactive session's disk. Everything you need lives in GitHub and in git;
there is no second store to sync.

Your job is to pick up `chore`- and `polish`-labelled issues and open PRs.

**REST query helpers — use these, not `gh issue/pr list/view --json` or `gh issue/pr edit`.**
Those all resolve through GraphQL in this `gh` version, and this environment's token serves
only a pinned set of PR-review GraphQL operations — everything else 403s, even though `gh api`
(plain REST) works fine on the same token. `{owner}/{repo}` below is `gh api`'s own placeholder
syntax, expanded from the checkout's git remote — write it literally.

- **List issues by label** (the REST issues endpoint also returns PRs, so filter those out):
  ```bash
  gh api "repos/{owner}/{repo}/issues?labels=<label>&state=open&per_page=100" --paginate \
    | jq '[.[] | select(has("pull_request")|not)
             | {number,title,labels,assignees,createdAt:.created_at,updatedAt:.updated_at}]'
  ```
- **List open PRs by label** (same endpoint, keep only entries that *are* PRs):
  ```bash
  gh api "repos/{owner}/{repo}/issues?labels=<label>&state=open&per_page=100" --paginate \
    | jq '[.[] | select(has("pull_request")) | {number,body}]'
  ```
- **PR review/CI snapshot for PR `<N>`** (replaces `gh pr view N --json comments,reviews,mergeable,statusCheckRollup`):
  ```bash
  gh api repos/{owner}/{repo}/pulls/<N> --jq '{mergeable, mergeable_state, sha: .head.sha}'
  gh api repos/{owner}/{repo}/pulls/<N>/reviews --paginate                    # → reviews[]
  gh api repos/{owner}/{repo}/issues/<N>/comments --paginate                  # → comments[] (PR conversation, not inline)
  gh api "repos/{owner}/{repo}/commits/<sha>/check-runs" --paginate --jq '.check_runs'  # → CI rollup
  ```
  `mergeable_state == "dirty"` is REST's equivalent of GraphQL's `mergeable: CONFLICTING`.
- **Claim/release an issue** (replaces `gh issue edit --add-assignee`/`--remove-assignee`):
  ```bash
  LOGIN=$(gh api user --jq .login)
  gh api repos/{owner}/{repo}/issues/<N>/assignees -f "assignees[]=$LOGIN"           # claim
  gh api -X DELETE repos/{owner}/{repo}/issues/<N>/assignees -f "assignees[]=$LOGIN" # release
  ```

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
   gh api user --jq .login >/dev/null || { echo "gh cannot authenticate — stopping." >&2; exit 1; }
   ```
   Pin `GH_VERSION`, and bump it by hand once the owner confirms a newer `gh` works — never
   chase `@latest`. **Deliberately do not `curl` `api.github.com/.../releases/latest` to
   discover the version:** that host has returned 403 in this sandbox, and the failure is
   silent — it feeds an empty version into the download URL and 404s the download itself.

   Verify with a real invocation, not just `command -v` — a shim on `PATH` is not a working
   binary with working credentials. **Use `gh api user` (REST) as that invocation, never
   `gh auth status`:** `gh auth status` validates the token through GraphQL under the hood, so
   under this environment's PR-review-only token it 403s and reports the token as *invalid*
   even though every REST call the routine actually makes succeeds with it — a false negative
   that would stop every run before it starts. `gh api user` returning a real login is what
   confirms `gh` is usable here. If *that* fails, fail hard and say so; do not proceed as
   though there were no work. See
   [docs/memory/gh-auth-status-false-negative-restricted-token.md](docs/memory/gh-auth-status-false-negative-restricted-token.md).
1. **There is nothing to close out.** A merged PR whose body carries `Closes #<N>` closes its
   issue by itself, so a run no longer starts by reconciling merged PRs against the tracker.
2. Check for open `claude-authored` PRs that need maintenance. Get all open PR numbers with the
   **list open PRs by label** helper (`<label>` = `claude-authored`), then `jq -r '.[].number'`.
   For each number N, a PR qualifies if either:
   - the **PR review/CI snapshot** helper's `mergeable_state` is `dirty`, **or**
   - `reviews[] | select(.body | length > 20)` is non-empty **and** `comments[] | select(.body | startswith("[worker]"))` is empty (reviews exist but worker hasn't replied yet).

   If any PR qualifies, perform maintenance (see **PR maintenance** below), then exit.
3. Check for open `claude-authored` PRs (any). If any exist, **exit immediately** — one at a time.
4. **Recover stale claims.** A prior run can crash between claiming an issue (rule 6) and
   opening its PR (rule 3 of "For each PR"), leaving it assigned forever and invisible to the
   claim query. This rule runs sequentially, before any new claim is made in *this* run, so
   there is no race with rule 6 — a "fresh" claim is always at least one full rule-4 pass old
   by the time rule 6 runs again next run.
   1. List candidates with the **list issues by label** helper (never `--search` or `--assignee`
      filtering — see the warning under rule 5) for both `chore` and `polish`.
      Keep the issues assigned to **your own login** (`gh api user --jq .login`) and not
      carrying the `blocked` label. **"Has an assignee" is not the test** — the assignee is an
      identity, not a status, so that wider filter sweeps the owner's own parked work as if it
      were a crashed claim. Compare logins.
   2. For each, check whether it has a live branch (`git ls-remote --heads origin 'worker/<N>-*'`)
      or an open PR referencing it (**list open PRs by label** helper, `<label>` = `claude-authored`,
      grepping bodies for `Closes #<N>`).
   3. If neither exists **and** `updatedAt` is more than 90 minutes old (long enough to cover a
      normal claim→PR cycle within one run, short enough that a crash isn't lost for days), the
      claim is orphaned.
   4. Release each orphaned issue with the **claim/release an issue** helper's release call.
5. Claim the next issue. Using the same **list issues by label** helper as rule 4, take `chore`
   first and fall back to `polish` if `chore` yields nothing; keep the issues with **no**
   assignee and without the `blocked` label; take the oldest by `createdAt`.

   ⚠️ **List by label, not `--search` or `--assignee`.** Those two are search-index backed
   and lag a mutation by 2–4s, so a claim or a release this run just made can be invisible to
   the very next read — which is exactly the shape of rules 4 and 5. The plain label listing
   reflects a mutation on the first read (measured against `gh issue list --label`, GraphQL —
   the REST endpoint used by the helper here hasn't been separately measured; treat it as
   presumptively the same class of read and re-verify if a claim/release race is ever observed).
   Details and measurements: [docs/memory/gh-issue-list-consistency.md](docs/memory/gh-issue-list-consistency.md).
6. Claim it with the **claim/release an issue** helper's claim call. **The assignee is the
   claim** — there is no separate status to set, and none to forget to clear.
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
   the **PR review/CI snapshot** helper to check for reviews. If there is no other useful work,
   run `sleep 270` then check. (`ScheduleWakeup` is only
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
1. Run the **PR review/CI snapshot** helper and inspect each review body for actionable inline comments not yet addressed (i.e. no `[worker]` reply in `comments`).
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
  4. **Do not sweep other worktrees. Ever.** An interactive session must never remove, or offer
     to remove, any worktree other than the one it's running in — not even one that passes every
     git-level check (clean `git status`, content landed by diff not just ancestry, no open PR).
     Passing those checks is not proof of idleness: a live session working in that worktree leaves
     no git trace of being live at all, and this has cost real, disruptive damage twice — once to
     the *scheduled* weekly cleanup task (2026-09-08, mid-review worktree removed) and once to an
     *interactive* session following this exact step (2026-09-11, a live worktree removed out from
     under another running session). Worktree cleanup is exclusively the job of the owner's weekly
     scheduled task now. If you notice a worktree that looks idle or stale, say so in your summary
     and stop there — do not act on it.

  This checklist item is otherwise scoped to your own branch/worktree only: never force-remove
  *any* worktree that still has staged/unstaged changes without flagging it first, and none of
  this is a general license for `branch -D`/force operations beyond your own finished branch.
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
[docs/CONVENTIONS.md § Issue label taxonomy](docs/CONVENTIONS.md#issue-label-taxonomy).

**Ask the owner before filing.** The backlog needs active triage, so a new issue is a decision,
not a side effect — propose it and let them say yes.
