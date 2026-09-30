# Worker routine — prompt source

Canonical, tracked home for the prompt used by the "Fuel Price Signal Chore and Polish Worker" scheduled Routine (`trig_01Mhkd4YLBpuGLhcXLMHaZVQ`, cron `0 9,20 * * *` UTC, environment `env_013cCwqEo6PFqNSNxsYgny1k` — an Anthropic-managed cloud environment that gets a fresh checkout every run).

## Why this file exists

Before this file, the actual pickup/PR rules were duplicated into two places outside the repo:

1. The Routine's stored `job_config` message (edited only through the Routines UI/API).
2. A dormant local `~/.claude/scheduled-tasks/fuel-price-signal-worker/SKILL.md` on the owner's Mac.

Both drifted out of step with CLAUDE.md, in opposite directions across a year: they were written against GitHub Issues, went stale when the 2026-08-06 Beads migration moved the tracker to `bd`, and are stale again now the 2026-09-07 cutover moved it back. Two untracked copies of the same instructions are two places to forget to update — twice over, here.

The fix is to keep exactly **one tracked copy of the shim text**, here. The scheduler's stored prompt — remote `job_config` or local `SKILL.md` — should hold nothing but a pointer to CLAUDE.md, never the rules themselves. When the pickup/PR process changes, only [§ Rules](#rules) below needs editing; a three-line shim has nothing substantive left to go stale.

The shim still says *"CLAUDE.md's section"*, and that is deliberate: CLAUDE.md's ["If you are the scheduled worker routine"](../../CLAUDE.md#if-you-are-the-scheduled-worker-routine) heading is now a one-line pointer to § Rules here. The rules moved out of CLAUDE.md on 2026-10-01 because Codex never reads CLAUDE.md, and 14 KB of repo process there was invisible to it; routing through the pointer meant the scheduler's stored prompt didn't need an owner edit.

This is the intended pattern for **every** scheduled routine in this project, not just this one — add `docs/routines/<name>.md` for each new routine (see PLAN_ml_signal.md's feature-pipeline routines) rather than writing instructions straight into the scheduler.

## The shim

This is the exact text that should be the Routine's stored prompt (`job_config.ccr.events[0].data.message.content` for the remote trigger; the `SKILL.md` body for a local scheduled task):

```
You are the fuel-price-signal worker routine.
Your working directory already has a fresh checkout of `edwinsteele/fuel-price-signal`.
Follow CLAUDE.md's "If you are the scheduled worker routine" section exactly.
```

Three lines, nothing else: who you are, where the repo is, where the rules live. No embedded steps, no `gh` commands, no reminders about what the rules say — [§ Rules](#rules) is the only place that gets to say what they are, so the shim doesn't paraphrase any of it.

## Applying it

- **Remote Routine** (`trig_01Mhkd4YLBpuGLhcXLMHaZVQ`): needs to be updated via the Routines UI (or `update_trigger`) with `prompt` set to the shim above. **Not done as part of this change** — the trigger was created via the Routines UI directly (`created_via: "http_api"`), and `update_trigger` only permits an agent session to modify a trigger it created itself via `create_trigger`. This is an owner action.
- **Local scheduled task** (`~/.claude/scheduled-tasks/fuel-price-signal-worker/SKILL.md` on the owner's Mac): not reachable from a repo PR or a cloud session — needs the owner to delete it (it's currently orphaned/non-firing) or replace its body with the shim above.
- **Done.** Both owner actions above were already complete by the time of the first fire (2026-09-08). That fire found a real blocker — the Routine's environment token only serves a pinned set of PR-review GraphQL operations, and every `gh issue/pr list/view --json`/`gh issue/pr edit` call the pickup rules used resolves through GraphQL, so they all 403'd — fixed by rewriting CLAUDE.md's pickup rules to use `gh api` (REST) instead (PR #401). A second live fire the same day confirmed the rewrite works against the real restricted token: claimed issue #367, opened PR #402 via REST, checked for reviews via REST, and the PR merged clean. This closes phase 7 of [#398](https://github.com/edwinsteele/fuel-price-signal/issues/398). One residual gap found on that run, fixed in [#403](https://github.com/edwinsteele/fuel-price-signal/issues/403): rule 0's own liveness check was `gh auth status`, which is GraphQL too and so false-negatives under this token — it now checks `gh api user` instead. See [docs/memory/gh-auth-status-false-negative-restricted-token.md](../memory/gh-auth-status-false-negative-restricted-token.md).

## Rules

The worker routine's pickup, PR and maintenance rules. A routine reaching this via the shim follows everything below exactly.


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
> [docs/memory/gh-auth-status-false-negative-restricted-token.md](../memory/gh-auth-status-false-negative-restricted-token.md).

You are a Sonnet worker running as a **Claude Code Routine** (see [docs/automation.md](../automation.md))
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
  gh api repos/{owner}/{repo}/pulls/<N>/reviews --paginate                    # → reviews[] (review bodies — boilerplate for Codex, findings for Sourcery)
  gh api repos/{owner}/{repo}/pulls/<N>/comments --paginate                   # → inline findings (Codex's ARE HERE, not in reviews[].body)
  gh api repos/{owner}/{repo}/issues/<N>/comments --paginate                  # → comments[] (PR conversation; Codex's clean-pass text lands here)
  gh api repos/{owner}/{repo}/issues/<N>/reactions --paginate                 # → a silent Codex clean pass can be JUST a chatgpt-codex-connector[bot] 👍 here, nothing else
  gh api "repos/{owner}/{repo}/commits/<sha>/check-runs" --paginate --jq '.check_runs'  # → CI rollup
  ```
  `mergeable_state == "dirty"` is REST's equivalent of GraphQL's `mergeable: CONFLICTING`. **Never
  treat `pulls/<N>/reviews` + `issues/<N>/comments` alone as "checked Codex" — its findings are
  inline comments and its clean pass can be a bare reaction with no comment or review at all.**
  Use the `codex-pr-review` skill (or [docs/memory/codex-pr-review-integration.md](../memory/codex-pr-review-integration.md)
  directly) to read Codex's status correctly; this bit a PR-status report on #422 (all four checks
  above ran except the reactions one, missing a silent clean pass). **A reaction on the PR persists
  across later pushes** — before trusting a `+1` as "clean", check its `created_at` against the
  timestamp of the commit currently at HEAD; a reaction older than the latest push reviewed a
  previous commit, not this one, and tells you nothing about what's there now (caught by Sourcery
  on PR #423, review round 1).
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
   [docs/memory/gh-auth-status-false-negative-restricted-token.md](../memory/gh-auth-status-false-negative-restricted-token.md).
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
   Details and measurements: [docs/memory/gh-issue-list-consistency.md](../memory/gh-issue-list-consistency.md).
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
   the **PR review/CI snapshot** helper to check for reviews — its inline-comments and reactions
   calls, not just `reviews[].body`, are what actually surface Codex's findings or clean pass; see
   [docs/memory/codex-pr-review-integration.md](../memory/codex-pr-review-integration.md). If
   there is no other useful work, run `sleep 270` then check. (`ScheduleWakeup` is only available
   in `/loop` mode — do not attempt it here.) Act on any actionable comments found. If Sourcery is
   rate-limited or absent, skip and move on — do not reschedule. **Codex auto-review is OFF** (owner,
   2026-09-29): it reviews only when asked, so right after opening the PR — and after each later push —
   post one `@codex review` (`gh api repos/{owner}/{repo}/issues/<N>/comments -f 'body=@codex review'`),
   once per head, never re-posted for a head it has already been asked about. If it has produced no
   signal after the wait, it hasn't reached the request yet — wait, don't repost. Implement comments, run `uv run ruff check . && uv run pytest -q`, push. Repeat until
   no actionable comments remain.

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
1. Run the **PR review/CI snapshot** helper and inspect the `pulls/<N>/comments` entries (inline
   findings — Codex's are here, not in `reviews[].body`, which is boilerplate) for actionable
   comments not yet addressed (i.e. no `[worker]` reply among the replies on that comment).
2. Read all such threads together to understand the full set of requested changes.
3. For any thread that is ambiguous or requires a design decision: reply `[worker] Needs owner input — <question>` and skip it. Do not make changes for that thread.
4. Make the minimal changes to address the remaining threads.
5. Run `uv run ruff check . && uv run pytest -q`. Fix any failures.
6. Push.
7. Reply to each addressed thread: `[worker] Done — <one sentence describing what changed>`.

Handle conflicts first, then review threads, in a single pass per PR.
