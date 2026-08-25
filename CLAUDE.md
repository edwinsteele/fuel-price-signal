# fuel-price-signal — Claude instructions

For project architecture, CLI patterns, data strategy, signal logic, and automation conventions, see [AGENTS.md](AGENTS.md).

## Orientation (read these when picking up cold)

- [AGENTS.md](AGENTS.md) — architecture, CLI pattern, data strategy, signal logic
- [docs/STATUS.md](docs/STATUS.md) — current build state; what's shipped vs pending
- [docs/CONVENTIONS.md](docs/CONVENTIONS.md) — code & workflow rules (the changeable how-we-do-things layer)
- `PLAN_ml_signal.md` — active ML-signal plan. **Lives at repo root and is gitignored** (despite some docs saying `docs/PLAN_ml_signal.md` — that path is wrong).
- Run `bd ready` for open work items (GitHub Issues retired in favour of Beads — see [AGENTS.md § Beads](AGENTS.md#beads)).
- Run `bd memories` for this repo's atomic technical gotchas (git discipline, DB-write timing, environment traps). These are short, load-bearing, and cheap to read — several are rules you will otherwise break before noticing.

## Model/effort guidance

- Sonnet for implementation (downloader, transformer, DB layer, tests)
- Opus for analytically hard design: cycle detection math, backtest engine architecture, leading indicator analysis

## Reuse from old projects

The port from the original repos is done. If you ever need to trace original logic, the source lives in `~/Code/ff-aws-backend` (primary: `recommendations.py`, `purchasing_strategy.py`, FuelCheck OAuth task, Click `cli.py`) and `~/Code/petrol_prices` (secondary: transformer/downloader/gap-fill commands, postcode→LGA map). Do not carry over the AWS/Django infrastructure (DynamoDB, S3, SQS, SNS, Serverless, Django ORM) or `jsonpickle` / `msrest`.

## Automated worker vs interactive session

Work items live in Beads (`bd`), not GitHub Issues — see [AGENTS.md § Beads](AGENTS.md#beads) for the general model. PRs still live on GitHub; only issue tracking moved.

### If you are the scheduled worker routine

> **Status (2026-08-15): this Routine is disabled.** See [docs/automation.md](docs/automation.md) and bd issue `fps-sk0` (`blocked`, P3) for why — a Claude Code Routines platform limitation blocks `bd dolt push` from a Routine sandbox, not something fixable here. This section is left intact in case the Routine is re-enabled once the upstream blocker moves; if you're reading this as an interactive session, don't expect the twice-daily schedule below to actually be firing.

You are a Sonnet worker running as a **Claude Code Routine** (see [docs/automation.md](docs/automation.md)) — this is an actively-running automation on a **twice-daily** schedule (`0 9,20 * * *` UTC — not hourly, despite older wording elsewhere in this doc), and it gets a fresh checkout each run rather than reusing a persistent interactive session's disk. The Dolt database under `.beads/` does not travel via ordinary git commits, so every run must sync explicitly with `bd dolt pull`/`bd dolt push`.

**Dolt remote push auth:** `bd` shells out to the real `git` binary for git-backed Dolt remotes ([Dolt's git-remote-support announcement](https://www.dolthub.com/blog/2026-02-13-announcing-git-remote-support-in-dolt/)), not a separate HTTP client — so it inherits [a known Dolt bug](https://github.com/dolthub/dolt/issues/10486) where push fails whenever credential resolution would need an interactive prompt, which any bare remote URL with no embedded credential hits in a non-interactive container. Ambient git credential helpers (the ones that make plain `git push` work) aren't guaranteed to reach bd's invocation. `DOLT_REMOTE_USER`/`DOLT_REMOTE_PASSWORD` (documented by `bd dolt push --help`) do **not** fix this — those apply only to Dolt's separate "Hosted Dolt" SaaS remotes, not generic git-backed ones; don't reach for them here.

**Fix, matching Dolt's own documented CI pattern:** embed a token directly in the remote URL instead of relying on ambient credentials. Pickup rule 0 below reconstructs `sync.remote` at the start of every run as `https://x-access-token:$(gh auth token)@github.com/edwinsteele/fuel-price-signal.git`, reusing the token `gh` already has (no new secret provisioned) — in-memory only, never written to `.beads/config.yaml` or any other tracked file. The tracked `sync.remote` stays a plain `https://github.com/...` URL so every other clone (including the owner's Mac, via its own `gh`-backed credential helper — `gh auth setup-git`) authenticates ambiently without this step. Verified working end-to-end live 2026-08-11 (fps-ddf): the Routine claimed an issue, pushed the claim, implemented it, and opened [PR #287](https://github.com/edwinsteele/fuel-price-signal/pull/287) — the first successful pickup cycle since the Beads migration.

**Known gap 2 (bd binary itself, not just its Dolt remote, can be absent):** the Routine's `environment_id` (see [docs/automation.md](docs/automation.md)) is a fixed Anthropic-managed environment — the routine-scheduling API has no field for a custom base image or a setup script, so there is no durable place to pre-bake `bd` into the container. Treat the binary as something you may have to install fresh every run rather than something the environment guarantees. That's what pickup rule 0 below does — do not skip it, and do not try to work around a missing `bd` by falling back to `gh issue` (that workflow was retired with the Beads migration; GitHub Issues are no longer the source of truth).

**bd version pin (deliberate, not a gap):** because the Routine reinstalls `bd` from scratch every run instead of reusing a persisted image, the version it gets is whatever pickup rule 0 asks for — it does not track the owner's Mac automatically. Rule 0 pins an exact version (`@beads/bd@1.1.2` on npm, which tracks the same version numbers as the Homebrew formula and GitHub releases — not a separate release train) rather than installing `@latest`. Reason: `bd`'s Dolt-backed database (`.beads/embeddeddolt/`) can require a schema migration across versions, and bd's own upgrade docs say exactly one designated clone should run `bd migrate` + `bd dolt push` while every other clone just updates its binary and runs `bd bootstrap`. An unattended hourly Routine left on `@latest` risks being the *first* environment to cross a migration boundary, unsupervised, against a database the owner's interactive sessions also write to — the pin prevents that. **When the owner runs `brew upgrade beads` locally and confirms it still works, bump the pinned version in pickup rule 0 below in the same change** — that's the entire sync "dance": one version string, moved deliberately, never automatically.

Your job is to pick up `chore` and `polish` labelled bd issues and open PRs.

**Pickup rules:**
0. Ensure `bd` is usable and version-pinned before anything else: `command -v bd >/dev/null 2>&1 || npm install -g @beads/bd@1.1.2`. npm's presence in this environment is confirmed (verified live 2026-08-11 — `bd` installed and pinned at 1.1.2 correctly on the first real run of this rule), so **do not fall back to the unpinned curl install script if the npm install fails** — that would silently run an unvalidated `@latest` `bd` against the shared Dolt database. Fail hard instead: surface the npm failure clearly and stop, do not proceed as if there were no work.

   **CI-env-var gotcha (found 2026-08-15, not yet live-verified):** `@beads/bd`'s npm postinstall skips downloading the native binary entirely when `process.env.CI` is set (`if (!process.env.CI) { install() } else { console.log('Skipping binary download in CI environment') }`), and its `bin/bd.js` wrapper has no fallback — it just prints "bd binary not found" and exits 1 on first real invocation. `command -v bd` only checks that the npm shim file exists, not that the binary actually downloaded, so it cannot catch this. After install, verify with a real invocation (`bd --version`), not just `command -v`. If that fails and `$CI` is set in this environment, retry with `env -u CI npm install -g @beads/bd@1.1.2 --force` before treating it as a hard npm failure.

   Then ensure `gh` is present too — install it if missing (this environment doesn't always have it pre-installed, unlike `bd`):
   ```
   command -v gh >/dev/null 2>&1 || {
     GH_VERSION="2.97.0"
     curl -fsSL -o /tmp/gh.tar.gz "https://github.com/cli/cli/releases/download/v${GH_VERSION}/gh_${GH_VERSION}_linux_amd64.tar.gz" \
       && tar -xzf /tmp/gh.tar.gz -C /tmp \
       && cp "/tmp/gh_${GH_VERSION}_linux_amd64/bin/gh" /usr/local/bin/gh \
       && chmod +x /usr/local/bin/gh
   }
   command -v gh >/dev/null 2>&1 || echo "gh install failed — TOKEN below will fall back to \$GH_TOKEN/\$GITHUB_TOKEN, which may be a proxy placeholder, not a real credential (fps-sk0)." >&2
   ```
   Pin `GH_VERSION` the same way `bd` is pinned below — bump it by hand when the owner confirms a newer `gh` works, never chase `@latest` here. **Deliberately not `curl`ing `api.github.com/.../releases/latest` to discover the version** (2026-08-15 revision, not yet live-verified): fps-sk0 found `api.github.com` returning 403 in this sandbox while `github.com/.../releases/download/...` — the exact domain+path `@beads/bd`'s own postinstall.js successfully downloads its binary from — was never actually confirmed blocked. The old two-step version showed a 404 on the *download*, but that's the predictable result of the *version-lookup* step against `api.github.com` failing silently and feeding an empty version into the download URL (`.../download/v/gh__linux_amd64.tar.gz`), not evidence that release downloads themselves are blocked. Pinning the version removes the `api.github.com` call entirely, sidestepping that failure mode regardless of which diagnosis is right.

   This matters because of fps-sk0 (root-caused 2026-08-14): Dolt has a documented upstream bug where its internal git push can't complete an interactive-style (STDIN/askpass) credential prompt — https://www.dolthub.com/blog/2026-02-13-announcing-git-remote-support-in-dolt/ says plainly "there is a bug in Dolt ... which prevents using Git remotes as Dolt remotes if your Git binary requires username and password credential inputs via STDIN, so you must use a Git authentication method that does not require these." This environment's ambient git auth uses exactly that interactive-style mechanism (`GIT_ASKPASS`) — which is why plain `git push` works fine but `bd dolt push` 403s regardless of network/proxy state. The only way around Dolt's bug is a token embedded directly in the URL, which needs `gh auth token` to actually resolve a real credential — hence installing `gh` here rather than treating its absence as tolerable.

   Once `bd` and `gh` are confirmed present, reconstruct the Dolt remote with a live token so `bd dolt push` can authenticate (see "Dolt remote auth" above — this is required every run, not a one-time setup step):
   ```
   TOKEN="$(gh auth token 2>/dev/null || echo "${GH_TOKEN:-$GITHUB_TOKEN}")"
   case "$TOKEN" in
     gho_*|ghp_*|ghs_*|ghu_*|github_pat_*) ;;
     *)
       echo "Token doesn't look like a real GitHub token (got '${TOKEN}', length ${#TOKEN}) — likely a proxy placeholder, not a usable secret. Skipping explicit Dolt remote reconfiguration rather than embedding a bogus credential — bd dolt push will still hit Dolt's askpass bug (fps-sk0) without a real token, so expect it to fail this run; report it, don't retry in a loop." >&2
       TOKEN=""
       ;;
   esac
   if [ -n "$TOKEN" ]; then
     bd dolt remote remove origin >/dev/null 2>&1 || true
     bd dolt remote add origin "https://x-access-token:${TOKEN}@github.com/edwinsteele/fuel-price-signal.git"
   fi
   ```
   Then run `bd dolt pull`.
1. Close out bd issues resolved by your own merged PRs since the last run: `gh pr list --label claude-authored --state merged --json number,body,mergedAt` (recent ones), pull the `Resolves: <id>` line out of each body, `bd close <id>` for each, then `bd dolt push`.
2. Check for open `claude-authored` PRs that need maintenance. Get all open PR numbers:
   ```bash
   gh pr list --label claude-authored --state open --json number | jq -r '.[].number'
   ```
   For each number N, a PR qualifies if either:
   - `gh pr view N --json mergeable | jq -r '.mergeable'` returns `CONFLICTING`, **or**
   - `gh pr view N --json reviews | jq '[.reviews[] | select(.body | length > 20)] | length'` is >0 **and** `gh pr view N --json comments | jq '[.comments[] | select(.body | startswith("[worker]"))] | length'` is 0 (reviews exist but worker hasn't replied yet).

   If any PR qualifies, perform maintenance (see **PR maintenance** below), then exit.
3. Check for open `claude-authored` PRs (any). If any exist, **exit immediately** — one at a time.
4. **Recover stale claims.** A prior run can crash between claiming an issue (rule 6) and opening its PR (rule 3 of "For each PR"), leaving it `in_progress` forever and invisible to `bd ready`. This rule runs sequentially, before any new claim is made in *this* run, so there's no race with rule 6 below — a "fresh" claim is always at least one full rule-4-pass old by the time rule 6 runs again next run.
   1. List candidates: `bd list --status in_progress --label-any chore,polish --json`.
   2. For each issue, check whether it has a live branch (`git ls-remote --heads origin 'worker/<id>-*'`) or an open PR referencing it (`gh pr list --label claude-authored --state open --json number,body` and grep bodies for `Resolves: <id>`).
   3. If neither exists **and** the issue's `updated_at` is more than 90 minutes old (long enough to cover a normal claim→PR cycle within one run, short enough that a crash isn't lost for days), the claim is orphaned.
   4. Release each orphaned issue found: `bd assign <id> ""`, `bd update <id> --status open`, then `bd dolt push`.
5. Query `bd ready --label chore --unassigned --sort oldest -n 1`; if empty, `bd ready --label polish --unassigned --sort oldest -n 1`. Take the first result.
6. `bd update <id> --claim` to mark it in_progress, then `bd dolt push`.
7. Create a branch `worker/<id>-<slug>` for the issue.

**For each PR:**
1. Implement the minimal change — do not scope-creep.
2. Run `uv run ruff check . && uv run pytest -q` locally before pushing. Fix any failures.
3. Open PR titled `fix: <issue title> (bd-<id>)` for a `chore` issue, `feat: <issue title> (bd-<id>)` for a `polish` issue — targeting `main` (`--base main`) with labels `claude-authored` + the issue's original label. For a `chore` issue, also add `auto-merge-ok` — this is what makes the auto-merge workflow (see below) actually fire; without it the PR sits green forever waiting for a manual merge (fps-hg7). PR body must include a 3–5 bullet plan (what changed, what didn't, what test was added) **and a `Resolves: <id>` line** — pickup rule 1 of the *next* run depends on finding it.
4. After opening the PR, do other useful sequenced work (update memory, file any follow-up issues via `bd create`). Once ≈270s of real elapsed time has passed, run `gh pr view N --json comments,reviews,mergeable,statusCheckRollup` to check for reviews. If there is no other useful work, run `sleep 270` then check. (`ScheduleWakeup` is only available in `/loop` mode — do not attempt it here.) Act on any actionable comments found in `reviews[].body`. If CodeRabbit is rate-limited or absent, skip and move on — do not reschedule. Implement comments, run `uv run ruff check . && uv run pytest -q`, push. Repeat until no actionable comments remain.

Note: this run does **not** close the bd issue — merging is gated by the separate `auto-merge.yml` workflow (≥900s age + green checks), which this run doesn't wait for. Closure happens in pickup rule 1 of a later run, once the PR shows up as merged.

**PR maintenance:**
When pickup rule 1 triggers, for each qualifying PR:

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

- **Do not pick up `chore` or `polish` issues yourself.** File a `bd create` instead (see below). If the user explicitly directs you to work one anyway, it's yours to finish — including closing it (next bullet) — don't leave it for the worker.
- **`design` issues are fair game** for interactive work. `bd update <id> --claim` when you start, `bd close <id>` when done, `bd dolt push` after either.
- **Post-merge checklist — the instant you have direct merge confirmation, run all of this in the same turn, unprompted.** "Direct confirmation" means you ran `gh pr merge` yourself, or the user just told you it merged. Don't wait to be asked for any of these, and don't split them across turns:
  1. `bd close <id>` + `bd dolt push`, for the `Resolves: <id>` line in the PR body. Do this regardless of the issue's label (`design`, `chore`, `polish`) and regardless of whether the worker routine is running — the worker's pickup-rule-1 auto-closure only scans `claude-authored` PRs, a label interactive sessions never use, so it structurally cannot see a PR you opened. Its "close next run" deferral is a workaround for *its own* async merge wait; it doesn't apply once you've confirmed the merge synchronously yourself.
  2. `git branch -D <branch>` for the now-local-only branch. Squash-merge means git won't recognize it as an ordinary merge (`branch -d` refuses), but the content is already in the squash commit on `main`, so force-deleting the local pointer loses nothing. The remote copy is usually already gone — this repo auto-deletes head branches on merge (`bd recall github-auto-deletes-merged-branch` if a manual delete surprises you with "remote ref does not exist").
     - **If the branch is checked out in the worktree this session is running from** (rather than a different, already-idle worktree), `branch -D` fails — git refuses to delete a branch checked out anywhere, including from another worktree's shell. This isn't rare: it's the normal case for a per-issue worktree session finishing its own PR. Don't force past it, and **don't ask the user how to proceed** — the owner's standing answer is always "leave it for a later cleanup session" (asked and answered 2026-08-23; removing your own worktree mid-session is disruptive and was never actually wanted). Just note in your final summary that the branch/worktree is stale and merged, and move on — no question needed.
  3. `git pull --ff-only` in any other worktree (including the primary one) that's now behind `main` and has a clean `git status --short` — a bare fast-forward on a clean tree can't lose anything.

  This checklist is scoped to branches/worktrees you've just confirmed are merged in this repo's per-issue-worktree workflow — not a general license for `branch -D`/force operations elsewhere.
- Do not open PRs with `claude-authored` label — that label is exclusively for the worker.
- After each commit + push, open a PR immediately without asking.
- After submitting a PR, wait 270s (4.5 min), then check for review comments (`gh pr view N --json comments,reviews,mergeable,statusCheckRollup`). Act on any actionable comments present. If CodeRabbit is rate-limited or absent, **skip it and move on — do not reschedule to wait for it**. Implement appropriate comments, push, repeat until no actionable comments remain.
- **`experiments/**` is exempt from the PR rule.** Lab book entries (per-experiment `README.md`, scripts, CSV outputs) and `experiments/INDEX.md` may be committed **and pushed** directly to `main` without a PR. This is the only path that bypasses review; all other paths still require one. **Direct-to-`main` includes the push** — a commit left on the local `main` is not landed, and unattended routines are exactly where that goes unnoticed (see [docs/CONVENTIONS.md](docs/CONVENTIONS.md) § the exemption for the 2026-08-26 incident). End any session that writes to `main` with `git status --short` empty and `git log --oneline origin/main..main` empty.

## spawn_task → bd create redirect

When `mcp__ccd_session__spawn_task` would normally be the right call (you noticed an out-of-scope issue while working), **do not spawn a session**. Instead:

```bash
bd create \
  --title "Short imperative title" \
  --labels "chore" \
  --description "$(cat <<'EOF'
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
bd dolt push
```

Use `--labels "polish"` or `--labels "design"` in place of `"chore"` as appropriate.
