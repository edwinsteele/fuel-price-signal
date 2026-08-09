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

You are a Sonnet worker running as a **Claude Code Routine** (see [docs/automation.md](docs/automation.md)) — this is an actively-running hourly automation, not a dormant one, and it gets a fresh checkout each run rather than reusing a persistent interactive session's disk. The Dolt database under `.beads/` does not travel via ordinary git commits, so every run must sync explicitly with `bd dolt pull`/`bd dolt push`.

**Known gap (unresolved as of this migration, check before trusting `bd ready` output):** the auto-configured Dolt remote — `git+ssh://git@github.com/edwinsteele/fuel-price-signal.git` — needs deploy-key or HTTPS-token auth reachable from the Routine's environment; it currently relies on the owner's personal 1Password-gated SSH key, which that environment does not have. Until this is fixed, `bd dolt pull`/`push` will likely fail in the Routine even though they work fine in an interactive session on the owner's Mac. If `bd dolt pull` fails, do not proceed as if the backlog is empty — surface the failure rather than silently finding "no ready work".

Your job is to pick up `chore` and `polish` labelled bd issues and open PRs.

**Pickup rules:**
0. `bd dolt pull` — first action, every run.
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
4. Query `bd ready --label chore --unassigned --sort oldest -n 1`; if empty, `bd ready --label polish --unassigned --sort oldest -n 1`. Take the first result.
5. `bd update <id> --claim` to mark it in_progress, then `bd dolt push`.
6. Create a branch `worker/<id>-<slug>` for the issue.

**For each PR:**
1. Implement the minimal change — do not scope-creep.
2. Run `uv run ruff check . && uv run pytest -q` locally before pushing. Fix any failures.
3. Open PR titled `fix: <issue title> (bd-<id>)` for a `chore` issue, `feat: <issue title> (bd-<id>)` for a `polish` issue — targeting `main` (`--base main`) with labels `claude-authored` + the issue's original label. PR body must include a 3–5 bullet plan (what changed, what didn't, what test was added) **and a `Resolves: <id>` line** — pickup rule 1 of the *next* run depends on finding it.
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

- **Do not pick up `chore` or `polish` issues yourself.** File a `bd create` instead (see below).
- **`design` issues are fair game** for interactive work. `bd update <id> --claim` when you start, `bd close <id>` when done, `bd dolt push` after either.
- Do not open PRs with `claude-authored` label — that label is exclusively for the worker.
- After each commit + push, open a PR immediately without asking.
- After submitting a PR, wait 270s (4.5 min), then check for review comments (`gh pr view N --json comments,reviews,mergeable,statusCheckRollup`). Act on any actionable comments present. If CodeRabbit is rate-limited or absent, **skip it and move on — do not reschedule to wait for it**. Implement appropriate comments, push, repeat until no actionable comments remain.
- **`experiments/**` is exempt from the PR rule.** Lab book entries (per-experiment `README.md`, scripts, CSV outputs) and `experiments/INDEX.md` may be committed directly to `main` without a PR. This is the only path that bypasses review; all other paths still require one.

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
