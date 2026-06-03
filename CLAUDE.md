# fuel-price-signal — Claude instructions

For project architecture, CLI patterns, data strategy, signal logic, and automation conventions, see [AGENTS.md](AGENTS.md).

## Model/effort guidance

- Sonnet for implementation (downloader, transformer, DB layer, tests)
- Opus for analytically hard design: cycle detection math, backtest engine architecture, leading indicator analysis

## Reuse from old projects

These local repos contain the source code that was ported. Check them if you need to trace original logic.

### `~/Code/ff-aws-backend` (primary)
- `ff_aws_backend/recommendations.py` — `PriceCycleDetector`, all signal classes, `RecommendationManager`
- `ff_analysis/purchasing_strategy.py` — backtest engine
- `frugalfuel/nswfuel/tasks/retrieve_price_snapshot_from_fuelapi.py` — OAuth API auth pattern
- `ff_aws_backend/cli.py` — CLI structure with Click

### `~/Code/petrol_prices` (secondary)
- `petrol_prices/management/commands/transformer.py` — CSV cleaner
- `petrol_prices/management/commands/downloader.py` — bulk CSV downloader
- `petrol_prices/management/commands/fill_daily_gaps.py` — forward-fill
- `postcode_council_map.py` — postcode → LGA mapping

### What NOT to carry over
- DynamoDB, S3, SQS, SNS, Serverless framework, Django ORM
- jsonpickle, `msrest` / AutoRest generated client

## Automated worker vs interactive session

### If you are the scheduled worker routine

You are a Sonnet worker. You run hourly. Your job is to pick up `chore` and `polish` issues and open PRs.

**Pickup rules:**
1. Check for open `claude-authored` PRs that need maintenance. Get all open PR numbers:
   ```bash
   gh pr list --label claude-authored --state open --json number | jq -r '.[].number'
   ```
   For each number N, a PR qualifies if either:
   - `gh pr view N --json mergeable | jq -r '.mergeable'` returns `CONFLICTING`, **or**
   - `gh pr view N --json reviews | jq '[.reviews[] | select(.body | length > 20)] | length'` is >0 **and** `gh pr view N --json comments | jq '[.comments[] | select(.body | startswith("[worker]"))] | length'` is 0 (reviews exist but worker hasn't replied yet).

   If any PR qualifies, perform maintenance (see **PR maintenance** below), then exit.
2. Check for open `claude-authored` PRs (any). If any exist, **exit immediately** — one at a time.
3. Query `gh issue list --label "chore,polish" --state open --no-assignee --json number,title,labels,createdAt` ordered by label (`chore` before `polish`), then by age (oldest first). Take 1.
4. Create a branch `worker/issue-<N>-<slug>` for the issue.

**For each PR:**
1. Implement the minimal change — do not scope-creep.
2. Run `uv run ruff check . && uv run pytest -q` locally before pushing. Fix any failures.
3. Open PR titled `fix: <issue title> (closes #N)` targeting `main` (`--base main`) with labels `claude-authored` + the issue's original label. PR body must include a 3–5 bullet plan (what changed, what didn't, what test was added).
4. After opening the PR, do other useful sequenced work (update memory, file any follow-up issues). Once ≈270s of real elapsed time has passed, run `gh pr view N --json comments,reviews,mergeable,statusCheckRollup` to check for reviews. If there is no other useful work, run `sleep 270` then check. (`ScheduleWakeup` is only available in `/loop` mode — do not attempt it here.) Act on any actionable comments found in `reviews[].body`. If CodeRabbit is rate-limited or absent, skip and move on — do not reschedule. Implement comments, run `uv run ruff check . && uv run pytest -q`, push. Repeat until no actionable comments remain.

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

- **Do not pick up `chore` or `polish` issues yourself.** File a `gh issue create` instead.
- **`design` issues are fair game** for interactive work.
- Do not open PRs with `claude-authored` label — that label is exclusively for the worker.
- After each commit + push, open a PR immediately without asking.
- After submitting a PR, wait 270s (4.5 min), then check for review comments (`gh pr view N --json comments,reviews,mergeable,statusCheckRollup`). Act on any actionable comments present. If CodeRabbit is rate-limited or absent, **skip it and move on — do not reschedule to wait for it**. Implement appropriate comments, push, repeat until no actionable comments remain.
- **`experiments/**` is exempt from the PR rule.** Lab book entries (per-experiment `README.md`, scripts, CSV outputs) and `experiments/INDEX.md` may be committed directly to `main` without a PR. This is the only path that bypasses review; all other paths still require one.

## spawn_task → gh issue create redirect

When `mcp__ccd_session__spawn_task` would normally be the right call (you noticed an out-of-scope issue while working), **do not spawn a session**. Instead:

```bash
gh issue create \
  --title "Short imperative title" \
  --label "chore"  # or polish or design \
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
