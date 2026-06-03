# Conventions

Code and workflow rules that any contributor (agent or human) should follow when changing this repo. The architectural shape lives in [AGENTS.md](../AGENTS.md); this file is the changeable how-we-do-things layer.

Each rule has a **Why** (the incident or constraint behind it) so edge cases can be judged rather than blindly followed.

## Code

### CLI modules — one file per command, `python -m` invocation

Each command is its own module in `fuel_signal/` with a `@click.command` named `main` and an `if __name__ == "__main__": main()` block. Invoke as `uv run python -m fuel_signal.<module>`. See [AGENTS.md § CLI pattern](../AGENTS.md#cli-pattern) for the full list.

Do **not** add new commands to a shared CLI group, and do **not** add `[project.scripts]` entries in `pyproject.toml`.

**Why:** `[project.scripts]` entry points rely on `.pth` files for editable installs, which broke under Python 3.14. The `python -m` pattern bypasses this entirely.

### SQL strings — plain literals use `%`, not `%%`

In `db.py`, SQL is built with plain string literals (not f-strings, not `str % args` formatting). In that context `%` is a literal character — write `(p.price_date/100)%100`, not `%%100`.

**Why:** `coverage_matrix` originally had `%%100` in a plain string, which passed two literal `%` characters to SQLite and caused `OperationalError: near "%": syntax error`. Only escape to `%%` inside f-strings or `str % (args,)` formatting calls.

### Comments document intent, not behaviour

Add a one-line comment when an invariant is non-obvious (e.g. why the YYYY-DD-MM date-swap condition skips the equality case in `history.py`). Don't restate what the code already says.

## Tests

### DB fixture and CliRunner pattern

See [AGENTS.md § Test patterns](../AGENTS.md#test-patterns) for the standard `conn` fixture and `CliRunner().invoke(main, [...])` pattern for module tests.

### Time-window tests use today-relative dates

Any test that feeds data into a function with a rolling window filter (e.g. `coverage_matrix(months=24)`, `gradient_by_lga`) must compute its test dates from `datetime.date.today()`, not hardcode `"2024-01-10"`.

**Why:** `test_coverage_matrix_returns_station_month_counts` originally inserted `"2024-01-10"` data; once the 24-month window passed that date, the test silently asserted on empty results instead of failing loudly. Hardcoded dates rot.

## Changing the production feature set

Adding, dropping, decomposing, or replacing a feature in the production model's resolved feature set requires paired walk-forward CV evidence before merge. Single-window comparisons — even multi-seed — do not generalise across regimes.

Minimum evidence, cited in the PR body or commit message:

- `uv run python -m fuel_signal.cv_report --drop-feature COL` (drop), or `--baseline OLD.joblib NEW.joblib` (add / swap / decompose)
- Per-fold CSV path under `experiments/<date>_<slug>/`
- Median Δ logloss, worst-fold Δ, fold-win count, fold count

Default decision rule: if any single fold regresses by more than the median improvement, keep the feature. The rule is asymmetric on purpose — a wide-mean, narrow-tail improvement is the win pattern; a regime that inverts the sign is the loss pattern.

Override is allowed when the regressing fold is known to be anomalous (a price-shock period, a labelling artefact, a regime explicitly out of scope). State the override reason in the PR body — a considered exception is fine; silently ignoring the rule is not.

**Why:** on 2026-06-03, `station_minus_last_max_cents` looked like a clean drop on one val window (5-seed Δ −0.0112 ± 0.0043), but a 14-fold paired walk-forward CV showed 7/14 fold-wins, mean Δ +0.0104, with fold 9 (2023-10→2024-01) regressing by +0.103. See `experiments/2026-06-03_drop_redundant_pair/`.

## Definition of done

Before considering a change complete, in this order:

1. **Re-read the issue** if the change closes one. Walk through the acceptance criteria / deliverables list and confirm each item is covered by the diff. Scope often drifts during implementation; the issue is the source of truth for what was promised, and the check catches gaps before review does. If something in the issue is no longer the right thing to build, say so in the PR body rather than silently dropping it.
2. **Run pre-commit checks locally:** `uv run ruff check . && uv run pytest -q`. The pre-commit hook runs the same pair, so a failing commit otherwise costs a fix-then-recommit cycle.
3. **Update README** if a user-facing command, flag, or invocation changed. The README is the first place a user looks; a stale one is actively misleading.
4. **Update tracking docs** if a module shipped or a project phase completed:
   - `PLAN_ml_signal.md` — mark items done with strikethrough + **DONE (date, PR#)**
   - `docs/STATUS.md` — current build state
   - `docs/ML_SIGNAL.md` — design decisions if any landed
5. **Commit `experiments/results.csv`** immediately after any `calibrate.py` or `score_phase2.py` run, as a standalone `chore: record experiment results` commit. The row is the permanent experiment log regardless of whether the model code survives.

## Decisions land in repo docs, not just memory

When a design decision is made during a session, capture it in [AGENTS.md](../AGENTS.md), [docs/ML_SIGNAL.md](ML_SIGNAL.md), or the relevant `PLAN_*.md` — **before** the work that depends on it. Private memory files complement repo docs but never substitute for them; decisions that govern code structure must be discoverable and version-controlled.

## Docs and memory: signal over sediment

Notes about completed work are fine briefly, then purge unless they inform future decisions. Closed GitHub issues are the authoritative record of "what was resolved and why"; markdown prose should not re-narrate them.

- **Keep** the durable principle, taxonomy, or constraint that came out of the work (e.g. "information value ≠ leadership"; "rolling-window stickiness lags during regime shifts").
- **Drop** the play-by-play: `RESOLVED YYYY-MM-DD` markers, script inventories from experiments, verification-gate write-ups, decision-option narratives, commit/PR archaeology.
- **Reference** closed issues by number for traceability (`tracked as #123`, `see #136`) — don't summarise their resolution.
- Memory files that document a known failure mode should be rewritten forward-looking once it's mitigated ("X is brittle when Y; current Z insulates against it; reappears if Z is dropped"), not stacked as `Finding → Resolution → How to apply`.
- When updating docs after work lands, the question is not "what happened?" but "what does a future reader need to know to make the next decision?"

## Git workflow

- **Fresh branch per PR.** Branch off `main` for each PR; do not continue committing to a previously merged branch even though GitHub diffs against `main` would still work.
- **Open the PR immediately** after the first commit+push — no need to ask first.
- Branch naming, PR title format, and PR body shape: see [AGENTS.md § Branch and PR conventions](../AGENTS.md#branch-and-pr-conventions).
- **Experiments lab book is exempt.** Changes confined to `experiments/**` may be committed directly to `main` without a PR. Each experiment dir is a self-contained lab book entry; iterate freely. `experiments/results.csv` (the formal graduated-experiment log) and `experiments/INDEX.md` (the lab book index) are also direct-to-`main`. Anything touching `fuel_signal/`, `tests/`, `docs/`, or top-level config still goes through a PR even if an experiment motivated it.

## PR feedback loop

Immediately after `gh pr create` returns a PR number, call `ScheduleWakeup(delaySeconds=270)` with a prompt that runs `gh pr view <N> --json comments,reviews,mergeable,statusCheckRollup`. This is a mandatory mechanical step, not a suggestion — do it before writing any response to the user. When the wakeup fires: act on any actionable comments present. If CodeRabbit is rate-limited or absent, **skip it and move on — do not reschedule to wait for it**. Use judgement on style nits that conflict with project conventions. Run `uv run ruff check . && uv run pytest -q`, push, and repeat until no actionable comments remain. The goal is a ready-to-merge deliverable.

## Code review caution

Before filing an issue from an agent-driven logic review:

- **Trace a concrete example** end-to-end, especially for format-handling code. The `history.py` YYYY-DD-MM date-swap condition was wrongly flagged because the agent didn't walk through a case where `raw_day == true_month`.
- **Check the docstring** for stated design intent before claiming inconsistency. `series.py`'s `brand:` resolver was wrongly flagged for using exact match — the docstring said exact was the intent.

## Shell tooling

Use `jq` for JSON slicing in bash, not `python3 -c "import json…"`. Idiomatic, cleaner output, no temp scripts.
