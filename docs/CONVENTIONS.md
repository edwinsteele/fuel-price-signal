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

### `db.py` owns SQL — compose helpers, don't inline queries

Command modules in `fuel_signal/` read data through `db.py` helpers (`get_daily_prices`, `average_price_series`, etc.), never by issuing their own SQL against the schema. If a helper doesn't expose what you need, add or extend one in `db.py` and call it — don't inline a one-off `SELECT`.

Experiments follow the same preference order, not an exemption. Reach for `load_features()` (the model-ready matrix) first, then a `db.py` read helper for raw/gap-filled series, then the `experiments/lib/features/` primitives for PIT-safe transforms. Compose SQL directly **only when no existing routine fits** — and if you find yourself recomputing something a helper already does (a market average, a gap-filled series), use the helper.

**Why:** the helpers encode invariants raw SQL silently skips — `daily_prices` is gap-filled and PIT-safe where `prices` is raw; the decicents/YYYYMMDD storage conversion happens at the `db.py` boundary; Sticky-exclusion and the 3-station aggregation floor live in one place. A hand-rolled `AVG(price_decicents) FROM daily_prices` (no Sticky exclusion) is the anti-pattern — `average_price_series` already does it correctly.

### LightGBM fit + predict with DataFrame slices, not NumPy

Pass `df[feature_columns]` (a DataFrame) to `.fit()` and `.predict_proba()` at the model boundary — avoids sklearn's feature-name mismatch warning (`X does not have valid feature names, but LGBMClassifier was fitted with feature names`).

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

Sign convention throughout: `Δ = proposed − baseline`. Negative is better (logloss is minimised).

Multi-feature changes — a cluster drop, or a composite-to-decomposition swap — are evaluated as a single joint CV run when the changes are conceptually one unit (e.g. dropping all members of a SHAP-redundancy cluster, or replacing one feature with two derived from it). Independent feature changes ride in separate CV runs and separate PRs.

Default decision rule: if any single fold regresses by more than the median improvement, keep the feature (or feature group). The rule is asymmetric on purpose — a wide-mean, narrow-tail improvement is the win pattern; a regime that inverts the sign is the loss pattern.

Override is allowed when the regressing fold is known to be anomalous (a price-shock period, a labelling artefact, a regime explicitly out of scope). State the override reason in the PR body — a considered exception is fine; silently ignoring the rule is not.

**Why:** on 2026-06-03, `station_minus_last_max_cents` looked like a clean drop on one val window (5-seed Δ −0.0112 ± 0.0043), but a 14-fold paired walk-forward CV showed 7/14 fold-wins, mean Δ +0.0104, with fold 9 (2023-10→2024-01) regressing by +0.103. See `experiments/2026-06-03_drop_redundant_pair/`.

### New constants must not silently diverge from a canonical equivalent

When a change introduces a numeric constant (a band width, a window length, a threshold) that has a canonical equivalent already in the codebase, either reuse the canonical one or **ablate the divergence before merge** — cite the measured cost of the new value vs the canonical value, same evidence bar as a feature change.

**Why:** #217 introduced `COMP_BAND_CENTS=5.0` for the dispersion cohort while the canonical Competitive band was `±10c`. The divergence went unmeasured and understated #212's lift by ~0.009 Δh25; #219→#221 later established the canonical ±10c was correct and dropped the constant. A new magic number that shadows an existing one is a silent regression surface.

### Choosing the gate metric — classify the candidate first

The gate metric is an **explicit per-experiment choice**, stated and justified in the experiment README. Do **not** default to `delta_ll_hard25_median` (or any single proxy) — for a whole class of features it is the wrong arbiter.

Before choosing the gate, classify the candidate on two axes:

- **Decision-bias carrier vs descriptive covariate.** A feature whose value is a *cost/timing preference* — it biases *when* you buy under asymmetric payoffs — belongs in τ / the cost model, not the feature set. A feature that adds *information* belongs in the feature set. The two are tested differently; a hedge dressed as a feature will look inert once each arm picks its own honest τ.
- **Is the part you're correcting the part the model leans on?** A "more accurate" version of an existing feature only helps if the model actually uses the component you're fixing. Check SHAP leverage *and what shape* of the feature the model uses (a slow drift / regime clock vs a level vs the estimator's error) before assuming a correctness fix moves the objective. The CPL-optimal estimator is often *biased* vs the accurate one by construction.

For **decision-timing / trough / cycle-phase** features, WFCV per-row log-loss is a **non-rejecting SCREEN, not a verdict** — flat or slightly-negative log-loss does NOT reject. Their value lands in realised buyer outcome, which a calibration average washes out. The arbiter is a **paired realised backtest at a held operating point** (don't let a τ move masquerade as a feature win).

Two cheap pre-screens (no retrain) before committing to a feature-regen → retrain → recalibrate:

- **Log-loss as the clock-vs-hedge fingerprint.** An *information* (clock) signal moves the threshold-free measure (log-loss); a *cost-preference* (hedge) does not and is absorbed once each arm picks its own honest τ. Flat log-loss ⇒ not a clock.
- **τ-sweep inertness check** over the saved WFCV row predictions (`rowpreds.parquet`): buy-rate-vs-τ, proxy-economics peak + local flatness, and per-fold decision-disagreement at a common τ (split by regime to close the "a regime-localized effect cancels in the pool" escape hatch). Near-coincident arms ⇒ the change is economically inert; don't pay the retrain.

**Why:** #250 (boundary fix) and #254 (regime cycle-length denominator) both showed flat WFCV log-loss. #250 was realised-positive (saving 3.04% → 3.37%) and would have been wrongly binned on the screen; #254's τ-sweep showed the apparent realised "win" was an operating-point artifact and the feature economically inert (fold 7 — where the denominators diverge most in value — had the *lowest* decision-disagreement, 1.3%). A single proxy promoted to a hard reject gate fails for any feature class whose value is orthogonal to the proxy. The held-τ realised backtest is a one-call paired walk-forward capability: `experiments/lib/realised.run_paired_realised_backtest` (#255) — use it as the arbiter for decision-timing features.

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

## One source of truth for current model state

[docs/STATUS.md](STATUS.md) is the **only** place that states the live model's feature count, on-disk artifact, calibration method, τ, and active phase. Other docs (AGENTS.md, ML_SIGNAL.md, README.md, `PLAN_*.md`) link to STATUS for those facts rather than restating them. Lock tables and historical results stay as a dated record; it's the *"currently on disk"* claims that must live in one file.

**Why:** before 2026-06-13 the current feature count was restated in four docs and drifted as the model moved 50→54 features and raw→isotonic calibration — STATUS said 50/raw while the artifact was 54/isotonic. A fact repeated in N places is a fact that's stale in N−1 of them after the next lock.

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

## Experiment scripts

Any experiment script that runs LightGBM fits **must** use `experiments/lib/` helpers — do not copy scaffolding from prior scripts. This includes `paired_wfcv.py` harnesses, step-level ablation scripts (`step*.py`), and oracle/diagnostic scripts that call `fit_score`. Import with `PYTHONPATH=.`.

These rules govern **new** scripts. `experiments/lib/` landed 2026-06-11 and `load_features()` postdates many existing experiment dirs; older scripts are frozen lab-book entries — some gitignored, untracked exploration — that are not retrofitted, not the template, and not the standard. Read them for their results, not as a pattern to copy.

### Load the feature matrix via `load_features()`, never raw CSV

In experiment scripts: `from fuel_signal.features import load_features` then `df = load_features()`. Do **not** `pd.read_csv("data/features.csv")` directly.

**Why:** `load_features()` goes through the parquet cache (PR #193); the raw CSV read bypasses it, paying the full parse every run and risking a stale CSV when the parquet is newer.

**Canonical skeleton:** `experiments/TEMPLATE_paired_wfcv.py` — copy, rename the dir, fill in the TODOs. Do not reverse-engineer the loop shape from a prior experiment.

### In-script / lib seam

**In-script (per-experiment):** `add_candidate_columns()`, run grid (`RUNS`), `GateSpec` thresholds, cohort/bucket boolean masks, `meta["definitions"]`.

**Lib (always import):** fold iteration, fitting, per-row loss, cohort mask, row-pred collection, seed-variance gate, aggregation, gate evaluation, meta I/O, timing, shared constants.

**Promotion rule:** if an `add_candidate_columns` block is copied into 2+ experiments unchanged, extract the primitive into `experiments/lib/features/` and import it.

| Module | Purpose |
|---|---|
| `constants.py` | `SEEDS`, `SHOCK_FOLDS`, `LGBM_DEFAULTS` — import; never redefine per-script |
| `fit.py` | `fit_score(train_df, val_df, cols, seed)`, `per_row_log_loss(y, p)` |
| `folds.py` | `iter_folds_with_baseline_fit(df, baseline_cols)` — yields baseline fit per fold; per-fold loop body stays in the script |
| `cohorts.py` | `hard_quantile_mask(prl, q)` — top-(1-q) fraction by per-row log-loss |
| `gates.py` | `GateSpec` + `evaluate_gates(fold_run, spec, run)` — single source for Δ sign (`run − R0`; negative = better; passes when `value <= threshold`); `seed_variance_gate(df_rows, cohort_ll_map)` — flags cells where seed_std > 5× cohort median |
| `aggregate.py` | `aggregate_with_deltas(df_rows, cohort_ll_map)` — groups by (fold, regime, run), appends delta_* vs R0 |
| `io.py` | `to_jsonable(o)`, `write_meta(out_dir, meta)` |
| `timing.py` | `time_block(label)` context manager — prints `  [label] N.Ns` |
| `rowpreds.py` | `RowPredCollector(ident_base)` — set `collector.ident_base = ident` each fold, call `collector.add(run, seed, proba)` per fit, `collector.to_parquet(path)` at the end |

### Feature-computation primitives

The inside of every `compute_features()` / `add_candidate_columns()` uses helpers from `experiments/lib/features/`. Do not inline the primitive; import and name the intent.

| Helper | Module | PIT-safety note |
|---|---|---|
| `cohort_std_by_date(df, mask)` | `dispersion` | mask must be same-date row attributes; no future rows enter |
| `cohort_agg_diff_by_date(df, mask_a, mask_b)` | `dispersion` | same constraint as `cohort_std_by_date` |
| `calendar_aware_delta(per_date_series, lag_days)` | `deltas` | reindexes to daily grid before shifting; gaps → NaN, not silent span |
| `rolling_baseline(per_date_series, window_days)` | `rolling` | `closed='left'` by default; today excluded from today's aggregate |
| `px_change_lag_diagnostic(df, lag_days)` | `diagnostics` | exact-date self-merge with `validate='m:1'`; never positional diff |

Signal C in `a_c_ablation` (row-wise std across LGA columns) is column-wise, not row-filtered — `cohort_std_by_date` does not apply; that computation stays inline.

Cross-reference: `feedback_experiment_scripts_pythonpath` (`PYTHONPATH=.` prefix); `feedback_instrument_walltime` (time + log per step); `feedback_throwaway_validation_scripts` (minimal one-off validators).

## Shell tooling

Use `jq` for JSON slicing in bash, not `python3 -c "import json…"`. Idiomatic, cleaner output, no temp scripts.
