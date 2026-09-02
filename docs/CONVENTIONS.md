# Conventions

Working rules for changing this repo. This file is Claude-facing: dense and
reference-shaped on purpose, not prose for a new human contributor. The
architectural shape lives in [AGENTS.md](../AGENTS.md); this is the changeable
how-we-do-things layer.

Each rule has a **Why** (the incident or constraint behind it) so edge cases can be
judged rather than blindly followed. Full derivations, decision logs, and forensic
detail live in bd issues and `experiments/` dirs, cited by ID — don't expect a
rule's full history to be re-narrated here; if you need it, follow the citation.

**Contents:** [Code](#code) · [Tests](#tests) · [Changing the production feature
set](#changing-the-production-feature-set) · [The baseline feature set is declared,
never discovered](#the-baseline-feature-set-is-declared-never-discovered) · [Noise
floor & placebo arity](#noise-floor--placebo-arity) · [The decision cadence is a
lock parameter](#the-decision-cadence-is-a-lock-parameter-declared--not-a-default) ·
[New constants must not silently
diverge](#new-constants-must-not-silently-diverge-from-a-canonical-equivalent) ·
[Choosing the gate metric](#choosing-the-gate-metric--classify-the-candidate-first)
· [Bucketed results](#bucketed-results--check-the-convention-spread-before-believing-an-ordering)
· [Definition of done](#definition-of-done) · [Decisions land in repo
docs](#decisions-land-in-repo-docs-not-just-memory) · [One source of
truth](#one-source-of-truth-for-current-model-state) · [Docs and memory: signal
over sediment](#docs-and-memory-signal-over-sediment) · [Git
workflow](#git-workflow) · [PR feedback loop](#pr-feedback-loop) · [Code review
caution](#code-review-caution) · [Experiment scripts](#experiment-scripts) ·
[Shell tooling](#shell-tooling)

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

**Pre-registered instance:** the fold whose val window is `2022-02-03 to 2022-05-03` spans the unrecoverable NSW source-data hole of 2022-03-12 to 2022-03-21 (`fps-tpy`; see [AGENTS.md § Known source data limitations](../AGENTS.md#known-source-data-limitations)). Decided 2026-08-27: flag it if it regresses, don't exclude it from the harness — the effect is small, partial, and confined to one fold, and this window already overlaps the Ukraine-invasion shock most regime-segmented runs treat as elevated-variance.

**Why:** on 2026-06-03, `station_minus_last_max_cents` looked like a clean drop on one val window (5-seed Δ −0.0112 ± 0.0043), but a 14-fold paired walk-forward CV showed 7/14 fold-wins, mean Δ +0.0104, with fold 9 (2023-10→2024-01) regressing by +0.103. See `experiments/2026-06-03_drop_redundant_pair/`.

### The baseline feature set is declared, never discovered

Code that needs "the production baseline columns" imports **`fuel_signal.features.LOCKED_FEATURE_COLUMNS`** — one symbol, re-exported for experiment scripts as `experiments.lib.constants.BASELINE_COLUMNS`. Never retype the group composition `FEATURE_COLUMNS + LGA_FEATURE_COLUMNS + NETWORK_FEATURE_COLUMNS`, and never derive the set by inspecting a features frame's header — header inspection can't distinguish "in the lock" from "evaluated and rejected, still computed" (e.g. Phase 4b brand-trough columns) from "evaluated, inconclusive" (e.g. `tgp_delta_7d`), so it silently promotes the wrong columns. `fuel_signal.features.non_model_columns(df)` names the held-out categories with a machine-readable reason code.

This binds analysis tooling too, not just the training path — `experiments/pipeline/redundancy.py` once retyped the composition and disagreed with the lock by one column, and it went undetected until a re-tested known graduate came back with the opposite sign (`fps-3jj.11`).

**Order is part of the contract, and new columns append — never sort.** LightGBM breaks equal-gain split ties by feature index; with 35 near-identical LGA trough columns, exact ties are common, so reordering fits a *different* model even with the same columns. Order-variance is small relative to seed-variance (~2%) but not zero at the arbiter (a sorted-vs-production permutation moved batch0's realised delta by 0.038 c/L).

**Graduating a held-out column is two edits, not one.** Adding it to `LOCKED_FEATURE_COLUMNS` is not enough — delete its `NON_MODEL_COLUMNS` entry in the *same* change, or `resolve_baseline_columns()` raises `NonModelColumnLeak`. This is a hard stop on purpose: a detector that skipped whatever the lock already claims couldn't have caught a column wrongly *in* the lock, which is the failure this guards against.

**Every result records which baseline it was measured against.** `baseline_fingerprint(columns)` returns `'<n>:<sha12>'` over the ordered list, stamped into experiment `meta.json`, batch `freeze.json`, and each run's `results.json`/`facts.json`. Two runs with differing fingerprints are not comparable, whatever their deltas say. A `null` fingerprint means the run predates the field — there is no safe fallback for it.

**Why:** `batch_freeze.resolve_baseline_columns()` once appended a rediscovered brand-column set "for completeness," giving batch0 a 64-column R0 (production + the rejected Phase 4b group) for every candidate — silently, with no trace in the run artifacts, for two months (`fps-sa1`, `fps-zci`, `fps-nor`). Full incident: `bd show fps-zci`.

### Noise floor & placebo arity

The AI-sourced feature pipeline's realised-CPL noise floor (placebo-column null, arity pricing, `TEXTURE_ICC_BOUND`, recompute/promote procedures) is documented in
[docs/feature-pipeline.md § Key contracts](feature-pipeline.md#key-contracts) — that's the authoritative copy, not restated here.

One rule is worth stating twice because it's easy to get wrong from memory: **the noise-floor bar is BATCH-SPECIFIC.** Never reuse a prior batch's number (e.g. batch1's −0.27 c/L) as a portable constant for a different batch — recompute per batch and cite that batch's own floor.

### The decision cadence is a lock parameter, declared — not a default

`TankParams.evaluation_interval_days` looks like a knob and behaves like part of the model contract — it moves realised economics by more than most feature decisions do (`experiments/2026-08-20_cadence_ceiling/`, bd `fps-fii`: same model/folds/seed/columns at 7 / 2 / 1 day gave headroom 1.60 / 2.94 / 3.06 c/L and realised CPL 190.06 / 188.28 / 188.04, re-measured 2026-09-02 — see the re-lock table below for the pre-#358 figures).

**The canonical cadence is 1 day** (re-locked 2026-08-22, `fps-oqz`). `TankParams.evaluation_interval_days` defaults to it and `score_phase2.py` constructs a bare `TankParams()`, so every row from here on inherits it. Do not change it because a finer grid scores better — that's discovering the lock rather than declaring it, the same failure class as resolving R0 from a frame header.

Moving it is a deliberate re-lock: a recorded rationale, a stated before/after, every subsequent row stamped so pre/post rows are mechanically distinguishable.

**The 2026-08-22 re-lock (7d → 1d):**

| | before | after |
|---|---|---|
| stamp | `50/3.571/7d/10%` | `50/3.571/1d/10%` |
| realised CPL (τ=0.25, same folds/stations/seed) | 189.67 | 187.82 |
| headroom vs oracle | 1.54 c/L | 2.97 c/L |
| chosen fills / fills-per-station-yr | 244 / 44 | 1832 / 135 |
| emergency (forced) fills | 67.6% | 21.4% |

**Those CPL and headroom figures are the ones the re-lock was decided on, and they stay as
the record.** They were measured with the pre-fps-2i4 price accessor, which forward-filled
a closed station's last price indefinitely. Re-running `model_cadence.py` on 2026-09-02
against the fixed accessor (fps-2i4) and the fixed oracle DP (PR #358) gives realised CPL
**190.06 → 188.04** and headroom **1.60 → 3.06 c/L** — the same direction and roughly the
same size, so the re-lock decision is unaffected (it rested on the owner's rationale, not
the economics, in any case). Quote the 2026-09-02 figures for anything forward-looking and
these for anything describing the 2026-08-22 decision.

**Caveat, outstanding as of 2026-09-02:** `model_cadence.py` currently reports
`ceiling_valid = False` at *every* cadence, because its run-dry gate tests
`dry_events == 0` — a proxy for "the oracle pruned run-dry paths and the model didn't",
which PR #358 made obsolete by having the oracle clamp and account for forgiven depletion
too. The dry litres are ~1000 L at every cadence and divide exactly into whole depletion
periods, i.e. one contiguous no-price span (station 414's closure), not cadence stranding.
Re-aiming that gate is tracked on `fps-32h`; until it lands, treat the headroom figures
above as measured-but-ungated.

The decision rests on the owner's rationale (uniformity with daily price cadence, max decision resolution, a choice every day as the intended product experience), not on the economics — `fps-929` found re-picking τ at daily cadence is worth only 0.062 c/L, so τ stays 0.25. It is **not a retrain** (the fit takes no `TankParams`) and **not a production code change** (`evaluation_interval_days` is consumed only by the backtest engines; the live daily signal in `fuel_signal/signal.py` is rule-based and already emits daily).

**Quote realised CPL and headroom with the cadence attached.** "1.60 c/L of headroom" is a fact about a specific cadence, not about the model.

**Every artifact that records a realised CPL must carry the `TankParams` it was produced at, and the writer must raise rather than silently omit it** (`fps-15c`). `fuel_signal.backtest.require_tank_stamp(tank, what=...)` is the one shared path — every writer routes through it:

| Site | What it stamps |
|---|---|
| `experiments/results.csv` (`log_experiment`) | `tank_params` column, empty when no backtest ran |
| `RealisedResult.meta` / `runner.py`'s `results.json` | `meta["tank_params"]`, always present |
| `dossier_tables.build_facts`'s `facts.json` | `provenance["tank_params"]`; **raises** if a CPL is about to be written with no source stamp |
| `batch_freeze`'s `freeze.json` / `noise_floor`'s `noise_floor.json` | the batch's declared cadence contract |

`dossier_tables._noise_band()` refuses (`available: False`) whenever a candidate run's `tank_params` doesn't match the batch's `noise_floor.json` — the cadence-axis twin of the `baseline_fingerprint` refusal above.

Tested generically, not per-site: `experiments.lib.io.artifact_has_unstamped_cpl(obj)` walks any JSON-shaped artifact for a `*cpl*` key with no `tank_params`/`tank` key alongside it, and `test_every_committed_experiment_json_artifact_carries_its_cadence_stamp` runs it over every git-tracked `experiments/**/*.json` — a generic backstop that catches the next mechanism without needing a dedicated test per site.

**A CLI default must be sourced from `TankParams()`, never a hand-copied literal** (`fps-q3p`) — `backtest.py`'s `--daily-use` once defaulted to a separately-rounded literal that rendered identically to `TankParams`'s own default at the stamp's 3dp precision while consuming marginally differently. Source CLI defaults from the dataclass field, not a literal, and let a test (`test_cli_daily_use_default_tracks_tankparams`) pin it.

**Cadence is not free to sweep** — a tank config can be unsafe (able to run dry before the next decision) at some cadences. `fuel_signal.backtest.validate_never_dry(tank)` enumerates the reachable decide-point lattice and the CLI runs it automatically, rejecting an unsafe `--tank-size`/`--daily-use`/`--eval-interval` combination rather than silently producing a wrong CPL. Default tank: 1–7 day cadence is safe, 8–14 is not; both the old 7d and current 1d canonical cadences are safe.

**To compare across the 2026-08-22 boundary, freeze a NEW batch at 1d.** Do not `noise_floor.py batch0 --force` — `check_freeze_cadence` refuses it outright (batch0's data was frozen at 7d and can't retroactively become a 1d batch); `batch_freeze.freeze_batch` also refuses to re-freeze an existing batch at all (`FileExistsError`).

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

### Bucketed results — check the convention spread before believing an ordering

Before reporting any aggregate **sliced into buckets** (cycle regime, volatility band, quarter, month), list the free choices its construction made — inventory basis, timestamp convention, boundary inclusivity, span trimming, tie-breaking — recompute it under each, and report:

```
spread ACROSS conventions, per bucket   (the ruler's sloppiness)
spread BETWEEN buckets, per convention  (the thing you want to see)
```

**A bucket comparison is readable only when the between-bucket gap exceeds the between-convention gap.** If it doesn't, do not pick a convention and proceed — that codifies an assumption as a fact, and the temptation is always to pick the one agreeing with the hypothesis you already hold. Change the measured quantity instead.

The convention spread is a **bias** term: it does not shrink with more stations, folds or seeds. A result can carry a tight bootstrap CI and still be meaningless, so report the spread *next to* the CI — they answer different questions. And an ordering claim is a **contrast**: bootstrap the difference rather than eyeballing two overlapping intervals (shared folds make the contrast tighter than either interval suggests), and run it for every axis you make a claim on, not just the headline one.

**The specific trap this generalises:** allocating a **path-coupled total cost** to sub-periods has no unique answer. In a tank-based backtest the oracle buys cheaply just *before* an expensive stretch and coasts through it; which period gets the credit is a choice. So `model_cpl − oracle_cpl` is well defined at **window level, per station** — the granularity `run_oracle_backtest` optimises — and **not identified at any sub-window zone**. Quantities natively stamped at a moment (per-row log-loss, per-decision accuracy, prices, predictions) are safe to bucket.

**Why:** the #262 headroom map's per-zone rows drove real decisions — "regime axis FLAT" retired the late-descent thread, and a 12–16c volatility "hump" was treated as a target. `experiments/2026-08-20_headroom_attribution/` recomputed them under six conventions: every zone moved 2.5–5.1 c/L while the zones differed by 0.5–3.4, no contrast separated on either axis, and the impossible negatives that motivated the fix reappeared under two of the six. Every per-zone row was withdrawn; the window-level number survives *as a window-level quantity* — but it is not a constant, and is itself conditional on cadence (1.60 c/L at 7-day, 3.06 c/L at daily — `fps-fii`).

**The mechanical corollary: cut economics on folds, not on row labels.** A fold is not a sub-period — `experiments/lib/realised.py` calls `aggregate_backtest` once per fold and `aggregate_backtest` calls `run_backtest` once per station with a fresh tank, so each (fold, station) is an **independent simulation** and any per-fold, per-station or per-fold-group figure is a sum of complete windows. A row-level label (cycle regime, day-of-week, volatility band, a candidate's `add_axis`) slices *through* a window instead, so a cost cut on one is unidentified no matter how many fills back it. Express a zone claim as a set of folds where you can; where the mechanism really is row-level, make the claim on a quantity stamped at a moment (per-row log-loss, per-decision accuracy) rather than on pooled CPL. `experiments/pipeline/` enforces this by attaching `ROW_AXIS_ECONOMICS_CAVEAT` to any `per_axis` delta or `CONFIDENCE_ZONE` grade that used a row label.

**The tell (impossible negatives) is often missing.** It only worked for #262 because the comparison was against an oracle, which cannot lose. The same defect measured against always-buy produces no impossible value at all — absence of an impossible value is not evidence of identification. Enumerate the free conventions and measure the spread; don't wait for the quantity to embarrass itself. (The 2026-06-18 gate-1 per-regime saving% sat unchallenged for two months for exactly this reason — withdrawn 2026-08-21, `experiments/2026-08-21_path_coupling_audit/`, bd `fps-grp`.)

## Definition of done

Before considering a change complete, in this order:

1. **Re-read the issue** if the change closes one (`bd show <id>` — see [AGENTS.md § Beads](../AGENTS.md#beads)). Walk through the acceptance criteria / deliverables list and confirm each item is covered by the diff. Scope often drifts during implementation; the issue is the source of truth for what was promised, and the check catches gaps before review does. If something in the issue is no longer the right thing to build, say so in the PR body rather than silently dropping it. Once merged, `bd close <id>` explicitly — there is no GitHub auto-close for a bd issue — then `bd dolt push`.
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

This rule applies to this file too — see [§ Noise floor & placebo arity](#noise-floor--placebo-arity) for where its own forensic history was moved out to (2026-09-02).

## Git workflow

- **Fresh branch per PR.** Branch off `main` for each PR; do not continue committing to a previously merged branch even though GitHub diffs against `main` would still work.
- **Open the PR immediately** after the first commit+push — no need to ask first.
- Branch naming, PR title format, and PR body shape: see [AGENTS.md § Branch and PR conventions](../AGENTS.md#branch-and-pr-conventions).
- **Experiments lab book is exempt.** Changes confined to `experiments/**` may be committed **and pushed** directly to `main` without a PR. Each experiment dir is a self-contained lab book entry; iterate freely. `experiments/results.csv` (the formal graduated-experiment log) and `experiments/INDEX.md` (the lab book index) are also direct-to-`main`. Anything touching `fuel_signal/`, `tests/`, `docs/`, or top-level config still goes through a PR even if an experiment motivated it.
- **"Direct to `main`" means committed AND pushed — a commit sitting on a local `main` is not landed.** The PR path can't get this wrong (you cannot open a PR without pushing), so every place the word "commit" appears alone is on an exempt path, and the exempt paths are the ones unattended routines use. That asymmetry bit us for real: on 2026-08-26 eight commits — two candidate dossiers, a post-hoc script, ledger/INDEX rows, and two interaction-log entries — were found sitting unpushed on the primary worktree's local `main`, the oldest two days old. One of them was itself the fix for the *commit* half of this same gap, written and then never pushed, so the fix reproduced the bug it fixed. **Postcondition for any routine or session that writes to `main`: `git status --short` is empty AND `git log --oneline origin/main..main` is empty before you report the work done.** Check it, don't remember it.
- **PNGs are gitignored under `experiments/**` — with one tracked exception.** `experiments/candidates/**/*.png` is explicitly un-ignored (a `!` negation after the blanket rule): those plots are only ever written once by `dossier_tables.py` for a run about to get a permanent, committed write-up, small and bounded in count (six per run), and the exact thing the README's `![](...)` embeds reference. Follow the same pattern (a scoped negation, not lifting the blanket rule) if another pipeline needs its plots tracked.

## PR feedback loop

Immediately after `gh pr create` returns a PR number, call `ScheduleWakeup(delaySeconds=270)` with a prompt that runs `gh pr view <N> --json comments,reviews,mergeable,statusCheckRollup`. This is a mandatory mechanical step, not a suggestion — do it before writing any response to the user. When the wakeup fires: act on any actionable comments present. If CodeRabbit is rate-limited or absent, **skip it and move on — do not reschedule to wait for it**. Use judgement on style nits that conflict with project conventions. Run `uv run ruff check . && uv run pytest -q`, push, and repeat until no actionable comments remain. The goal is a ready-to-merge deliverable.

**A re-review after a fix commit can re-post identical comments against stale line numbers.** Observed on PR #311 (fps-3jj.9): CodeRabbit's second review, triggered by the fix-commit push, posted the same 4 comments verbatim — including inline diff suggestions quoting the pre-fix code — even though the fix commit had already addressed every one of them. Don't assume a repeated comment is a new/unaddressed finding; `grep`/`sed` the current file at the cited path and check whether the flagged code still looks like what the comment describes before touching anything again.

**The opposite timing failure also happens, and its findings can still be valid.** Observed on PR #313 (fps-3jj.8): a manually-pasted external review was addressed and pushed as a fix commit, then CodeRabbit's own review posted ~20 minutes later — but its own metadata showed it had been comparing against the *original* pre-fix commit, not current HEAD. Its 6 findings didn't overlap with the fix already pushed, and 2 of them were real (confirmed against current code) and worth fixing anyway. The commit range a review cites is not a reliable signal of whether its findings are live or stale in either direction — verify every finding against current code regardless of what the review says it diffed against, the same discipline as the stale-repost case above, not a reason to auto-dismiss a review that looks behind.

**CodeRabbit's "Docstring Coverage" pre-merge check is a bot metric, not a finding — treat it like a style nit, not an actionable comment.** Observed on PR #344 (bd-fps-b5p): CodeRabbit's own review said "No actionable comments were generated," but its pre-merge-checks table separately flagged "Docstring Coverage 66.67% (threshold 80%)." This repo's convention is WHY-comments over formal docstrings (see top of this doc) — writing docstrings purely to clear the bot's threshold would fight that convention for no reader benefit. Left unaddressed; the PR merged clean anyway (the check is advisory, not a merge gate).

**Sourcery can rate-limit ("used your own review budget") the same as CodeRabbit can be absent — treat it identically: skip and move on, don't reschedule to wait for it.** Observed on PR #344.

## Code review caution

Before filing an issue from an agent-driven logic review:

- **Trace a concrete example** end-to-end, especially for format-handling code. The `history.py` YYYY-DD-MM date-swap condition was wrongly flagged because the agent didn't walk through a case where `raw_day == true_month`.
- **Check the docstring** for stated design intent before claiming inconsistency. `series.py`'s `brand:` resolver was wrongly flagged for using exact match — the docstring said exact was the intent.
- **Distrust prose numbers when verifying a code constant.** CodeRabbit flagged `snapshot_retire.py`'s `DEFAULT_TOLERANCE = 0.05` as "should be 5.0 cents", citing nearby AGENTS.md prose that loosely said "agrees within 5c". The prose was the imprecise one (the actual check used `<0.05`, effectively an exact-match test since prices are 0.1c-quantized) — the constant was correct. Verify against what the code actually does, not how a nearby doc rounds it off in words.
- **This applies to a review from another Claude session too, not just bots.** An independent Claude session reviewing PR #301 (fps-3jj.6) found a genuinely severe bug (`launch.py`/`runner.py` give every candidate in a batch the same `out_dir`, so candidate 2+ silently overwrites candidate 1) but also one over-extended claim: it flagged a test fixture's `price_date` dtype as mismatched against `runner.py`'s real `ident_base`, reading only the diff hunk rather than the unchanged lines above it (`runner.py`'s `_run_wfcv_screen` does `pd.to_datetime(...)` before writing `ident_base` — the fixture was already correct). Checking the *full* function a diff hunk lives in, not just the added/changed lines, would have caught this before it needed a round-trip to correct. Verification discipline doesn't relax because the reviewer is another agent instead of a bot — if anything a thorough, well-argued review is more persuasive and needs the same check-before-acting.

When a review's findings come back reported as fixed, re-verify against the diff rather than closing on the report:

- **Re-read the diff, don't accept the summary.** In the fps-hvi review of PR #299, the fixer reported one finding as "already fixed before your review ran, stale" and cited a commit that had only touched `pit_test.py`/`validate.py` — the finding was live when raised and was fixed by a later commit. Cheap to check with `git show --stat`; the conclusion happened to be right, but the reasoning would have discredited a valid finding.
- **Re-review the fix itself for regressions.** The same round's fix for "write artifacts before grading" made `_grade_run` return `effect_delta = None` on failure, which `_summarise_for_comment` then fed to a `:+.4f` format spec — a `TypeError` on exactly the path the fix existed to survive. A full green suite (1038 passed, ruff clean) did not catch it, because the new test didn't exercise the one argument that triggers it. A fix lands in code that the original review already mapped; re-run that map over it.

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
