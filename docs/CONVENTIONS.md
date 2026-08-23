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

### The baseline feature set is declared, never discovered

Code that needs "the production baseline columns" imports **`fuel_signal.features.LOCKED_FEATURE_COLUMNS`** — one symbol, re-exported for experiment scripts as `experiments.lib.constants.BASELINE_COLUMNS`. Never retype the group composition `FEATURE_COLUMNS + LGA_FEATURE_COLUMNS + NETWORK_FEATURE_COLUMNS`, and never derive the set by inspecting a features frame's header.

`data/features.csv` is deliberately a **superset** of the model contract. A column sits in it for one of three reasons, and only the first puts it in the baseline:

| Reason | Example |
|---|---|
| In the lock | the 54 (see [AGENTS.md § Canonical feature set](../AGENTS.md#canonical-feature-set-54-feat-baseline-locked-issue-216)) |
| Evaluated and **rejected**, still computed | the 10 `days_since_trough_entry_<brand>` (Phase 4b, 2026-06-02) |
| **Held out** pending graduation | `tgp_delta_7d` (#271) |

Header inspection cannot tell these apart, so it silently promotes rejected and held-out columns into the baseline. `fuel_signal.features.non_model_columns(df)` names the second and third categories with a machine-readable reason code, so "outside the lock" is an assertable condition rather than something a reviewer has to already know.

**The rule binds analysis tooling, not just the training path (`fps-3jj.11` pre-start check, 2026-08-23).** Anything that asks "what does the model already have" resolves the set the same way. `experiments/pipeline/redundancy.py` retyped it as `FEATURE_COLUMNS + LGA_FEATURE_COLUMNS + NETWORK_FEATURE_COLUMNS + TGP_FEATURE_COLUMNS` (55) for its block-R² predictor set, and `docs/routines/generator.md` named the same four symbols, so the code and the spec agreed with each other and disagreed with the lock. The screen therefore regressed candidates on `tgp_delta_7d`, a column R0 never sees — inflating measured redundancy for anything correlated with the wholesale signal, i.e. against the lead the ledger rates highest among still-open ground (`fps-x0f`). It scored `batch0`'s own `tgp_delta_7d` at block R² = **1.000** and flagged it `REDESIGN`: a predictor of itself. Against the real lock it is 0.288. A *reported* number rather than a gate, so nothing was mis-graded — but it is the same defect class as `fps-sa1`, caught by a live run rather than by the tests, one of which asserted the wrong invariant outright.

Note that `train_lgbm`'s *default* resolution is not the lock — the lock is `train_lgbm --no-brand-features`. "Mirrors train_lgbm's default" is not a justification for a baseline resolution.

**Order is part of the contract, and new columns append.** Keep the production order — the group concatenation above, which is exactly what `data/models/lgbm_calibrated.joblib`'s `feature_columns` holds — and never sort it. LightGBM breaks equal-gain split ties by feature index, and with 35 near-identical LGA trough columns exact ties are common, so the same columns in a different order fit a *different* model. Sorting looks tidier and is the wrong trade: it makes the contract file diff-stable while making the model unstable to feature additions, since one alphabetically-early insertion reshuffles every index after it. Appending leaves existing indices untouched.

How much does order matter? **Pin it; don't sweep it.** Measured over 5 draws each on one train/val slice, order-variance is ~2% of seed-variance (log-loss std 0.000258 vs 0.015643; decision flips at τ=0.25 of 8–29 rows vs 1743–2069). It does not need the multi-seed discipline that seeds get. But it is *not* nothing at the arbiter: the sorted-vs-production permutation moved batch0's pooled realised delta by 0.038 c/L, ~0.8 of a decision flip — which says more about the arbiter differencing two nearly-identical quantities than about order.

**Graduating a held-out column is two edits, not one.** Adding it to `LOCKED_FEATURE_COLUMNS` is not enough — you must delete its `NON_MODEL_COLUMNS` entry in the *same* change. Until you do, `resolve_baseline_columns()` raises `NonModelColumnLeak`, because `non_model_columns()` deliberately does **not** consult the lock: a detector that skipped whatever the lock already claims could not detect a column wrongly *in* the lock, which is the entire fps-sa1 failure. This is a hard stop rather than a warning, on purpose — graduation is a declaration, not something the code should infer from a list it cannot audit. The standing example is `tgp_delta_7d` at the #271 chip-4 re-lock (bd `fps-1785999729707-1-0301bf82`, component 5) — **now hypothetical, not live**: #271 closed as superseded on 2026-08-20 and that re-lock will not happen, so `tgp_delta_7d` stays in `NON_MODEL_COLUMNS` (as `evaluated-inconclusive`) indefinitely. The two-edit rule is unchanged and applies to whatever graduates next.

**Every result records which baseline it was measured against.** `baseline_fingerprint(columns)` returns `'<n>:<sha12>'` over the **ordered** list; it is stamped into experiment `meta.json` (automatically, by `experiments.lib.io.write_meta`), batch `freeze.json`, and each run's `results.json` → `facts.json`. Two runs whose fingerprints differ are not comparable, whatever their deltas say — check the fingerprints before comparing numbers across runs. A `null` fingerprint means the run predates the field, which is exactly the population where a wrong R0 could be hiding; there is no safe fallback, so nothing substitutes today's constants for it.

**Why:** `batch_freeze.resolve_baseline_columns()` appended `discover_brand_feature_columns(df)` on exactly that rationale, giving batch0 a 64-column R0 — production plus the rejected Phase 4b group — for every candidate in the batch, on both the log-loss screen and the realised arbiter. It surfaced only because `tgp_delta_7d` was re-tested as a known graduate and came back with the opposite sign; the candidate arms agreed to 0.0013 c/L while the baselines differed by 0.132 (fps-sa1, fps-nor). Neither defect left any trace in the artifacts that recorded the runs, which is why two incommensurable baselines were compared head-to-head for two months — hence the single symbol and the fingerprint above (fps-zci).

**A re-lock invalidates the batch's noise floor — but which recovery path applies depends on WHAT re-locked.** `experiments/pipeline/noise_floor.py` fits the frozen baseline at a single fixed seed and takes the realised-CPL delta against ~20 placebo-column draws (`fps-awz`, reworked from an earlier paired-seed-group design — see `docs/feature-pipeline.md`) — that fit is against a specific `baseline_fingerprint` AND a specific `tank_params` cadence, same as every candidate run. Every successful graduation changes `LOCKED_FEATURE_COLUMNS` (a 55th column changes the model whose noise was measured), so a floor computed before a graduation does not grade runs made after it: `dossier_tables._noise_band()` compares the floor's own `baseline_fingerprint` (and, separately, its `tank_params`, `fps-v8o`) against the candidate run's and refuses (`available: False`) on any mismatch, the same style as its existing `partial`-fold-subset refusal. `batch_freeze.freeze_batch()` computes the floor as its own final step, so a fresh batch always starts with a matching one.

For a **column-lock** graduation, recompute an existing batch's floor in place with `PYTHONPATH=. uv run python -m experiments.pipeline.noise_floor <batch> --force` (or after any `--skip-noise-floor` freeze, fps-cf8) — the batch's `freeze.json` still declares the right cadence, only the fingerprint moved. For a **cadence** re-lock this does NOT work: `check_freeze_cadence` (`fps-oqz`) refuses `--force` outright when the batch's `freeze.json` and the current `TankParams()` default disagree on cadence, rather than silently leaving the batch split-brained (floor at the new cadence, manifest still declaring the old one). Freeze a **new** batch at the new cadence instead — this is what happened to `batch0` at the 2026-08-22 7d→1d re-lock (`fps-oqz`); its floor was never recomputed, `batch1` was frozen fresh at 1d instead (`fps-aay`).

**The gate is distance from the band, not empirical rank (`fps-awz`).** `dossier_tables.py`'s `family_wise_z_threshold` (moved there from `retrospective.py` by `fps-3jj.17` so `_noise_band()` can attach a per-run threshold to `facts.json`; `retrospective.py` imports it) is a Bonferroni-corrected t-critical value in band-standard-deviation space; `clears_family_wise_threshold` compares each candidate's `candidate_z_vs_band` against it. The empirical percentile (`candidate_percentile_better_than_noise`, `family_wise_percentile_threshold`) is still computed and reported, but as descriptive colour only — with a draw count in the tens, the empirical-rank statistic has too few distinct values to resolve anything finer than "beat every draw."

**The honest single-candidate bar, as a number (`fps-awz`, resolved via `fps-aay`'s batch1 freeze, 2026-08-22).** `batch1`'s noise floor (`experiments/batches/batch1/noise_floor.json`, 20 placebo-column draws at the batch's 1d cadence): `band_mean_delta_cpl_held = -0.0089 c/L`, `band_std_delta_cpl_held = 0.0800 c/L` (n=20, sample std). At batch1's own size (**5 candidates**, `docs/routines/generator.md` § Batch sizing), `family_wise_z_threshold` requires a candidate's `effect_delta_cpl_held ≤ -0.22 c/L` to clear as surprising at the BATCH level — a single candidate judged alone needs only `≤ -0.15 c/L`. This is a DETECTION threshold, not a target: `experiments/results.csv`'s history shows single-column features landing between 0.03 and 0.26 c/L, so a bar above that range describes the instrument's resolution, not what a good feature looks like — do not read this as "propose nothing below -0.22 c/L." The number itself is **batch-specific** (a new batch's own floor depends on its own draws, columns and cadence) — don't reuse -0.22 c/L as a constant the way the withdrawn `0.05 c/L` TGP figure was wrongly reused; recompute per batch and cite that batch's own floor.

### The decision cadence is a lock parameter, declared — not a default

`TankParams.evaluation_interval_days` looks like a knob and behaves like part of
the model contract. `experiments/2026-08-20_cadence_ceiling/` (bd `fps-fii`)
measured the same model, folds, seed and columns at 7 / 2 / 1 day and got
**headroom 1.54 / 2.69 / 2.97 c/L and realised CPL 189.67 / 187.85 / 187.82**.
Cadence moves realised economics by more than most feature decisions do.

**The canonical cadence is 1 day** (re-locked 2026-08-22, `fps-oqz` — see the
before/after below). `TankParams.evaluation_interval_days` defaults to it, and
`score_phase2.py` constructs a bare `TankParams()` with no CLI override, so every
row written from here on inherits it. Do not change it because a finer grid scores
better — that is discovering the lock rather than declaring it, the same failure as
resolving R0 from a frame header.

**Moving it is a deliberate re-lock**, with the same ceremony as changing
`LOCKED_FEATURE_COLUMNS`: a recorded rationale, a stated before/after, and every
subsequent row stamped so pre- and post-change rows are mechanically
distinguishable.

#### The 2026-08-22 re-lock: 7d → 1d (`fps-oqz`)

| | before | after |
|---|---|---|
| `evaluation_interval_days` | 7 | 1 |
| stamp | `50/3.571/7d/10%` | `50/3.571/1d/10%` |
| realised CPL (τ=0.25, same folds/stations/seed) | 189.67 | 187.82 |
| headroom vs oracle | 1.54 c/L | 2.97 c/L |
| chosen fills | 244 | 1832 |
| fills/station/yr | 44 | 135 |
| emergency (forced) fills | 67.6% | 21.4% |

Note what did and did not argue for it. `fps-929` closed as a *signal-content
limit* — re-picking τ at daily cadence is worth only 0.062 c/L, so the economics
did **not** make the case, and τ stays 0.25. The decision rests on the owner's
rationale: uniformity with the daily price cadence (one clock, not two), maximum
decision resolution, and a choice every day as the intended product experience.
The 1.85 c/L is a consequence, not the argument. 2d was recommended by `fps-929`
on the hassle trade-off (188.00 c/L, 83 fills/yr, 37.5% forced) and explicitly
overridden by the owner on that rationale; anyone revisiting this should argue
against the rationale, not re-derive the table. Source:
`experiments/2026-08-21_tau_cadence/tau_sweep.csv`.

**The 1.85 c/L is conditional on the driver actually behaving this way.** Cadence
is an *evaluation* parameter — an assumption about how often the owner checks and
acts — not a property of the model. It is **not a retrain** (the fit takes no
`TankParams`; `fps-929` replayed 36 cells off one fit) and **not a production code
change** (`evaluation_interval_days` is consumed only by the backtest engines; the
live daily signal in `fuel_signal/signal.py` is rule-based, never loads the model,
and already emits daily).

**Quote realised CPL and headroom with the cadence attached.** "1.66 c/L of
headroom" is not a fact about the model; "1.54 c/L at 7-day cadence" is.

**Ordering rule — stamp before you decide.** The `tank_params` column (`fps-xx1`)
now lives in `results.csv`, alongside `baseline_fingerprint`, derived inside
`log_experiment` from the `TankParams` the backtest actually ran with — same rule
as the fingerprint, the stamp cannot disagree with what produced the row. It is
empty when no backtest ran (`tank=None`). Every row predating the 2026-08-22
re-lock reads `50/3.571/7d/10%`; every row after it reads `50/3.571/1d/10%`. That
column is what makes the two eras comparable at all — without it a daily-cadence
row would be indistinguishable from the 7-day rows it must be compared against,
which is why `fps-15c` was a hard precondition for `fps-oqz` rather than a tidy-up.

**A recorded CPL without its cadence is a refusal, not a convention (`fps-15c`).**
The rule above is a stated CONTRACT, not just a habit `log_experiment` happens to
follow: every artifact that records a realised CPL must carry the `TankParams` it
was produced at, and the writer must **raise** rather than silently omit it.
`fuel_signal.backtest.require_tank_stamp(tank, what=...)` is the one shared path —
raises if `tank` is `None`, otherwise returns `format_tank_params(tank)`. Every
writer routes through it rather than reimplementing the check:

| Site | What it stamps |
|---|---|
| `experiments/results.csv` (`log_experiment`) | `tank_params` column, empty when no backtest ran |
| `RealisedResult.meta` (`run_paired_realised_backtest`) | `meta["tank_params"]`, always present — `tank` is resolved (never `None`) before the stamp |
| `runner.py`'s `results.json` | `meta["tank_params"]`, read straight off `RealisedResult.meta` |
| `dossier_tables.build_facts`'s `facts.json` | `provenance["tank_params"]`; **raises** if `status=="graded"` (a CPL is about to be written) and the source `results.json` has none |
| `batch_freeze.freeze_batch`'s `freeze.json` | `tank_params`, the batch's declared cadence contract — the same role `baseline_fingerprint` plays for column identity |
| `noise_floor.compute_noise_floor`'s `noise_floor.json` | `tank_params`, cross-checked between both halves of every seed pair the same way `n_windows` already is |

`experiments/lib/io.write_meta` (the shared `meta.json` writer ~14 hand-written
experiment scripts already call) also accepts an optional `tank=` and stamps it the
same way, so a *new* mechanism inherits the discipline by using the existing shared
helper instead of reinventing it.

Tests enforce this generically rather than one assertion per site
(`tests/test_exp_lib_io.py`): `experiments.lib.io.artifact_has_unstamped_cpl(obj)`
walks a JSON-shaped dict/list looking for any key containing `cpl` (case-
insensitive — `cpl_own`, `delta_cpl_held`, `band_mean_delta_cpl_held`, ...) with no
`tank_params`/`tank` key anywhere in the same document.
`test_every_committed_experiment_json_artifact_carries_its_cadence_stamp` runs it
over **every git-tracked `experiments/**/*.json`**, not a hand-picked list — a
future artifact-writing mechanism that builds its own dict from scratch and skips
the shared helper is caught here without needing its own dedicated test, which is
the generic "catches the next mechanism" backstop `fps-15c` exists to provide. Two
deliberate, commented exceptions are allowlisted rather than silently exempted: a
cadence-*sweep* experiment (`2026-08-20_cadence_ceiling/stage2_model/meta.json`)
that self-documents cadence per row as `cadence_days` — it IS the independent
variable there, just under a different key name than this scan looks for — and
`retrospective_facts.json`, a derived artifact built entirely from already-stamped
`facts.json`/`noise_floor.json` whose own propagation is `fps-aam`.

The consuming side of this contract lives in `dossier_tables._noise_band()`
(`fps-v8o`): it refuses (`available: false`) whenever a candidate run's
`tank_params` doesn't match `noise_floor.json`'s, mirroring the existing
`baseline_fingerprint`-mismatch refusal (`fps-cf8`) along the cadence axis.

`batch0`'s `freeze.json`, `noise_floor.json`, and `tgp_delta_7d`'s `results.json` /
`facts.json` were backfilled with `tank_params: "50/3.571/7d/10%"` — the batch was
frozen and every candidate in it ran before this field existed, entirely at the
then-canonical 7-day cadence, so the backfilled value is a historical fact, not a
guess. Since the 2026-08-22 re-lock those stamps no longer match the default, so a
*new* candidate run against batch0's floor refuses with `available: false` on the
`tank_params` axis. That is the guard working, not a regression: batch0 is a 7-day
batch, and grading a 1-day candidate against a 7-day floor is exactly the silent
comparison `fps-v8o` exists to block.

**To compare across the boundary, freeze a NEW batch at 1d** — that is what `fps-aay`
already plans. Do *not* reach for `noise_floor.py batch0 --force`: `freeze_batch`
refuses to re-freeze an existing batch (`FileExistsError`, "batches are frozen once"),
so `--force` is the only in-place mechanism, and it would recompute the floor at the
current default while `freeze.json` kept declaring 7d — leaving the batch split-brained
with no reader to notice, since nothing consumes `freeze.json`'s stamp and `_noise_band`
compares only run-vs-floor. `compute_noise_floor` now refuses that outright
(`check_freeze_cadence`, `fps-oqz`): a batch's floor and its freeze manifest must agree
on cadence. Editing the stamps is not a fix either.

**And cadence is not free to sweep.** `run_backtest`'s emergency rule forces a
half-fill whenever a wait would leave `tank_level < depletion` (bd `fps-5mn`,
fixed 2026-08-20 — it previously tested only `tank_level / size < floor_fraction`,
which left a gap: a level above the floor could still fail to survive one
interval, and the tank silently ran dry via a `max(0.0, ...)` clamp). A tank
config can still be unsafe if the emergency half-fill target itself
(`0.5 * size`) can't cover one interval's depletion — `fuel_signal.backtest
.validate_never_dry(tank)` enumerates the reachable decide-point lattice
(starting from the true 50%-full start, not just wait-chain levels — a config can
be broken at the very first decision) and flags any config where that happens;
the CLI runs it automatically and rejects an unsafe `--tank-size`/`--daily-use`/
`--eval-interval` combination. Default tank: 1-7 days are safe; 8-14 are not
(`D > 0.5 * tank_size`). Both the old canonical 7d and the current canonical 1d
are safe, so the re-lock did not cross that boundary.

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

**Why:** the #262 headroom map's per-zone rows drove real decisions — "regime axis FLAT" retired the late-descent thread, and a 12–16c volatility "hump" was treated as a target. `experiments/2026-08-20_headroom_attribution/` recomputed them under six conventions: every zone moved 2.5–5.1 c/L while the zones differed by 0.5–3.4, no contrast separated on either axis, and the impossible negatives that motivated the fix reappeared under two of the six. Every per-zone row was withdrawn; the window-level number survives *as a window-level quantity* — but it is not a constant. `experiments/2026-08-20_cadence_ceiling/` (bd `fps-fii`) showed it is conditional on the evaluation cadence: 1.54 c/L on a 7-day decision grid, 2.97 c/L on a daily one. Quote it with its cadence attached. Note the tell is not always available — that map's negatives exposed it only because it compared against an oracle; the same defect measured against always-buy is silent.

**The mechanical corollary: cut economics on folds, not on row labels.** A fold is not a
sub-period — `experiments/lib/realised.py` calls `aggregate_backtest` once per fold and
`aggregate_backtest` calls `run_backtest` once per station with a fresh tank, so each
(fold, station) is an **independent simulation** and any per-fold, per-station or
per-fold-group figure is a sum of complete windows. Nothing is allocated, so nothing is
unidentified. A row-level label (cycle regime, day-of-week, volatility band, a candidate's
`add_axis`) slices *through* a window instead, so a cost cut on one is unidentified no
matter how many fills back it. Express a zone claim as a set of folds where you can; where
the mechanism really is row-level, make the claim on a quantity stamped at a moment
(per-row log-loss, per-decision accuracy, "did the model pick a worse week than the
oracle?") rather than on pooled CPL. `experiments/pipeline/` enforces this by attaching
`ROW_AXIS_ECONOMICS_CAVEAT` to any `per_axis` delta or `CONFIDENCE_ZONE` grade that used a
row label, so the caveat travels with the number instead of living only here.

**And note the tell is often missing.** #262's per-zone rows exposed themselves through
*impossible negatives* — a perfect-foresight oracle cannot lose — but that only worked
because the comparison was against an oracle. The same defect measured against always-buy
produces no impossible value at all. The 2026-06-18 gate-1 per-regime saving% sat unchallenged
for two months for exactly that reason (withdrawn 2026-08-21,
`experiments/2026-08-21_path_coupling_audit/`, bd `fps-grp`): one free convention already in
that experiment — whether forced emergency fills count — moved each bucket 8.15–21.30 c/L
against a between-bucket spread of 6.73–10.40, and *reversed* the ordering. **Absence of an
impossible value is not evidence of identification.** Enumerate the free conventions and
measure the spread; don't wait for the quantity to embarrass itself.

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

## Git workflow

- **Fresh branch per PR.** Branch off `main` for each PR; do not continue committing to a previously merged branch even though GitHub diffs against `main` would still work.
- **Open the PR immediately** after the first commit+push — no need to ask first.
- Branch naming, PR title format, and PR body shape: see [AGENTS.md § Branch and PR conventions](../AGENTS.md#branch-and-pr-conventions).
- **Experiments lab book is exempt.** Changes confined to `experiments/**` may be committed directly to `main` without a PR. Each experiment dir is a self-contained lab book entry; iterate freely. `experiments/results.csv` (the formal graduated-experiment log) and `experiments/INDEX.md` (the lab book index) are also direct-to-`main`. Anything touching `fuel_signal/`, `tests/`, `docs/`, or top-level config still goes through a PR even if an experiment motivated it.

## PR feedback loop

Immediately after `gh pr create` returns a PR number, call `ScheduleWakeup(delaySeconds=270)` with a prompt that runs `gh pr view <N> --json comments,reviews,mergeable,statusCheckRollup`. This is a mandatory mechanical step, not a suggestion — do it before writing any response to the user. When the wakeup fires: act on any actionable comments present. If CodeRabbit is rate-limited or absent, **skip it and move on — do not reschedule to wait for it**. Use judgement on style nits that conflict with project conventions. Run `uv run ruff check . && uv run pytest -q`, push, and repeat until no actionable comments remain. The goal is a ready-to-merge deliverable.

**A re-review after a fix commit can re-post identical comments against stale line numbers.** Observed on PR #311 (fps-3jj.9): CodeRabbit's second review, triggered by the fix-commit push, posted the same 4 comments verbatim — including inline diff suggestions quoting the pre-fix code — even though the fix commit had already addressed every one of them. Don't assume a repeated comment is a new/unaddressed finding; `grep`/`sed` the current file at the cited path and check whether the flagged code still looks like what the comment describes before touching anything again.

**The opposite timing failure also happens, and its findings can still be valid.** Observed on PR #313 (fps-3jj.8): a manually-pasted external review was addressed and pushed as a fix commit, then CodeRabbit's own review posted ~20 minutes later — but its own metadata showed it had been comparing against the *original* pre-fix commit, not current HEAD. Its 6 findings didn't overlap with the fix already pushed, and 2 of them were real (confirmed against current code) and worth fixing anyway. The commit range a review cites is not a reliable signal of whether its findings are live or stale in either direction — verify every finding against current code regardless of what the review says it diffed against, the same discipline as the stale-repost case above, not a reason to auto-dismiss a review that looks behind.

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
