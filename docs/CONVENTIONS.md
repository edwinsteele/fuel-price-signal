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

**Pre-registered instance:** the fold whose val window is `2022-02-03 to 2022-05-03` spans the unrecoverable NSW source-data hole of 2022-03-12 to 2022-03-21 (`fps-tpy`; see [AGENTS.md § Known source data limitations](../AGENTS.md#known-source-data-limitations)). Decided 2026-08-27: flag it if it regresses, don't exclude it from the harness — the effect is small, partial, and confined to one fold, and this window already overlaps the Ukraine-invasion shock most regime-segmented runs treat as elevated-variance.

**Why:** on 2026-06-03, `station_minus_last_max_cents` looked like a clean drop on one val window (5-seed Δ −0.0112 ± 0.0043), but a 14-fold paired walk-forward CV showed 7/14 fold-wins, mean Δ +0.0104, with fold 9 (2023-10→2024-01) regressing by +0.103. See `experiments/2026-06-03_drop_redundant_pair/`.

### The baseline feature set is declared, never discovered

Code that needs "the production baseline columns" imports **`fuel_signal.features.LOCKED_FEATURE_COLUMNS`** — one symbol, re-exported for experiment scripts as `experiments.lib.constants.BASELINE_COLUMNS`. Never retype the group composition `FEATURE_COLUMNS + LGA_FEATURE_COLUMNS + NETWORK_FEATURE_COLUMNS`, and never derive the set by inspecting a features frame's header.

`data/features.csv` is deliberately a **superset** of the model contract. A column sits in it for one of three reasons, and only the first puts it in the baseline:

| Reason | Example |
|---|---|
| In the lock | the 54 (see [AGENTS.md § Canonical feature set](../AGENTS.md#canonical-feature-set-54-feat-baseline-locked-issue-216)) |
| Evaluated and **rejected**, still computed | the 10 `days_since_trough_entry_<brand>` (Phase 4b, 2026-06-02) |
| Evaluated, **inconclusive** (below the arbiter's resolution), still computed | `tgp_delta_7d` — the #271 re-lock was retracted 2026-08-19 and will not recur; nothing is pending |

Header inspection cannot tell these apart, so it silently promotes rejected and inconclusive columns into the baseline. `fuel_signal.features.non_model_columns(df)` names the second and third categories with a machine-readable reason code (`evaluated-and-rejected` / `evaluated-inconclusive`), so "outside the lock" is an assertable condition rather than something a reviewer has to already know.

**The rule binds analysis tooling, not just the training path (`fps-3jj.11` pre-start check, 2026-08-23).** Anything that asks "what does the model already have" resolves the set the same way. `experiments/pipeline/redundancy.py` retyped it as `FEATURE_COLUMNS + LGA_FEATURE_COLUMNS + NETWORK_FEATURE_COLUMNS + TGP_FEATURE_COLUMNS` (55) for its block-R² predictor set, and `docs/routines/generator.md` named the same four symbols, so the code and the spec agreed with each other and disagreed with the lock. The screen therefore regressed candidates on `tgp_delta_7d`, a column R0 never sees — inflating measured redundancy for anything correlated with the wholesale signal, i.e. against the lead the ledger rates highest among still-open ground (`fps-x0f`). It scored `batch0`'s own `tgp_delta_7d` at block R² = **1.000** and flagged it `REDESIGN`: a predictor of itself. Against the real lock it is 0.288. A *reported* number rather than a gate, so nothing was mis-graded — but it is the same defect class as `fps-sa1`, caught by a live run rather than by the tests, one of which asserted the wrong invariant outright.

Note that `train_lgbm`'s *default* resolution is not the lock — the lock is `train_lgbm --no-brand-features`. "Mirrors train_lgbm's default" is not a justification for a baseline resolution.

**Order is part of the contract, and new columns append.** Keep the production order — the group concatenation above, which is exactly what `data/models/lgbm_calibrated.joblib`'s `feature_columns` holds — and never sort it. LightGBM breaks equal-gain split ties by feature index, and with 35 near-identical LGA trough columns exact ties are common, so the same columns in a different order fit a *different* model. Sorting looks tidier and is the wrong trade: it makes the contract file diff-stable while making the model unstable to feature additions, since one alphabetically-early insertion reshuffles every index after it. Appending leaves existing indices untouched.

How much does order matter? **Pin it; don't sweep it.** Measured over 5 draws each on one train/val slice, order-variance is ~2% of seed-variance (log-loss std 0.000258 vs 0.015643; decision flips at τ=0.25 of 8–29 rows vs 1743–2069). It does not need the multi-seed discipline that seeds get. But it is *not* nothing at the arbiter: the sorted-vs-production permutation moved batch0's pooled realised delta by 0.038 c/L, ~0.8 of a decision flip — which says more about the arbiter differencing two nearly-identical quantities than about order.

**Graduating a held-out column is two edits, not one.** Adding it to `LOCKED_FEATURE_COLUMNS` is not enough — you must delete its `NON_MODEL_COLUMNS` entry in the *same* change. Until you do, `resolve_baseline_columns()` raises `NonModelColumnLeak`, because `non_model_columns()` deliberately does **not** consult the lock: a detector that skipped whatever the lock already claims could not detect a column wrongly *in* the lock, which is the entire fps-sa1 failure. This is a hard stop rather than a warning, on purpose — graduation is a declaration, not something the code should infer from a list it cannot audit. The standing example is `tgp_delta_7d` at the #271 chip-4 re-lock (bd `fps-1785999729707-1-0301bf82`, component 5) — **now hypothetical, not live**: #271 closed as superseded on 2026-08-20 and that re-lock will not happen, so `tgp_delta_7d` stays in `NON_MODEL_COLUMNS` (as `evaluated-inconclusive`) indefinitely. The two-edit rule is unchanged and applies to whatever graduates next.

**Every result records which baseline it was measured against.** `baseline_fingerprint(columns)` returns `'<n>:<sha12>'` over the **ordered** list; it is stamped into experiment `meta.json` (automatically, by `experiments.lib.io.write_meta`), batch `freeze.json`, and each run's `results.json` → `facts.json`. Two runs whose fingerprints differ are not comparable, whatever their deltas say — check the fingerprints before comparing numbers across runs. A `null` fingerprint means the run predates the field, which is exactly the population where a wrong R0 could be hiding; there is no safe fallback, so nothing substitutes today's constants for it.

**Why:** `batch_freeze.resolve_baseline_columns()` appended `discover_brand_feature_columns(df)` on exactly that rationale, giving batch0 a 64-column R0 — production plus the rejected Phase 4b group — for every candidate in the batch, on both the log-loss screen and the realised arbiter. It surfaced only because `tgp_delta_7d` was re-tested as a known graduate and came back with the opposite sign; the candidate arms agreed to 0.0013 c/L while the baselines differed by 0.132 (fps-sa1, fps-nor). Neither defect left any trace in the artifacts that recorded the runs, which is why two incommensurable baselines were compared head-to-head for two months — hence the single symbol and the fingerprint above (fps-zci).

**A re-lock invalidates the batch's noise floor — but which recovery path applies depends on WHAT re-locked.** `experiments/pipeline/noise_floor.py` fits the frozen baseline at a single fixed seed and takes the realised-CPL delta against ~20 placebo-column draws (`fps-awz`, reworked from an earlier paired-seed-group design — see `docs/feature-pipeline.md`) — that fit is against a specific `baseline_fingerprint` AND a specific `tank_params` cadence, same as every candidate run. Every successful graduation changes `LOCKED_FEATURE_COLUMNS` (a 55th column changes the model whose noise was measured), so a floor computed before a graduation does not grade runs made after it: `dossier_tables._noise_band()` compares the floor's own `baseline_fingerprint` (and, separately, its `tank_params`, `fps-v8o`) against the candidate run's and refuses (`available: False`) on any mismatch, the same style as its existing `partial`-fold-subset refusal. `batch_freeze.freeze_batch()` computes the floor as its own final step, so a fresh batch always starts with a matching one.

For a **column-lock** graduation, recompute an existing batch's floor in place with `PYTHONPATH=. uv run python -m experiments.pipeline.noise_floor <batch> --force` (or after any `--skip-noise-floor` freeze, fps-cf8) — the batch's `freeze.json` still declares the right cadence, only the fingerprint moved. For a **cadence** re-lock this does NOT work: `check_freeze_cadence` (`fps-oqz`) refuses `--force` outright when the batch's `freeze.json` and the current `TankParams()` default disagree on cadence, rather than silently leaving the batch split-brained (floor at the new cadence, manifest still declaring the old one). Freeze a **new** batch at the new cadence instead — this is what happened to `batch0` at the 2026-08-22 7d→1d re-lock (`fps-oqz`); its floor was never recomputed, `batch1` was frozen fresh at 1d instead (`fps-aay`).

**The band's ARITY must be at least the candidate's (`fps-3jj.14`).** A candidate is one MECHANISM and may be **any number of columns** — this entry historically said 2–4, and `fps-3jj.21` briefly narrowed it to 1–3 behind a `MAX_RULER_ARITY` cap, but `fps-3jj.25` removed that cap outright and `docs/routines/generator.md` (the copy the generator actually reads) now constrains candidate shape in no way. See the next entry for how reuse is priced instead — every feature win in this project's history arrived as a group. The placebo null was fixed at ONE column until batch1 made that load-bearing: three of its five candidates are 3-column groups. A k-column arm has more chances for the fit to find something than a j-column placebo arm does when k > j, so a wide arm graded against a narrow ruler is biased **in the candidate's favour** by an amount that is not knowable from the run. `compute_noise_floor` therefore takes `--arity` and stamps `n_placebo_columns`; `dossier_tables._noise_band()` refuses (`available: False`) when the floor's arity is BELOW the run's candidate column count, same failure class as the `baseline_fingerprint` / `null_method` / `tank_params` refusals above. The guard is **one-sided on purpose**: a floor at or above the run's arity can only be as wide or wider — a harder bar — so it is allowed and disclosed as `floor_arity_exceeds_run`, rather than forcing one ~2h calibration per distinct arity in a batch. A floor with no `n_placebo_columns` key predates this and *was* arity 1, so its absence pins the value at 1 rather than leaving it unknown; such a floor keeps grading 1-column candidates and only stops grading multi-column ones. ~~Practical ceiling: the draw pools bind on `n_draws × arity`, so the maximum draw count at arity k is `floor(30 / k)`.~~ **Superseded by `fps-3jj.21`, 2026-08-26 — see the next entry.**

**The draw count no longer trades off against arity, and arity is capped at 4 (`fps-3jj.21`).** The old ceiling was `min(n_columns, n_seeds) / arity` — every draw needed `arity` DISTINCT source columns out of 54, and `arity` distinct seeds out of a fixed pool of 30. That is the wrong constraint in both halves, and it did real damage: because `family_wise_z_threshold` carries `df = n_draws - 1`, a thinner band means a sharply harder bar, so a wider candidate was penalised for its width twice over. `experiments/2026-08-23_placebo_arity/` measured **two thirds** of batch1's k=1 → k=3 bar move as draw count, not the arity effect it was being read as. Extrapolated, the ruler self-destructed: arity 10 → 3 draws, arity 18 → no band possible at all, arity 35 → `select_draws` raised outright. A per-LGA candidate — the project's own motivating precedent for group candidates, per `docs/routines/generator.md` — was ungradeable.

Both pools are unbounded now. `placebo.block_seed(i)` supplies a seed at any flat position (the historical 30 primes are its first 30 values, so any bank of ≤30 picks stays byte-identical), and `placebo.candidate_pool` lays repeated laps over the column list instead of stopping at one. **Pick `n_draws` for the certainty you want and pay the compute.**

**Arity is not capped; reuse is PRICED (`fps-3jj.25`).** There was briefly a `MAX_RULER_ARITY` constant refusing any candidate wider than it. It is gone, on the owner's standing position — restated twice, on `fps-3jj.14` and again on `fps-3jj.25` — that **constraining candidate shape to suit the ruler distorts the thing being measured**. A cap made the instrument dictate what the generator was allowed to propose, and cost a wide candidate its verdict entirely (a night of compute, then an unbounded nightly re-pick).

The replacement is a correction, not a refusal. Two draws sharing a source column share that column's TEXTURE exactly — same values, same NaN pattern, only re-dated — so their deltas are correlated and the bank carries less information than its draw count suggests. `placebo.effective_n_draws` prices that from the bank that actually got built, and `family_wise_z_threshold` takes the effective count in place of the nominal one (`scipy.stats.t.ppf` accepts a fractional `df`; do not round). A wider candidate reuses more, so its band is worth fewer draws, so its bar is automatically wider.

**A pair sharing `s` of `arity` columns is charged `icc × s / arity`, not the full `icc`, and the difference is the whole design.** Charging any overlap the full ICC put a 20-draw arity-10 bank at 3.2 effective draws — which is what made a cap look necessary in the first place. Scaled by share it is 10.81, and the bar degrades gently enough that no cap is needed. Single-candidate bar on batch1's band, by candidate width, at the **measured** `TEXTURE_ICC_BOUND = 0.274` (see the next entry; the bracketed figures are the same table at the 0.391 that shipped until 2026-08-27):

| candidate columns | effective draws | bar (1 candidate) | at the old 0.391 | pre-`fps-3jj.21` |
|---|---|---|---|---|
| 1–2 | 20.00 | −0.152 | −0.152 | −0.225 |
| 3 | 18.17 | −0.153 | −0.154 | −0.271 |
| 10 | 10.81 | −0.165 | −0.171 | −0.778 |
| 35 | 4.31 | −0.226 | −0.287 | **no band possible** |

Harder as it widens, which is correct, and usable at every width. The only arity limit left is physical: a draw needs `arity` distinct source columns, so it cannot exceed the baseline column count.

**The ICC is MEASURED by column, and it is a 95% upper bound — not a point estimate (`fps-3jj.23`, 2026-08-27).** `placebo.TEXTURE_ICC_BOUND = 0.274` comes from a one-way ANOVA of 32 pinned-source draws on batch1 — 8 source columns × 4 block seeds, `noise_floor --same-source-column` — grouped **by source column**: `F(7,24) = 0.735`, `p = 0.65`, point estimate −0.071 (i.e. 0), one-sided 95% upper bound **0.274**, and the design resolves down to **0.262**. A point estimate of ~0 therefore reads "could not see it", never "it is not there": an ICC of 0.05 and one of 0.25 are both consistent with what was measured, which is why the constant carries the pessimistic end.

**What it replaced.** 0.391 was a bound on the ICC by texture **FAMILY**, used as a stand-in for the ICC by **COLUMN**. Family is coarser — two different `days_since_trough_entry_<lga>` columns are same-family but different-column, while two placebos from the *same* column are more alike than that — so 0.391 was never established as an upper bound on the quantity actually used, and the direction of its error was unknown. The by-column measurement closes that gap rather than restating it. Separately, `fps-8o0` was checked against the old artifact in the same pass: batch1's `network` family is a **singleton**, and `texture_channel.py` dropped it from the sums of squares while keeping `n=20`, `k=5` in the degrees of freedom and in `k_bar`. Recomputed correctly the by-family bound is **0.226**, not 0.391 — so the old constant was overstated on its own terms as well. Two estimators of two different quantities, both landing under 0.391.

**The residual, stated rather than buried.** The 8 pinned columns were chosen by `select_draws`' own even spread over the 49 usable columns — sampled the way the bank samples, deliberately not curated for texture diversity, which would have overstated the between-column variance and hence the ICC. But their within-column (alignment) variance ran **2.2×** the committed 20-draw multi-column floor's *total* variance, which an equal-alignment-variance model forbids. It is not resolvable (95% CI on the ratio [0.90, 5.18], `p = 0.08`; that design cannot call a ratio significant below 0.574), so there is no established evidence the pinned set is unrepresentative — but if the lean is real the ANOVA's ICC is biased **low** and 0.274 is optimistic. This is also the second, independent reason the bead's originally-specified variance-ratio construction could not have discharged it: its stated assumption is the one the data leans against.

**Lowering this constant LOOSENS wide bars — do not move it again without a measurement.** Less correlation → more effective draws → narrower band → easier bar. Arity 35 went from −0.287 to −0.226 c/L on this change; every candidate batch1 has actually run (arity 1 and 3) moved by 0.001. To tighten it further, buy draws: the bound is limited by the design's resolution, and 12 columns × 5 seeds (60 draws, ~11.4h) lands near 0.23 even if the point estimate again comes out at 0. `experiments/2026-08-27_texture_icc/power.py` prices any shape before you pay for it. Full working: `experiments/2026-08-27_texture_icc/`; the superseded by-family analysis is `experiments/2026-08-26_placebo_draw_independence/`.

**"It sets every bar" is true but misleading — it sets the WIDE bars.** Sweeping the constant across its entire possible range on batch1's live ruler (`experiments/2026-08-27_texture_icc/power.py`, shipped `effective_n_draws`, 20-draw bank):

| arity | ICC 0 | 0.274 (live) | 1.0 | most the bar can move |
|---|---|---|---|---|
| 1–2 | −0.152 | −0.152 | −0.152 | **0.000** |
| 3 | −0.152 | −0.153 | −0.157 | 0.005 |
| 4 | −0.152 | −0.155 | −0.163 | 0.011 |
| 10 | −0.152 | −0.165 | −0.211 | 0.059 |
| 20 | −0.152 | −0.185 | −0.431 | 0.280 |
| 35 | −0.152 | −0.226 | no band | 0.383 |

At arity 1–2 the value is irrelevant by construction (no two draws share a column at all), and at arity 3–4 — every candidate batch1 has ever run — its whole range moves the bar by ≤ 0.011 c/L, well under the realised arbiter's own decision quantum. It is genuinely load-bearing only for the wide groups `docs/routines/generator.md` invites. Quote the qualifier; a bare "it sets every bar" overstates it.

**Price a design before paying for it — the obvious cheap version of this measurement was worthless.** One pinned column × 10 seeds compared against the committed 20-draw floor — the natural reading of "a same-column noise floor", and what `fps-3jj.23` originally specified — is an unpaired variance ratio on `F(9,19)`: it cannot call anything under **0.661** significant, and its best possible outcome is a 95% upper bound of **0.587**, *looser than the 0.391 it would have replaced* (all figures at the ONE-SIDED 5% tail, the tail 0.391 itself was derived at). Two hours of fits for a worse number, whatever the deltas came out as. The design that worked is a one-way ANOVA of delta grouped **by column** over several pinned columns (`noise_floor --same-source-column`, repeatable), which estimates the by-column quantity directly rather than through a second artifact; **8 columns × 4 seeds = 32 draws (~5.7h measured) was the smallest shape of any kind that could leave a bound tighter than 0.391**, and it landed at 0.274. The arithmetic that ruled the cheap design out cost 6 seconds: `experiments/2026-08-27_texture_icc/power.py`.

**A shared block SEED is what actually destroys a bank's independence — not a shared column (`fps-3jj.20`, fixed by `fps-3jj.21`).** A block seed selects a *rearrangement* of the date blocks. Two placebos built with the same seed get the identical rearrangement, which destroys each column's alignment to the target while leaving the two columns' alignment **to each other** intact — so a pair of near-duplicate source columns comes out of the shuffle as correlated as it went in. This is live in batch1's committed grading ruler: `candidate_pool`'s fallback tail used to restart the seed cycle at index 0, so draw 10 (built entirely of substitutes, after three primaries landed on all-NaN LGA columns) reuses draw 1's seeds 97/101/103 and sits at placebo correlations 0.965 / 0.778 / 0.765 against it. One of ten draws is a near-copy of another, deterministically, on every recompute. Measured on batch1, placebo-pair |ρ| by what the pair shares:

| pair shares | median | max |
|---|---|---|
| same source column, **different** seed | 0.038 | 0.218 |
| different source column, different seed | 0.015 | 0.175 |
| different source column, **same** seed | 0.097 | **0.971** |

Reusing a column is cheap; reusing a seed is not. `block_seed`'s never-restarting counter closes the third row by construction rather than making it less likely.

**Consequence: a floor recomputed after 2026-08-26 is not comparable to one computed before it, wherever substitution fired.** The seeds assigned to substituted draws changed. Committed floors on disk are untouched (nothing recomputes them), and the common case where nothing is substituted is unaffected — the first 30 seeds are the historical pool verbatim. But treat a recompute like a re-lock: recompute, don't mix.

**`effective_n_draws` does NOT price the seed channel, and on batch1 it reads clean when it isn't (`fps-3jj.24`).** The two entries above sit badly together and the gap is worth stating outright: `placebo.effective_n_draws` charges a pair for shared source COLUMNS, and batch1's collided pair shares none — draw 1 is `cycle_pct_through / cycle_mean_length / cycle_last_max_cents`, draw 10 is `cycle_days_since_peak / cycle_last_min_cents / station_price_cents`, and the bank is 30 picks over 30 distinct columns. So `dossier_tables._resolve_effective_n_draws` returns the nominal **10.0 exactly**, and `facts.json` emits `effective_n_draws: 10.0` for a bank that is worth about 8.6. Read against the table above, the correction charges the cheap channel (same column, different seed: median |ρ| 0.038) at `icc = TEXTURE_ICC_BOUND` — 0.391 when this was written, 0.274 since `fps-3jj.23` measured it — and the expensive one (different column, same seed: max |ρ| 0.971) at zero. **This is not a bug to fix.** `block_seed`'s never-restarting counter closes the seed channel by construction, so no floor computed after 2026-08-26 can have one; the only affected artifacts are batch1's `noise_floor.json` and its retired `noise_floor_k1.json` (20 draws, two collided pairs on seeds 97 and 101, `n_eff` likewise reading a nominal 20.0). Changing `effective_n_draws` to chase a closed channel would add a code path nothing can reach and move the ruler under five already-written dossiers. Disclose the two artifacts instead; `docs/routines/retrospective.md` carries the standing check.

**Decision (`fps-3jj.24`, 2026-08-27): batch1's retrospective ranks against the COMMITTED ruler, and the recompute was refused as a worse instrument — not as a cost saving.** Priced by hand at the collided pair's observed 0.836 correlation, the bank is worth 8.57 effective draws and the bar is too EASY by 0.0057 c/L at one candidate and **0.0144 c/L at the 5-candidate batch gate**. Ranking on the committed numbers keeps the batch verdict in the same units as the per-candidate write-ups. The recompute would swap that known, signed, closed-form bias for a larger unknown one: a fresh 10-draw band estimates its std to ±23.6% (`1/sqrt(2(n-1))`), i.e. ±0.0697 c/L on the batch-gate bar — **4.8× the bias it removes** — so its "difference" would be dominated by resampling luck, on top of not being comparable draw-for-draw per the entry above. Because the correction is monotone-HARDENING it can only demote, never promote, so the committed ruler wrongly fails nobody; it can only wrongly PASS a candidate inside |`z`| ∈ [1.9226, 1.9797) at 1 candidate or [2.9591, 3.1030) at 5. That window is checked per row at write-up time at zero cost. The k1 floor's shared defect leaves the "two thirds is thinness" attribution below intact (the k1→k3 gap moves −0.0536 → −0.0637 c/L when both are corrected).

**Promoting a wider floor is a RENAME, and there is no selector.** `_noise_band` reads one hardcoded filename, `noise_floor.json`; nothing reads an arity-suffixed side-file. So "compute a wider floor" does not by itself unblock grading — the promotion is the other half, and both halves belong in any instruction that sends someone to build one:

```
# 1. compute it beside the current ruler, at the SAME n_draws as the floor it replaces —
#    there is no longer an arity-dependent ceiling (fps-3jj.21), and matching the draw
#    count is what keeps the new ruler comparable to the old one.
PYTHONPATH=. uv run python -m experiments.pipeline.noise_floor <batch> \
    --arity 3 --n-draws 20 --out-name noise_floor_k3.json
# 2. promote it, keeping the old ruler under its own arity's name
mv <batch>/noise_floor.json    <batch>/noise_floor_k1.json
mv <batch>/noise_floor_k3.json <batch>/noise_floor.json
```

**Do not `--force` a recompute over `noise_floor.json` instead.** It is mechanically safe for *grading* (the one-sided guard means a wider ruler only ever raises the bar), which is exactly what makes it tempting — but it destroys two things: the ruler this batch's already-written dossiers were graded against, and the k=1 baseline any arity comparison needs. `noise_floor.json` is tracked in git, so the loss is recoverable, but a recovery is not a procedure. Rename; keep both.

**A fresh freeze always produces an arity-1 floor.** `batch_freeze.freeze_batch` ends by calling `compute_noise_floor(batch_dir, tank=tank)` with the default arity, and it must — at freeze time the batch's candidates do not exist yet, so its arity is unknowable. That is not an oversight to route around: the bar and the arity are mutually dependent (the generator reads the batch's floor to calibrate its `CONFIDENCE` language, and the floor's arity depends on what the generator proposes). The freeze's k=1 floor breaks that circle by giving the generator an approximate bar to write against; the post-generation run at the batch's real modal arity produces the *grading* ruler. Budget one extra floor run per batch (~2h at k=3) as part of the recipe, not as a surprise — see bd `fps-3jj.19`.

**The gate is distance from the band, not empirical rank (`fps-awz`).** `dossier_tables.py`'s `family_wise_z_threshold` (moved there from `retrospective.py` by `fps-3jj.17` so `_noise_band()` can attach a per-run threshold to `facts.json`; `retrospective.py` imports it) is a Bonferroni-corrected t-critical value in band-standard-deviation space; `clears_family_wise_threshold` compares each candidate's `candidate_z_vs_band` against it. The empirical percentile (`candidate_percentile_better_than_noise`, `family_wise_percentile_threshold`) is still computed and reported, but as descriptive colour only — with a draw count in the tens, the empirical-rank statistic has too few distinct values to resolve anything finer than "beat every draw."

**The honest single-candidate bar, as a number (`fps-awz`, resolved via `fps-aay`'s batch1 freeze 2026-08-22; re-cut at arity 3 by `fps-3jj.14` 2026-08-24).** `batch1`'s noise floor (`experiments/batches/batch1/noise_floor.json`, **10 draws at arity 3**, at the batch's 1d cadence): `band_mean_delta_cpl_held = +0.0251 c/L`, `band_std_delta_cpl_held = 0.0999 c/L` (n=10, sample std). At batch1's own size (**5 candidates**, `docs/routines/generator.md` § Batch sizing), `family_wise_z_threshold` requires a candidate's `effect_delta_cpl_held ≤ -0.27 c/L` to clear as surprising at the BATCH level — a single candidate judged alone needs only `≤ -0.17 c/L`. **Both carry the `fps-3jj.20` seed collision and are therefore marginally too easy** — `-0.285` / `-0.173` once it is priced (`fps-3jj.24`, which decided to keep quoting the committed pair; see the entry above for why the recompute was refused).

**Those replaced a `-0.15` / `-0.22` pair, and the reason the bar moved is worth carrying.** The superseded numbers came from the arity-1 floor now retained as `noise_floor_k1.json`. Decomposing the batch-level move (`-0.2170` → `-0.2706`): the band mean rose `+0.0340` (three junk columns dilute the fit more than one — an *easier* bar), the band std widened `-0.0519` (three junk columns give the fit more places to find spurious structure — the arity effect proper), and `-0.0357` is **not arity at all** but the t-penalty for estimating the band from 10 draws instead of 20 (`df = n_draws - 1`, so z goes 2.602 → 2.959 at 5 candidates). Two-thirds of the net hardening is therefore draw count. The widening itself is **unresolved, not demonstrated** — std ratio 1.25x with a 95% interval of [0.74, 2.40] against a design that could only have called 1.70x. The promotion rests on the guard (a k=1 floor cannot grade a 3-column candidate at all), not on the statistic. Full working: `experiments/2026-08-23_placebo_arity/README.md`.

**This is a DETECTION threshold, not a target.** `experiments/results.csv`'s history shows single-column features landing between 0.03 and 0.26 c/L, so a bar above that range describes the instrument's resolution, not what a good feature looks like — do not read this as "propose nothing below -0.27 c/L." The number itself is **batch-specific** (a new batch's own floor depends on its own draws, columns, cadence and **arity**) — don't reuse -0.27 c/L as a constant the way the withdrawn `0.05 c/L` TGP figure was wrongly reused; recompute per batch and cite that batch's own floor.

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

**A CLI default must be sourced from `TankParams()`, never a hand-copied literal
(`fps-q3p`).** `backtest.py`'s `--daily-use` option defaulted to
`round(50.0 / 14, 3)` while `TankParams.daily_consumption_litres` defaults to the
unrounded `50.0 / 14` — different values that `format_tank_params`'s `f"{daily:.3f}"`
rendered identically, so a CLI-default run and a bare-`TankParams()` run stamped the
same while consuming marginally differently. The stamp contract above only compares
rendered strings; it cannot catch two sources of the same lock parameter drifting
under the hood. Fixed by sourcing the default from `TankParams().daily_consumption_litres`
directly (`--eval-interval` already did this after `fps-oqz`), pinned by
`test_cli_daily_use_default_tracks_tankparams`. Before adding a new CLI option that
shadows a `TankParams` field, source its default from the dataclass, not a literal.

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
- **Experiments lab book is exempt.** Changes confined to `experiments/**` may be committed **and pushed** directly to `main` without a PR. Each experiment dir is a self-contained lab book entry; iterate freely. `experiments/results.csv` (the formal graduated-experiment log) and `experiments/INDEX.md` (the lab book index) are also direct-to-`main`. Anything touching `fuel_signal/`, `tests/`, `docs/`, or top-level config still goes through a PR even if an experiment motivated it.

- **"Direct to `main`" means committed AND pushed — a commit sitting on a local `main` is not landed.** The PR path can't get this wrong (you cannot open a PR without pushing), so every place the word "commit" appears alone is on an exempt path, and the exempt paths are the ones unattended routines use. That asymmetry bit us for real: on 2026-08-26 eight commits — two candidate dossiers, a post-hoc script, ledger/INDEX rows, and two interaction-log entries — were found sitting unpushed on the primary worktree's local `main`, the oldest two days old. One of them was itself the fix for the *commit* half of this same gap, written and then never pushed, so the fix reproduced the bug it fixed. **Postcondition for any routine or session that writes to `main`: `git status --short` is empty AND `git log --oneline origin/main..main` is empty before you report the work done.** Both halves — the first catches a `bd`-written `.beads/interactions.jsonl` row left dirty, the second catches the unpushed commit. Check it, don't remember it.
- **PNGs are gitignored under `experiments/**` — with one tracked exception.** `.gitignore`'s type-based rule (any `experiments/**/*.png`) exists so binary plots don't bloat every iteration of a lab-book dir, but it silently kept the AI-sourced pipeline's dossier plots off GitHub even after a candidate's `README.md` was committed and embedded them by filename (`fps-3jj.6`, fixed 2026-08-24). `experiments/candidates/**/*.png` is explicitly un-ignored (a `!` negation after the blanket rule) since those plots are only ever written once by `dossier_tables.py` for a run that's about to get a permanent, committed write-up — small, bounded in count (six per run), and the exact thing the README's `![](...)` embeds reference. Follow the same pattern (a scoped negation, not lifting the blanket rule) if another pipeline needs its plots tracked.

## PR feedback loop

Immediately after `gh pr create` returns a PR number, call `ScheduleWakeup(delaySeconds=270)` with a prompt that runs `gh pr view <N> --json comments,reviews,mergeable,statusCheckRollup`. This is a mandatory mechanical step, not a suggestion — do it before writing any response to the user. When the wakeup fires: act on any actionable comments present. If CodeRabbit is rate-limited or absent, **skip it and move on — do not reschedule to wait for it**. Use judgement on style nits that conflict with project conventions. Run `uv run ruff check . && uv run pytest -q`, push, and repeat until no actionable comments remain. The goal is a ready-to-merge deliverable.

**A re-review after a fix commit can re-post identical comments against stale line numbers.** Observed on PR #311 (fps-3jj.9): CodeRabbit's second review, triggered by the fix-commit push, posted the same 4 comments verbatim — including inline diff suggestions quoting the pre-fix code — even though the fix commit had already addressed every one of them. Don't assume a repeated comment is a new/unaddressed finding; `grep`/`sed` the current file at the cited path and check whether the flagged code still looks like what the comment describes before touching anything again.

**The opposite timing failure also happens, and its findings can still be valid.** Observed on PR #313 (fps-3jj.8): a manually-pasted external review was addressed and pushed as a fix commit, then CodeRabbit's own review posted ~20 minutes later — but its own metadata showed it had been comparing against the *original* pre-fix commit, not current HEAD. Its 6 findings didn't overlap with the fix already pushed, and 2 of them were real (confirmed against current code) and worth fixing anyway. The commit range a review cites is not a reliable signal of whether its findings are live or stale in either direction — verify every finding against current code regardless of what the review says it diffed against, the same discipline as the stale-repost case above, not a reason to auto-dismiss a review that looks behind.

**CodeRabbit's "Docstring Coverage" pre-merge check is a bot metric, not a finding — treat it like a style nit, not an actionable comment.** Observed on PR #344 (bd-fps-b5p): CodeRabbit's own review said "No actionable comments were generated," but its pre-merge-checks table separately flagged "Docstring Coverage 66.67% (threshold 80%)" on the touched functions. This repo's convention is WHY-comments over formal docstrings (see top of this doc) — writing docstrings purely to clear the bot's threshold would fight that convention for no reader benefit. Left unaddressed; the PR merged clean anyway (the check is advisory, not a merge gate).

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
