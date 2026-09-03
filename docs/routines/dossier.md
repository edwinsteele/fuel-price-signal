# Dossier routine — prompt source

Canonical, tracked home for the prompt used by the **dossier session** (`fps-3jj.6`) in the
AI-sourced feature engineering pipeline (`fps-3jj`). Like `docs/routines/worker.md` and unlike
`docs/routines/generator.md`, this **is** a scheduled task: nightly, same night as the launch
routine (`fps-3jj.5`), Claude active only briefly. The scheduler's stored prompt is a three-line
shim — see "The shim" below.

## What this session does, in order

### Step 0 — deterministic pass (no judgement, no arithmetic)

Run the whole queue through the code-only step first:

```bash
PYTHONPATH=. uv run python -m experiments.pipeline.dossier_tables --scan experiments/candidates
```

This does one thing, mechanically (`experiments/pipeline/dossier_tables.py`, fps-3jj.6): **facts.json
+ PNGs for every completed, undossiered run.** For every directory whose dossier isn't current —
no `README.md` yet, or a `facts.json` that wasn't built from the `results.json` sitting there now
(see `find_pending_runs`; fps-jrl replaced fps-0yd's mtime comparison, which called every run in a
fresh checkout stale) — it writes `facts.json`
(the only source of numbers you are allowed to use below)
and the applicable plots (always: `per_fold_delta_bars.png`, `seed_mean_vs_median.png`,
`realised_cpl_by_fold.png`, `tau_sweep.png`, `candidate_over_time.png`; conditional:
`axis_breakdown.png`, `cycle_phase_breakdown.png`, `external_series_overlay.png`). One bad/partial
run doesn't stop the rest of the scan — it's logged and skipped; check the command's own output
for any `failed, skipping` lines and flag them rather than silently ignoring.

**Retryable aborts are skipped entirely, before anything is written (fps-g31).** `find_pending_runs`
excludes any run whose `results.json` carries a `RETRYABLE_STATUSES` status (`aborted_pipeline` /
`aborted_environment`). Those runs never got a fair hearing, their bd claim goes back on the queue,
and the re-run **reuses the same directory** — so a `facts.json`/`README.md` written for a
retryable status is exactly the wrong content to leave behind (the run never reached the scoring
stages; there is no real verdict to write up), regardless of what the successful re-run's
`results.json` mtime later does. No `facts.json`, no `README.md`, no ledger row, no INDEX row:
wait for the re-run.
`aborted_candidate` and `disqualified` are NOT retryable — those are real verdicts and get written
up normally.

**Stale-claim recovery is NOT this routine's job.** It runs entirely inside the launch routine
(`fps-3jj.5`, merged) as part of every nightly `launch` invocation — `recover_stale_claims()`
checks `bd list --status in_progress --label experiment` for issues whose `run.log` ends in a
traceback and releases them. An earlier version of `dossier_tables.py` had its own,
weaker, parallel stale-claim mechanism keyed on a `claim.json` file nothing ever wrote — removed
once launch.py's real implementation was confirmed to already cover this correctly. Don't
reintroduce it here.

**FIXED — run-directory convention (fps-icv).** This routine's queue model (`find_pending_runs`: a
directory with no `README.md`, or a `facts.json` not built from the `results.json` on disk — see
fps-jrl) assumes one subdirectory per candidate. That now holds: launch.py's `launch_detached` and runner.py's `run_candidate` both default a candidate's
`out_dir` to `runner.default_out_dir(candidate_path)` (candidate_path with its `.py` suffix
stripped), not `candidate_path.parent` (the whole **batch** directory shared by every candidate
module filed against it). Before this fix, candidate 2+ in any batch would overwrite candidate 1's
`results.json`/`rowpreds.parquet`/`fills.parquet` in place, and once this routine had written a
`README.md` into that shared directory for candidate 1, the directory would be permanently
excluded from the queue — candidate 2+ silently never dossiered. `find_pending_runs`'s
`root.rglob(RESULTS_FILENAME)` already recurses into per-candidate subdirectories, so `batch1`
(5 candidates) is expected to dossier correctly.

If `--scan` reports nothing to process, **exit quietly** — this mirrors the launch routine's "if
the run is still going" case, just one step later in the pipeline.

### Step 1 — for each run with a fresh facts.json and no README.md yet

Everything in this step is judgement, and everything you write must trace back to a `facts.json`
field or be visibly marked as your own reasoning. **Do not compute or restate a number that isn't
already in facts.json** — if a number you want isn't there, that's a gap in `dossier_tables.py`
worth a follow-up bead (`bd create`), not something to compute in-session.

1. **Read `facts.json` in full**, including `plots` (the list of PNG filenames actually written for
   this run — reference only those, don't assume one exists).

2. **Grade the run against its own `PREDICTED_SIGNATURE`** (`facts["grading"]["predicted_signature"]`).
   This is the one place this session forms a verdict, not a transcription — decided KISS
   (fps-3jj.6): no structured `PREDICTIONS` companion, just your read of whether `headline` +
   `breakdowns` match what the candidate predicted. Write your verdict directly back into
   `facts.json`'s `grading` block (`verdict`: one of `"matched"` / `"partially_matched"` /
   `"contradicted"` / `"inconclusive"`; `explanation`: one or two sentences; `pending`: `false`).
   This is the only field in `facts.json` this session ever edits — everything else stays exactly
   as `dossier_tables.py` wrote it, so the file stays traceable to its deterministic source plus
   this one recorded judgement.

   **A later regeneration keeps what you write here (fps-jrl).** `process_run` reads the
   `facts.json` already on disk and carries its `grading` block forward verbatim, because nothing
   can rebuild it — it is judgement, not a derived fact. It is dropped only when the run itself
   changed (a different `predicted_signature` or a different `status`), and it says so out loud
   when it does. This was NOT true until fps-jrl: every regeneration reset `verdict` to `null`,
   which is how PR #351 silently wiped all five of batch1's verdicts.

   A verdict carried across a regeneration whose NUMBERS moved (a pipeline fix re-running the
   candidate, as #359 did to batch1) keeps answering the question asked — the verdict grades
   `predicted_signature`, a claim about the SHAPE of the result — but the file is then
   mixed-vintage, so it says so: `grading.carried_from_fingerprint` records the `results.json`
   the verdict was actually formed against. Compare it to `provenance.results_fingerprint`; if
   they differ, the judgement predates the numbers printed beside it. Absent means never carried.

   In the same spirit, a regeneration that would replace `noise_band`,
   `per_fold[].regime`/`per_regime` or `decision_flips` with an "unavailable" value the current
   environment can no longer compute (a missing `noise_floor.json`, a `results.json` predating
   `meta.shock_folds`) refuses and writes nothing — it shows up in `--scan`'s output as a
   `failed, skipping` line naming the fields. Restore the missing input, or re-run **that one
   run_dir** with `--allow-field-loss` if the downgrade is what you actually want. The flag is
   rejected with `--scan` on purpose: scan-wide it would downgrade every refusing run in the
   pass, on one line of stdout each, in an unattended overnight log.

   **A `PREDICTED_SIGNATURE` making a claim about SHAP importance/sign/orientation grades
   `"inconclusive"`, always** (decided `fps-1l1` — no SHAP field will exist in `facts.json`,
   see `docs/routines/generator.md` item 8 in "What this session must be handed"). Don't
   reach for a proxy (a column's raw value trend, a per-fold delta) and call it a SHAP
   read — say plainly that the pipeline cannot test this part of the claim, same as an
   untestable per-axis CPL claim is written up as untestable rather than guessed at.

   **This rule is a backstop, and it is the expensive place to catch the problem.** By the
   time a run reaches this step the compute is already spent and the verdict is already
   forfeit: an ungradeable signature does not grade "untested pending an improvement", it
   grades `inconclusive` permanently, which reads like a measurement and is not one. The
   authoring-side rule lives in `experiments/candidates/TEMPLATE.py`'s `PREDICTED_SIGNATURE`
   comment (`fps-i2o` — the positive constraint, the list of gradeable `facts.json` families,
   and a worked BAD/GOOD pair from `stickiness_phase_saddle`) and in
   `docs/routines/generator.md` item 8. If you find yourself applying this rule to a real
   run, the signature should have been caught upstream — note it in the README's
   `not_tested` with the feasibility category (`fps-1p1`), not as an open question.

3. **Write `README.md`** in the run directory, in the house style (`experiments/TEMPLATE.md`), with
   an explicit **Facts / Judgement split**:

   - **Facts** (traceable to `facts.json`, verbatim or lightly tabulated — no new numbers):
     provenance (candidate, batch, snapshot date, git SHA, seeds, bead) — and state plainly
     that those seeds are the **WFCV screen only**, while the realised arbiter ran on the
     single seed at `facts["provenance"]["realised_seed"]` (fps-qbv — distinct from the
     `seeds` list beside it). Do **not** gloss that as "one draw with no error bar": the
     realised delta is a PAIRED difference — both arms share the seed, the data, the folds and
     every baseline column, differing only by the added one, so the fit's seed idiosyncrasy is
     common-mode and cancels (see `experiments/pipeline/noise_floor.py`'s docstring; on batch0
     the two arms agreed on 741/752 fills). Its error bar is the batch noise floor, built the
     same paired way at the same fixed seed (`PLACEBO_SEED_DEFAULT = SEEDS[0]`, matching the
     runner's `realised_seed`) — that matching is what makes the two comparable at all.
     Multi-seeding the arbiter alone would narrow the candidate against an unchanged ruler and
     manufacture significance; fps-awz retired exactly that unpaired seed-swap null. Report the
     single realised seed as provenance, not as a limitation — the
     headline realised-CPL
     delta and its `effect_resolved` verdict, the WFCV log-loss delta labelled as descriptive
     colour, the per-fold / per-regime / per-axis breakdown tables (mark suppressed cells plainly —
     "n=18, below the min-cell-n guard, not a finding"; a run whose `results.json` predates
     fps-3tu's empirical shock-fold set has `per_fold[].regime: null` and `per_regime:
     {"computed": false, "reason": ...}` — `breakdowns["per_fold_regime_unavailable_reason"]`
     carries that same reason and explains BOTH symptoms; state it plainly ("regime unknown —
     this run predates the empirical shock-fold split; re-run to backfill it") rather than
     rendering a blank regime column with no note, which reads as "no regime effect" and
     inverts this whole contract; if `per_axis` is present, also carry
     BOTH `breakdowns["per_axis_identification_note"]` and `breakdowns["per_axis_coverage_note"]`
     verbatim — the first says the per-axis **CPL** cells are a path-coupled cost allocated to a
     sub-period and so are **not identified** (report them as colour, never as a finding; the
     `per_fold` and `per_regime` cells beside them are fine, because a fold is an independent
     simulation rather than a slice through one — see `experiments/2026-08-21_path_coupling_audit/`
     and `docs/CONVENTIONS.md` § *Bucketed results*), the second says the axis lookup is built from
     `rowpreds.parquet`'s WFCV-validation-window rows, not the full frame, so it can under-cover
     relative to `headline.zone` and by how much. Never let a per-axis delta reach the Judgement
     section as evidence — if a candidate's whole claim rests on one, say plainly that the claim is
     untestable as posed and that the zone should be re-expressed as a set of folds), the
     `seed_flags` list (if empty,
     say so — "no cells exceeded 5× cohort median seed_std" is itself a fact worth stating).
     **Every `seed_std` is WFCV LOG-LOSS, not CPL** (`experiments/lib/gates.py`'s
     `seed_variance_gate` runs on `ll_all`/`ll_hard25`) — say so, and never let a seed flag
     be read as instability in a `delta_cpl_own`, which is not seed-measured at all.
     Quote the raw `seed_std` beside the ratio: the ratio self-normalises against the run's
     own spread (right call) but cannot be compared to the effect size, and the comparison
     that matters is raw seed_std against `delta_ll_*` (0.5359 against -0.0003 is ~1700×,
     which a bare "75.4×" hides). **If any flagged cell is on the CANDIDATE arm, render a
     fold × arm table rather than prose** — the diagnostic is the within-fold R0-vs-candidate
     comparison (a jump there means the added columns destabilised an otherwise-ordinary fit,
     a redundancy tell), and prose destroys it. R0-only flags are a property of the window,
     not the candidate; one line is enough for those,
     validation (NaN rate, PIT test result, INPUTS check result) — including
     `facts["validation"]["extra_feature_provider_miss_rate"]` (fps-qbv): this is the REPLAY's
     own coverage, distinct from `candidate_column_nan_rate` beside it, which describes the
     offline features frame and says nothing about how much of the realised backtest actually
     ran with the candidate's extra-feature columns present — a candidate can pass the
     provider-health check (which only errors at 100% misses) while running most of the
     replay without its own columns, and that must be stated plainly rather than left implicit
     in a clean-looking NaN rate — and the noise-floor delta
     (`facts["noise_band"]`) — if `floor_arity_exceeds_run` is `true`, say so: this run was
     graded against a ruler built from MORE columns than the candidate has, which is allowed
     (a wider ruler only raises the bar, never lowers it — `docs/CONVENTIONS.md` § The band's
     ARITY) but means a candidate that failed here has not been shown to fail against its own
     arity's band — and in that case `facts["noise_band"]["comparable_banks"]` (`fps-30p`) is
     the field that answers it, so report it rather than writing a `not_tested` line wondering.
     It carries every OTHER bank in the batch dir with the same `baseline_fingerprint` and
     `tank_params`, each scored against this same delta, plus any non-comparable bank with the
     reason it was skipped. Render them as a small table (name, arity, draws, `null_method`,
     z, threshold) whenever any bank is comparable — and carry `narrower_than_run` into that
     table, because a bank NARROWER than the run is biased in the candidate's favour (the
     canonical path refuses that case outright; siblings only disclose it, since a narrower
     bank is often exactly what answers the arity question). A narrower sibling's z is a
     LOWER BOUND on the bar, never a verdict. Two rules: they are **corroboration only**
     — `candidate_z_vs_band` off the canonical bank is what the Step 3 outcome keys on, and a
     sibling that disagrees is a fact to report, never a grade to substitute — and a differing
     `null_method` must be quoted beside the z, since a pinned-source bank measures a
     genuinely different null. An empty list means no siblings exist, not that none were
     checked. If `available: false`, say plainly "noise-floor band: not
     available" and quote `facts["noise_band"]["reason"]` verbatim rather than omitting the
     line or paraphrasing it (as of `fps-cf8` there is more than one refusal reason — no floor
     yet, a partial `--fold-subset` floor, or a `baseline_fingerprint` mismatch against a
     stale/re-locked floor, or a floor narrower than the run's own arity — and the reason string
     already says which); if available, report `candidate_percentile_better_than_noise` as-is
     (already oriented so higher = better — don't re-derive or re-sign it).

     **Also print the bar in c/L, not only the percentile and z** (`fps-3jj.21`). A z against a
     z-threshold says whether this candidate cleared; it says nothing about how strict the
     ruler that judged it was, so two runs graded by different-shaped floors read identically.
     Four fields make that legible and all four belong in the write-up:
     - `single_candidate_bar_cpl_held` — the threshold in the units the rest of the project
       quotes, comparable against `experiments/results.csv`'s 0.03–0.26 c/L history and against
       a batch-mate. It is the **favourable side only**; a candidate can also be surprisingly
       BAD, so judge the two-sided question on `abs(candidate_z_vs_band)` against
       `single_candidate_z_threshold`, never on this number alone.
     - `bar_draw_count_shift_cpl_held` — how much of that bar is band THINNESS rather than band
       WIDTH (the `df = n_draws - 1` t-penalty), quoted against the large-sample limit and
       computed on the NOMINAL draw count. Signed on the same scale as the bar, so a larger
       penalty is a more negative number. Print it whenever it is a material fraction of the
       bar: two thirds of batch1's k=1 → k=3 bar move was this, not the arity effect it was
       being read as.
     - `bar_reuse_shift_cpl_held` — the REST of the bar's hardening: how much comes from the
       bank's draws reusing source columns and so being worth less than their count. **Print it
       alongside the thinness shift, never instead of it, and never omit it** — on a wide
       candidate it is the larger of the two by an order of magnitude. On batch1's band a
       35-column candidate splits −0.0127 thinness against **−0.1345** reuse, so quoting only
       the thinness figure would explain 9% of why the bar moved from −0.152 to −0.287 and
       silently attribute the other 91% to nothing. The two sum to the whole distance between
       the large-sample bar and the one applied; a reuse-free bank reads 0.0.
     - `floor_arity_excess` — how many columns wider the ruler is than the run, the magnitude
       behind the `floor_arity_exceeds_run` flag above. If `facts["decision_flips"]["computed"]` is `true` (arity 2+
     and the noise-band call isn't a clean reject — `experiments/pipeline/dossier_tables.py`'s
     `_decision_flips`, fps-gez), report the per-fold table — this is the buy/wait divergence
     between R0 and candidate, diffed directly off `fills.parquet` (not a rowpreds
     proba-threshold crossing — see `experiments/lib/flips.py`'s module docstring for why that
     substrate is the wrong one here: rowpreds' WFCV-screen proba is a different, uncalibrated
     fit from the one that actually drove the realised decision). Columns, in fold order (never
     re-sorted by population — see fps-e1w's "Explicitly rejected": an n-vs-magnitude ordering
     implies a trust hierarchy no cell in this table can actually earn):
       - `flips` (`n_baseline_only + n_candidate_only`) and `decns` (`n_decisions`) side by
         side, never `flips` alone. `flips` overstates independent decisions — one real flip can
         cascade into a run of later differing fills at the same station as the tank's state
         diverges, which `n_decisions` collapses (`experiments/lib/flips.py`'s
         `_collapse_cascades`, window derived from the run's own tank cadence via
         `cascade_window_days`, never a hardcoded literal). fps-6yi: 303 flips are 88 decisions
         (report the RUN-level `n_flips`/`n_decisions` alongside the table's totals for exactly
         this reason — never just the per-fold column's sum), and fold 8's 36 flips are 5 —
         quoting 36 alone materially overstates that fold's independent evidence. **`decns` is
         not comparable across runs at different cadences** (`cascade_window_days`'s docstring)
         — a coarser cadence needs proportionally more elapsed quiet time before a gap counts
         as a new decision, so it systematically reports a LOWER `decns` for the identical
         underlying divergence. Never read a lower `decns` on a 7d-cadence run as weaker
         evidence than a similar `decns` on a 1d-cadence one.
       - `litres_baseline` / `litres_candidate` — flip-only litres per arm. `pooled_cpl` is
         spend-weighted, so a 3.57 L top-up and a 42.86 L fill count identically in a flip count
         but not in the CPL; a large litres gap between arms (fps-6yi: 2064 vs 2982, a 44%
         difference) is itself worth surfacing, not just a weighting detail.
       - **`regret_cpl_baseline` / `regret_cpl_candidate` from `decision_flips["regret"]` — the
         rendered timing column** (fps-2js). Regret is `price paid − the cheapest price that
         same station reached inside the tank's feasible wait`; `0` means the arm bought the
         best price it could actually have reached, and lower is better. Unlike the flip CPL
         columns it is **defined identically on both arms and cycle-normalised**, so cells are
         comparable to each other and it POOLS — always report the `all` row, which is the only
         run-level statement this table can make (fps-6yi at its 13-day horizon: baseline
         10.03 vs candidate 9.38 c/L, i.e. both arms buy about equally well, consistent with
         the -0.0735 headline). **Report `horizon_days` AND `cadence_days` with it, always.**
         The horizon is a fixed ceiling on the tank's feasible wait (`regret_horizon_days`),
         NOT a cycle length; it is deliberately a ceiling no real fill attains (fps-6yi's
         fills top out at 11.6 days of headroom), so regret levels are conservative by
         construction and slightly conservative toward whichever arm buys in larger fills.
         `cadence_days` matters just as much: the window is walked on the run's own evaluation
         grid, so a 1d run samples 14 candidate prices inside it and a 7d run samples 2. On
         fps-6yi the same flips give -0.644 at 1d, -0.695 at 2d and **+0.143 at 7d — the level
         nearly halves and the sign flips.** Regret is comparable only WITHIN one cadence;
         never compare it across dossiers at different cadences. (Same defect class as the
         `decns` quantisation caveat above.)
         - Print `regret_cpl_delta` **beside its own interval**
           (`regret_cpl_delta_interval` = delta ± 2×`regret_cpl_delta_se`, sized on cascade-collapsed
           DECISIONS), and apply exactly the citation rule below: fps-6yi's `all` row is
           -0.64 c/L against a ±4.24 interval, and 12 of 12 resolvable folds sit inside their
           own. **Regret is a legibility fix, not a power fix — it does not rescue a thin
           cell, it just stops the cell lying about its size.** Never present it as having
           resolved something `flip_cpl_delta` could not.
         - It is a **WHEN measure, not a WHAT measure** (same station only, by construction).
           When regret and `run_contribution_cpl` disagree in sign — fps-6yi's fold 4
           contributes +0.0437 c/L while its regret says the candidate timed 0.26 c/L better —
           that is not a contradiction, and it is the most useful thing this pair says: the
           difference came from WHAT was bought (volume and composition), not WHEN.
         - Report `dark_fill_days` whenever it is non-zero (fps-6yi: 56 of 303, folds 4-7, all
           one station's outage). Those fills ARE scored, at the forward-filled price the tank
           simulator itself bought at — the outage does not bias the estimate, but it thins and
           tilts specific folds, so a fold's evidence weight is not what its row count suggests.
         - When `decision_flips["regret"]["computed"]` is `false`, state its `reason` verbatim
           (usually: the price DB is gitignored and absent from this checkout) and render the
           rest of the table normally — it is not a reason to skip the flip section.
       - `flip_cpl_delta` — **kept in `facts.json`, no longer rendered in the table** (fps-2js).
         `flip_cpl_baseline` and `flip_cpl_candidate` pool DISJOINT fills on different days at
         different points in the price cycle: their difference has no stable denominator, so it
         explodes in thin cells and routinely cannot be distinguished from zero (fps-6yi: 0 of
         12 resolvable folds cleared their own interval — fold 13's headline-grabbing -32.73
         sits inside a 2×SE of ±41.05, effective n sized on its 3 cascade-collapsed DECISIONS,
         not its 6 raw flips; under regret that same fill reads 0.00, the exact optimum).
         Showing it beside regret would put two measures of the same thing on screen that
         disagree, and the louder, less trustworthy one wins that contest — so render regret
         and leave this in the facts for anyone who wants to check it. **The citation rule
         below applies to it regardless of whether it is rendered.**
         **A cell marked `inside_own_se` must never be cited as evidence in the Judgement
         section** — "favourable flips dominate, N of M folds" is a vote count over cells that
         cannot resolve their own numbers, not corroboration (this is exactly how fps-6yi's
         original Judgement went wrong; see that run's post-dossier review addendum). Whenever
         a delta is `None`, state its `_reason` verbatim (`"no flips"` / `"one arm only"`)
         rather than a blank cell; when a delta exists but no interval could be estimated,
         state its `_interval_reason` verbatim instead of printing a hairline interval — both
         are honest "unmeasurable," never the same as a resolved zero-width one.
       - `run_contribution_cpl` — **the highest-value column.** What this fold's divergence is
         actually worth at the scale of the whole run: a litres-weighted shift-share of the
         fold's own `delta_cpl_own` (`facts["breakdowns"]["per_fold"]`, the whole fold's pooled
         delta, matching and flipped fills alike — NOT `flip_cpl_delta`, the flip-only figure
         above), weighted by that fold's share of the run's TOTAL litres. Fold 13's -32.73 c/L
         flip delta contributes only -0.0644 c/L to the run; folds 2 and 4 hand back +0.0995
         between them. Sum the column and report `run_contribution_total_cpl` alongside
         `composition_residual_cpl` as its own line (fps-6yi: -0.0606 total, -0.0129 residual,
         reconciling exactly to the -0.0735 headline `delta_cpl_held`) — the residual is real
         (a litres-weighted average of per-fold ratios isn't exactly the ratio of pooled
         totals whenever the two arms' litres mix differs fold to fold), not a rounding
         artefact, and hiding it would be its own distortion. It can also absorb tau
         divergence between arms (`delta_cpl_own` is own-tau, `delta_cpl_held` is held-tau —
         see `_attach_run_contributions`'s docstring). **Branch the caveat on
         `decision_flips["tau_diverges_any"]` (fps-4je)** rather than stating it
         unconditionally: `true` — at least one fold that actually entered
         `run_contribution_total_cpl` (i.e. not suppressed) had its own-tau genuinely differ
         between arms, so state the residual as "composition drift AND tau drift"; `false` —
         own-tau agreed in every such fold, so state it as "composition drift" alone; `None` —
         UNAVAILABLE, never read as "checked, agrees": either the key is present but every
         unsuppressed fold's own flag was itself unavailable, or — the common case for any run
         dossiered before fps-4je landed (PR #354 review finding #2:
         every batch1 facts.json committed as of fps-4je has `decision_flips.computed: true`
         with no `tau_diverges_any` key at all, i.e. a **missing key**, not a `null` value) —
         `results["realised_deltas"]` doesn't exist yet. A missing `tau_diverges_any` key reads
         exactly the same as an explicit `None`: fall back to "composition drift, and possibly
         tau drift" in both cases. The same flag is available per fold as
         `facts["breakdowns"]["per_fold"][i]["tau_diverges"]` if a specific fold's contribution
         needs the same caveat — note that field is populated even for a suppressed fold (it
         just doesn't feed the run-level roll-up), so don't cite a suppressed fold's own
         `tau_diverges` as evidence about the residual either.
     **Same rule as `per_axis`: flip detail is colour illustrating a mechanism, and must never
     reach the Judgement section as a second arbiter overriding the noise-band call** — one vivid
     flip (a station catching a big cheap fill one day early) is easy to find more convincing than
     the aggregate test, and it isn't. Flip counts are also a LOWER bound on decision divergence
     (a WAIT day writes no fill row, so two arms can diverge in daily probability without ever
     producing a differing fill — `experiments/lib/flips.py`'s module docstring). If `computed`
     is `false`, state the `reason` verbatim rather than omitting the field — a candidate with
     only 1 column, or a clean reject, not having this breakdown is expected, not a gap in this
     run.
     If `facts["redundancy"]["available"]` is `true` (fps-qbv), and a candidate declares a
     falsification test naming which of its own columns is the suspect, **quote
     `per_column_r2` for that specific column, not just `block_r2`** — the block figure is the
     MEAN across the candidate's members and dilutes a single lock-redundant column out of
     view (a member at 0.855 alongside others near 0.10 gives a block figure near 0.3, which
     reads as "not redundant" while the named column is 85% reconstructible from the lock).
     `max_column_r2` is the same worst-case figure at a glance. If `available` is `false`,
     state the `reason` verbatim (no batch record for this batch, or this candidate is absent
     from it) rather than silently skipping the redundancy question.
   - **Judgement** (your reasoning, visibly separated — a `## Judgement` heading is enough):
     - The `PREDICTED_SIGNATURE` grading verdict + explanation you just wrote.
     - **`not_tested`** — adjacent ground this run does *not* rule out. This is the highest-value
       sentence in the whole dossier (see the ledger's own docstring) — a rejected candidate whose
       series still has room ("the raw gap failed, but its velocity might not") is exactly what
       lets the next generator session revisit it legitimately. Actually think about this; don't
       write "none" as a default.

       Two rules make an entry sound (`fps-1p1`). This field is copied verbatim into
       `experiments/ledger.yaml` and handed to the next generator session, so a wrong line is
       not inert — it either sends work at closed ground or hides open ground.

       1. **State whether the test is possible, and with what.** "X was not measured" is only
          half a fact. Put every line in one of three registers, because a reader acts
          differently on each:
          - **Measurable now** — name the artifact it would come from (`decision_flips.rows`,
            a sibling `noise_floor*.json`, the batch DB). If it is cheap, consider just doing it.
          - **Blocked on a named artifact** — say which, and what would unblock it.
          - **Permanently untestable by this pipeline** — say so plainly. This is a different
            outcome from "pending", and conflating them is what produces a dossier that
            recommends waiting for something that will never arrive.
       2. **Check before you write.** Before any line of the form "Y would resolve this": look
          for Y. Enumerate the batch directory rather than assuming the canonical filename is
          the only artifact (`fps-30p`), and read the **current status and close reason** of any
          bead you cite — not just its ID. **Never stake a recommendation on the outcome of a
          bead filed in this same session**: you do not know how it will close, and "worth a
          second look once `<bead>` lands" is worthless if it lands as "won't build".

       Worked failure, `stickiness_phase_saddle` (`fps-6yi`, batch1, dossiered 2026-08-27):
       all three of its `not_tested` lines were wrong, one per failure mode. It filed `fps-1l1`
       and staked its recommendation on the outcome in the same document, on the same day —
       fps-1l1 closed "don't capture", making that line *permanently untestable*, not pending
       (rule 1, and see `experiments/candidates/TEMPLATE.py` / `fps-i2o` for why the signature
       was ungradeable in the first place). It said a same-arity noise floor "could resolve it
       either way" while two comparable banks sat in the directory it had just read
       `noise_floor.json` from, both answering it (rule 2). And it declared the mechanism
       uncheckable when `decision_flips.rows` plus the batch DB localised it cleanly (rule 1,
       measurable-now). The corrected entry is in that run's README under
       *Addendum — post-dossier review*.
     - An overall recommendation — **not a binary verdict**. Something like "worth a second look
       once the batch has more shock-fold coverage" or "dead end on this series, but see
       not_tested" is the right register. The dossier exists to support a human (or a follow-up
       session) interrogating it further, not to close the question. A conditional recommendation
       is only as good as its condition: gate on something checkable (data coverage, a run that
       has not happened) and verify the condition is actually still open before you write it —
       never on a bead outcome you have not read, per rule 2 above.
   - Embed the plots that exist for this run (`![](per_fold_delta_bars.png)` etc. — only ones
     listed in `facts["plots"]`).

4. **Prepend one entry to `experiments/ledger.yaml`** (newest-first, matching its existing
   convention) using the same fields as every prior entry:
   - `candidate`: `facts["candidate"]["name"]`
   - `inputs`: `facts["candidate"]["inputs"]`
   - `claim`: `facts["candidate"]["hypothesis"]`
   - `falsified`: empty string only if the grading verdict is `"matched"` **and**
     `facts["headline"]["realised"]["effect_resolved"]` is true (i.e. the claim held up and the
     arbiter moved the right way) — otherwise, one or two sentences on what the run actually
     showed, in the same style as existing entries (see `tgp_delta_7d` vs
     `station_minus_tgp_cents` in the file for the contrast).
   - `not_tested`: the same judgement you wrote into the README — reuse it verbatim, don't redo it.
   - `outcome`: mechanical, from `facts["provenance"]["status"]` and, for a completed run,
     `facts["noise_band"]` — **not a judgement call** (`fps-3jj.17`, decided 2026-08-23):
     - `status == "disqualified"` → `outcome: disqualified`.
     - `status == "aborted_candidate"` → `outcome: aborted_candidate`.
     - `status == "graded"` — the pipeline's own name for "ran to completion and reached the
       scoring stages" (`runner.py`'s outcome taxonomy comment; it says nothing yet about the
       claim's merit) — apply the **rejected-vs-inconclusive split**:
       - `facts["noise_band"]["reason_code"] == "floor_arity_below_run"` → **STOP. Do not write
         a README.md, do not write a ledger entry, do not pick an outcome for this run at all.**
         Report that this batch needs a wider-arity ruler (quote the `reason`, which carries the
         exact two commands) and move on to the next pending run. This is the one unavailability
         that is NOT about the candidate: the run completed and graded fine, and it becomes a real
         measurement the moment a floor with enough columns exists (`fps-3jj.14`). Falling through
         to the `rejected` rule below would stamp DEAD GROUND — what the next generator reads as
         "don't re-propose this" — onto a claim nothing has measured, which is precisely the harm
         `fps-3jj.17` exists to prevent. Writing no README.md is deliberate and load-bearing:
         `find_pending_runs` treats an absent (or empty) README.md as pending — it is the marker
         that a session actually wrote the run up (fps-jrl) — so staying silent leaves the run to
         be picked up correctly on a later night. Check this
         BEFORE the `available` test below — the arity case sets `available: false` too, so
         ordering is what keeps it out of the `rejected` branch.

         (There was briefly a second arity code, `run_arity_above_cap`, for a candidate wider
         than the ruler could grade. **Retired in `fps-3jj.25`** — reuse is priced into the
         threshold now rather than capped, so a wide candidate gets a wider bar instead of no
         verdict and an unbounded nightly re-pick. If you ever see that code, the run predates
         the change; treat it exactly like the case above.)

       - `facts["noise_band"]["available"]` is `false` (any other reason) → `outcome: rejected`. No
         noise floor means no ruler to grade "below the instrument's resolution" against, so there
         is nothing to call inconclusive — same as before this rule existed. Note the unavailability
         in the README as usual (Step 1 above); don't invent a floor to unblock this decision.
       - `facts["noise_band"]["available"]` is `true` but `candidate_z_vs_band` or
         `single_candidate_z_threshold` is `null` → `outcome: rejected`, same fallback and same
         reasoning as the `available: false` case above. This state is real, not hypothetical: the
         batch has a `noise_floor.json`, but `_noise_band` couldn't estimate a band std from it
         (fewer than 2 draws, or — even with `n_draws >= 2` — every draw landing on the same
         value; that reads as `float(np.ptp(deltas)) == 0.0`, not `band_std == 0.0` — `np.std` of
         identical floats is a tiny nonzero number like `1.78e-18`, not exact zero, `fps-tnz`), so
         there is no ruler to compare `delta_cpl_held` against
         despite `available` reading `true`. Don't treat a present-but-null `z`/`t` as "inside the
         band" (that's what `-t < z < t` is for, and it doesn't apply when either side is `null`)
         — this branch must be checked **before** the three regions below, not folded into them.
       - Otherwise read `z = facts["noise_band"]["candidate_z_vs_band"]` and
         `t = facts["noise_band"]["single_candidate_z_threshold"]` (both non-null at this point) —
         already computed by
         `dossier_tables._noise_band` (`t` is `family_wise_z_threshold(n_candidates=1, n_draws=...)`,
         i.e. the *single*-candidate detection bar, docs/CONVENTIONS.md's "-0.17 c/L judged
         singly" for `batch1` (was -0.15 against the retired arity-1 floor, `fps-3jj.14`); deliberately **not** the batch-level `clears_family_wise_threshold`
         gate, which needs the whole batch dossiered and exists to answer a different question —
         "is any candidate surprising after correcting for picking the best of N", the retrospective
         routine's job, not this one's):

         **`single_candidate_z_threshold` is the authoritative `t`.** A
         `single_candidate_z_threshold_nominal` sits beside it in the same object and exists only
         so the bar's thinness and reuse components can be reported separately (see the
         noise-floor fields above). **Never grade against the nominal one** — it ignores the
         reuse correction and is therefore too easy, which is the whole bias this machinery
         removes.

         - `z >= t` → `outcome: rejected`. `delta_cpl_held` is clearly the WRONG SIGN by more than
           this batch's own noise band — measurably worse than a placebo column, not merely
           unresolved.
         - `-t < z < t` → `outcome: inconclusive`. The effect sits inside the noise band — this run
           could not distinguish the candidate from noise. Per the ledger header, this is **not**
           `rejected`: the ground may still be open (more data, a different arm of the same
           mechanism, etc. is a legitimate follow-up, not a re-proposal of dead ground).
         - `z <= -t` → `outcome: rejected`, same as every run before this rule existed. The
           candidate clears single-candidate noise in the GOOD direction, but this dossier step
           still never decides graduation (see below) — a human who reads a strongly negative `z`
           and wants to pursue it upgrades this same entry's `outcome` to `graduated` by hand.
     The retryable statuses are filtered out at Step 0 and never reach a ledger entry. **Never
     write `graduated` here.** Per the parent design's decision boundary, this pipeline never
     touches `fuel_signal/features.py`; graduation is a separate, human-initiated PR, and a human
     updates this same ledger entry's `outcome` to `graduated` by hand when that PR lands (see how
     the `tgp_delta_7d` entry already reads `graduated` even though every pipeline run of it would
     have closed with status `graded`).
   - `evidence`: the run directory's path, relative to the repo root.

5. **Prepend one row to `experiments/INDEX.md`'s table** in the same pass (its own convention: the
   two files must never drift, because neither is derived from the other) — Date / Name /
   Hypothesis / Result / Status, `Status` = `open` while the pipeline still has follow-ups queued
   for this candidate's series, `done` otherwise, `abandoned` for `disqualified` /
   `aborted_candidate` (never `graduated` — same human-edit-later rule as the ledger). Retryable
   statuses can't reach this step at all — see Step 0.

6. **Commit `README.md` + the run's updated `facts.json` (`grading` filled in) + the PNGs +
   `experiments/ledger.yaml` + `experiments/INDEX.md` together, straight to `main`, and
   `git push`** — `experiments/**` is PR-exempt (`CLAUDE.md`), and `ledger.yaml`/`INDEX.md` are
   prose/data files under the same exemption since they only ever get written by this routine or
   hand-backfills of experiment content. One commit per run if several are queued in one session
   (keeps `git blame` readable per candidate). **The push is not optional and not deferrable to
   "later" or "a human":** direct-to-`main` means landed on the remote (docs/CONVENTIONS.md §
   the exemption), and this routine is unattended, so a commit it leaves behind is a commit
   nobody is watching for.

7. **Close the candidate's bead: `bd close <bead_id>` (from `facts["provenance"]["bead_id"]`),
   then `bd dolt push`.** Mechanical, every time step 6 fires — this is not a judgement about the
   candidate's merit, only a record that the run is finished and written up (a rejected or
   inconclusive candidate gets closed exactly the same as any other; `outcome` in the ledger is
   the merit verdict, bead status is not). Found missing 2026-08-24 (fps-2l9 sat `in_progress`
   after being fully dossiered, with no close step anywhere) — a real process hole, not tidiness:
   a batch's retrospective bead (`fps-3jj.18.1`-style, one per batch — see § batch parent bead
   below) is `blocks`-blocked on every candidate bead being CLOSED, not on its README existing on
   disk, so a missed close here silently leaves the retrospective looking blocked forever even
   once every candidate has actually run. Skip this step only in the arity-refusal case above,
   where step 6 itself is skipped (no README, no ledger entry, no bead touch — the run goes back
   in the queue, not to closure).

   **`bd close` also writes a `field_change` row to the git-tracked
   `.beads/interactions.jsonl`** (`bd dolt push` syncs the separate Dolt remote — a different
   system, see `feedback_beads_commit_tracked_state` — and does not touch git at all). Step 6's
   commit already landed before this bead close happens, so that row is left as an uncommitted
   change in the primary worktree at the end of every run unless committed separately. Found
   2026-08-25 (fps-6to's close left `.beads/interactions.jsonl` dirty, caught only by the owner
   checking `git status` after the run) — until now this was patched after the fact by a human or
   a later session noticing and pushing a standalone `chore: record <id> closure in the
   interaction log` commit (see git history for several examples). Fold that into this step
   instead of leaving it dangling: immediately after `bd close` (and whether or not `bd dolt push`
   succeeds — the two are independent), run
   `git add .beads/interactions.jsonl && git commit -m "chore: record <id> closure in the
   interaction log" && git push`. If several candidates are dossiered in one session, one commit
   per close (or a single trailing commit covering all of them) both work — just don't let the run
   end with `.beads/interactions.jsonl` dirty.

   **The push belongs in this step too, for the reason the step itself demonstrates.** The
   revision that added the commit half above (2026-08-25) was written, committed to a local
   `main`, and never pushed — so the instruction telling this routine not to leave work dangling
   was itself left dangling, and shipped only on 2026-08-26 when the eight-commit backlog was
   found. A rule about finishing that doesn't itself get finished is the sharpest possible
   demonstration that "commit" and "landed" are different things.

### Step 2 — summary

Print one line per run processed (candidate name, outcome, one-clause verdict) and one line per
run the deterministic scan logged as `failed, skipping`.

**Before printing that summary, verify the run actually landed** — `git status --short` empty
(nothing left dirty, including a `bd`-written `.beads/interactions.jsonl` row) and
`git log --oneline origin/main..main` empty (nothing left unpushed). Both are one command and
neither depends on remembering what this run happened to touch, which is the point: the two
failures this routine has actually had — a dirty interactions log (2026-08-25) and an unpushed
backlog (2026-08-26) — are each caught by one of them. Report the run as done only once both are
clean; if either isn't, finish the commit/push before summarising rather than noting it as a
caveat.

No PR, no further action — the next thing
that touches this candidate is either a human reading the README, or (once every candidate bead in
a batch is closed) the batch's retrospective bead — see `docs/routines/generator.md` § Filing for
the current parent/child bead structure; earlier revisions of this doc named a standalone
`fps-3jj.8`, superseded by the per-batch parent-bead convention.

## Known gaps to flag, not silently work around

- **`noise_floor.json` per-batch step.** `batch_freeze.py` computes this as its own final step
  (`fps-cf8`), so a batch frozen normally always has one. `facts["noise_band"]["available"]`
  still reads `false` in three cases: an older batch frozen before `fps-cf8` or with
  `--skip-noise-floor` (never computed), a floor computed with `--fold-subset` (partial fold
  coverage), or a floor whose `baseline_fingerprint` no longer matches this run's (a re-lock —
  graduation or `LOCKED_FEATURE_COLUMNS` change — invalidates an existing batch's floor; see
  `docs/CONVENTIONS.md`). If you see `available: false`, that means this batch is missing a
  *matching* floor, not that the capability is unbuilt. Report it plainly in the README (see
  Step 1 above), quoting `facts["noise_band"]["reason"]`, rather than inventing a noise
  estimate.

## The shim

This is the exact text that should be the Routine's stored prompt (three lines, same pattern as
`docs/routines/worker.md`'s shim — no embedded steps, no reminders about what this file says):

```
You are the fuel-price-signal dossier routine.
Your working directory is /Users/esteele/Code/fuel-price-signal (the
fuel-price-signal repo, primary worktree — this is a persistent local checkout, not
a fresh clone).
Follow docs/routines/dossier.md exactly.
```

**This is a LOCAL routine.** An earlier revision of this shim said "your working directory already
has a fresh checkout", copied from the cloud worker. That is wrong here and not merely untidy: a
fresh clone has neither the run artifacts (`results.json`, `rowpreds.parquet`, `fills.parquet` are
produced on this machine) nor the batch snapshot (`features.parquet` and the cloned DB are
gitignored), so a fresh-checkout dossier routine would find an empty work queue every night and
exit quietly, forever. It must name the primary worktree, exactly as `docs/routines/launch.md`'s
shim does.

**Registered** (2026-08-22, `fps-3jj.11` prep) as the local scheduled task
`fuel-price-signal-dossier`, cron `0 5 * * *` local — 5:00 AM AEST daily. Separate from both the
chore/polish worker and the launch routine: three independent schedules against the same repo.

**Why 5 AM and not "later the same night".** Two constraints pull against each other. The run must
have finished (launch fires at 9 PM, a candidate takes tens of minutes at 1-day cadence, so ~10 PM
is already enough), but this routine uses Claude *actively* to write the dossier, and
`docs/routines/launch.md` puts the 10 PM–4 AM AEST window off limits for exactly that reason —
it is why launch itself sits at 9 PM. Same-night dossiering lands squarely inside it. 5 AM clears
the window, leaves hours of headroom over even a badly overrunning candidate, and has the dossier
waiting when the owner wakes up. If the run is still going, this routine exits quietly and picks
the candidate up the next morning, so a late finish costs a day rather than a write-up.
