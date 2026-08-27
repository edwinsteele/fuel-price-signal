# Generator session — prompt source

Canonical, tracked home for the prompt used by the **candidate-generator session**
(`fps-3jj.7`) in the AI-sourced feature engineering pipeline (`fps-3jj`).

Unlike `docs/routines/worker.md`, this is not a scheduler shim. **The generator is
invoked by the owner interactively, not on a schedule** — see "Decided: interactive,
not scheduled" below. This file *is* the prompt: read it in full at the start of a
generator session, or paste it as the session's opening instruction.

## When to invoke

Once per batch, when the experiment queue has drained (all of the previous batch's
candidates show a closed outcome in `experiments/ledger.yaml`). Cadence is an event,
not an interval — do not turn this into a scheduled task.

**Decided: interactive, not scheduled (2026-08-17).** Launch and dossier are genuinely
nightly; the generator runs once per 5–15 nights. Scheduling "once the queue drains"
means building and maintaining a trigger condition, which is more machinery than
starting a session. It is also the step most worth a human look before it commits —
its output determines what N nights of compute run on, and batch setup already has a
human checkpoint (the DB refresh is hard-gated: abort on failure, never run on stale
data). Generation itself is still AI-sourced; only the invocation is human, so aim (b)
— measuring what the model proposes and whether its confidence predicts its hits — is
unaffected by who pressed go.

## What this session must be handed

Read all of the following before proposing anything:

1. **The canonical column list** — what already exists, so candidates don't duplicate
   it. From `fuel_signal/features.py`:
   - `FEATURE_COLUMNS` (15, the trained-model contract): `cycle_pct_through`,
     `cycle_days_since_peak`, `cycle_mean_length`, `cycle_last_min_cents`,
     `cycle_last_max_cents`, `cycle_peak_count`, `station_price_cents`,
     `station_minus_last_min_cents`, `station_minus_last_max_cents`,
     `station_minus_sydney_avg_cents`, `lga_mean_cents`,
     `station_minus_lga_mean_cents`, `brand_mean_cents`,
     `station_minus_brand_mean_cents`, `stickiness_score`.
   - `LGA_FEATURE_COLUMNS` (35 `days_since_trough_entry_<lga>` event features, in the
     trained model).
   - `NETWORK_FEATURE_COLUMNS` (4: `network_px_std`, `network_px_std_delta_3d`,
     `lga_phase_std`, `lga_phase_std_delta_3d`, in the trained model, #216).
   - `TGP_FEATURE_COLUMNS` (`tgp_delta_7d` — **computed into `features.csv`, NOT in
     `FEATURE_COLUMNS`, and no longer on a path to it.** Its June 2026 realised-arbiter
     graduation was **retracted 2026-08-19 as inert**: −0.039 c/L does not reproduce
     (+0.0059 c/L on identical columns, folds and seed), and it was never resolvable
     either way, being smaller than the value of one buy/wait decision flip — **~0.05 c/L
     at the 7-day cadence that measurement ran on, and smaller at today's 1-day cadence**
     (1832 chosen fills instead of 244, so each flip carries less of the total). Do not
     reuse 0.05 as a general resolution bar; it is a 7-day quantity, not a constant.
     Against the batch0 noise floor it sits at the 40th percentile of pure fit noise.
     The column is registered in `features.NON_MODEL_COLUMNS` under
     `NON_MODEL_REASON_INCONCLUSIVE` — measured below the arbiter's resolution, so
     neither graduated nor dead ground. #271 is closed as superseded; the successor,
     bd `fps-x0f`, asks whether *any* TGP expression earns a place rather than assuming
     one has. Fair game as an `INPUTS` read; do **not** cite it as a proven feature.
   - Brand trough columns (`days_since_trough_entry_<brand>`) — computed, evaluated
     and rejected for the trained model (#187, Phase 4b), but still present in
     `features.csv`. Not part of the model contract; fair game as an `INPUTS` read,
     not as something to re-propose unmodified.
2. **The negative-results ledger** — `experiments/ledger.yaml`. Read every entry, not
   just the outcome codes. The `not_tested` field on each is deliberate: it is the
   field that keeps the generator from being timid, and a candidate that revisits a
   rejected series with a genuinely different mechanism is legitimate (`tgp_delta_7d`
   graduated after the raw gap `station_minus_tgp_cents` failed on the same series).
   The ledger's job is **not** deduplication — the column list already stops literal
   duplicates. It exists for the three failure modes duplication can't catch:
   - Re-proposing a **rejected** candidate (it's not in `features.csv`, so the column
     list gives no protection).
   - **Reparameterisation** (e.g. `tgp_delta_14d` vs `tgp_delta_7d`, ~0.95 correlated)
     — this is a mechanical, free check (§ Redundancy checks), not a prompt rule.
   - **Redundancy against a combination** of existing columns — also mechanical
     and free.
3. **`docs/STATUS.md`** — current model state, what's shipped, what's pending.
4. **Headroom — where recoverable value (`model_cpl − oracle_cpl`) lives.**
   **The #262 per-zone map no longer answers this. Every per-zone row of it was
   withdrawn** on 2026-08-20 (`experiments/2026-08-20_headroom_attribution/`, bd
   `fps-1785999730023-4-264564ac`; re-confirmed by
   `experiments/2026-08-21_path_coupling_audit/`, bd `fps-grp`). If you have seen
   `regime axis is FLAT`, `late_descent 1.00 / normal 1.50 / overdue 1.33`, or
   `12–16c 7.09` quoted as guidance — including in an older copy of this file —
   **those numbers are withdrawn and must not steer a candidate.** What is left:
   - **Window-level headroom stands, with its cadence attached.** The production
     decision grid is **daily** since the 2026-08-22 re-lock (bd `fps-oqz`), so the
     headroom a candidate aims at is **2.97 c/L at 1-day cadence** — not the 1.54
     c/L that the same measurement gives on the retired 7-day grid
     (`experiments/2026-08-20_cadence_ceiling/`, bd `fps-fii`). At 1d the model
     captures ~64% of the always-buy → perfect-foresight distance, down from ~71%
     at 7d: the target got bigger *and* the model's share of it got smaller,
     because at 7d the TANK was the binding constraint and at 1d the SIGNAL is.
     Headroom is not a constant, so never quote one without its cadence.
   - **There is no known zone of concentration.** Not "the concentration is
     elsewhere" — the per-zone question is **not identified**, so no axis tried so
     far can name one. A candidate is therefore *not* expected to aim at a
     headroom hot spot, and "helps in the 12–16c band" is no longer a sharper
     claim than a uniform one; it is a claim about a withdrawn measurement.
   - **The cycle-phase axis is open, not closed.** The finding that rested the
     late-descent thread ("regime axis FLAT") is one of the withdrawn rows, so
     that closure argument is gone. Don't treat cycle-phase as settled dead
     ground *or* as a known win — see bd `fps-x0f`, which is re-opening it, and
     note the axis itself has a known unfixed defect (`cycle_mean_length` is an
     expanding all-history mean).
   - **Surviving lead is external wholesale/crude, not more cycle-derived features**
     — "signal-from-cycle in diminishing returns". Weigh this against the
     no-new-datasets constraint below: it's a reason to look harder at what's
     derivable from the TGP series already in the frame, not a licence to pull in
     a new series. **Weakened but not withdrawn:** it drew partly on the
     now-withdrawn flat regime axis; its other support (#214/#231, log-loss based)
     is unaffected. Note also that no TGP feature is in the lock — `tgp_delta_7d`'s
     June graduation was retracted 2026-08-19 as inert.
   - **The rule underneath all of this**, because it constrains what a candidate
     may claim: `docs/CONVENTIONS.md` § *Bucketed results — check the convention
     spread before believing an ordering*. Allocating a **path-coupled total cost**
     (any tank-backtest CPL) to a sub-period has no unique answer, so per-zone
     economics is unidentified while window-level and per-fold economics are fine.
5. **The no-new-datasets constraint.** A candidate's `INPUTS` must be columns already
   present in the frozen features frame (`features.csv`) plus `price_date` — nothing
   that requires a new downloader, a new table, or a new API. This is enforced
   mechanically at launch time (`INPUTS` declares reads; `add_columns` is called
   against a frame restricted to exactly those columns, and any undeclared read
   raises `KeyError` — see `fps-3jj` parent design, "Validation"), but the generator
   should not spend a slot proposing something that can't pass it. `add_axis`
   (optional per-row grouping function) is under the same restriction.

   **"Present in the frame" is necessary, not sufficient — the label columns are in
   there too.** `fuel_signal.features.TARGET_COLUMNS` (`future_min_cents`, `label`) sit
   in `features.csv` and satisfy the rule above by its letter, but they are computed
   from prices *after* `price_date`: they are what the model predicts, never an input.
   Naming either in `INPUTS` aborts the candidate at validation
   (`validate.py`'s `TargetColumnInInputs`), so a slot spent on one is a slot wasted.
   Note this is the one hazard the differential PIT test cannot catch for you — a
   forward-looking value stamped *on* the row survives date-truncation unchanged — so
   don't treat "the leak test would have caught it" as a reason to be casual here.
6. **The fold/regime taxonomy.** Two distinct axes, don't conflate them:
   - **Fold-level**: 14-fold paired walk-forward CV; `SHOCK_FOLDS = {1, 4, 9, 13}`
     (`experiments/lib/constants.py`), everything else `normal`.
   - **Row-level cycle regime**: `experiments/lib/zones.py` `assign_regime()` over
     `cycle_pct_through` — `normal` (0.0–0.6), `late_descent` (0.6–1.0), `overdue`
     (≥1.0).
   - **`TARGET["axis"] = "regime"` is a RESERVED NAME meaning the FOLD-level one.**
     The runner implements it as `"shock" if fold in SHOCK_FOLDS else "normal"`
     (`runner.py`, `_resolve_zone`) — it does **not** call `assign_regime()`. The two
     axes share a level named `normal`, so a candidate that writes
     `{"axis": "regime", "expect_concentration_in": ["normal"]}` meaning *cycle-phase
     normal* is silently graded against *non-shock folds* instead. No error is raised;
     the grade is just wrong. To target the row-level cycle regime, supply your own
     `add_axis` returning those labels and name the axis something else
     (`"cycle_regime"`), accepting the row-level COST caveat below.
   - **The cycle-phase axis has a known unfixed defect** and bd `fps-x0f` is
     re-examining whether it measures anything: `cycle_pct_through` divides by
     `cycle_mean_length`, still an expanding all-history mean over every confirmed peak
     to date. A candidate is free to target it, but should not treat "late descent is a
     weak zone" as established — three prior findings on that axis disagree with each
     other.
   - A candidate's `TARGET` axis is not limited to these two — `add_axis` lets a
     candidate declare its own per-row grouping (day-of-week, TGP direction,
     competition density, cycle amplitude, brand class are all live, uncut
     candidates per the parent design) — but any axis must be a function of columns
     already in the frame, and goes through the same differential PIT test as
     `add_columns` (an axis leaks into the *conclusion*, not the model, if it uses
     future information — e.g. "days that turned out to be near the trough").
   - **A row-level axis cannot carry a COST claim.** `CONFIDENCE_ZONE` and the
     dossier's `per_axis` table grade a zone claim on pooled realised CPL, and
     pooled CPL is a **path-coupled total** — a buy now changes what is possible
     later — so cutting it on a per-row label allocates that total to a
     sub-period, which has no unique answer. This is the defect that withdrew
     every per-zone row of #262 (`docs/CONVENTIONS.md` § *Bucketed results*;
     `experiments/2026-08-21_path_coupling_audit/`, bd `fps-grp`). The fold cuts
     above are safe for a mechanical reason, not a stylistic one: each (fold,
     station) is an **independent simulation with its own tank**, so a fold-cut
     number is a sum of complete windows, while a row label is a slice through
     one. So:
     - **Prefer expressing a zone claim as a set of folds** (`TARGET["folds"]`)
       or as the built-in `regime` (a grouping of whole folds). These are graded
       on identified quantities.
     - **If the mechanism really is row-level**, declare `add_axis` — it is still
       genuinely useful for the *descriptive* per-axis tables (feature values,
       NaN rates, log-loss) — but expect the per-axis **CPL** cell to be reported
       with an identification caveat and treat it as colour, not as a result.
       Don't build the candidate's headline claim on it.
7. **The arbiter is realised CPL, a decision-timing objective — not accuracy.**
   WFCV log-loss is descriptive colour, not the gate. Verbatim, from
   `docs/CONVENTIONS.md` (the cautionary example — read it as written, don't
   paraphrase it away):

   > #250 (boundary fix) and #254 (regime cycle-length denominator) both showed flat
   > WFCV log-loss. #250 was realised-positive (saving 3.04% → 3.37%) and would have
   > been wrongly binned on the screen; #254's τ-sweep showed the apparent realised
   > "win" was an operating-point artifact and the feature economically inert (fold
   > 7 — where the denominators diverge most in value — had the *lowest*
   > decision-disagreement, 1.3%). A single proxy promoted to a hard reject gate
   > fails for any feature class whose value is orthogonal to the proxy.

   `cycle_mean_length` is the standing example of a feature that looked more accurate
   and produced worse/flat realised CPL (`project_cycle_length_accuracy_not_objective`
   — the denominator turned out to be economically inert because the model already
   uses the drift-clock). A candidate whose `PREDICTED_SIGNATURE` is phrased in
   accuracy terms ("more precisely estimates X") without a decision-timing story is
   weaker than one phrased in terms of *when the model should act differently*.
8. **`PREDICTED_SIGNATURE` cannot make a claim about SHAP structure — `facts.json`
   carries no SHAP field, and the dossier session grades that field alone (`facts.json`
   is "the only source of numbers you are allowed to use", `docs/routines/dossier.md`
   Step 1).** Decided `fps-1l1`: capturing per-column SHAP importance/sign routinely
   was rejected, not deferred. Two distinct fit paths run per candidate, and neither
   persists a model to disk: the **WFCV screen** (`runner.py`'s `_run_wfcv_screen`)
   fits raw LGBM only, 5 seeds × ~15 outer folds × 2 arms, purely for log-loss; the
   **realised arbiter** (`experiments/lib/realised.py`'s `_train_calibrate_select_tau`)
   runs on a single seed but fits a full calibrated pipeline per (fold, arm) — a raw
   fit plus an inner walk-forward OOF sweep (`fuel_signal/calibrate.py`'s
   `pool_oof_predictions`) to fit the isotonic calibrator and pick τ, so each
   (fold, arm) cell is itself several fits. Every one of those fitted pipelines is
   discarded once its predictions are pooled into `rowpreds.parquet`/`fills.parquet`;
   nothing survives for a post-hoc SHAP pass to run against. Adding it for real would
   mean either a second fit pass purely to explain the first, or teaching
   `realised.py`/`runner.py` to persist a fitted pipeline per (fold, arm, seed) — a
   change to the fit path itself, not to `dossier_tables.py`'s post-hoc "read
   committed artifacts, no judgement" boundary. A `stickiness_phase_saddle`
   (`fps-6yi`, batch1) prediction phrased as "the SIGNED and ABSOLUTE columns disagree
   in orientation somewhere in the SHAP surface" could only be graded `inconclusive`
   for exactly this reason. Phrase a mechanism prediction in terms of `decision_flips`,
   `per_fold`, or `per_axis` data instead — all of which `facts.json` already carries
   and the dossier session can actually check.

## Rules for candidates

Every candidate must:

- State a **mechanism** in price-formation terms — not "the tree might like it."
- Say **why an existing feature doesn't already carry it**.
- **Engage with prior art** if adjacent to anything in the ledger (`PRIOR_ART` field;
  see format below) — cite the rejected candidate and name what differs mechanically.
- Declare `TARGET`, `PREDICTED_SIGNATURE`, `CONFIDENCE_EFFECT`, `CONFIDENCE_ZONE` (see
  "Two CONFIDENCE fields" below).
- Declare a **mechanism family** (a short label, e.g. "wholesale-lead",
  "competition-density", "cycle-shape"). This is a **disclosure, not a gate** — see
  below.

### The honest detection bar (fps-cds, resolved via fps-aay's batch1 freeze)

This section was blank until 2026-08-22 — the bar wasn't knowable before `fps-awz`'s
placebo-column ruler was rebuilt and a real batch was frozen against it (`fps-aay`).

`batch1`'s own noise floor puts the family-wise gate (`docs/CONVENTIONS.md` § The gate
is distance from the band) at `effect_delta_cpl_held ≤ -0.27 c/L` for a batch this size
(5 candidates) to read as surprising at the BATCH level, not just individually — a
single candidate judged alone needs only `≤ -0.17 c/L`.

**Updated 2026-08-24 (`fps-3jj.14`), from `-0.22` / `-0.15`.** batch1's ruler was
re-cut at arity 3 to match its own candidates' column counts, and the bar hardened. If
you are reading a batch1 candidate file written before that date, its `CONFIDENCE`
language was calibrated against the older, easier bar.

**This is a DETECTION threshold, not a target.** `experiments/results.csv`'s own
history shows single-column features landing between 0.03 and 0.26 c/L — a bar above
that range describes what this instrument can currently resolve, not what a good
feature looks like. Do not propose only candidates you expect to clear -0.27 c/L on
their own, and do not read a candidate that lands below the bar as a failed proposal —
group candidates in particular (§ "A candidate is one MECHANISM" below) routinely beat
their own noise band as a group even when no single column in it would. The bar is
useful for calibrating confidence language (`CONFIDENCE_EFFECT`), not for pre-filtering
ideas.

**The number itself is `batch1`-specific**, not a constant — it depends on that batch's
own draws, columns and cadence (`docs/CONVENTIONS.md` § A re-lock invalidates the
batch's noise floor). Check the current batch's own `noise_floor.json` before citing a
number here; don't carry `-0.27` forward into a later batch by habit. It also depends
on the batch's **arity** — see `docs/CONVENTIONS.md` § The band's ARITY must be at least
the candidate's — so a batch whose candidates are shaped differently gets a different
bar even on the same data. It no longer depends on the batch's *draw count* as a side
effect of arity, which is what used to make that dependence so much larger than the
arity effect itself (`fps-3jj.21`).

### A candidate is one MECHANISM — it may be one column or a group of related columns

**Both shapes are explicitly fine. Do not constrain yourself to one column per
candidate.** `COLUMNS` is a list. A candidate that expresses its mechanism as 2–4
related columns is as legitimate a proposal as a single column, and should be filed as
one candidate with one `HYPOTHESIS`, one `TARGET` and one pair of `CONFIDENCE` fields —
because it is one idea.

**The noise band does not constrain this, and must not** (`fps-3jj.25`). There was briefly a
cap: the ruler could only grade up to a fixed arity, so this section was rewritten to tell you
to propose fewer columns. That was backwards. Constraining candidate shape to suit the
instrument distorts the very thing the batch is measuring — and it constrains the one shape of
feature work that has actually succeeded here.

The instrument was changed instead. Two defects, both now fixed:

- **Draw count used to collapse as arity rose.** The placebo bank needed `arity` distinct source
  columns *per draw* out of 54, so a wider candidate got a thinner band — and a thinner band
  means a much harder bar, because `family_wise_z_threshold` carries `df = n_draws − 1`. A
  4-column candidate's bar was **0.04 c/L harder** than a 3-column batch-mate's purely from band
  thinness, and nothing said so. Draw count no longer depends on arity at all (`fps-3jj.21`).
- **What reuse remains is now PRICED, not forbidden.** Source-column reuse degrades a bank
  *gradually*, so `placebo.effective_n_draws` feeds the effective draw count into the threshold.
  A wider candidate gets a wider, harder bar — automatically, in proportion — instead of no
  verdict (`fps-3jj.25`).

Concretely, on batch1's band, the single-candidate bar by candidate width: 1 column **−0.152**
c/L, 10 columns **−0.171**, 35 columns **−0.287**. Harder as it widens, which is correct, and
*usable* at every width. Before `fps-3jj.21` the same progression was −0.22, −0.78, and **no
band at all** above 17 columns.

**So propose the honest shape and let the bar adjust.** The one number still resting on an
estimate is the texture ICC behind the pricing — bounded at ≤ 0.391 by an underpowered ANOVA and
measured directly in bd `fps-3jj.23`. It is an upper bound on a *coarser* quantity (grouped by
texture family, not by column), so it is **not** established as erring on the safe side — if the
true value is higher, bars on wide candidates are currently too easy. Either way every bar moves
when `fps-3jj.23` lands.

### The one hard limit, and how to check it

It is physical, not statistical: a placebo draw needs **one distinct source column per candidate
column**, and each of those has to survive screening. So the ceiling is the count of *usable*
baseline columns — **not** the nominal 54. On batch1 it is **49**; five
`days_since_trough_entry_<lga>` columns are all-NaN (no station was ever classified into that
LGA) and are skipped as unverifiable.

**Do not read that number off `baseline_columns.json`.** It is a flat list of the 54 declared
names and carries no NaN information, so it will tell you 54 and you will believe you have
headroom you do not have. `screen_draw_groups` only guards against exceeding the *nominal*
count, so a candidate sitting between the two numbers passes that guard, burns a night of
compute, and then cannot have a floor built — the no-verdict failure this whole change exists to
eliminate.

The usable count lives in the frozen frame. To get it for a batch:

```bash
PYTHONPATH=. uv run python -c "
import json, pathlib, sys
from fuel_signal.features import load_features
b = pathlib.Path(sys.argv[1])
if not (b / 'features.parquet').exists():
    raise SystemExit(
        f'{b}/features.parquet is not in this checkout. The frozen frame is gitignored '
        '(.gitignore: experiments/**/*.parquet), so it exists only in the clone that ran '
        'the freeze — not in a fresh clone or a git worktree. Re-run this from that clone.'
    )
f = load_features(b / 'features.csv')  # load_features reads the .parquet sibling
cols = json.loads((b / 'baseline_columns.json').read_text())
usable = [c for c in cols if f[c].notna().any()]
print(f'{len(usable)} usable of {len(cols)} declared')
" experiments/batches/batch1
```

The guard is there because the frozen frame is **gitignored**. Without it the command dies on
`FileNotFoundError: .../features.csv` — a file no writer ever creates, since `load_features`
swaps in the `.parquet` sibling — which says nothing about what is actually missing and leaves
you back at 54 as the only number in reach.

In practice this only binds on a candidate with **dozens** of columns, and no proposal has come
close. If you are within a few columns of the usable count, say so in the candidate's
`PRIOR_ART` rather than guessing — it is a fact about the batch, and the owner would rather
know before a night of compute than after.

**Why this is stated so plainly:** every feature win in this project's history arrived as
a *group*, not a lone column. The 35 LGA event features went in together; the 4
`NETWORK_FEATURE_COLUMNS` went in together and moved realised CPL by ~1 c/L. The only
feature ever measured alone on the paired realised arbiter (`tgp_delta_7d`) came in at
+0.006 c/L. An earlier reading of this file implied one column per candidate; that would
have cut out the only shape of feature work that has actually worked here.

Pick the shape the mechanism needs. If the story is "the network is converging and here
is how you can tell", and telling it honestly takes a level, a velocity and a dispersion
term, propose all three as one candidate. If the story is one number, propose one number.
Padding a single idea out to three columns to look substantial is the failure mode in the
other direction — the test is whether removing a column breaks the *mechanism*, not
whether it lowers the score.

**A candidate is a proposal, not a commitment.** Iterating on a candidate — or discarding
it outright — after its run is normal and expected; a group that wins can be ablated
afterwards to find which column carried it, as its own follow-up. That is a reason to
propose the honest version of an idea rather than the safest one.

## Diversity: |rho| is a hard gate, mechanism family is a disclosure

**Decided 2026-08-17.** The original rule required both "≥3 mechanism families" and
"pairwise |rho| below a threshold." These fail in opposite directions and neither
subsumes the other:

- **|rho| catches reparameterisation** — two candidates ~0.95 correlated waste a
  slot. Mechanical, checkable, free.
- **Mechanism family catches what rho can't** — several statistically uncorrelated
  candidates that all encode the same underlying story ("the network is converging
  on a trough"). Reads as diversity, isn't.

But only rho is actually checkable — "mechanism family" has no definition that isn't
the generator's own say-so, and a generator that wants five slots will assert three
families. So:

- **Pairwise |rho| is a HARD GATE — across candidates.** A batch cannot be filed with a
  *cross-candidate* pair above the threshold (see Redundancy checks — same mechanism,
  same run). Correlation between columns *inside* one multi-column candidate is
  disclosed, not gated.
- **Mechanism family is a REQUIRED DISCLOSURE, not a threshold.** Name it per
  candidate; it is stored in the batch record. If all five land in one family, that
  is visible in the dossier and the retrospective and becomes a *finding about the
  generator*, not a gate it learned to game. Don't retroactively relabel a family to
  hit a count — the honest label is the useful one.

## Redundancy checks — run here, not at launch

**Decided 2026-08-17.** Both free mechanical checks run in *this* session, before any
bead is filed, against **live `data/features.parquet`** (NOT the frozen batch
snapshot — a redundancy screen doesn't depend on which day's data it uses; a column
0.95-correlated with `tgp_delta_7d` on Monday is 0.95-correlated on Thursday):

**Run them with the module, not by hand** (`fps-3jj.15`) — the threshold is a declared
constant rather than a per-session judgement call, and the output is the batch record's
raw material:

```bash
PYTHONPATH=. uv run python -m experiments.pipeline.redundancy \
  experiments/candidates/<batch>/*.py
```

It exits non-zero if the batch fails, and validates each candidate (including the
`TARGET_COLUMNS` label guard) before correlating anything, so a candidate reaching for
the target is caught here rather than five nights later.

1. **Pairwise |rho| ACROSS candidates.** Compute each candidate's column(s) via its
   `add_columns` against live data, then the pairwise correlation matrix. Reject/redesign
   any *cross-candidate* pair above threshold before filing either.

   **The threshold is `|rho| >= 0.85`** (`redundancy.PAIRWISE_RHO_THRESHOLD`). Set from
   the locked features' own correlation structure, not by feel: 25% of the 54 locked
   columns' pairs already sit at |rho| >= 0.7 and the highest is 0.971
   (`cycle_pct_through` vs `cycle_days_since_peak` — the same quantity in two units, both
   locked), so a 0.7 gate would reject candidates for being no more similar than a
   quarter of production. Only ~1% of locked pairs reach 0.85. A false reject costs one
   redesign in this session; a false accept costs a night of a five-night batch.

   **Two candidates may not declare the same column NAME.** The screen refuses the batch
   (`DuplicateCandidateColumn`) rather than gating it, because a shared name cannot be
   attributed to a candidate: the pair would be silently classified within-candidate and
   skip the threshold entirely, even at |rho| = 1.0. Remediation is to rename one of
   them and re-run — nothing downstream catches this, since the runner only ever sees one
   candidate at a time.

   **Correlation WITHIN a multi-column candidate is not gated.** A group whose members
   are related is the normal case — that is usually what makes them one mechanism rather
   than three — and gating it would ban the group shape by the back door. Report the
   within-candidate correlations in the batch record as a disclosure; only cross-candidate
   pairs are a hard gate. Two *different* candidates that are 0.95 correlated waste a
   slot; two columns inside one candidate do not.
2. **R² of each candidate regressed on the existing column set.** A candidate that's
   90% reconstructible from `LOCKED_FEATURE_COLUMNS` — the 54 columns the R0 arm actually
   trains on, single-sourced from the same symbol `batch_freeze` writes into a batch's
   `baseline_columns.json` — should be reworked now, not filed, queued, and disqualified
   five nights later.

   **The predictor set is the lock and nothing else.** Computed-but-non-model columns are
   deliberately excluded: the Phase 4b brand troughs, and `tgp_delta_7d` (registered
   `NON_MODEL_COLUMNS`, `evaluated-inconclusive`). Being reconstructible from a column R0
   never sees says nothing about whether a candidate is redundant with the model. An earlier
   revision of this file named `FEATURE_COLUMNS + LGA_FEATURE_COLUMNS +
   NETWORK_FEATURE_COLUMNS + TGP_FEATURE_COLUMNS` here, and the screen scored `tgp_delta_7d`
   itself at block R² = 1.000 as a result — biasing the check against the very ground bd
   `fps-x0f` exists to revisit. They remain fair game as an `INPUTS` read; they are just not
   evidence of redundancy.

   **For a multi-column candidate, regress the group as a block** (its columns jointly
   against the existing set) — that matches how it will be evaluated. A member that is
   individually reconstructible is not disqualifying if the group as a whole is not.

   **"As a block" means the MEAN of the members' R², and can only mean that.** For OLS,
   regressing several targets jointly against a shared predictor set gives coefficients
   identical to regressing them separately — so "jointly" cannot refer to the fit, only to
   how the members are aggregated afterwards. The mean is the standard redundancy index,
   and it is what makes the preceding sentence true arithmetically: one member at 0.95
   alongside two at 0.10 gives a block figure of 0.38, not a rejection. `max` would make
   that sentence false. This is reported and flagged at 0.9, never auto-rejected — only
   |rho| is a gate.

   Five locked LGA trough columns (`bayside`, `waverley`, `hunters_hill`, `botany_bay`,
   `lane_cove`) are entirely NaN on the real frame and are dropped from the predictor set
   with the drop disclosed. Do not read their absence as the screen going wrong.

Reasons these run here and not in the launch routine: pairwise rho is a **batch-level**
property — launch validation sees one candidate per night and structurally cannot
compute it. And redesign beats rejection — the whole value of "checked before any bead
is filed" is that a fixable candidate gets fixed before it costs a night, not five
nights later.

Residual risk accepted: a candidate module hand-edited after filing is unchecked. Low
risk on a one-machine setup; not worth a second implementation of the same check.

## Candidate module format

One Python file per candidate, written to `experiments/candidates/<batch>/`.
(`experiments/**` is exempt from the PR rule — commit **and push** these straight to `main`; the push is part of the exemption, not a separate step to remember.)

```python
NAME = "tgp_accel_7d"
TARGET = {"axis": "regime", "expect_concentration_in": ["shock"], "folds": [4, 9]}
HYPOTHESIS = "..."              # mechanism in price-formation terms
PREDICTED_SIGNATURE = "..."     # what the numbers should look like if the mechanism is real
CONFIDENCE_EFFECT = 0.35        # P(pooled realised CPL delta < 0) — does it move the arbiter at all?
CONFIDENCE_ZONE = 0.20          # P(the effect concentrates where TARGET says) — scored only if EFFECT resolves true
MECHANISM_FAMILY = "wholesale-lead"   # disclosure, not a gate — see Diversity above
COLUMNS = ["tgp_accel_7d"]       # one column, or several — see "A candidate is one MECHANISM"
INPUTS = ["tgp_delta_7d", "price_date"]   # declared reads -> restricted-frame check + correlation/R^2
PRIOR_ART = "adjacent to station_minus_tgp_cents (rejected, ledger) — differs in ..."

def add_columns(df): ...   # pure fn of the features frame; no DB, no network
def add_axis(df): ...      # OPTIONAL — per-row group labels; omit to use standard fold/regime cuts
```

Full field definitions and scoring rules for the two `CONFIDENCE` fields live in
`fps-3jj.4` (module format / runner); this file states only the generator-side
obligations.

## Two CONFIDENCE fields (decided 2026-08-17)

State two priors per candidate, not one:

- **`CONFIDENCE_EFFECT`** — P(pooled realised CPL delta < 0). "Does it move the
  arbiter at all?"
- **`CONFIDENCE_ZONE`** — P(the effect concentrates where `TARGET` says). Scored only
  on candidates where `CONFIDENCE_EFFECT` resolved true. **Gradeable on an identified
  quantity only when `TARGET` names folds or `axis: "regime"`** — a row-level
  `add_axis` grade is a path-coupled cost allocated to a sub-period and the runner
  marks it as such (see the fold/regime taxonomy above). Prefer a fold-expressed zone
  claim if you want this prior to actually mean something in the calibration record.

Both are scored against outcomes in the batch retrospective (`fps-3jj.8`) — this is
the payoff artifact for aim (b) (gaining experience with an AI-sourced pipeline), so
**spreading these numbers is the point**. A batch where every candidate is rated 0.5
produces no calibration signal at all. Don't hedge to the middle; state a real prior
and be gradeable on it.

`TARGET` carries the machine-resolvable zone claim; `PREDICTED_SIGNATURE` stays prose
for the morning reader and for the ledger's `falsified` line if the candidate is
rejected. There is no separate structured `PREDICTIONS` companion — `TARGET` already
does that job.

## Calibration accumulates across batches

Five candidates cannot calibrate anything on their own. The retrospective (`fps-3jj.8`)
appends confidences and outcomes to a running record and reads calibration across
batches, not within one. Don't expect — or claim — a confident calibration read from
`batch1` alone.

## Batch sizing

**Batches are numbered by their directory name**, and the directory name is the batch
name — `batch0`, `batch1`, `batch2`. Nothing in the code constrains this
(`parse_candidate_ref` takes free-form paths), so the only thing keeping it coherent is
using one scheme everywhere. An earlier revision of this file numbered batches from 1
while the directories numbered from 0; if you meet "batch 2 = the first AI-sourced set"
in an old note or bead title, it means `batch1` here.

- **`batch0` = 1 candidate** — `tgp_delta_7d` as a known graduate, to check the pipeline
  gives the right answer before trusting it on anything new. **Done 2026-08-19.**
- **`batch1` = 5 candidates** — the first genuinely AI-sourced set. Small enough that a
  runner bug costs 5 nights, not 10; re-freezes onto fresher data sooner.
- **`batch2`+ = 10–15.**

**Ordering constraint — DISCHARGED 2026-08-21. It no longer gates this session.**

This file used to require **`batch0` → finish the TGP re-lock (`fps-1785999729707-1` /
gh#271) → `batch1`**, so that the known-graduate check could not be run against a
baseline already containing `tgp_delta_7d` (R0 and the candidate arm would be identical
and `batch0` would test nothing). Both halves are now settled and the gate is void:

- **`batch0` ran and reached a verdict** on 2026-08-19. That verdict was the actual
  gate, and it held.
- **The TGP re-lock will never happen.** #271 is closed as superseded and no TGP feature
  is in the lock, so there is no proven-but-missing feature for the next batch to be
  unfairly judged against — which was the entire reason for the ordering. Successor is
  bd `fps-x0f`.

Nothing now orders the next batch behind anything else in the TGP thread. **If you are
reading a copy of this file that still states that order as a precondition, it is stale
— proceed.**

## Filing

**Create the batch's parent bead FIRST** (decided 2026-08-23, batch1). Every batch gets
one tracking bead — type `epic`, parented to `fps-3jj`, labelled `design` and **never**
`experiment` (a bead carrying that label enters the launch queue, and the parent has no
`Batch:`/`Module:` lines to parse, so the routine would claim it and abort it on the
first night). The candidate beads are its children, and so are the batch-level tasks that
only become possible once the queue drains. `fps-3jj.18` is the worked example; it holds
the freeze parameters, the batch's own detection bar, the candidate table in launch order,
and the expected-yield statement written *before* any result existed.

Two children go in at filing time, not later, because neither has a scheduled task and
both are easy to forget once the nightly runs start producing dossiers:

- **Run the batch retrospective** (`fps-3jj.18.1` is the template) — blocked by every
  candidate bead. This is the aim-(b) payoff artifact and the only place the leaderboard,
  the family-wise gate, the confidence calibration and the mechanism-family concentration
  get written down.
- **Act on the outcome** (`fps-3jj.18.2`) — blocked by the retrospective. Graduate,
  iterate, or record as mapped ground; a graduation is a separate `features.py` PR, never
  an inline edit.

Per-candidate *dossier* runs get no bead: the dossier routine writes those unsupervised.

Then, for each candidate that clears the diversity gate and the redundancy checks:

1. Write `experiments/candidates/<batch>/<NAME>.py` in the format above; commit
   **and `git push`** straight to `main` (`experiments/**` is PR-exempt). The launch
   routine claims these candidates from a bead queue that any clone can serve — a
   module still sitting on one machine's local `main` is a candidate the launcher
   cannot run.
2. `bd create` one issue per candidate, labelled `experiment`, `--parent` the batch bead
   — the launch routine
   (fps-3jj.5, `experiments/pipeline/launch.py`) queries `bd ready --label experiment`
   and is structurally invisible to the chore/polish worker without this label.
   Description must carry `HYPOTHESIS`, `TARGET`, `PREDICTED_SIGNATURE`, both
   `CONFIDENCE` fields, `MECHANISM_FAMILY`, `PRIOR_ART`, and **exactly** two
   machine-parsed lines (`parse_candidate_ref()` in `launch.py` regex-matches these,
   line-anchored — any other shape fails to parse and the bead gets immediately
   aborted the first time the launch routine claims it):
   ```text
   Batch: experiments/batches/<batch>
   Module: experiments/candidates/<batch>/<NAME>.py
   ```
   Both paths must resolve inside `experiments/` (`<batch>` matching the batch dir
   this candidate's module was just written under is what makes that true).

   **Creation order is run order.** `launch.py` claims with
   `bd ready --label experiment --unassigned --sort oldest -n 1`, so the sequence you
   `bd create` in is the sequence the nights run in. File highest-conviction first, so a
   runner bug or a genuine surprise lands on the candidate that matters most rather than
   on the one you cared least about.

   Also record each candidate's **column count** in its description — `fps-3jj.14`'s
   arity calibration is sized from the batch's real arities, and reading them back off
   the modules later is strictly worse than having the generator state them.
3. `bd dolt push` after filing the batch.
4. **Confirm the batch actually left this machine**: `git status --short` empty (the
   `batch.md`/`batch.json` the redundancy screen wrote, and any `bd`-written
   `.beads/interactions.jsonl` row, are both committed) and
   `git log --oneline origin/main..main` empty (everything pushed). `bd dolt push` in
   step 3 syncs the Dolt database only — it is a different system and never touches git,
   so a green `bd dolt push` says nothing about whether the candidate modules landed.
   That split is the exact shape of the failure: beads visible to the launch routine,
   modules still on one laptop. See docs/CONVENTIONS.md § the exemption.

## Batch record

**Written for you** by the redundancy screen — `batch.md` and `batch.json` land beside the
candidate modules in `experiments/candidates/<batch>/` on every run (pass
`--no-write-record` to suppress). Don't hand-write one; commit what the screen produced.

It holds every candidate's `MECHANISM_FAMILY` (so a same-family batch is visible per the
Diversity decision above — the record calls it out explicitly when every candidate shares
one label, and marks `NOT DECLARED` when a module omits the field), the pairwise |rho|
matrix split into cross-candidate (the gated ones) and within-candidate (disclosure only),
and each candidate's block R² against the existing column set. This is what makes "the
generator produced five variants of one idea" legible later, rather than lost.

**Two formats on purpose.** `batch.json` is the machine-readable half — prose is not a
data format, and the question this record exists to answer later is a group-by over
mechanism families, not a close reading. `batch.md` is the same numbers for a human.

Note that **nothing reads `batch.json` automatically yet.** Family concentration reaches
the batch retrospective by a different route (`MECHANISM_FAMILY` -> `results.json` ->
`facts.json` -> leaderboard rows, `fps-3jj.16`). `batch.json` is the durable evidence
artifact for the screen itself — the pairwise matrix and block R² that justified filing,
which live nowhere else — and the format a cross-batch reader would use. Don't go hunting
for a consumer; there isn't one.

**Written at SCREEN time, not at filing time**, because these are the numbers that
justified filing. A record regenerated afterwards would measure whatever the modules say
*then* — which is exactly the residual risk this file already accepts when it notes that a
module hand-edited after filing goes unchecked.
