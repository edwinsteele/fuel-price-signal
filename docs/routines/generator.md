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
   - `TGP_FEATURE_COLUMNS` (`tgp_delta_7d` — **computed into `features.csv`, graduated
     the realised arbiter, but deliberately NOT yet in `FEATURE_COLUMNS`**; see
     "Ordering constraint" below).
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
   - **Window-level headroom stands, with its cadence attached.** The model is
     ~⅔ of the way from always-buy to the perfect-foresight floor: **1.54 c/L on
     the production 7-day decision grid, 2.97 c/L on a daily one**
     (`experiments/2026-08-20_cadence_ceiling/`, bd `fps-fii`). It is not a
     constant, so never quote a headroom number without its cadence.
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
6. **The fold/regime taxonomy.** Two distinct axes, don't conflate them:
   - **Fold-level**: 14-fold paired walk-forward CV; `SHOCK_FOLDS = {1, 4, 9, 13}`
     (`experiments/lib/constants.py`), everything else `normal`.
   - **Row-level cycle regime**: `experiments/lib/zones.py` `assign_regime()` over
     `cycle_pct_through` — `normal` (0.0–0.6), `late_descent` (0.6–1.0), `overdue`
     (≥1.0).
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

- **Pairwise |rho| is a HARD GATE.** A batch cannot be filed with a pair above the
  threshold (see Redundancy checks — same mechanism, same run).
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

1. **Pairwise |rho| across the batch.** Compute each candidate's column(s) via its
   `add_columns` against live data, then the full pairwise correlation matrix across
   the batch. Reject/redesign any pair above threshold before filing either.
2. **R² of each candidate regressed on the existing column set.** A candidate that's
   90% reconstructible from `FEATURE_COLUMNS + LGA_FEATURE_COLUMNS +
   NETWORK_FEATURE_COLUMNS + TGP_FEATURE_COLUMNS` should be reworked now, not filed,
   queued, and disqualified five nights later.

Reasons these run here and not in the launch routine: pairwise rho is a **batch-level**
property — launch validation sees one candidate per night and structurally cannot
compute it. And redesign beats rejection — the whole value of "checked before any bead
is filed" is that a fixable candidate gets fixed before it costs a night, not five
nights later.

Residual risk accepted: a candidate module hand-edited after filing is unchecked. Low
risk on a one-machine setup; not worth a second implementation of the same check.

## Candidate module format

One Python file per candidate, written to `experiments/candidates/<batch>/`.
(`experiments/**` is exempt from the PR rule — commit these straight to `main`.)

```python
NAME = "tgp_accel_7d"
TARGET = {"axis": "regime", "expect_concentration_in": ["shock"], "folds": [4, 9]}
HYPOTHESIS = "..."              # mechanism in price-formation terms
PREDICTED_SIGNATURE = "..."     # what the numbers should look like if the mechanism is real
CONFIDENCE_EFFECT = 0.35        # P(pooled realised CPL delta < 0) — does it move the arbiter at all?
CONFIDENCE_ZONE = 0.20          # P(the effect concentrates where TARGET says) — scored only if EFFECT resolves true
MECHANISM_FAMILY = "wholesale-lead"   # disclosure, not a gate — see Diversity above
COLUMNS = ["tgp_accel_7d"]
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
batch 2 alone.

## Batch sizing

- **Batch 1 = 1** — `tgp_delta_7d` as a known graduate, to check the pipeline gives
  the right answer before trusting it on anything new.
- **Batch 2 = 5** — the first genuinely AI-sourced set. Small enough that a runner bug
  costs 5 nights, not 10; re-freezes onto fresher data sooner.
- **Batch 3+ = 10–15.**

**Ordering constraint — do not let batch 1 and the TGP re-lock overlap.** `tgp_delta_7d`
graduated the arbiter but is deliberately held out of `FEATURE_COLUMNS` (see
`fuel_signal/features.py` comment on `TGP_FEATURE_COLUMNS`). Batch 1 uses it as the
known-graduate check; if it's already in the trained baseline by the time batch 1
runs, R0 and the candidate arm are identical and batch 1 tests nothing. Required
order: **batch 1 → finish the TGP re-lock (`fps-1785999729707-1` / gh#271) →
batch 2.** Don't invoke this session for batch 2 candidates until that order has held.

## Filing

For each candidate that clears the diversity gate and the redundancy checks:

1. Write `experiments/candidates/<batch>/<NAME>.py` in the format above; commit
   straight to `main` (`experiments/**` is PR-exempt).
2. `bd create` one issue per candidate, labelled `experiment` — the launch routine
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
3. `bd dolt push` after filing the batch.

## Batch record

Store, alongside the batch (e.g. `experiments/candidates/<batch>/batch.md` or the
batch dir's own note): every candidate's `MECHANISM_FAMILY` (so a same-family batch is
visible per the Diversity decision above), the computed pairwise |rho| matrix, and the
R² of each candidate against the existing column set. This is what makes "the
generator produced five variants of one idea" legible later, rather than lost.
