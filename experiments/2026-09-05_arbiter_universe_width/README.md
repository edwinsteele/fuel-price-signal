# Arbiter universe width — is grading on a broad Sydney sample affordable, and is it the same measurement?

- **Date:** 2026-09-05
- **SHA:** `2d0ecdf` (sampler landed in PR #361, squash `8e39b7a`)
- **Status:** complete — criteria 1, 3, 4, 5 answered; criterion 6 measured and costed, discharged by `fps-916` / `fps-ajs`
- **Bead:** `fps-nas`

## Hypothesis

The realised arbiter replays five stations because every caller leaves `station_codes=None`
and `experiments/lib/realised.py` resolves it to `PREFERRED_STATIONS`. That is **~92
independent decisions** for a whole 14-fold run (`fps-e6i`), and no feature in
`experiments/results.csv` history has ever delivered 0.5 c/L on its own — so most of what
the batch pipeline measures is graded against a ruler too coarse to read it.

`fps-nas` proposes grading on a **broad Sydney sample** while still reporting the five.
That is only worth doing if two things hold, and each is a separate measurement:

1. **It is cheap.** The bead's costing argument is structural, not measured: per fold per
   arm, `_train_calibrate_select_tau` fits a full LGBM on all 714 stations' fold-train plus
   a walk-forward OOF loop for the isotonic calibrator, and none of that scales with the
   replay universe. Only `load_history` and the `aggregate_backtest` tank walk do.
2. **It is the same measurement.** Grading Sydney-wide only serves the owner's own CPL if a
   feature that helps Sydney-wide helps at the five. The current support for that is thin
   and on the wrong statistic — `fps-e6i` measured per-row **log-loss** gain at the five as
   0.33 SE from the network-wide figure, which is not CPL homogeneity.

**Computing both CPLs is the homogeneity test.** If they track, one arbiter serves both
goals at far higher resolution. If they diverge, that is a real finding about the commute
and the two goals genuinely need separating.

Reason to doubt (1) before running it: `ModelStrategy.decide` calls `predict_proba` on
**one row** per station-day, and at the locked 1d cadence a 90-day fold is 90 decisions per
station — so a 200-station fold is 18,000 single-row sklearn calls per arm, each paying full
per-call overhead. The structural argument may not survive contact.

## How to invoke these scripts

Both need the sampler from PR #361 on the checkout, and a frozen batch with its
gitignored `fuel_signal.db` / `features.parquet` present (batch1 in the primary worktree).

```bash
PYTHONPATH=. uv run python experiments/2026-09-05_arbiter_universe_width/timing.py 2>&1 | tee experiments/2026-09-05_arbiter_universe_width/timing.log
```

Then, with a width the timing run justifies:

```bash
PYTHONPATH=. uv run python experiments/2026-09-05_arbiter_universe_width/homogeneity.py --n-stations 410 2>&1 | tee experiments/2026-09-05_arbiter_universe_width/homogeneity.log
```

`--n-stations` is required and has no default on purpose: the universe width is a costing
decision, and `homogeneity.py` must not be where it gets made by accident.

**`meta.json` is the homogeneity run's, not the timing run's.** Both scripts call
`write_meta` on this directory, so the second to run clobbers the first. Nothing is lost —
each script also writes its own `timing.json` / `homogeneity.json` carrying the identical
payload — but read those, not `meta.json`, when you want the timing provenance.

## Setup

**Sampling scheme** (`experiments/lib/universe.py`, PR #361) — acceptance criterion 2.
Stratified by **council**, proportional to each council's eligible-station count (largest
remainder), drawn from one seeded shuffle of a canonically sorted pool. Council rather than
brand or price level because the cycle propagates geographically (the whole `lga_*` feature
family exists because LGAs lead and lag each other), because it is the axis the reference
five are narrowest on (2 councils, both on the lagging edge), and because price level is
downstream of both — stratifying on something that close to the measured outcome risks
conditioning the estimate rather than describing the population.

Eligibility gates: council in `SYDNEY_METRO_COUNCILS`; `daily_prices` coverage over the span
≥ 0.90, applied **per val window** rather than over the envelope; Sticky share of classified
days ≤ 0.50. Coverage is a **correctness** gate, not tidiness — `aggregate_backtest` silently
skips a station with no data in a window and `run_backtest` clamps a dry tank, so a dark
station reweights the pooled CPL without announcing itself. On batch1's frozen DB over its
real 14-fold span (**2021-11-05 → 2025-04-17**, 1260 days) this admits **410 of 599 stations
across 29 councils** (`eligible_pool_digest` `410:4a9920301726`).

> The span above is the geometry `_plan_folds` actually produces. A hand derivation gives
> 2021-11-06 → 2024-12-29 and a pool of 607/751 across 32 councils; that is **wrong**, and an
> earlier draft of this README carried it. Build the spec from `_plan_folds`, never by hand —
> and note `sample_station_universe` now refuses a spec with no `windows`.

**Known asymmetry, carried not fixed.** Station 414 (BP Valley Heights) reads 0.742 coverage
across the span, which looks merely thin. Its **per-fold** vector is
`[0.96, 1.00, 1.00, 0.33, 0.00, 0.00, 0.10, 1.00 ×7]`: folds 5 and 6 are entirely dark and
folds 4 and 7 barely covered, so the incumbent five-station arbiter is a four-station
instrument across two of its fourteen folds — `aggregate_backtest` skips those cells
silently. `describe_universe` reports this for both populations rather than papering over it
(`five` comes back `n_eligible_of_supplied: 4`, `worst_window_coverage: 0.0`); the five stay
exogenous and are never filtered.

This is corroboration, not discovery — `project_preferred_station_outages` already recorded
folds 4-7 as "414 genuinely closed (rebuild), an honest 4-station market" and fold 1 as the
worse case (a **3-station** shock fold, 414 and 18517 both dark). Shock set is {1, 3, 4, 14};
fold 1 was already known, fold 4 is the increment.

**Coverage is not feature availability (`fps-wst`).** The gate counts `daily_prices` rows.
`fill.py` forward-fills gaps ≤ 28d and `labels.py` strips 7d before / 90d after a gap from the
feature frame, so the two disagree by construction: fold 1 passes the per-window gate at 0.96
coverage while holding **zero feature rows** for both Blue Mountains stations. A widened
universe gated only on coverage can therefore admit feature-dark stations. Prefer `fps-wst`'s
reported-field approach over bolting on a second gate.

**Staleness, stated rather than gated.** Coverage sees only >28-day darkness, so everything
shorter is a stale price the replay trades on as live (up to 27 consecutive days). Roughly
**two-thirds of every day the arbiter has ever replayed is forward-filled**, for every
population — pre-existing and unchanged by any of this work. Medians by population:

| population | `observed_fraction` median |
|---|---|
| the five | 0.3365 |
| 599 stations, span gate only | 0.3212 |
| **410 stations, per-window gate (used here)** | **0.3325** |

Medians sit within 1.02x of the five, so composition does not bite there; the tail does
(64.8% of the broad pool carries a ≥21-day filled run vs 40% of the five).

**Candidate:** `tgp_cycle_displacement` — batch1's cleanest signal, i.e. the best available
signal-to-noise for a homogeneity read.

**How the two populations are compared.** Never by differencing their CPLs: five stations and
410 stations fill different tanks on different days at different points in the cycle, so the
difference has no shared denominator (`feedback_disjoint_basket_comparison`). Each
population's delta is read against **its own fold-clustered 2·SE** — the fold is the
independent replicate, since stations inside one 90-day window share a cycle and a shock, and
sizing an interval on station-days would understate it by roughly √(universe width), which
would manufacture exactly the resolution improvement being measured. The one paired
comparison the two populations can legitimately share is per-fold, because the 14-fold grid
is identical: sign agreement and Pearson r across folds.

## Results

### Criterion 1 — cost is fit-dominated, but not for the stated reason

One fold (11), one arm, `fit` timed once because it is station-independent by construction.

| n | fit | `load_history` | replay | one arm-fold | fit share |
|---:|---:|---:|---:|---:|---:|
| 5 | 44.1s | 28.3s | 0.30s | 72.8s | 0.61 |
| 50 | 44.1s | 26.8s | 2.92s | 73.8s | 0.60 |
| 200 | 44.1s | 27.4s | 11.61s | 83.2s | 0.53 |

**The conclusion holds — 40× the stations costs +14% wall clock — but both halves of the
hypothesis's mechanism were wrong:**

- **The 18,000-call worry was physically real.** Replay is dead linear: 38.4× cost for 40×
  stations, ≈ 0.058 s/station-fold. It simply starts from 0.30s, so linear growth off that
  base never catches the fixed work.
- **`load_history` does not scale.** 28.3s at n=5 vs 27.4s at n=200, across a 40× widening.
  The hypothesis listed it as one of the two scaling terms. It is fixed per-fold overhead, so
  the constant is fit **+ history ≈ 71s**, not fit's 44s alone — and stations are charged
  only through replay.

### Width actually run: 410, the whole eligible pool

`timing.py` justified n=200 at +14%. The owner then set the compute budget explicitly — up to
5× on batch freezes and candidate runs is cheap — which removes cost as a tiebreak, so the run
used the **entire eligible pool** instead of a sample of it. The 0.90 coverage gate is a
correctness filter and not a cost one, so 410 is the ceiling at any budget rather than a
compromise; at census width the council stratification is a no-op by construction.

Measured at 410: **2024s** for 14 folds × 2 arms ≈ 72 s/arm-fold, against **1349s** (≈ 48
s/arm-fold) at five. +50% for 82× the stations.

### Criteria 3-4 — homogeneity holds

`tgp_cycle_displacement`, batch1, held-τ, 1d cadence, 14 folds, seed 42.

| | five | broad (410) |
|---|---:|---:|
| always-fill CPL | 193.4498 | 184.7827 |
| baseline held CPL | 188.0376 | 179.7224 |
| candidate held CPL | 187.8299 | 179.5932 |
| **pooled Δ** | **−0.2077 c/L** | **−0.1292 c/L** |
| per-fold mean | −0.2099 | −0.1354 |
| per-fold sd | 0.7982 | 0.3992 |
| own fold-clustered 2·SE | ±0.4267 | ±0.2134 |
| own interval | [−0.6366, +0.2168] | [−0.3488, +0.0780] |
| decisions | 83 | 8,476 |
| councils | 2 | 29 |
| wall | 1349s | 2024s |

Paired across the shared 14-fold grid: **same headline sign**, **12/14 folds agree on sign**,
**r = 0.704**, intervals overlap, and each population's point estimate falls inside the
other's interval. **The two goals are not at cross purposes.**

**Harness validation.** The five-station arm returned **−0.2077 c/L**, which is batch1's
committed headline for this candidate to the digit. Nothing below is an artifact of a new
code path.

### The resolution gain is 2×, not 10×

102× the decisions bought exactly **1.999× tighter** — the per-fold sd halved, 0.7982 →
0.3992, and the interval with it. That is the correct answer and the trap the comparison
contract above was written to avoid: the independent unit is the fold, and widening the
universe does not create folds. Sizing on station-days would have reported √102 ≈ 10× and
manufactured the very improvement under measurement.

### Neither width clears its own fold-clustered interval

`inside_own_2se` is **True for both**. As a t-statistic on 14 per-fold deltas:

| | t |
|---|---:|
| five | 0.98 |
| broad (410) | 1.27 |

Widening moved it the right way and did not arrive. This ruler disagrees with the placebo band
the dossier actually grades on, where the five-station delta scores z = −2.33 against a bar of
1.923 and **passes**. The two ask different questions — the placebo band asks *could a
meaningless column have produced this pooled number*, the fold-clustered SE asks *does the
effect repeat across folds reliably enough to expect it next time*. The candidate passes the
first at both widths and fails the second at both widths.

Recommendation recorded, not acted on here: keep the placebo band as the grading ruler (it is
the paired, common-mode-cancelling design `fps-awz` deliberately chose over the seed-swap
null) and **report the fold-clustered interval beside it** as a generalisation check. It has
never been shown in a dossier, and it does not say pass.

### Criterion 6 — the yardstick is five-station, and does not say so

Grading the broad delta against the **existing** five-station placebo band
(`noise_floor.json`: mean 0.0251, sd 0.0999, bar 1.923 at 10 draws):

| population | Δ | z vs the five-station band | verdict |
|---|---:|---:|---|
| five | −0.2077 | −2.330 | clears |
| broad (410) | −0.1292 | −1.544 | **fails** |

Same feature, same required conclusion, opposite verdict — from the yardstick alone. A
410-station measurement is inherently steadier, so both its effects and its noise are smaller;
held against the noisier five-station band it reads as unimpressive. This is very likely the
wrong ruler rather than a real verdict (if the band narrows with width as the fold sd just
did, the broad delta would score ≈ −3.1), but that must be **measured, not assumed**.

**And the mismatch would be silent.** `compute_noise_floor()` accepts `station_codes`
(`noise_floor.py:223`) and never stamps it into the payload. Every other identity axis is
written down *and refused on* by `_bank_admissibility` — `baseline_fingerprint`,
`tank_params`, `null_method`, `n_placebo_columns`, `partial`. Station population is the one
axis that is neither recorded nor checked, and the CLI does not expose it either. It is the
same defect shape as `fps-15c`'s cadence stamp, with population as the missing second axis.

**Costed.** batch1's committed floor is 10 draws / 8368s ≈ 154 arm-folds. At 410's measured
72 s/arm-fold that is ≈ **3.1h at 10 draws**, ≈ **11h at 40**. Draw count is now the binding
constraint on resolution, not station count — the bar falls 1.923 → 1.772 → 1.706 at 10 → 20
→ 40 draws and is flat beyond (1.669 at 100).

## Conclusion

**Criterion 5 — decided: grade candidates on the broad 410, report the five.** Homogeneity
holds on every axis measured (sign, 12/14 folds, r = 0.70, mutually contained point
estimates), so one arbiter serves both goals. The broad estimate is simultaneously more
precise (2× tighter) and less flattering (−0.129 vs −0.208 c/L), which makes it a stricter
bar in c/L terms — the right direction to be wrong in. The five remain the **reported**
outcome and stay exogenous: never filtered, never sampled, never gated.

**Criterion 6 — addressed here, discharged elsewhere.** The implication is now measured
(a verdict flip), costed (3-11h), and root-caused (an unstamped, unchecked identity axis).
Two beads carry the fix, in order:

- **`fps-916`** — stamp the station population into `noise_floor.json` and add it to
  `_bank_admissibility`; expose a universe option on the CLI. `experiments/pipeline/` code,
  so it needs a PR. Must land **first**, or the first wide floor is an artifact that cannot
  say what it is.
- **`fps-ajs`** — recompute the floor at 410. Blocked by `fps-916`. Writes beside the
  existing ruler (`--out-name`), never over it: every dossier to date was graded against that
  file.

**What this does not establish** (`reference_negative_results_ledger` shape — `not_tested` is
the payload):

- ~~Whether the **placebo band** narrows with universe width the way the fold-clustered sd did.
  Assumed ≈ 2× above; `not_tested`, and it is the whole of `fps-ajs`.~~
  **Retired 2026-09-06** (`fps-ajs`, `experiments/2026-09-06_noise_floor_n410`): it narrows
  **1.437× [0.993, 2.198]** at matched arity — less than the ≈2× assumed above, and the
  interval does not exclude 1×. The broad delta nonetheless clears at **z = −3.330** vs a bar
  of 1.706, near the ≈−3.1 predicted here but **by the wrong mechanism**: the 410-station
  placebo null carries a **+0.0561 c/L mean (t = +6.37, p = 1.6e-07)** that five stations does
  not. Decomposed over the +1.826 of |z| this buys, the mean shift and the narrowing contribute
  comparably (order-dependent: +1.168/+0.812 vs +0.658/+1.013) — roughly half the improvement
  comes from an effect this write-up did not hypothesise.
  Criterion 5 survives. The bank is arity 1 and grades no batch1 candidate.
- Whether homogeneity holds for **any candidate other than `tgp_cycle_displacement`**. One
  candidate, chosen as the best available signal-to-noise. `not_tested`.
- Whether the broad and five deltas genuinely **differ in magnitude** (−0.129 vs −0.208).
  Disjoint baskets, no shared denominator — deliberately `not_computed`, and no amount of
  widening makes it computable.
- Whether **feature-dark stations** (`fps-wst`) materially reweight the 410-station pooled
  CPL. The coverage gate cannot see them by construction. `not_tested`.
- The **fold-clustered t** is a generalisation check that no dossier currently reports, and it
  does not clear at either width. Whether that should gate anything is `not_decided`.

## Followups

- `fps-916`, `fps-ajs` — criterion 6, above.
- `fps-enx` — `r0_cache.joblib` is keyed only on `batch_dir`, so alternating station universes
  thrashes it. Found while writing `homogeneity.py`: the obvious implementation (call
  `run_candidate` twice with different `station_codes`) is unusable for that reason, so the
  script calls `run_paired_realised_backtest` directly instead. Now on the critical path — the
  criterion-5 decision means two populations coexist routinely.
- `fps-wst` — coverage is not feature availability.
- `fps-4z6` — the opposite direction: report the WFCV **screen** restricted to the five. The
  bead notes these are alternatives worth deciding between together; this entry answers the
  widen-the-arbiter side and does not close that one out.
