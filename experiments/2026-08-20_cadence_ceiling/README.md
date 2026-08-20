# Resizing the oracle ceiling: is 1.66 c/L an artifact of the 7-day decision cadence?

- **Date:** 2026-08-20
- **Branch:** main
- **SHA:** 71444e3
- **Status:** done — the ceiling moves a lot; the diminishing-returns judgement it supported does not survive
- **Beads:** `fps-fii` (this issue); resizes the prize that gates `fps-x0f` step 3

## Hypothesis

The "we are near the floor, stop doing feature work" judgement rests on one
window-level number from #262:

```
always-buy      193.41 c/L
model           189.79   <- captured 3.62 (69%)
perfect oracle  188.14   <- 1.66 still available (31%)
```

But 1.66 is measured on a **7-day decision grid**, and that grid is a
`TankParams` default, not a fact about the world. With a 50 L tank burning 25 L
per weekly interval the car must fill at least every second interval, so the
strategy's whole freedom is *which of two adjacent weeks to buy in*. An oracle
allowed to check more often can exploit within-week dips the weekly grid cannot
see. Expected: the true ceiling sits well below 188.14 and the prize is larger
than 1.66 — but not automatically, because a finer grid hands the **model** extra
chances too.

## How to invoke these scripts

```bash
PYTHONPATH=. uv run python experiments/2026-08-20_cadence_ceiling/oracle_cadence.py
PYTHONPATH=. uv run python experiments/2026-08-20_cadence_ceiling/model_cadence.py
PYTHONPATH=. uv run python experiments/2026-08-20_cadence_ceiling/plot_decision_freedom.py
```

Stage 1 (`oracle_cadence.py`, ~25s) sweeps cadence on the fit-free arms only and
answers "how much bigger could the pot be?" on its own. Stage 2
(`model_cadence.py`, ~684s) adds the model arm so headroom is like-for-like.
`CADENCE_FOLDS=13,14` runs stage 2 as a ~3min smoke; a smoke writes everything to
`smoke/` so it can never be mistaken for a full run.

## Setup

- **Folds/stations/seed/columns identical to #262**: 14 walk-forward windows,
  5 `PREFERRED_STATIONS`, seed 42, isotonic, `BASELINE_COLUMNS`
  (`54:1a6ec2d84a69`), `inner_fold_params={"train_min_days":1095,...}` (the fold-1
  inner-OOF gotcha). Fold windows come from `realised._plan_folds`, not re-derived.
- **Cadences 7 / 3 / 2 / 1 day.** The issue asked for 7/3/1; 2 was added because
  3 turns out to be structurally unusable (below), and 2 is the nearest provably
  safe cadence.
- **The fit is paid once.** `_train_calibrate_select_tau` takes no `TankParams`,
  so a fold's calibrated pipeline and its OOF-selected tau are **cadence-invariant**.
  Stage 2 runs the real harness once at 7d, keeps its `RealisedResult.baseline_cache`
  (which captures each fold's `cal_pipe` + `own_tau`), and replays those same fitted
  models at the other cadences through the same `aggregate_backtest` economics the
  harness uses internally. Exact, not an approximation; ~684s instead of ~2200s.
  tau came out at **0.25 in all 14 folds** — the production lock.
- **Cross-check:** stage 1 and stage 2 compute the oracle independently and agree
  to full float precision (7d = 188.13534777651083 in both).

## The engine defect this surfaced (why 3-day is not reported)

`run_backtest` clamps a tank that would go negative to zero and carries on — its
emergency rule tests the **current** level, not the post-depletion overshoot,
despite its own comment saying it exists "to avoid running dry before next
evaluation". `run_oracle_backtest` prunes run-dry paths outright. So the oracle is
only a ceiling over **never-dry** strategies, and whether the model stays never-dry
depends on the reachable level lattice, which depends on `D = daily x interval`.

At 3-day cadence a reachable level (7.14 L) sits **above** the 10% emergency floor
yet **cannot survive** a 10.71 L interval — the model strands the tank, buys fewer
litres than it drove, and its CPL denominator is understated on exactly the paths
where waiting looked cheapest. Verified both ways:

- exact rational arithmetic over the reachable lattice: cadences **1, 2 and 7 are
  safe**; 3, 4, 5, 6 and 8-14 are not (on the default tank);
- an empirical run-dry audit replaying the engine's own level arithmetic over each
  arm's ledger: **3d has 109 dry events / 389 L, worst single event 3.571 L =
  exactly one day's burn.** 7d, 2d and 1d: zero.

3-day is therefore reported as evidence of the defect and **its headroom is not
quoted** (it comes out at 0.29 c/L, and at one-fold smoke scale it went outright
negative at -1.47 — impossible for a true ceiling). Engine fix tracked separately;
it does **not** affect the numbers below, since at 1-day cadence D = 3.571 L sits
below the 5 L floor and the gap is empty.

## Why this conclusion survives the open engine defect

The engine fix is **not** landed at the time of this run, so a reader is right to
ask how any number here can be trusted. The answer is that the defect's code path
was never executed on the rows reported, and that is checkable rather than assumed:

1. **The defect is bounded.** It fires under exactly one condition — a reachable
   decide-level in the half-open interval `[floor*size, D)`: high enough that the
   emergency rule stays quiet, too low to survive one interval. Outside that
   interval the clamp is unreachable.
2. **At 7d / 2d / 1d that interval holds no reachable level.** Enumerated in exact
   rational arithmetic over the full lattice (both the wait-chain from full and the
   one from a post-emergency half-fill): 29 reachable levels, none of them dry-
   capable. At 1d it is not even marginal — `D = 3.571 L` is *below* the 5 L floor,
   so `[5, 3.571)` is empty and the defect is unreachable for any lattice at all.
   Only 3d has a level in the gap (7.14 L), and 3d is excluded from every claim.
3. **The audit confirms it empirically.** `dry_audit.csv`: **0 dry events across
   10,150 depletion steps** (840 at 7d + 3,080 at 2d + 6,230 at 1d), both arms.
   The 3d row logging 109 events / 389 L is the positive control proving the audit
   detects the condition rather than always returning zero.
4. **The proposed fix is a no-op at these cadences.** Comparing the current
   emergency predicate (`level/size < floor_fraction`) against the proposed one
   (`level < depletion or ...`) at every reachable level: identical decision on all
   29 rows. A fixed engine returns bit-identical CPLs, because it differs from the
   current engine only on paths none of these runs took.

**The residue, stated plainly:** point 4 is *derived, not executed*. It becomes a
test the moment the fix lands, which is why that issue's merge gate is "7d must
reproduce 193.41 / 189.67 / 188.14 exactly". **If that gate ever fails, the
conclusions in this file are retracted, not patched.** Nothing here claims the
engine is correct in general — only that this specific, fully characterised defect
could not have touched these rows.

## Results

Window level, pooled over 14 folds x 5 stations. `dry_audit.csv` gates each row.

| cadence | always-buy | model | oracle | captured | **headroom** | pot | captured % | forced fills |
|---|---|---|---|---|---|---|---|---|
| 7d | 193.41 | 189.67 | 188.14 | 3.74 | **1.54** | 5.28 | 71% | 68% |
| 3d | *invalid — 109 dry events* | | | | | | | |
| 2d | 192.95 | 187.85 | 185.16 | 5.10 | **2.69** | 7.79 | 65% | 36% |
| 1d | 192.99 | 187.82 | 184.85 | 5.17 | **2.97** | 8.14 | 64% | 21% |

**The 7-day grid was hiding roughly half the prize. Headroom nearly doubles at
daily cadence — 1.54 -> 2.97 c/L.** Robust, not a pooling accident: headroom grew
in **12/14 folds**, median +1.70 c/L. The two that shrank (folds 8 and 9, the
expensive 204 c/L shock windows) are the ones where the model itself gained most
(204.58 -> 199.70 and 200.43 -> 196.07) — a finer grid rescues the model's worst
fold, where at 7d it actually **lost** to always-buy.

**The model plateaus where the oracle does not — this is the real finding.** The
model improves in **14/14 folds** going 7d -> 1d, so it does benefit. But it stops:
189.67 -> 187.85 (2d) -> **187.82 (1d)**. Three hundredths of a cent for doubling
its decision opportunities, while the oracle keeps going (185.16 -> 184.85). So
captured share falls monotonically, **71% -> 65% -> 64%**, as the model is handed
more chances. Headroom grows mainly because the oracle exploits the finer grid and
the model does not.

**The forced-fill constraint is cadence-conditional, and the forced count is
invariant.** Deferral latitude is measured in decide-points, not litres: at 7d
`D = 25 L` leaves the reachable lattice at {50, 25, 0} — one rung where "wait" is
still a choice; at 1d there are 12. Measured on the model's own ledger:

| cadence | fills | forced | **chosen** |
|---|---|---|---|
| 7d | 752 | 508 | **244** |
| 2d | 1395 | 499 | **896** |
| 1d | 2332 | 500 | **1832** |

Forced fills barely move (508 / 499 / 500) because hitting the floor is a function
of **distance driven**, which is fixed by the window. Checking more often spares
you no forced fill at all — it stacks discretionary decisions on top of an
unchanged forced baseline. See `decision_freedom.png`.

This retires an explanation that has been doing real work: "~250 real decisions
across the whole experiment, so feature effects are structurally small" is true
**at weekly cadence** and largely dissolves at daily (~1832 chosen fills, ~7x).
The ~0.05 c/L decision-flip quantum is a 7-day quantity.

**#262's 189.79 does not reproduce.** always-buy (193.41) and oracle (188.14) match
to the cent — both are price-only — but the model comes in at **189.67**, so 7d
headroom is **1.54, not 1.66**. 0.12 c/L is ~2-3 decision flips, consistent with the
features.csv data vintage moving since 2026-06-19 (the same order as the +0.045
vintage effect batch0 measured). Nothing else differs: same fingerprint, folds,
seed, tau.

## Conclusion

**Open -> the diminishing-returns judgement does not survive, but not because the
model is worse than thought.** The pot is ~2x larger than the number that
justified stopping, *and* the model already captures more absolute cents at fine
cadence (3.74 -> 5.17 c/L) than it did at coarse. What changed is which constraint
binds: at 7 days the **tank** gagged the signal, and at 1 day the **signal** is
the limit. That is a live target, not a floor.

**Leaky-ceiling caveat unchanged and load-bearing.** 2.97 c/L is perfect foresight,
so it is an upper bound on a partly-unforecastable prize — necessary, not
sufficient. **Window level only**; no zone slicing (`fps-1785999730023-4-264564ac`).

**Cadence realism:** the owner confirmed (2026-08-20) that checking daily is
behaviourally acceptable, so the 1-day row is a real operating point and not a
thought experiment. It does imply topping up ~10 L at a time rather than filling
weekly (mean fill 9.86 L at 1d vs 28.79 L at 7d).

## Followups

- **Re-pick tau for daily operation before any feature work.** tau was held at the
  production 0.25 across every cadence — correct for isolating the cadence effect,
  but it means the daily arm is the production model *operated as-is*, not tuned
  for a regime with 7x more decisions. An unknown slice of the 2.97 c/L may be
  recoverable by re-running the OOF tau selection against daily-cadence economics,
  with no new features at all. Cheapest next move by a distance.
- **Fix the engine's run-dry clamp** (separate bd issue) — make the emergency rule
  test what its comment claims (`tank_level < depletion`), which is a no-op on the
  default 7-day lattice {0, 25, 50} and makes every cadence legal. Gate before
  merge: 7d must reproduce 193.41 / 189.67 / 188.14 exactly.
- Answers the `not_tested` line left open by `2026-08-20_headroom_attribution`
  ("whether the window-level 1.66 c/L ceiling itself moves under a finer
  evaluation cadence"). It does.
