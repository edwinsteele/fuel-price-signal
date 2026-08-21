# 2026-08-21 — τ sweep vs decision cadence (bd `fps-929`)

**Verdict: the daily-cadence plateau is a SIGNAL-CONTENT limit, not an operating-point
limit. Re-picking τ buys 0.06 c/L of the 2.97 c/L available at 1-day cadence — 2% of it.
No τ re-lock is argued for. Feature work is the right lever.**

## Question

`fps-fii` (`experiments/2026-08-20_cadence_ceiling/`) found the model plateaus where the
oracle does not: going 7d→2d the model gains 1.82 c/L, then 2d→1d gains **0.03**, while
the oracle keeps improving and captured share falls monotonically (71% → 65% → 64%).

Two explanations, implying completely different work:

1. **Operating-point limit.** τ=0.25 was OOF-selected under weekly economics where
   nearly every decision is forced (67.6% of 7d fills are emergency fills). A model with
   ~12 wait-rungs instead of 1 should plausibly be choosier. Free to fix.
2. **Signal-content limit.** The features carry no more exploitable information. Fixing
   it means features.

## Method

`tau_sweep.py`. One paid walk-forward fit at 7d through the real harness (14 folds,
5 stations, seed 42, 54 columns `54:1a6ec2d84a69`), then its `baseline_cache` replayed
across 12 τ values × 3 cadences. Exact, not approximate: `_train_calibrate_select_tau`
takes no `TankParams`, and τ is only a threshold applied to already-calibrated
probabilities, so the fit is both tank- and τ-invariant and every cell is replay-only.
Total **12.8 min**, of which the single fit is **9.8** — the 36 replay cells cost
0.6–4.2 s each.

**Self-check:** every fold's own OOF-selected τ came out at exactly 0.25, the production
lock, and the 7d/τ=0.25 cell reproduces `fps-fii`'s 189.6717 to three decimals
(189.672). The replay path is faithful to the harness it borrowed the fits from.

3d is not swept — `fps-fii` established it is structurally unusable (a reachable
decide-level sits above the emergency floor yet cannot survive one interval).

**Run-dry audit is per CELL, not per cadence.** τ changes dryness: a higher τ buys less
often and can strand a tank that τ=0.25 kept wet, and `run_backtest` clamps a negative
tank to 0 and carries on rather than erroring. **All 36 cells came back with 0 dry
events**, so every number below is quotable.

## The asymmetry — read this before quoting the result

This sweep picks τ **in hindsight**, on the same rows it scores. That makes it
deliberately asymmetric evidence:

- A **flat** curve RULES OUT the operating-point explanation. If the best τ available
  with hindsight is no better than 0.25, no honest out-of-sample rule beats it either.
- A **dip** would only UPPER-BOUND a realisable gain, never establish one.

The curve is flat, so the definitive half is the half that applies.

## Result

| cadence | best τ (hindsight) | CPL at best | CPL at τ=0.25 | **gain from re-picking** | oracle | headroom |
|---|---|---|---|---|---|---|
| 7d | 0.20 | 189.663 | 189.672 | **0.009** | 188.135 | 1.54 |
| 2d | 0.30 | 187.916 | 188.001 | **0.086** | 185.163 | 2.84 |
| 1d | 0.35 | 187.759 | 187.821 | **0.062** | 184.852 | 2.97 |

The hypothesis is **directionally right and economically irrelevant**: the optimum does
move up as the grid gets finer (0.20 → 0.30 → 0.35), exactly as "a model with more
wait-rungs should be choosier" predicts. It is worth 0.06 c/L.

### The decision-flip count is the decisive line

`fps-929` asked for flip counts behind any delta under ~0.1 c/L. They invert the usual
reading:

| cadence | τ move | decisions changed | CPL moved |
|---|---|---|---|
| 7d | 0.25 → 0.20 | **8** | 0.009 c/L |
| 2d | 0.25 → 0.30 | 132 | 0.086 c/L |
| 1d | 0.25 → 0.35 | **422** | 0.062 c/L |

Normally a tiny delta off one or two flips means *noise* ([[feedback_realised_arbiter_
decision_flip_quantum]]). Here the opposite: at 1-day cadence **422 buy/wait decisions
change and the pooled economics barely moves**. The changes cancel. That is not an
underpowered measurement of a real effect — it is a real measurement of no effect, and
it is much stronger evidence than a small-flip-count null would have been.

Plateau width says the same thing. Over τ ∈ [0.15, 0.50] the model's CPL spans 1.061 c/L
at 7d, 0.461 at 2d, **0.318 at 1d**. The finer the grid, the less the operating point
matters.

### Direction matters, and the plot title says so

Flatness is **not** symmetric. Above the optimum — the choosier side, where the
hypothesis pointed — 1d is far flatter than 7d (0.21 vs 1.06 c/L at τ=0.50). *Below* the
optimum it reverses (1.01 vs 0.13 at τ=0.05): at 7d an over-permissive model is forced
to fill anyway, while at 1d it genuinely can buy constantly at bad prices. Only the
upper side answers this experiment. See `tau_sweep.png`.

## What this does and does not license

- **Does:** close out the operating-point hypothesis. The plateau is real. The binding
  constraint at fine cadence is signal, so feature work is the right lever — which is
  what `fps-3jj`'s batch1 is for.
- **Does:** leave the production lock alone. τ=0.25 is within 0.01 c/L of optimal at 7d
  and within 0.06 at 1d. **No re-lock, no `docs/STATUS.md` change.**
- **Does not:** say anything about whether to move *cadence*. That question was untouched
  by this experiment and is much larger than τ — the model is 1.85 c/L cheaper at 1d than at 7d
  (187.82 vs 189.67) regardless of τ. But it also fills 2332 times instead of 752 over
  the same window, at a 9.9 L mean fill instead of 28.8 L: roughly three times as many
  servo visits for smaller amounts. That is an operational trade the CPL column does not
  price, and moving cadence is a deliberate re-lock (`docs/CONVENTIONS.md`).
  **Addendum 2026-08-22:** that re-lock has since happened — `fps-oqz` moved the
  canonical cadence 7d → 1d, on the owner's rationale (uniformity, resolution, a choice
  every day) rather than on this experiment's economics. The 2d option recommended here
  on the hassle trade-off was considered and overridden. τ stays 0.25, as concluded.
- **Does not:** slice by zone. Window level only — per-zone economics is not identified
  (`fps-grp`). Per-fold is safe (each fold/station is an independent tank).

## Scale check, for calibration

At the then-production 7-day cadence the whole remaining headroom is 1.54 c/L on ~1300 L
of annual consumption per vehicle: **about $20/year**. (Since the 2026-08-22 re-lock the
production cadence is 1 day, where the headroom is 2.97 c/L — roughly $39/year on the
same consumption. The order of magnitude, which is the point of this section, is
unchanged.) The τ gain measured here is under
$1/year. This does not make the work pointless — `fps-3jj`'s stated primary aim is
experience with an AI-sourced pipeline, not the pump savings — but any argument that
reaches for economic urgency should be checked against this number first.

## Files

| file | what |
|---|---|
| `tau_sweep.py` | harness — one fit, 36 replay cells, per-cell dry audit |
| `plot_tau_sweep.py` → `tau_sweep.png` | CPL vs τ per cadence, and cost-of-wrong-τ |
| `tau_sweep.csv` | the 36 cells |
| `tau_sweep_per_fold.csv` | per (cadence, τ, fold) |
| `tau_fills.parquet` | every fill, for the flip counts |
| `meta.json`, `run.log` | provenance + timings |

`_dry_audit` is imported from `fps-fii`'s `model_cadence.py` rather than copied — its
correctness rests on mirroring `run_backtest`'s level arithmetic exactly, and a second
transcription is a second chance to get it wrong. That is its **second use**, this
repo's stated trigger for hoisting shared glue into `experiments/lib` — filed as a
follow-up rather than done here.
