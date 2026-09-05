# Noise floor at the 410-station grading population

- **Date:** 2026-09-06
- **Branch:** main (run launched from the main checkout)
- **SHA:** 091def7 (stamped into `noise_floor_n410.json.git_sha`)
- **Status:** done
- **Bead:** `fps-ajs` (blocked on, and unblocked by, `fps-916`)

## Hypothesis

`fps-nas` decided criterion 5 — grade candidates on the broad 410-station Sydney
population, report the five commute stations — and logged one thing as `not_tested`:
**whether the placebo band narrows with universe width the way the fold-clustered sd did**
(exactly 1.999x, 0.7982 → 0.3992, for 102x the decisions). That assumption is load-bearing.
If the band narrows ~2x, `tgp_cycle_displacement`'s broad delta of −0.1292 scores ≈ −3.1 and
clears more convincingly than the five-station figure ever did. If it does not narrow,
criterion 5 made grading strictly harder for nothing.

## Setup

```bash
PYTHONPATH=. uv run python -m experiments.pipeline.noise_floor batch1 \
    --n-stations 410 --n-draws 40 --out-name noise_floor_n410.json
```

11.8h (42572s), 40 draws x 14 folds, written to
`experiments/batches/batch1/noise_floor_n410.json` beside the existing ruler, never over it.

Analysis (this dir):

```bash
PYTHONPATH=. uv run python experiments/2026-09-06_noise_floor_n410/analyse.py
```

Three committed banks are compared. All three share `baseline_fingerprint`
(`54:1a6ec2d84a69`), `tank_params` (`50/3.571/1d/10%`), `null_method` (`placebo_column`),
and all 14 windows with `partial: false` — **population and arity are the only axes that
differ**, which is what makes the comparison readable at all.

| bank | population | arity | draws | mean | sd |
|---|---:|---:|---:|---:|---:|
| `noise_floor.json` (canonical ruler) | 5 | 3 | 10 | +0.0251 | 0.0999 |
| `noise_floor_k1.json` | 5 | 1 | 20 | −0.0089 | 0.0800 |
| `noise_floor_n410.json` (new) | 410 | 1 | 40 | **+0.0561** | **0.0556** |

Draw hygiene of the new bank is clean: 40 draws, 40 distinct source columns, 40 distinct
block seeds, `|self_correlation|` max 0.4695 against a 0.60 cap, so
`effective_n_draws = 40.0 = ` nominal. No reuse to price.

**The bead nominated the wrong comparator.** Its acceptance criterion says to compare against
"the five-station band's 0.0999" — but that is the **arity-3** bank, and the run defaults to
arity 1 (`DEFAULT_PLACEBO_ARITY`, and no `--arity` was passed). 0.0999 → 0.0556 is 1.796x
with arity and width moving together. The only clean comparison is `noise_floor_k1.json`
→ `noise_floor_n410.json`, both arity 1.

## Results

### 1. The band narrows — 1.44x, not 2x — and the run cannot tell 1x from 2x

Same arity, k=1: sd **0.0800 → 0.0556**, a ratio of **1.437x**.

F = 2.066 on df (19, 39), two-sided **p = 0.055**. 95% CI on the sd ratio: **[0.993, 2.198]**.

That interval contains the assumed 2.00x **and** contains 1.00x (no narrowing at all). So
the point estimate is short of the assumption, and 40 draws against 20 do not have the
resolution to reject either end. **The headline answer to the bead's question is "probably,
by less than assumed, and not decisively."** Pairing on source column (20 columns both k=1
banks drew) gives 1.379x, consistent with the full-bank figure.

Note the shape of the claim: this is a variance ratio estimated from 20 and 40 draws, and
`relSE(sd)` is 0.162 and 0.113 respectively. Buying a decisive answer means more draws on
*both* sides, not more stations — the five-station bank is now the noisier half.

### 2. The unpredicted finding: at 410 stations the placebo null is not centred on zero

| bank | mean | t(mean = 0) | p |
|---|---:|---:|---:|
| 5-stn k=3 | +0.0251 | +0.79 | 0.45 |
| 5-stn k=1 | −0.0089 | −0.50 | 0.62 |
| **410-stn k=1** | **+0.0561** | **+6.37** | **1.6e-07** |

At five stations the placebo band sits on zero, as a null should. At 410 it sits a full
band-sd above zero: a **meaningless column costs +0.056 c/L** on the broad population.

This is not sampling noise and not an artifact of which columns were drawn:

- Paired on the 20 source columns both k=1 banks used, the shift is **+0.0676 c/L,
  t = +2.66, p = 0.015**. Column texture is held; only the population changes.
- Splitting the 410 bank into columns `noise_floor_k1.json` also drew (mean +0.0588) versus
  columns only the 410 run drew (mean +0.0534): **Welch t = +0.30, p = 0.77**. The shift is
  in both halves.

Mechanistically this is what you would expect once you say it out loud — a noise column
perturbs which station the strategy picks, and picking wrongly among 410 stations costs more
than picking wrongly among five, where there is less price dispersion to lose. Worth stating
that this is a *post hoc* reading of a result that was not predicted; it is not tested here.

**The grading path already handles it.** `_score_bank` computes
`z = (delta − band_mean) / band_std`, so the mean is subtracted, not assumed away. Nothing is
broken. But the **bar in c/L** moves a long way:

| | mean | sd | z_gate | bar (c/L) |
|---|---:|---:|---:|---:|
| 5-stn k=1 | −0.0089 | 0.0800 | 1.7718 | −0.1506 |
| 410-stn k=1 | +0.0561 | 0.0556 | 1.7058 | **−0.0388** |

**+0.1118 c/L easier.** `bar = mean − z·sd` is additive, so this split is exact and does not
depend on the order taken: the **mean shift supplies +0.0650** and the **sd narrowing +0.0468**.
For the bar, the mean shift is the larger effect, and it was nobody's hypothesis.

### 3. `tgp_cycle_displacement` re-graded — right answer, wrong mechanism

Not a dossier-legal grade (see §4); reported as a property of the ruler.

| | Δ | z | bar | verdict |
|---|---:|---:|---:|---|
| five vs 5-stn k=3 (as published) | −0.2077 | −2.330 | 1.923 | clears |
| broad vs 5-stn k=3 (`fps-nas`, wrong ruler) | −0.1292 | −1.544 | 1.923 | fails |
| **broad vs 410-stn k=1 (right width)** | **−0.1292** | **−3.330** | **1.706** | **clears** |

`fps-nas` predicted ≈ −3.1 and the measured figure is **−3.330** — but a material part of
that comes from the mean shift it did not predict, not from the narrowing it argued.
Decomposing from the broad delta graded on the five-station k=1 band (z = −1.504) to its
grade on the 410 band (z = −3.330), a total of **+1.826 of |z|**:

| order | narrowing | mean shift |
|---|---:|---:|
| sd first | +0.658 | +1.168 |
| mean first | +1.013 | +0.812 |

**z is a ratio, so this decomposition is order-dependent and neither term can be called
dominant.** Averaging the two orders (Shapley) gives ≈ +0.84 to narrowing and ≈ +0.99 to the
mean shift — comparable contributions. The defensible claim is the weaker one: **roughly half
the improvement comes from an effect that was not in the hypothesis.** The verdict flip
`fps-nas` identified is real and resolves in the direction it hoped — the broad delta clears
comfortably once graded at its own width.

(The **bar** decomposition in §2 is a different matter: `bar = mean − z·sd` is *additive*, so
the +0.0650 / +0.0468 split there is exact and order-independent. The mean shift genuinely is
the larger term for the bar; it is only for `z` that the ordering ambiguity bites.)

### 4. The bank cannot grade a single batch1 candidate

The run defaulted to **arity 1**. Every batch1 candidate is arity 2 or 3:

| candidate | arity | as committed | if re-run at 410 |
|---|---:|---|---|
| `lga_trough_propagation` | 3 | refused: `station_population` | refused: `arity` |
| `network_move_breadth` | 3 | refused: `station_population` | refused: `arity` |
| `station_descent_dynamics` | 3 | refused: `station_population` | refused: `arity` |
| `stickiness_phase_saddle` | 2 | refused: `station_population` | refused: `arity` |
| `tgp_cycle_displacement` | 2 | refused: `station_population` | refused: `arity` |

Both refusals verified by calling `_bank_admissibility` directly, not asserted from reading
the code. The `station_population` refusal is **fps-916 working exactly as designed** — the
five-station candidate runs must not be graded against a 410-station band, and now they
cannot be, silently or otherwise. That axis is healthy.

The `arity` one is the gap: even after a candidate is re-run at 410, `_noise_band` refuses a
floor whose arity is below the run's, so an arity-1 band grades nothing in this batch. The
11.8h buys the scientific answer above and a bank that is not yet a usable ruler. A grading
bank needs `--arity 2` (and 3 for the wider three) at 410.

## Conclusion

**Iterate.** The bead's question is answered — the band does narrow, by **1.44x
[0.99, 2.20]**, less than the 2.00x `fps-nas` assumed and not separable from either 1x or 2x
at these draw counts. Criterion 5 survives: grading on the broad population is not made
harder, because the broad delta clears its own band at z = −3.33.

The finding worth carrying forward is the one nobody filed a bead for: **the placebo null
acquires a +0.056 c/L positive bias at 410 stations that it does not have at five**
(t = +6.37, and t = +2.66 paired on column). It is correctly handled by the existing
`(delta − band_mean)/band_std` grading, so no dossier to date is wrong. It is not handled by
anyone's *intuition* about where the bar sits: it is the larger of the two terms moving the
410 bar (+0.0650 of +0.1118, exactly, since the bar is additive in mean and sd).

## Followups

- **Arity.** `noise_floor_n410.json` grades no batch1 candidate. A usable broad ruler needs
  `--arity 2` at minimum; ~11.8h each at 40 draws.
- **Draws, not stations.** If the 1.44x-vs-2.00x question needs settling, the binding
  constraint is now the **five-station** bank's 20 draws, not the 410 bank's 40.
- **`fps-nas`'s `not_tested` entry for band narrowing is retired** with 1.437x [0.993, 2.198].
- The mean-shift mechanism (noise-driven misallocation costs more across 410 stations than
  across 5) is a *post hoc* reading and is `not_tested`.
