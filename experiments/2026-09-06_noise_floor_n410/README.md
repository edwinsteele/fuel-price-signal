# Noise floor at the 410-station grading population

- **Date:** 2026-09-06
- **Branch:** main (run launched from the main checkout)
- **SHA:** 091def7 (stamped into `noise_floor_n410.json.git_sha`)
- **Status:** done (phase 1) · done (phase 2, 2026-09-06)
- **Bead:** `fps-ajs` (blocked on, and unblocked by, `fps-916`)

> **Phase 2 correction, 2026-09-06.** A second bank was run at the same width and
> **arity 3** (`noise_floor_n410_k3.json`) — the first 410-station ruler that is legally
> admissible for any batch1 candidate. It overturns two things below. **§3's
> `tgp_cycle_displacement` grade (z = −3.330, "clears") does not survive correcting the
> ruler's arity: the legal grade is z = −1.695 against a bar of 1.733, and it FAILS.**
> And **§2's +0.056 c/L positive bias does not reproduce at arity 3** (+0.0298,
> p = 0.10 on effective draws). Everything in §1–§4 is correct *for an arity-1 bank*;
> read [Phase 2](#phase-2--the-arity-3-410-bank) before quoting any of it.

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
| `noise_floor_n410.json` (phase 1) | 410 | 1 | 40 | **+0.0561** | **0.0556** |
| `noise_floor_n410_k3.json` (phase 2) | 410 | 3 | 40 | +0.0298 | 0.0938 |

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

---

## Phase 2 — the arity-3 410 bank

Phase 1's bank grades **nothing** (§4): it is arity 1, every batch1 candidate is arity 2 or
3, and `_noise_band` refuses a floor narrower than the run. So a second bank was run at the
same width and **arity 3** — a floor grades any run of arity ≤ its own, so one arity-3 bank
covers all five candidates, and arity costs nothing in wall clock (the fit loop is per
*draw*, not per column, `noise_floor.py:409`).

```bash
PYTHONPATH=. uv run python -m experiments.pipeline.noise_floor batch1 \
    --n-stations 410 --n-draws 40 --arity 3 --out-name noise_floor_n410_k3.json
```

12.3h (44,316s), 40 draws × 14 folds, `partial: false`, SHA `0cdbccf`. Identity axes match
the other three banks (`baseline_fingerprint` `54:1a6ec2d84a69`, `tank_params`
`50/3.571/1d/10%`, `null_method` `placebo_column`, 14 windows) — the analysis script now
refuses to run if any of them drifts. Population digest `410:5bbff5bf61d3`, pool digest
`410:4a9920301726`, both matching phase 1.

**Draw hygiene.** 40 draws × arity 3 = 120 column-slots drawn from the 49-column pool: all
49 columns used, 2–3 times each, no column repeated inside a draw, 120 distinct block
seeds, `|self_correlation|` max 0.4701 against the 0.60 cap. The reuse is priced, not
ignored — `effective_n_draws` is **28.076** of 40 nominal, which raises the bar's `z_gate`
from 1.7058 to **1.7332**.

### 5. Two of four predictions held, and they were the arithmetic ones

The predictions were extrapolations recorded before the run so it could falsify them.

| quantity | predicted | measured | basis of the prediction |
|---|---:|---:|---|
| `effective_n_draws` | 28.08 | **28.076** | 120 slots from a 49-column pool |
| `z_gate` | 1.7332 | **1.7332** | `family_wise_z_threshold(1, n_eff)` |
| sd | 0.0695 | **0.0938** (+35%) | 0.0556 × the five-station k=1→k=3 ratio |
| mean | +0.0561 | **+0.0298** | assumed the arity-1 410 bias carries |

Both arithmetic predictions are exact. **Both statistical ones are wrong, in opposite
directions** — the band is a third wider than extrapolated *and* less biased. The
five-station arity ratio (1.249x, on 20 and 10 draws) did not transfer; at 410 the same
ratio is **1.686x [1.226, 2.319]**.

### 6. The +0.056 c/L bias does not reproduce at arity 3

Phase 1's headline was that the 410 placebo null sits well above zero. At arity 3 it does
not:

| bank | mean | t(mean = 0) | p |
|---|---:|---:|---:|
| 5-stn k=3 | +0.0251 | +0.79 | 0.45 |
| 5-stn k=1 | −0.0089 | −0.50 | 0.62 |
| 410-stn k=1 | **+0.0561** | **+6.37** | **1.6e-07** |
| **410-stn k=3** | **+0.0298** | +1.68 (on `effective_n_draws`) | **0.104** |

(+2.01 / p = 0.051 on the nominal 40; the honest figure discounts for source-column reuse.)

The 410 arity-3 mean is **indistinguishable from the five-station arity-3 mean**
(+0.0298 vs +0.0251, Welch p = 0.894) and from zero. The bank is homogeneous — its two
halves agree (+0.0294 vs +0.0303, p = 0.98) — so this is not a drifting run.

What is **not** established is that the bias is *gone*: the arity-1 and arity-3 410 means
are also not distinguishable from each other (Welch p = 0.133), because the arity-3 sd is
1.69x wider. The 2×2 interaction — arity moved the mean +0.034 at five stations and −0.026
at 410 — is **−0.060 ± 0.040, z = −1.50, p = 0.134**. Under-powered.

**The defensible statement is the narrow one: the width-driven positive bias is an
arity-1 result and has not been shown to exist at the arity that grades candidates.** The
post-hoc mechanism offered in §2 (a noise column misallocates among 410 stations, which
costs more than misallocating among 5) predicts the bias should *grow* with arity, since
three noise columns perturb the pick more than one. The point estimate moves the other way.
That is evidence against the mechanism, not proof; it remains `not_tested`.

### 7. The band narrowing is itself arity-dependent, and vanishes at arity 3

The whole of `fps-ajs` was "does the band narrow with universe width". Phase 1 answered
1.437x at arity 1. At arity 3:

| arity | 5-stn sd | 410 sd | narrowing | p | 95% CI |
|---:|---:|---:|---:|---:|---|
| 1 | 0.0800 | 0.0556 | **1.437x** | 0.055 | [0.993, 2.198] |
| 3 | 0.0999 | 0.0938 | **1.065x** | 0.726 | [0.679, 1.995] |

At the arity that actually grades candidates there is **no measurable narrowing at all**.
Both CIs contain 1.00x, so the two rows are not distinguishable from each other either —
but the direction of the point estimate matters for how phase 1's answer should be quoted:
**1.44x is an arity-1 figure, and should not be restated as "the band narrows 1.44x"
without it.**

The consequence for grading is smaller than phase 1 implied. Comparing the canonical
five-station ruler with the new one at the same arity:

| bank | mean | sd | `z_gate` | bar (c/L) |
|---|---:|---:|---:|---:|
| 5-stn k=3 (canonical) | +0.0251 | 0.0999 | 1.9226 | −0.1670 |
| **410 k=3** | +0.0298 | 0.0938 | 1.7332 | **−0.1328** |
| 410 k=1 (phase 1) | +0.0561 | 0.0556 | 1.7058 | −0.0388 |

`bar = mean − z·sd` is additive, so the +0.0342 c/L the bar moves is split exactly:

| term | c/L |
|---|---:|
| mean shift | +0.0047 |
| sd narrowing (at the five-station `z_gate`) | +0.0117 |
| **more draws** (`z_gate` 1.9226 at 10 → 1.7332 at 28.08) | **+0.0178** |

**The largest term is the draw count, not the population width.** Most of phase 1's
"+0.1118 c/L easier" was the arity-1 band's narrowness, and it is not available to a run
that grades a real (arity ≥ 2) candidate.

### 8. `tgp_cycle_displacement` fails its first legal 410-station grade

| grade | Δ | z | bar | verdict | legal? |
|---|---:|---:|---:|---|---|
| **broad vs 410 k=3** | **−0.1292** | **−1.695** | **1.733** | **FAILS** | **yes** (conservative) |
| broad vs 410 k=1 (phase 1 §3) | −0.1292 | −3.330 | 1.706 | clears | no — arity refused |
| broad vs 5-stn k=3 (`fps-nas`) | −0.1292 | −1.544 | 1.923 | fails | no — wrong width |
| five vs 5-stn k=3 (as published) | −0.2077 | −2.330 | 1.923 | clears | yes |

Predicted for this run: z ≈ −3.2. **Measured: −1.695.** The prediction inherited phase 1's
arity-1 sd; correcting the arity destroys it. The error direction was called correctly
(a positive band mean helps a negative delta) but it is worth far less than the band
widening costs.

**Two qualifications, both load-bearing, in opposite directions.**

*It is a conservative fail.* The candidate is arity 2 and the bank is arity 3. Admissible —
a floor grades any run of arity ≤ its own — but the band is wider than a matched arity-2
ruler's would be. Interpolating between the measured k=1 and k=3 sds gives ≈ 0.0747 for
k=2, at which the same delta scores **z ≈ −2.13 and would clear**. The matched ruler has
not been built, so the arity-2 answer is unknown, not negative.

*It is a very near miss.* At the stamped bar the delta needs a band sd ≤ 0.0918; the bank
has 0.0938 — **2.3% over**. Equivalently the delta would have to reach −0.1328 c/L and it
is −0.1292. A 2.3% miss on a variance estimated from 40 draws (relSE 0.113) is not a
verdict anyone should lean on.

**The honest summary is uncomfortable and should be stated plainly: at broad width this
candidate's verdict is currently decided by the arity of the ruler, over a range
(z = −3.33 at arity 1, −1.70 at arity 3) far wider than the width question phase 1 was
built to settle.** Arity is not a free parameter — `_noise_band` fixes it to ≥ the run's —
but nothing in the pipeline yet pins it to *equal* the run's, and the two admissible
choices here (2 and 3) straddle the bar.

### 9. All five candidates are now gradable at 410 — after they are re-run there

Every batch1 candidate is refused today on `station_population` (that is `fps-916` working
as designed) and every one of them is admissible against this bank once re-run at 410.
Verified by calling `_bank_admissibility` directly, not read off the code:

| candidate | arity | as committed | if re-run at 410 |
|---|---:|---|---|
| `lga_trough_propagation` | 3 | refused: `station_population` | **OK** |
| `network_move_breadth` | 3 | refused: `station_population` | **OK** |
| `station_descent_dynamics` | 3 | refused: `station_population` | **OK** |
| `stickiness_phase_saddle` | 2 | refused: `station_population` | **OK** |
| `tgp_cycle_displacement` | 2 | refused: `station_population` | **OK** |

The re-runs are ≈0.56h each (14 folds × 2 arms at ~72 s/arm-fold), ≈2.8h for all five:

```bash
PYTHONPATH=. uv run python -m experiments.pipeline.runner \
    --batch-dir experiments/batches/batch1 \
    --candidate experiments/candidates/batch1/<name>.py --n-stations 410
```

**Run them consecutively, not interleaved with five-station runs.** `r0_cache.joblib` is one
file per batch dir fingerprinted on `station_codes`; a width flip falls back to a full refit
(never a correctness risk, but ~17 min: 2024s uncached vs 975s cached). Grouped by width you
pay that once.

## Phase 2 conclusion

The bead's question is answered, with an arity qualifier it did not anticipate: **the band
narrows 1.44x at arity 1 and 1.06x at arity 3, and neither is separable from 1.00x.** The
practical gain from grading at 410 is +0.034 c/L of bar at the grading arity, and most of
that is the draw count rather than the width.

`fps-nas`'s criterion 5 is **not** vindicated the way phase 1 reported. Phase 1 concluded
"the broad delta clears its own band at z = −3.33"; at the only arity that can legally grade
it, the broad delta **fails**, by 2.3%. Criterion 5 is not thereby refuted either — the
matched arity-2 ruler is the one that decides, and it has not been run. What is settled is
that phase 1's reassurance was an artifact of an inadmissible bank.

## Followups

- **`noise_floor_n410.json` (arity 1) should not be used to grade anything**, and its §3
  grade should not be quoted. It remains a valid ruler-property measurement.
- **The matched arity-2 410 bank is the open question**, and it is the one that decides
  `tgp_cycle_displacement` at broad width. ~12h at 40 draws.
- **Re-run the five candidates at 410** (≈2.8h, consecutively) before any of them can be
  graded on this bank.
- **Draws, not stations**, still: settling 1.44x vs 2.00x is bound by the five-station
  banks' 20 and 10 draws.
- The §2 mean-shift mechanism is now *doubted as well as* `not_tested` — it predicts the
  bias grows with arity and the point estimate falls.
