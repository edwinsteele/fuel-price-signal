# tau-distance of the log-loss gain — where TGP's screen win actually lives

- **Date:** 2026-08-31
- **Branch:** `claude/fps-e6i-480b5a`
- **SHA:** see `git log` for this dir; inputs are frozen batch1 artifacts (`git_sha` 4a2016d in the candidate's `results.json`)
- **Bead:** `fps-e6i`
- **Status:** done
- **Subject:** `experiments/candidates/batch1/tgp_cycle_displacement`

## Hypothesis (the bead's)

> Both TGP expressions improve the WFCV log-loss screen and neither reaches the realised
> arbiter. Hypothesis: the probability improvement lands on rows nowhere near tau = 0.25, so
> the decision rule never reads it. […] If the gain is concentrated far from tau, the whole TGP
> track is answered: not "the feature is wrong" but "it improves a quantity the decision rule
> does not consult."

## Verdict

**The hypothesis is falsified as an explanation, and a different channel is identified and
measured.** The gain is not threshold-remote — it is threshold-*indifferent*. What actually
separates the screen from the arbiter is **population**: the two statistics share **0.71%** of
their evidence base, and once that is equalised the disagreement the bead opened on largely
disappears.

| | |
|---|---|
| Rows within one isotonic step of tau | 0.83% of rows carrying **0.8%** of the gain — **1.01x**, exactly proportional |
| Rows that change side of the boundary | **~4.0%** of all rows (5-seed sd 0.45pp); 232 of the arbiter's own 5,693 rows |
| Screen rows the arbiter ever acts on | **5,693 of 798,649 = 0.71%** |
| per-fold corr(arbiter dCPL, screen dLL) — **all 714 stations** | **-0.096** paired / **-0.176** published convention |
| per-fold corr(arbiter dCPL, screen dLL) — **the arbiter's 5 stations** | **+0.478** paired / **+0.543** published convention |
| gap between those two correlations | **+0.574**, 95% CI **[+0.101, +1.033]**, P(gap>0) = **0.991** |

This is **not** a TGP finding. Nothing in it is specific to `tgp_cycle_displacement`; it is a
property of the screen/arbiter seam that every candidate in every batch runs through. The
bead's acceptance criterion 4 ("if the gain is threshold-remote […] close the TGP track") does
**not** fire, and closing the TGP track on this evidence would be wrong.

## How to invoke this script

```bash
PYTHONPATH=. uv run python experiments/2026-08-31_tau_distance_of_logloss_gain/run.py 2>&1 | tee experiments/2026-08-31_tau_distance_of_logloss_gain/run.log
```

Pure lookup over committed artifacts — **no model is fit**, ~2 min wall. Reads
`experiments/candidates/batch1/tgp_cycle_displacement/rowpreds.parquet` (gitignored, present in
the primary checkout) and `experiments/batches/batch1/r0_cache.joblib`.

## Setup

Two artifacts, and the whole method rests on the relationship between them.

- **rowpreds.parquet** — the WFCV screen's per-row output: 798,649 rows x 2 arms x 5 seeds,
  714 stations, 14 folds. `proba` is the **raw** LightGBM output; the screen scores log-loss on
  it and never calibrates.
- **r0_cache.joblib** — the realised arbiter's own cached per-fold baseline: the fitted
  isotonic calibrator and the OOF-selected `own_tau`. Its `station_codes` fingerprint is
  `[261, 414, 429, 585, 18517]` — the five preferred stations.

### The two are the same fit — asserted, not assumed

`experiments.lib.fit.fit_score` builds `LGBMClassifier(random_state=seed, **LGBM_DEFAULTS)`;
`fuel_signal.train_lgbm.build_pipeline(seed)` builds the same estimator with the same three
hyperparameters; and `realised._plan_folds` walks the same `walk_forward_folds` geometry (fold
1 val = 2021-11-05..2022-02-02 in both). The script **checks this rather than reasoning about
it**: it re-scores folds 1/7/14 with the cache's `cal_pipe.base_pipeline` and compares against
the screen's stored R0/seed-42 probabilities.

```
max |screen R0 proba - arbiter base proba| over folds 1/7/14: 2.980e-08
```

That is float32 epsilon. The screen's R0 probabilities **are** the arbiter's baseline scores, so
the cache's isotonic map converts between the two scales *exactly*, with no fitting and no
approximation. The run asserts `worst < 1e-6` and aborts otherwise.

## Result 1 — the bead's ruler was the wrong ruler

tau = 0.25 is a threshold on **calibrated** probability. The screen's probabilities are **raw**.
Isotonic is monotone, so `p_cal >= 0.25` is exactly equivalent to `p_raw >= tau_raw` for a
`tau_raw` recoverable from the map:

```
 fold  own_tau  tau_raw  band_lo  band_hi  band_width  n_iso_levels
    1  0.25000  0.11359  0.10736  0.11484     0.00748           107
    4  0.25000  0.13119  0.11744  0.13193     0.01449           129
    9  0.25000  0.15899  0.15554  0.16198     0.00645           161
   13  0.25000  0.15899  0.13858  0.16396     0.02538           165
   14  0.25000  0.15576  0.13858  0.15899     0.02040           183
```

tau is 0.25 on **14/14 folds**; `tau_raw` spans **[0.1136, 0.1590]**. Measuring `|p_raw - 0.25|`
as the bead literally asks would have aimed ~0.10 wide of the actual boundary — landing in the
`(0.05, 0.1]` bin, which is the single most gain-dense bin in the whole profile. The mistake
would have manufactured a confirmation.

See `where_tau_sits.png` (left: the raw histogram with both points marked; right: the isotonic
maps for folds 1/7/14 crossing 0.25).

## Result 2 — the gain is threshold-INDIFFERENT, not threshold-remote

Binned by `|p_raw(R0) - tau_raw|`, seed-mean, all 714 stations (`tau_distance_profile.png`):

| `|p - tau_raw|` | rows | row % | sum dLL (nats) | % of gain | mean\|dp\| |
|---|---|---|---|---|---|
| [0.00, 0.01] | 12,478 | 1.56% | -174.2 | 1.7% | 0.0319 |
| (0.01, 0.02] | 12,332 | 1.54% | -186.9 | 1.8% | 0.0323 |
| (0.02, 0.05] | 40,221 | 5.04% | -764.3 | 7.5% | 0.0306 |
| (0.05, 0.10] | 94,003 | 11.77% | -2746.3 | 26.8% | 0.0227 |
| (0.10, 0.20] | 411,614 | 51.54% | -1213.6 | 11.8% | 0.0067 |
| (0.20, 0.40] | 60,795 | 7.61% | -2265.1 | 22.1% | 0.0708 |
| (0.40, 1.00] | 167,206 | 20.94% | -2904.3 | 28.3% | 0.0441 |

**Acceptance criterion 2 — share of the gain within one isotonic step of tau:**

```
rows inside one isotonic step of tau : 6,609 (0.83% of rows)
their share of the log-loss gain     : 0.8%  (-85.93 of -10254.54 nats)
concentration ratio (gain% / row%)   : 1.01x
```

Across the 5 seeds individually that share is 0.02%–1.10% (sd 0.43pp) against a row share of
0.83–0.89%. So the near-threshold neighbourhood carries **its proportional share and no less**.
The gain is spread across the probability axis rather than avoiding the boundary.

**Threshold crossings.** 22,659–37,177 rows (3.60%–4.65% across seeds, sd 0.45pp) land on
opposite sides of `tau_raw` under the two arms. The count is flat under a +/-30% sweep of
`tau_raw` (2.69%–2.85% at seed-mean), which matters because the candidate arm's own calibrator
was not cached — see Limitations. **The decision rule is not blind to this feature.**

At the arbiter's own five stations: **232 crossings** on average (5 seeds: 187/209/226/255/284).
The realised run's own ledger reports **423 flips / 92 decisions** — the same order of magnitude,
independently derived. The screen's decision-relevant signal at the arbiter's stations is real
and roughly the size the arbiter actually saw.

**Acceptance criterion 1 — where the biggest movers live** (seed-mean; `dist` = `|p - tau_raw|`):

| cohort | n | median dist R0 | median dist cand | in one-step band | % of gain |
|---|---|---|---|---|---|
| all rows | 798,649 | 0.1383 | 0.1383 | 0.83% | 100.0% |
| top 10% by \|dp\| | 79,865 | 0.3313 | 0.3120 | 0.56% | 79.5% |
| top 1% by \|dp\| | 7,987 | 0.4115 | 0.2453 | 0.16% | 24.5% |
| top 0.1% by \|dp\| | 799 | 0.4835 | 0.1934 | 0.38% | 4.5% |

**Here the bead is right, but only about the tail.** The largest probability movements *are*
threshold-remote — the top 1% by |dp| sit 3x farther from the boundary than a typical row
(0.41 vs 0.14) and are 5x under-represented in the one-step band (0.16% vs 0.83%). But that
cohort carries only ~25% of the gain. The bulk of the improvement is small movements spread
everywhere, including at the boundary. Threshold-remoteness of the extreme tail is a true
observation that does not explain the outcome.

## Result 3 — the channel that does explain it: population

The arbiter replays exactly five stations. The screen scores 714.

```
group                 rows    row %    sum d_ll  % of gain    d_ll/row
preferred 5          5,693    0.71%    -62.7444       0.6%  -1.102e-02
other 709          792,956   99.29% -10191.7939      99.4%  -1.285e-02
```

The per-row gain at the arbiter's stations is **the same size as everywhere else** — the feature
is not failing there. It is simply that 99.3% of the measured improvement happens at stations
where no fuel is ever bought.

**Power (fold-clustered, mean of 14 per-fold means, SE across folds):**

| group | mean dLL/row | 2·SE | resolvable? |
|---|---|---|---|
| preferred 5 | -1.075e-02 | 1.459e-02 | **NO** |
| other 709 | -1.319e-02 | 8.877e-03 | yes |

The preferred-5 mean sits **0.33 SE** from the network-wide mean — no evidence the five stations
differ in kind. What differs is resolution: at 5,693 rows the effect cannot clear its own 2·SE
even on the *smooth* statistic. The arbiter's statistic is realised CPL through a path-dependent
tank, which is strictly noisier, so this bounds the arbiter's power from above. (On the single
realised seed 42, neither group resolves — network-wide resolution needs the 5-seed average
*and* the 141x row count.)

## Result 4 — the crux: align the populations and the disagreement goes away

The bead's actual puzzle was per-fold disagreement between screen and arbiter. Restricting the
screen to the five stations the arbiter replays is the **only** change between these two rows —
same model, same folds, same seeds, same statistic (`crux_screen_vs_arbiter.png`):

| comparison | Pearson r | Spearman | sign agreement |
|---|---|---|---|
| paired conv. \| screen ALL 714 stations | **-0.096** | -0.051 | 7/14 |
| paired conv. \| screen PREFERRED 5 only | **+0.478** | +0.503 | 10/14 |
| published conv. \| screen ALL 714 stations | **-0.176** | -0.130 | 5/14 |
| published conv. \| screen PREFERRED 5 only | **+0.543** | +0.521 | 11/14 |

The published-convention all-station figure reproduces the bead's cited **-0.176 exactly**,
which is the provenance check on the whole recomputation.

n = 14 folds, so the individual correlations are weak evidence on their own (all-station 95% CI
[-0.596, +0.458]; preferred-5 [-0.070, +0.805]). **The claim is the gap, and the gap is what
resolves** — the two correlations share the same 14 folds and the same arbiter series, so a
fold-level bootstrap (20,000 resamples) prices the difference directly:

```
gap (pref5 - all)   r = +0.574   95% CI [+0.101, +1.033]   P(gap>0) = 0.991
```

Lead with the gap, not with `r = +0.478`.

## Conclusion

The screen is not measuring "a quantity the decision rule does not consult". It is measuring the
**right quantity on a 141x larger population**, of which the arbiter reads 0.71%. The two are
not in conflict; they are differently scoped, and the screen's per-fold delta was never a valid
forecast of the arbiter's per-fold delta.

Neither scope is wrong on its own terms — the screen is network-wide because that is where the
statistical power is, and the arbiter is five stations because that is the owner's actual
commute (`reference_preferred_stations_context`). The defect is the **missing seam**: nothing in
the pipeline reports the screen restricted to the population the arbiter acts on, so a
network-wide log-loss win has been read as a forecast of a five-station CPL outcome it has no
mechanical reason to predict. Filed as `fps-4z6`.

**Do not close the TGP track on this.** Acceptance criterion 4 was conditional on the gain being
threshold-remote; it is not. And the finding is generic — it applies identically to every
candidate in batch0 and batch1, so it cannot single out TGP. The prior TGP expressions'
inconclusive arbiter readings are now better explained by five-station resolution than by
anything about wholesale price series.

## Limitations

1. **The candidate arm's own `tau_raw` is unknown.** Only R0's calibrator was cached, so both
   arms are ruled against R0's `tau_raw`. That is the right instrument for this question — it
   isolates movement of the probabilities from relocation of the boundary, and the boundary
   demonstrably does not relocate (`own_tau_median = held_tau_median = 0.25` on both arms). The
   crossing count is reported under a +/-30% sweep of `tau_raw` so the conclusion does not rest
   on one point; it moves by less than 0.2pp.
2. **n = 14 folds.** The correlation gap's CI is wide ([+0.101, +1.033]) and the upper end is
   unphysical, an artefact of bootstrapping a bounded statistic at small n. It excludes zero,
   which is the claim; it does not pin the magnitude.
3. **The screen's own headline is seed-noisy.** Total gain across the 5 seeds ranges
   -6,573 to -15,378 nats (mean -10,255, sd 3,418 — a 33% coefficient of variation). Every
   per-bin decomposition inherits that; single-seed and seed-mean profiles disagree in sign on
   individual bins ([B] in `run.log`), which is why the load-bearing numbers here are the ones
   stable across all five seeds (one-step share, crossing share, population share) and not the
   per-bin profile.
4. **Log-loss is not CPL.** Result 3's power argument bounds the arbiter from above rather than
   measuring it: it shows the *smoother* statistic already fails to resolve at five stations.

## Incidental finding — paired vs unpaired seed aggregation

`experiments/lib/aggregate.py:46` computes `delta_<cohort>_median` as
`median_over_seeds(candidate_ll) - median_over_seeds(R0_ll)` — a **difference of medians**. The
seeds are paired (same data, same folds; only `random_state` differs), so the **median of the
paired differences** is the better estimator: it cancels the seed-level fluctuation the two arms
share. The two statistics disagree by up to 0.0127 nats per fold and **flip sign on fold 2**
(paired +0.0036, published -0.0090) — enough to change a per-fold sign-agreement count, which is
exactly the statistic this bead was opened to interpret. Every correlation above is therefore
reported under both conventions; the conclusion is unaffected. Filed as `fps-2kc`.

## Followups

- `fps-4z6` — report the screen restricted to the arbiter's replay universe alongside the
  network-wide figure, so the dossier's screen and arbiter numbers are commensurable.
- `fps-2kc` — paired seed aggregation in `experiments/lib/aggregate.py`.

## Files

- `run.py` — the whole analysis; `run.log` — full output.
- `facts.json` — every number above, machine-readable.
- `per_fold_screen_vs_arbiter.csv` — the 14-row table behind Result 4.
- `crux_screen_vs_arbiter.png` — the money plot.
- `tau_distance_profile.png` — Result 2.
- `where_tau_sits.png` — Result 1.
