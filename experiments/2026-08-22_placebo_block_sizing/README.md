# Placebo block sizing — the two measured constants behind `placebo.py` (fps-d7m)

**Status: done (graduated — the constants these tables set are shipped in
`experiments/pipeline/placebo.py`, PR #324).**

## Why this dir exists

`fps-d7m` replaced the noise floor's placebo construction: a circular time-shift became a
**block permutation** (cut the date axis into contiguous blocks, reorder whole blocks). That
construction has two numbers in it that came out of a *measurement* rather than a principle:

| constant | shipped | what it controls |
|---|---|---|
| `PLACEBO_BLOCK_DAYS` | 61 | how long each contiguous block is |
| `MIN_BLOCKS` | 24 | floor on block COUNT; short series get shortened blocks instead |

Both were justified in `placebo.py`'s docstring by tables — and the scripts that produced
those tables lived in a scratch dir that gets wiped. That is the exact setup
`docs/CONVENTIONS.md` warns about: a constant derived from a measurement is not verified
until that measurement can re-run. **Anyone moving either constant should re-run
`block_count_sweep.py` first and paste the new tables into `placebo.py` alongside the
change.** The script asserts that its own conclusion still matches the shipped `MIN_BLOCKS`,
so a silent divergence fails loudly rather than rotting.

## Scripts

```bash
# synthetic, deterministic, ~30s, no batch data needed — regenerates both constants' tables
PYTHONPATH=. uv run python experiments/2026-08-22_placebo_block_sizing/block_count_sweep.py

# real batch0, a few minutes — fps-d7m's acceptance-criterion-2 evidence
PYTHONPATH=. uv run python experiments/2026-08-22_placebo_block_sizing/batch0_screen.py
```

Outputs: `block_count_sweep.csv`, `block_length_sweep.csv`, `batch0_screen.csv`.

## Table 1 — decorrelation vs block COUNT (sets `MIN_BLOCKS`)

500 seeds against a deliberately **adversarial** probe series: monotone drift plus a 31-day
cycle, i.e. drift-dominated. That is the hardest shape for any time-axis reordering, and the
shape that broke the circular shift. A gentler series would flatter these numbers.

| blocks | median | p95 | p99 | max | clears 0.6 |
|---|---|---|---|---|---|
| 8 | 0.238 | 0.642 | 0.753 | 0.792 | no |
| 12 | 0.190 | 0.472 | 0.652 | 0.751 | no |
| 16 | 0.168 | 0.460 | 0.575 | 0.740 | no |
| 20 | 0.153 | 0.426 | 0.539 | 0.657 | no |
| **24** | **0.145** | **0.392** | **0.472** | **0.555** | **yes** |
| 30 | 0.115 | 0.325 | 0.396 | 0.529 | yes |
| 59 | 0.078 | 0.244 | 0.315 | 0.332 | yes |

**24 is the smallest count whose WORST case over 500 seeds still clears the screen.** At 12
the p99 is 0.652 — roughly one draw in a hundred would fail and be substituted away, and it
would be a *level-like* column being substituted away, which is precisely the systematic
exclusion fps-d7m exists to remove, arriving by a different route.

Binds only below `24 × 61 = 1464` positions, so never on batch0's 3572-date market-wide
branch, and on roughly the thinnest tenth of its stations. batch0's thinnest station (389
dates) drops to a 16-position block — less texture preserved, but a station with a short
history has less texture to preserve, and unreliable decorrelation is the worse defect.

## Table 2 — fidelity vs block LENGTH (sets `PLACEBO_BLOCK_DAYS`)

A placebo must *resemble* a real feature, so the permutation should preserve local texture. A
circular shift introduced exactly one seam (the wrap); block permutation introduces one per
boundary. Shorter blocks decorrelate better but retain less autocorrelation. Held at a
batch0-sized 3572 positions.

| block days | blocks | seams | lag-1 autocorr | median self-corr |
|---|---|---|---|---|
| *original* | — | 0 | 0.999 | — |
| 14 | 256 | 255 | 0.927 | 0.039 |
| 29 | 124 | 123 | 0.962 | 0.052 |
| **61** | **59** | **58** | **0.977** | **0.083** |
| 127 | 29 | 28 | 0.990 | 0.124 |
| 251 | 15 | 14 | 0.993 | 0.158 |

**The trade runs in both directions, so this is a choice, not an optimum.** 61 sits above the
project's own cycle length (batch0 `cycle_mean_length`: mean 30.5, max 35.3 days) so a block
holds at least one whole cycle, while still leaving ~58 blocks on a batch0-sized range — well
clear of Table 1's floor of 24.

## Table 3 — full batch0 screen (fps-d7m acceptance criterion 2)

54 locked columns × 5 seeds, shipped code:

- **5** columns all-NaN (`days_since_trough_entry_<lga>` for LGAs no station classifies into) — unverifiable, skipped
- **49** usable columns, **245** checks, **0** correlation failures
- median of per-column max **0.046**, worst **0.470**

All eight columns the circular shift excluded now pass: `cycle_last_min_cents` 0.128,
`cycle_peak_count` 0.033, `cycle_mean_length` 0.066, `cycle_last_max_cents` 0.054,
`station_price_cents` 0.050, `lga_mean_cents` 0.037, `brand_mean_cents` 0.040,
`stickiness_score` 0.465 (medians).

### Two skip reasons, and why the distinction matters

The script reports them separately because **conflating them was a real error in this work's
own prose**, caught in review. At the production default of 20 draws:

- **correlation-driven substitution: never fires** on batch0 under block permutation. Under
  the circular shift it fired on eight columns, every run.
- **all-NaN substitution: fires twice, every run.** `select_draws` spreads its 20 primaries
  by position and lands on `days_since_trough_entry_lane_cove` (seed 157) and `...waverley`
  (seed 181), which are replaced from the fallback tail by `cycle_days_since_peak` and
  `cycle_mean_length`.

The published bank composition (11 trough-counter / 3 cycle-magnitude / 1 price-level / 5
other) is only reached *because of* those two substitutions. Behaviour is correct and
graceful; the original write-up simply claimed "no substitution needed", from a check that
could never fail.

## Named limitation (carried forward, not closed)

Block permutation cannot decorrelate a column whose information is mostly **cross-sectional**
rather than temporal — each station keeps its own values, so no time-axis reorder touches the
between-station structure. batch0's top two are `stickiness_score` (0.470) and
`station_minus_sydney_avg_cents` (0.328), against a 0.046 median. Both still pass, so they
enter the bank as its **weakest draws**.

`MAX_SELF_CORRELATION` is deliberately **not** tightened to exclude them: that would
re-introduce a systematic category exclusion along the station axis instead of the level
axis — the same bug wearing a different hat. `noise_floor.json` stamps every draw's
`self_correlation` so this is visible per floor rather than only in prose.

## Not measured here

These tables are **correlation** only. The model is a tree ensemble and the arbiter is
realised CPL, so what any of this costs in c/L is settled by
`noise_floor.py <batch> --force`, not by anything in this dir. That recompute is still
outstanding and is what `fps-cds` (the honest single-candidate bar, as a number) needs.

Related: `experiments/pipeline/placebo.py`, `experiments/pipeline/noise_floor.py`,
`docs/feature-pipeline.md`, bd `fps-d7m` / `fps-awz` / `fps-cds` / `fps-aay`.
