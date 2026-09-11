# WFCV fold-streaming benchmark

- **Date:** 2026-09-11
- **Branch:** main
- **SHA:** recorded in each generated metrics artifact
- **Status:** done

## Hypothesis

`experiments.lib.folds.iter_folds_with_baseline_fit` materialises all 14 expanding
train/validation frames and retains them until every seed has fitted. Consuming the same
canonical folds one at a time should produce exactly the same predictions and losses while
substantially reducing peak resident memory. Wall time may improve if the current path swaps;
prediction parity and memory reduction are the primary tests.

## Design

This is an execution-engine benchmark, not a candidate-quality experiment. It therefore uses
the experiments lab book and frozen batch data, but does not invoke the candidate/noise-floor
runner or realised arbiter.

Two variants are compared in separate processes so peak RSS is meaningful:

- `current`: the production `_run_wfcv_screen` path, including
  `iter_folds_with_baseline_fit`'s `list(...)` and retained `fits` dictionary.
- `streaming`: the same fold generator, LightGBM fit helper, hard-25 calculation and
  `RowPredCollector`, but each fold is fully consumed and released before requesting the next.

The model comparison is the locked baseline versus the locked baseline plus the already-
computed, non-model `tgp_delta_7d` column. This gives the production two-arm shape without
adding candidate-feature computation to the measurement. The batch's cached shock-fold set is
required, matching the normal candidate-run path and allowing regime assignment immediately.

## How to invoke

Run the cheap fold-construction comparison first:

```bash
PYTHONPATH=. .venv/bin/python experiments/2026-09-11_wfcv_streaming/bench.py folds current
PYTHONPATH=. .venv/bin/python experiments/2026-09-11_wfcv_streaming/bench.py folds streaming
PYTHONPATH=. .venv/bin/python experiments/2026-09-11_wfcv_streaming/bench.py compare-folds
```

Then prove end-to-end parity with one seed:

```bash
PYTHONPATH=. .venv/bin/python experiments/2026-09-11_wfcv_streaming/bench.py wfcv current --seeds 42
PYTHONPATH=. .venv/bin/python experiments/2026-09-11_wfcv_streaming/bench.py wfcv streaming --seeds 42
PYTHONPATH=. .venv/bin/python experiments/2026-09-11_wfcv_streaming/bench.py compare-wfcv --seeds 42
```

Only after that passes, repeat the last three commands with `--seeds 42,43,44,45,46`.

Generated profiles, row predictions and per-fit rows are written under `artifacts/` and are
gitignored. `comparison_*.json` records the measured result in a compact form suitable for
transcribing here when the experiment concludes.

## Acceptance criteria

- Exact equality (`check_exact=True`) of fold identities, per-fit losses, hard-25 losses,
  row identities and float32 predictions. Timing columns are excluded from equality.
- Substantial peak-RSS reduction on the frozen production-scale batch.
- No material end-to-end wall-time regression. A neutral wall-time result is acceptable if
  the memory reduction is large, because the change removes swap/OOM sensitivity.

## Results

All parity checks passed with `check_exact=True`.

| stage | current | streaming | result |
|---|---:|---:|---|
| fold construction, peak RSS | 7,496 MiB | 6,161 MiB | streaming −17.8% (−1,335 MiB) |
| fold construction, wall | 5.16 s | 7.14 s | order/cache-confounded; streaming ran first |
| one-seed WFCV predictions | 1,597,298 | 1,597,298 | exact parity |
| one-seed WFCV screen | 85.86 s | 89.85 s | streaming 4.7% slower |
| one-seed summed fit time | 82.23 s | 85.80 s | +3.58 s of the +3.99 s screen delta |
| one-seed peak RSS | 8,614 MiB | 7,692 MiB | streaming −10.7% (−922 MiB) |
| five-seed WFCV predictions | 7,986,490 | 7,986,490 | exact parity |
| five-seed WFCV screen | 468.86 s | 502.85 s | streaming 7.3% slower |
| five-seed summed fit time | 464.31 s | 499.14 s | +34.83 s of the +33.99 s screen delta |
| five-seed non-fit overhead | 4.55 s | 3.71 s | streaming −0.84 s |
| five-seed peak RSS | 8,029 MiB | 8,383 MiB | streaming +4.4% (+354 MiB) |

The five-seed timing difference is fit-time drift, not fold orchestration: subtracting
`sum_fit_seconds` leaves streaming 0.84 seconds faster over the entire 14-fold screen. That
quantity is too small to matter beside 464–499 seconds of LightGBM work and too small to call
reliably from one ordered pair of runs.

The isolated fold phase confirms that retaining all folds has a real physical-memory cost,
but the full workload does not show a stable reduction in process-lifetime peak RSS. Even the
control's one-seed peak (8,614 MiB) exceeded its five-seed peak (8,029 MiB), demonstrating at
least ~585 MiB of run-to-run variation in this measurement. During the full screen, transient
LightGBM allocations, OS memory compression/reclamation and `RowPredCollector`'s additional
live accumulation obscure the fold-only saving. The loaded five-seed row-prediction artifact is
415 MiB in pandas, so the collector is material but is not large enough by itself to explain the
full reversal.

This environment runs pandas 3.0.2, where copy-on-write is always enabled. Consequently, the
sum of `DataFrame.memory_usage(deep=True)` over all retained folds substantially overstates the
physical resident cost: shared backing data are counted once per logical frame by that estimate,
but need not be resident once per frame. Peak RSS, not the earlier ~10 GiB logical sum, is the
relevant measurement.

An additional shape check found that narrowing the 70-column source frame to the 58 columns
needed by this benchmark would reduce its pandas-reported size by only 16.8% (1,133 MiB to
942 MiB), so column projection alone is unlikely to transform the result.

## Conclusion

**Do not promote streaming folds as a WFCV speed optimisation on this evidence.** It is
output-safe and reduces the isolated fold-retention footprint by 1.33 GiB, but saves less than
one second of non-fit work and does not reduce full five-seed peak RSS. The original hypothesis
is therefore only partially confirmed: the retained frames are wasteful, but they are not the
end-to-end bottleneck.

If memory is pursued further, profile current RSS at phase/fold boundaries before choosing the
next target. Incremental row-prediction persistence is one candidate because
`RowPredCollector` copies the identity block for every `(arm, seed)` and retains all blocks until
one final concat/write, but its measured 415 MiB output footprint says it is not the whole
answer. For wall time, the remaining target is LightGBM fitting, not fold orchestration.
