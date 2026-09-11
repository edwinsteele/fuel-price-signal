---
name: wfcv-fold-streaming-not-bottleneck
description: Streaming the 14 WFCV folds is output-exact but does not improve full five-seed wall time or stable peak RSS; LightGBM fitting dominates.
metadata:
  type: project
---

Do not propose lazy fold consumption as a WFCV speed optimisation without a new
measurement showing that the workload has changed. The production-scale experiment at
`experiments/2026-09-11_wfcv_streaming/` compared the current retained-fold path with a
fold-at-a-time path on frozen batch1 (2,084,203 rows, 70 columns, locked baseline
`54:1a6ec2d84a69`). They were exactly equal across all 140 loss rows and 7,986,490 float32
row predictions (`pandas.assert_frame_equal(..., check_exact=True)`).

Retaining the folds has a measurable isolated cost: fold-construction peak RSS fell from
7,496 to 6,161 MiB when streamed, and one-seed WFCV fell from 8,614 to 7,692 MiB. But that
did not survive the full five-seed workload: peak RSS was 8,029 MiB current versus 8,383
MiB streamed, while the control itself varied by at least 585 MiB between one- and
five-seed runs. The five-seed screen took 468.86 versus 502.85 seconds, entirely explained
by fit-time drift (summed fits 464.31 versus 499.14 seconds); non-fit orchestration was only
4.55 versus 3.71 seconds. At most 0.84 seconds of a roughly eight-minute screen was
available to this optimisation.

The earlier ~10 GiB estimate from summing `DataFrame.memory_usage(deep=True)` across
logical folds was not a physical-memory estimate. This environment uses pandas 3.0.2,
where copy-on-write is always enabled, so shared backing data are counted repeatedly by
that sum without necessarily being resident repeatedly. Use isolated-process peak RSS for
this question. For wall time, profile LightGBM fitting; for memory, sample RSS at phase and
fold boundaries before choosing a target rather than inferring it from logical DataFrame
sizes.
