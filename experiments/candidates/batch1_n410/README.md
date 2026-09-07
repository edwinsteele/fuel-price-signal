# batch1 candidates, re-run at the 410-station replay population

Symlinks to `../batch1/*.py` — **not copies**, so there is exactly one definition of each
candidate and the two widths cannot drift apart.

## Why this directory exists

`runner.py`'s output directory is `default_out_dir(candidate_path)` = the candidate path
with `.py` stripped, and `run_candidate` **unlinks the existing `results.json` before it
runs** (`runner.py:393`). Pointing `--candidate` at `../batch1/<name>.py` with
`--n-stations 410` would therefore destroy each candidate's five-station `results.json`
in place, leaving it desynchronised from the tracked `facts.json`, `README.md` and PNGs
beside it — which together are batch1's dossier record.

Giving the wide runs their own candidate path gives them their own out-dir. Nothing else
is coupled to the candidate's parent directory name: the batch is carried by `--batch-dir`
and stamped as `meta.batch_dir`.

## Running them

```bash
cd ~/Code/fuel-price-signal
for c in lga_trough_propagation network_move_breadth station_descent_dynamics \
         stickiness_phase_saddle tgp_cycle_displacement; do
  echo "=== $c ==="
  PYTHONPATH=. uv run python -m experiments.pipeline.runner \
      --batch-dir experiments/batches/batch1 \
      --candidate experiments/candidates/batch1_n410/$c.py \
      --n-stations 410 2>&1 | tee experiments/candidates/batch1_n410/$c.log
done
```

≈34 min each, ≈2.8h total, plus a one-off ~17 min `r0_cache` refit on the first run
(the cache is one file per batch dir fingerprinted on `station_codes`). Each run ends with
`<name>: graded (wall=…s)`.

**`r0_cache.joblib` in `experiments/batches/batch1/` is now 410-fingerprinted** — these
runs left it that way. The next *five*-station run in that batch dir pays the ~17 min
refit. Expected, not a fault.

**Run these consecutively and do not interleave a five-station run**, or you pay that
17 min refit on every width flip.

## What grades them

`experiments/batches/batch1/noise_floor_n410_k3.json` — 410 stations, arity 3, 40 draws.
It is the matched ruler for the three arity-3 candidates and a deliberately conservative
one for the two arity-2 candidates. See `experiments/2026-09-06_noise_floor_n410/README.md`
§ "Phase 2".

## Results — all five ran, 2026-09-06/07

All five returned `status: "graded"`, all stamping `station_population = 410:5bbff5bf61d3`,
`baseline_fingerprint = 54:1a6ec2d84a69`, `tank_params = 50/3.571/1d/10%`, `n_windows = 14`
— matching `noise_floor_n410_k3.json` on every admissibility axis, with no silent fallback
to five stations.

| candidate | arity | 5-stn Δ | 410 Δ | z vs 410 k=3 band |
|---|--:|--:|--:|--:|
| `lga_trough_propagation` | 3 | −0.0672 | +0.0666 | +0.392 |
| `network_move_breadth` | 3 | −0.1627 | −0.1004 | −1.388 |
| `station_descent_dynamics` | 3 | −0.0237 | +0.0604 | +0.326 |
| `stickiness_phase_saddle` | 2 | −0.0761 | +0.0544 | +0.262 |
| `tgp_cycle_displacement` | 2 | −0.2077 | −0.1292 | −1.695 |

**Nothing clears.** Single-candidate gate 1.7332; family-wise gate for five candidates
2.5159. Three of five turned positive — a cost, not a saving.

**These directories are NOT dossier-gradable.** `_noise_band` picks its bank by the fixed
filename `noise_floor.json` (the five-station bank), so a dossier over any of them prints
`refused: station_population` as its headline verdict and reaches the 410 bank only as a
corroborating sibling. The grades above were computed by hand in
`experiments/2026-09-06_noise_floor_n410/analyse.py` § 10. Full write-up:
`experiments/2026-09-06_noise_floor_n410/README.md` § "Phase 3".

Only `results.json` is tracked here; `fills.parquet`, `rowpreds.parquet` and `*.log` are
gitignored.
