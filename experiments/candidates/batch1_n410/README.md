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
(the cache is one file per batch dir fingerprinted on `station_codes`, and batch1's is
currently five-station). Each run ends with `<name>: graded (wall=…s)`.

**Run these consecutively and do not interleave a five-station run**, or you pay that
17 min refit on every width flip.

## What grades them

`experiments/batches/batch1/noise_floor_n410_k3.json` — 410 stations, arity 3, 40 draws.
It is the matched ruler for the three arity-3 candidates and a deliberately conservative
one for the two arity-2 candidates. See `experiments/2026-09-06_noise_floor_n410/README.md`
§ "Phase 2".
