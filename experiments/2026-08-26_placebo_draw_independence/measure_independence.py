"""What does relaxing the placebo bank's DISTINCT-SOURCE-COLUMN rule cost? (bd fps-3jj.21)

`placebo.screen_draw_groups` requires `n_draws * arity` DISTINCT source columns and
`n_draws * arity` distinct entries from a 30-long seed pool, so the maximum draw count at
arity k is `min(floor(54/k), floor(30/k))`. Since `family_wise_z_threshold` uses
`df = n_draws - 1`, the bar explodes as arity rises and most of the explosion is the
collapsing draw count, not arity. A 35-column (per-LGA) candidate cannot be graded at all.

This script measures the thing the bead says is missing: **the independence cost of
letting draws reuse source columns**, on the batch1 frozen frame, with no fits.

Four measurements:

  A. Source diversity of the lock itself — does "distinct column" mean "distinct source"?
  B. corr(placebo(c, s1), placebo(c, s2)) — the reuse channel, head on. Compared against
     the two quantities already trusted: the screen's own self-correlation, and the
     cross-draw correlation the CURRENT distinct-column construction produces.
  C. Whole-bank comparison: current 10x3 bank vs a proposed 20x3 reuse bank, over every
     cross-draw column pair.
  D. Texture-family reuse already present in the current bank (the channel correlation
     cannot see).

Reads the frozen frame from the PRIMARY worktree read-only; writes nothing but its own CSVs.
"""
from __future__ import annotations

import itertools
import json
import os
import pathlib
import time

import numpy as np
import pandas as pd

from experiments.pipeline.placebo import (
    PLACEBO_BLOCK_SEED_POOL,
    make_placebo_series,
)
from fuel_signal.features import load_features

# Repo-relative, like every sibling experiment (`2026-08-22_placebo_block_sizing`,
# `2026-08-23_placebo_arity`) — run from the repo root, as the README's commands do.
# The frozen frame it needs is GITIGNORED, so it exists only in the clone that ran the
# freeze — a git worktree will not have it. FUEL_BATCH_DIR overrides for that case rather
# than baking one machine's absolute path into the script.
BATCH = pathlib.Path(os.environ.get("FUEL_BATCH_DIR", "experiments/batches/batch1"))
OUT = pathlib.Path(__file__).parent

# Seeds beyond PLACEBO_BLOCK_SEED_POOL's 30, for the proposed construction. Kept as an
# explicit extension of the same pool so the first 30 are byte-identical to today's.
EXTENDED_SEEDS: tuple[int, ...] = PLACEBO_BLOCK_SEED_POOL + tuple(
    int(p) for p in (
        257, 263, 269, 271, 277, 281, 283, 293, 307, 311,
        313, 317, 331, 337, 347, 349, 353, 359, 367, 373,
        379, 383, 389, 397, 401, 409, 419, 421, 431, 433,
    )
)


def texture_family(column: str) -> str:
    """Coarse source-texture label. Deliberately crude — the point of measurement D is
    that even this crude a partition shows the lock has far fewer families than columns."""
    if column.startswith("days_since_trough_entry_"):
        return "lga_trough_counter"
    if column.startswith("network_"):
        return "network"
    if column in {"cycle_pct_through", "cycle_days_since_peak"}:
        return "cycle_phase"
    if column.startswith("cycle_"):
        return "cycle_magnitude"
    return "price_level_other"


def deck_picks(columns: list[str], n_picks: int) -> list[str]:
    """`n_picks` source columns from an UNBOUNDED extension of the OLD `candidate_pool`.

    Written against the pre-fps-3jj.21 code, and deliberately kept that way: this is the
    prototype whose measurement justified the change, so it must stay runnable against what
    it was arguing about rather than importing the thing it proposed. `placebo` now ships
    this shape as `candidate_sequence`.

    The old `candidate_pool` returned exactly `len(columns)` entries: `select_draws`'s even
    spread (the primaries) followed by every remaining column in natural order (the fallback
    tail). Together that is one LAP — a permutation of the whole column list. This extends it
    by appending further laps, each the natural order rotated one position further, so a lap
    boundary does not reproduce the previous lap's adjacency. For `n_picks <= len(columns)`
    it is that old `candidate_pool` byte for byte.
    """
    n_cols = len(columns)
    # `select_draws`'s own even spread, inlined so the prototype is not blocked by that
    # function's seed-pool guard — the seeds are assigned by `deck_draws` below instead.
    n_primary = min(n_picks, n_cols)
    seen: set[int] = set()
    primary_idx: list[int] = []
    for i in range(n_primary):
        idx = round(i * n_cols / n_primary) % n_cols
        while idx in seen:
            idx = (idx + 1) % n_cols
        seen.add(idx)
        primary_idx.append(idx)
    primary = [columns[i] for i in primary_idx]
    picks = primary + [c for c in columns if c not in set(primary)]
    lap = 1
    while len(picks) < n_picks:
        picks += [columns[(i + lap) % n_cols] for i in range(n_cols)]
        lap += 1
    return picks[:n_picks]


def deck_draws(columns: list[str], n_draws: int, arity: int) -> list[list[tuple[str, int, float]]]:
    """`n_draws` groups of `arity` (column, seed, nan) triples off `deck_picks`.

    Within a draw, columns are forced distinct (skip to the next pick) so a draw straddling a
    lap boundary cannot double up. Every column in the bank gets its own seed off a flat
    counter that never wraps, so fps-3jj.20's primary/fallback seed collision cannot occur.
    """
    picks = deck_picks(columns, n_draws * arity + len(columns))
    draws: list[list[tuple[str, int, float]]] = []
    cursor = counter = 0
    for _ in range(n_draws):
        members: list[tuple[str, int, float]] = []
        used: set[str] = set()
        while len(members) < arity:
            col = picks[cursor]
            cursor += 1
            if col in used:
                continue
            used.add(col)
            members.append((col, EXTENDED_SEEDS[counter], float("nan")))
            counter += 1
        draws.append(members)
    return draws


def main() -> None:
    t0 = time.time()
    if not (BATCH / "baseline_columns.json").exists():
        raise SystemExit(
            f"{BATCH} has no baseline_columns.json. Run this from the repo root, and note "
            "that the frozen frame (features.parquet) is gitignored — it lives only in the "
            "clone that ran the freeze, not in a worktree. Point FUEL_BATCH_DIR at that "
            "clone's batch dir if you are running from elsewhere."
        )
    frame = load_features(BATCH / "features.csv")
    baseline = json.loads((BATCH / "baseline_columns.json").read_text())
    print(f"[{time.time()-t0:6.1f}s] frame {frame.shape}, {len(baseline)} baseline columns")

    usable = [c for c in baseline if frame[c].notna().any()]
    print(f"           {len(usable)} usable (non-all-NaN) source columns")

    # ---- A. does "distinct column" mean "distinct source"? --------------------------
    src = frame[usable].corr().abs()
    pairs = [
        (a, b, float(src.loc[a, b]))
        for a, b in itertools.combinations(usable, 2)
        if np.isfinite(src.loc[a, b])
    ]
    pair_vals = np.array([p[2] for p in pairs])
    fam_counts = pd.Series([texture_family(c) for c in usable]).value_counts()
    print("\n=== A. source-column diversity of the lock ===")
    print(f"  {len(usable)} columns, {len(pairs)} pairs; |rho| median {np.median(pair_vals):.3f}, "
          f"p95 {np.quantile(pair_vals, 0.95):.3f}, max {pair_vals.max():.3f}")
    for thr in (0.7, 0.9, 0.95):
        print(f"  pairs with |rho| >= {thr}: {(pair_vals >= thr).sum()}")
    print(f"  texture families ({len(fam_counts)}): {dict(fam_counts)}")
    pd.DataFrame(pairs, columns=["col_a", "col_b", "abs_rho"]).sort_values(
        "abs_rho", ascending=False
    ).to_csv(OUT / "source_column_correlations.csv", index=False)

    # ---- B. the reuse channel, head on ----------------------------------------------
    # One representative column per texture family, plus the lock's most-correlated pair.
    probe_cols: list[str] = []
    for fam in fam_counts.index:
        probe_cols += [c for c in usable if texture_family(c) == fam][:3]
    probe_cols = list(dict.fromkeys(probe_cols))
    probe_seeds = EXTENDED_SEEDS[:4]
    print(f"\n=== B. reuse channel ({len(probe_cols)} probe columns x {len(probe_seeds)} seeds) ===")

    cache: dict[tuple[str, int], np.ndarray] = {}
    for c in probe_cols:
        for s in probe_seeds:
            cache[(c, s)] = make_placebo_series(frame, c, s).to_numpy(dtype=np.float32)
    print(f"[{time.time()-t0:6.1f}s] built {len(cache)} placebo series")

    def corr(x: np.ndarray, y: np.ndarray) -> float:
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 2:
            return float("nan")
        return float(np.corrcoef(x[ok], y[ok])[0, 1])

    rows = []
    for c in probe_cols:
        source = frame[c].to_numpy(dtype=np.float32)
        for s in probe_seeds:  # the screen's own quantity
            rows.append(("self (screened)", c, c, s, s, abs(corr(cache[(c, s)], source))))
        for s1, s2 in itertools.combinations(probe_seeds, 2):  # THE REUSE CHANNEL
            rows.append(("same source, diff seed", c, c, s1, s2,
                         abs(corr(cache[(c, s1)], cache[(c, s2)]))))
    for (c1, c2) in itertools.combinations(probe_cols, 2):  # what the bank does today
        for s1, s2 in ((probe_seeds[0], probe_seeds[1]), (probe_seeds[2], probe_seeds[3])):
            rows.append(("diff source, diff seed", c1, c2, s1, s2,
                         abs(corr(cache[(c1, s1)], cache[(c2, s2)]))))
    for (c1, c2) in itertools.combinations(probe_cols, 2):  # fps-3jj.20's failure mode
        for s in probe_seeds[:2]:
            rows.append(("diff source, SAME seed", c1, c2, s, s,
                         abs(corr(cache[(c1, s)], cache[(c2, s)]))))

    reuse = pd.DataFrame(rows, columns=["kind", "col_a", "col_b", "seed_a", "seed_b", "abs_rho"])
    reuse.to_csv(OUT / "reuse_channel.csv", index=False)
    print(reuse.groupby("kind")["abs_rho"].agg(
        n="size", median="median", p95=lambda s: s.quantile(0.95), max="max"
    ).to_string())
    del cache

    # ---- C. whole-bank comparison ----------------------------------------------------
    print("\n=== C. whole-bank cross-draw correlation ===")

    def bank_cross_draw(draws: list[list[tuple[str, int, float]]], label: str) -> pd.DataFrame:
        series = {
            (i, j): make_placebo_series(frame, col, seed).to_numpy(dtype=np.float32)
            for i, draw in enumerate(draws)
            for j, (col, seed, _c) in enumerate(draw)
        }
        out = []
        for (ia, ja), (ib, jb) in itertools.combinations(series, 2):
            if ia == ib:
                continue  # within-draw pairs are disclosed, not gated (generator.md)
            out.append((label, ia, ib, draws[ia][ja][0], draws[ib][jb][0],
                        draws[ia][ja][1], draws[ib][jb][1],
                        abs(corr(series[(ia, ja)], series[(ib, jb)]))))
        return pd.DataFrame(out, columns=[
            "bank", "draw_a", "draw_b", "col_a", "col_b", "seed_a", "seed_b", "abs_rho"])

    # Bank A is read from the COMMITTED ruler, not re-derived by calling
    # `screen_draw_groups`. That call would now return the POST-fix construction, so the
    # script would silently stop measuring the thing it is arguing against — and the headline
    # 0.965 would quietly become ~0.26, unreproducible from its own write-up. The committed
    # `noise_floor.json` is the bank batch1's dossiers were actually graded against, which is
    # the honest object of comparison anyway.
    floor = json.loads((BATCH / "noise_floor.json").read_text())
    k = int(floor["n_placebo_columns"])
    members = [
        (m["source_column"], m["block_seed"], float(m["self_correlation"]))
        for m in floor["placebo_draws"]
    ]
    current = [members[i:i + k] for i in range(0, len(members), k)]
    cur = bank_cross_draw(
        current, f"A: committed {len(current)}x{k} ruler (distinct columns, seed pool)"
    )
    print(f"[{time.time()-t0:6.1f}s] bank A done")

    # Proposed: a DECK. Each lap is `select_draws`'s own even spread over the whole column
    # list (a permutation of all of them, so the lap preserves today's texture
    # anti-clustering); when the deck empties it is reshuffled by rotating the spread, so
    # columns are reused as uniformly as possible instead of at random. Every column in the
    # bank gets its own seed off a flat counter that never wraps, so fps-3jj.20's
    # primary/fallback seed collision cannot occur either. For m <= n_cols picks this is
    # exactly one lap, i.e. byte-identical to `select_draws` today.
    n_draws_b, arity_b = 20, 3
    proposed = deck_draws(usable, n_draws_b, arity_b)
    prop = bank_cross_draw(proposed, "B: proposed 20x3 (reuse across draws)")
    print(f"[{time.time()-t0:6.1f}s] bank B done")

    banks = pd.concat([cur, prop], ignore_index=True)
    banks.to_csv(OUT / "bank_cross_draw.csv", index=False)
    print(banks.groupby("bank")["abs_rho"].agg(
        n="size", median="median", p95=lambda s: s.quantile(0.95), max="max"
    ).to_string())
    print("\n  worst cross-draw pair per bank:")
    for label, grp in banks.groupby("bank"):
        w = grp.loc[grp["abs_rho"].idxmax()]
        print(f"    {label}: {w.abs_rho:.3f}  d{w.draw_a} {w.col_a}@{w.seed_a} <-> "
              f"d{w.draw_b} {w.col_b}@{w.seed_b}")

    # ---- D. texture reuse the current bank ALREADY has -------------------------------
    print("\n=== D. texture-family reuse already present ===")
    for label, draws in (("A: current 10x3", current), ("B: proposed 20x3", proposed)):
        fams = [{texture_family(c) for c, _s, _r in d} for d in draws]
        shared = sum(1 for a, b in itertools.combinations(fams, 2) if a & b)
        total = len(fams) * (len(fams) - 1) // 2
        allf = pd.Series([texture_family(c) for d in draws for c, _s, _r in d]).value_counts()
        print(f"  {label}: {shared}/{total} draw pairs share >=1 texture family; "
              f"families used {dict(allf)}")

    print(f"\n[{time.time()-t0:6.1f}s] done")


if __name__ == "__main__":
    main()
