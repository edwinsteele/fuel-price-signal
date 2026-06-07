"""Step 1: do the three candidate signals separate extended-descent from
normal late-descent rows? Plus side probes for #4 (cycle persistence) and #5
(drop-size separability).

Usage: PYTHONPATH=. uv run python experiments/2026-06-06_late_descent_triplet/step1_information_probe.py
"""
from __future__ import annotations

import json
import time
import pathlib

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from fuel_signal.features import load_features

OUT_DIR = pathlib.Path("experiments/2026-06-06_late_descent_triplet")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Phase proxies (from cycle_pct_through = days_since_last_peak / mean_cycle_length)
# Per memory project_cycle_pct_through_semantics: empirical trough is ~pct=0.5
# (peak-anchored ratio; cycle is peak→trough→peak, non-monotonic).
# So normal late-descent rows are the descending arm leading into trough.
PHASE_NORMAL_LATE = (0.30, 0.50)   # descending arm in normal-length cycles
PHASE_EXTENDED = (0.90, np.inf)    # cycle past expected length — elongation regime

# Stickiness percentile to split sticky vs competitive cohorts
STICKY_PCTL = 0.75


def standardised_mean_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d using pooled std. Positive = a > b."""
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) < 50 or len(b) < 50:
        return float("nan")
    pooled = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / (len(a) + len(b) - 2))
    if pooled == 0:
        return float("nan")
    return (a.mean() - b.mean()) / pooled


def report_split(name: str, normal: np.ndarray, extended: np.ndarray, direction: str) -> dict:
    """Print + return a comparison summary for one signal."""
    smd = standardised_mean_diff(extended, normal)  # positive = extended > normal
    out = {
        "signal": name,
        "n_normal": int((~np.isnan(normal)).sum()),
        "n_extended": int((~np.isnan(extended)).sum()),
        "mean_normal": float(np.nanmean(normal)),
        "mean_extended": float(np.nanmean(extended)),
        "std_normal": float(np.nanstd(normal)),
        "std_extended": float(np.nanstd(extended)),
        "smd_extended_minus_normal": float(smd),
        "expected_direction": direction,
        "verdict": "SEPARATES" if abs(smd) >= 0.3 else ("weak" if abs(smd) >= 0.15 else "no"),
    }
    print(f"\n--- {name} ---")
    print(f"  normal late-descent rows  : n={out['n_normal']:>8,}  mean={out['mean_normal']:+.4f}  std={out['std_normal']:.4f}")
    print(f"  extended-descent rows     : n={out['n_extended']:>8,}  mean={out['mean_extended']:+.4f}  std={out['std_extended']:.4f}")
    print(f"  SMD (extended − normal)   : {smd:+.3f}  (expected: {direction})")
    print(f"  verdict                   : {out['verdict']}")
    return out


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
t0 = time.perf_counter()
print("Loading features (via load_features helper)...")
df = load_features()
df["price_date"] = pd.to_datetime(df["price_date"])
print(f"  loaded {len(df):,} rows in {time.perf_counter()-t0:.1f}s")
print(f"  date range: {df.price_date.min().date()} to {df.price_date.max().date()}")
print(f"  stations: {df.station_code.nunique():,}, dates: {df.price_date.nunique():,}")
t_load = time.perf_counter() - t0

# Phase partition masks (computed once)
normal_mask = (df.cycle_pct_through >= PHASE_NORMAL_LATE[0]) & (df.cycle_pct_through < PHASE_NORMAL_LATE[1])
extended_mask = df.cycle_pct_through >= PHASE_EXTENDED[0]
print(f"\nPhase partition (proxy for regime):")
print(f"  normal late-descent (pct ∈ [{PHASE_NORMAL_LATE[0]}, {PHASE_NORMAL_LATE[1]})): {normal_mask.sum():>9,} rows")
print(f"  extended descent    (pct ≥ {PHASE_EXTENDED[0]}                  ): {extended_mask.sum():>9,} rows")

results: list[dict] = []

# ---------------------------------------------------------------------------
# Probe A: Cross-station dispersion (competitive cohort only)
# Per (date), std of station_price_cents over non-sticky stations.
# Expectation: lower in normal late-descent (compression into trough);
# higher / flat in extended descent.
# ---------------------------------------------------------------------------
t = time.perf_counter()
print("\n[A] computing per-date dispersion over competitive stations...")
sticky_threshold = df["stickiness_score"].quantile(STICKY_PCTL)
print(f"    sticky threshold (p{int(STICKY_PCTL*100)} of stickiness_score): {sticky_threshold:.2f}c")
competitive = df[df["stickiness_score"] < sticky_threshold].copy()
date_dispersion = competitive.groupby("price_date")["station_price_cents"].agg(
    n_stations="size",
    px_std="std",
    px_iqr=lambda x: x.quantile(0.75) - x.quantile(0.25),
)
# Join dispersion back per row by date
df = df.join(date_dispersion[["px_std", "px_iqr"]], on="price_date")
print(f"    computed in {time.perf_counter()-t:.1f}s")

results.append(report_split(
    "A1: network px_std (competitive)",
    df.loc[normal_mask, "px_std"].values,
    df.loc[extended_mask, "px_std"].values,
    direction="positive (extended > normal)",
))
results.append(report_split(
    "A2: network px_iqr (competitive)",
    df.loc[normal_mask, "px_iqr"].values,
    df.loc[extended_mask, "px_iqr"].values,
    direction="positive (extended > normal)",
))

# ---------------------------------------------------------------------------
# Probe B: Sticky-floor reference gap
# Per (date), gap = competitive_min − sticky_median.
# Expectation: smaller in normal late-descent (competitive floor near sticky
# baseline); larger / flat in extended descent (floor itself sliding down,
# so competitive prices are well below sticky baseline persistently).
# ---------------------------------------------------------------------------
t = time.perf_counter()
print("\n[B] computing per-date sticky-floor gap...")
sticky = df[df["stickiness_score"] >= sticky_threshold]
sticky_median_by_date = sticky.groupby("price_date")["station_price_cents"].median()
competitive_min_by_date = competitive.groupby("price_date")["station_price_cents"].quantile(0.05)
# 5th percentile is robust to outliers vs raw min
gap_by_date = (competitive_min_by_date - sticky_median_by_date).rename("sticky_floor_gap")
df = df.join(gap_by_date, on="price_date")
print(f"    computed in {time.perf_counter()-t:.1f}s")

results.append(report_split(
    "B: sticky_floor_gap (comp_p05 − sticky_med)",
    df.loc[normal_mask, "sticky_floor_gap"].values,
    df.loc[extended_mask, "sticky_floor_gap"].values,
    direction="negative (extended gap more negative, comp floor further below stickies)",
))

# ---------------------------------------------------------------------------
# Probe C: Leader divergence
# Per (date), spread of days_since_trough_entry across LGAs.
# Expectation: smaller in normal late-descent (LGAs roughly in phase);
# larger in extended descent (LGAs out of phase, some have troughed,
# others haven't).
# ---------------------------------------------------------------------------
t = time.perf_counter()
print("\n[C] computing per-date LGA-leader divergence...")
lga_cols = [c for c in df.columns if c.startswith("days_since_trough_entry_")
            # Brand cols also use this prefix in features.csv — strip those
            and c.replace("days_since_trough_entry_", "")
            not in {"7_eleven", "ampol_foodary", "bp", "budget", "eg_ampol",
                    "independent", "metro_fuel", "reddy_express", "shell", "speedway"}]
print(f"    LGA columns identified: {len(lga_cols)}")
# Per-date these values are constant across stations — take first row per date
per_date = df.drop_duplicates("price_date").set_index("price_date")[lga_cols]
# Spread metrics
per_date_spread = per_date.std(axis=1).rename("lga_phase_std")
per_date_iqr = (per_date.quantile(0.75, axis=1) - per_date.quantile(0.25, axis=1)).rename("lga_phase_iqr")
df = df.join(per_date_spread, on="price_date")
df = df.join(per_date_iqr, on="price_date")
print(f"    computed in {time.perf_counter()-t:.1f}s")

results.append(report_split(
    "C1: lga_phase_std (days_since_trough)",
    df.loc[normal_mask, "lga_phase_std"].values,
    df.loc[extended_mask, "lga_phase_std"].values,
    direction="positive (extended > normal)",
))
results.append(report_split(
    "C2: lga_phase_iqr (days_since_trough)",
    df.loc[normal_mask, "lga_phase_iqr"].values,
    df.loc[extended_mask, "lga_phase_iqr"].values,
    direction="positive (extended > normal)",
))

# Save per-row signal sample (only the new columns) for downstream plotting
sample_cols = ["station_code", "price_date", "cycle_pct_through",
               "px_std", "px_iqr", "sticky_floor_gap",
               "lga_phase_std", "lga_phase_iqr"]
df[sample_cols].sample(min(200_000, len(df)), random_state=42).to_csv(
    OUT_DIR / "step1_signals_sample.csv", index=False)

# ---------------------------------------------------------------------------
# SIDE PROBES (user requested rough data)
# ---------------------------------------------------------------------------

# --- #4 cycle persistence (autocorr of cycle length per station) ---
t = time.perf_counter()
print("\n[#4] cycle-length persistence (per-station autocorr)...")
# Detect troughs from station_price_cents per station; lighter sample
station_counts = df.groupby("station_code").size()
long_stations = station_counts[station_counts > 1000].index
rng = np.random.default_rng(42)
sample_stations = rng.choice(long_stations, size=min(50, len(long_stations)), replace=False)
acorrs1, acorrs2 = [], []
all_cycle_lens = []
all_troughs_dow = []
for sc in sample_stations:
    s = df[df.station_code == sc].sort_values("price_date")
    if len(s) < 200:
        continue
    smoothed = s["station_price_cents"].rolling(3, center=True, min_periods=1).mean().values
    trough_idx, _ = find_peaks(-smoothed, distance=15, prominence=5)
    if len(trough_idx) < 6:
        continue
    trough_dates = s["price_date"].iloc[trough_idx].values
    cycle_lens = np.diff([pd.Timestamp(d).toordinal() for d in trough_dates])
    all_cycle_lens.extend(cycle_lens.tolist())
    all_troughs_dow.extend([pd.Timestamp(d).day_name() for d in trough_dates])
    if cycle_lens.std() == 0:
        continue
    if len(cycle_lens) >= 3:
        acorrs1.append(np.corrcoef(cycle_lens[:-1], cycle_lens[1:])[0, 1])
    if len(cycle_lens) >= 5:
        acorrs2.append(np.corrcoef(cycle_lens[:-2], cycle_lens[2:])[0, 1])

acorrs1 = np.array(acorrs1); acorrs2 = np.array(acorrs2)
cycle_persistence = {
    "n_stations_sampled": int(len(sample_stations)),
    "n_stations_with_acorr": int(len(acorrs1)),
    "pooled_cycle_len_n": int(len(all_cycle_lens)),
    "pooled_cycle_len_mean": float(np.mean(all_cycle_lens)),
    "pooled_cycle_len_median": float(np.median(all_cycle_lens)),
    "pooled_cycle_len_p25": float(np.quantile(all_cycle_lens, 0.25)),
    "pooled_cycle_len_p75": float(np.quantile(all_cycle_lens, 0.75)),
    "pooled_cycle_len_p90": float(np.quantile(all_cycle_lens, 0.90)),
    "lag1_acorr_mean": float(acorrs1.mean()),
    "lag1_acorr_median": float(np.median(acorrs1)),
    "lag1_pct_positive": float((acorrs1 > 0).mean()),
    "lag1_pct_strong": float((acorrs1 > 0.3).mean()),
    "lag2_acorr_mean": float(acorrs2.mean()),
    "lag2_acorr_median": float(np.median(acorrs2)),
}
print(f"    cycle length pool: n={cycle_persistence['pooled_cycle_len_n']}, "
      f"median={cycle_persistence['pooled_cycle_len_median']:.1f}d, "
      f"p75={cycle_persistence['pooled_cycle_len_p75']:.1f}d, "
      f"p90={cycle_persistence['pooled_cycle_len_p90']:.1f}d")
print(f"    lag-1 acorr: mean={cycle_persistence['lag1_acorr_mean']:+.3f}, "
      f"pct>0={cycle_persistence['lag1_pct_positive']*100:.0f}%, "
      f"pct>0.3={cycle_persistence['lag1_pct_strong']*100:.0f}%")
print(f"    lag-2 acorr: mean={cycle_persistence['lag2_acorr_mean']:+.3f}")

# Tuesday sanity
dow_counts = pd.Series(all_troughs_dow).value_counts().reindex(
    ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
)
print(f"    trough day-of-week distribution (expect uniform={len(all_troughs_dow)/7:.0f}/day):")
for d, c in dow_counts.items():
    bar = "#" * int(40 * c / dow_counts.max())
    print(f"      {d:<10}: {c:>4} {bar}")
print(f"    [#4 took {time.perf_counter()-t:.1f}s]")

# --- #5 drop-size separability (bimodality check) ---
t = time.perf_counter()
print("\n[#5] drop-size separability — are big drops a distinct class?")
# Daily delta per station (sample for speed)
sub = df[df.station_code.isin(sample_stations)].sort_values(["station_code", "price_date"])
sub["delta"] = sub.groupby("station_code")["station_price_cents"].diff()
drops = sub[sub["delta"] < 0].copy()
drops["drop_size"] = -drops["delta"]
print(f"    drop-day n={len(drops):,}")
print("    quantiles of drop size (cents):")
qs = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
for q in qs:
    print(f"      p{int(q*100):02d}: {drops['drop_size'].quantile(q):.2f}c")
print(f"      mean: {drops['drop_size'].mean():.2f}c, std: {drops['drop_size'].std():.2f}c")
print("    histogram (1c bins, 0–20c) — bimodal would show a shoulder near 5–10c:")
edges = np.arange(0, 21, 1)
hist, _ = np.histogram(drops['drop_size'], bins=edges)
max_h = hist.max()
for i, c in enumerate(hist):
    bar = "#" * int(60 * c / max_h)
    print(f"      [{edges[i]:>2}–{edges[i+1]:>2}c): {c:>7,} ({100*c/len(drops):5.1f}%) {bar}")

drop_separability = {
    "n_drop_days": int(len(drops)),
    "quantiles": {f"p{int(q*100)}": float(drops['drop_size'].quantile(q)) for q in qs},
    "mean": float(drops["drop_size"].mean()),
    "std": float(drops["drop_size"].std()),
    "histogram_1c_bins": hist.tolist(),
    "histogram_edges": edges.tolist(),
}
print(f"    [#5 took {time.perf_counter()-t:.1f}s]")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
total_elapsed = time.perf_counter() - t0
print(f"\n=== TOTAL ELAPSED: {total_elapsed:.1f}s (load: {t_load:.1f}s) ===")

# Persist machine-readable summary
summary = {
    "experiment": "2026-06-06_late_descent_triplet/step1_information_probe",
    "phase_partition": {
        "normal_late_descent": {"cycle_pct_through": list(PHASE_NORMAL_LATE),
                                "n_rows": int(normal_mask.sum())},
        "extended_descent": {"cycle_pct_through": [PHASE_EXTENDED[0], None],
                             "n_rows": int(extended_mask.sum())},
    },
    "sticky_threshold_cents": float(sticky_threshold),
    "probe_results": results,
    "side_probe_cycle_persistence": cycle_persistence,
    "side_probe_trough_dow": dow_counts.to_dict(),
    "side_probe_drop_separability": drop_separability,
    "wall_times_sec": {"total": total_elapsed, "load_features": t_load},
}
with open(OUT_DIR / "step1_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)
print(f"\nSummary written to {OUT_DIR / 'step1_summary.json'}")
print(f"Per-row signal sample written to {OUT_DIR / 'step1_signals_sample.csv'}")

# Decision rule
graduated = [r for r in results if r["verdict"] == "SEPARATES"]
print("\n=== STEP 1 DECISION ===")
if graduated:
    print(f"GRADUATE to Step 2. Signals separating extended vs normal late-descent:")
    for r in graduated:
        print(f"  - {r['signal']}: SMD={r['smd_extended_minus_normal']:+.3f}")
else:
    weak = [r for r in results if r["verdict"] == "weak"]
    if weak:
        print(f"INCONCLUSIVE — only weak signals (0.15 ≤ |SMD| < 0.3):")
        for r in weak:
            print(f"  - {r['signal']}: SMD={r['smd_extended_minus_normal']:+.3f}")
        print("  Recommend manual review of histograms before deciding Step 2.")
    else:
        print("DO NOT graduate. No signal showed meaningful separation.")
        print("Write up negative result; close late-descent track in favour of external data.")
