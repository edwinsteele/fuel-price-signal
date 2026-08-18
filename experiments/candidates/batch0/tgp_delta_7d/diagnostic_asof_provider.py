import json
import pathlib
import time

import pandas as pd

from experiments.lib.constants import SHOCK_FOLDS
from experiments.lib.realised import ArmSpec, run_paired_realised_backtest
from fuel_signal.features import load_features

BATCH_DIR = pathlib.Path("experiments/batches/batch0")
COL = "tgp_delta_7d"

t0 = time.perf_counter()
frame = load_features(BATCH_DIR / "features.csv")  # no .csv in this dir -> falls back to features.parquet, same as runner.py
baseline_columns = json.loads((BATCH_DIR / "baseline_columns.json").read_text())
candidate_cols = baseline_columns + [COL]

nuniq = frame.groupby("price_date")[COL].nunique(dropna=False)
print(f"tgp_delta_7d distinct values per date: max={nuniq.max()} (expect 1 -- market-wide, not station-specific)")

pit = frame[["price_date", COL]].drop_duplicates("price_date").set_index("price_date")[COL].sort_index()
pit.index = pd.to_datetime(pit.index)

stats = {"hits": 0, "misses": 0}


def asof_provider(as_of, station_code, station_price):
    val = pit.asof(pd.Timestamp(as_of))
    if pd.isna(val):
        stats["misses"] += 1
        return {COL: None}
    stats["hits"] += 1
    return {COL: float(val)}


arms = [
    ArmSpec("R0", frame, feature_columns=baseline_columns),
    ArmSpec("candidate", frame, feature_columns=candidate_cols, extra_feature_provider=asof_provider),
]

res = run_paired_realised_backtest(
    arms, baseline_columns, seed=42, held_tau=None,
    inner_fold_params={"train_min_days": 1095, "val_days": 90, "step_days": 90},
    db_path=BATCH_DIR / "fuel_signal.db", collect_fills=True,
)

print("\n=== aggregate (pooled CPL per arm, own + held tau) ===")
print(res.aggregate.to_string(index=False))

base = res.aggregate.set_index("arm")
d_held = base.loc["candidate", "cpl_held"] - base.loc["R0", "cpl_held"]
print(f"\ndelta_cpl_held={d_held:+.4f}  (batch0 harness's exact-key provider got +0.0941)")
print(f"asof provider hits={stats['hits']} misses={stats['misses']}  (harness's exact-key provider: 822 hits / 88 misses)")

print("\n=== per-fold delta_cpl_held (held tau), tagged shock/normal ===")
piv = res.deltas.pivot_table(index="fold", columns="arm", values="delta_cpl_held")
piv["regime"] = ["shock" if f in SHOCK_FOLDS else "normal" for f in piv.index]
print(piv.to_string(float_format=lambda x: f"{x:+.4f}"))

print(f"\n[wall] {time.perf_counter() - t0:.1f}s")
