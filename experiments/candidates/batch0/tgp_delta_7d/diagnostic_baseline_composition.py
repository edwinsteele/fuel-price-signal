"""Lead 1 (bd fps-nor): does the R0 baseline's composition explain the sign flip?

June (2026-06-20_leading_indicators/realised_tgp.py) graduated `tgp_delta_7d`
against a 54-column BASE and got pooled delta_cpl_held = -0.039 c/L. batch0
re-tested it against a 64-column R0 -- June's exact 54 PLUS 10
`days_since_trough_entry_<brand>` columns that did not exist in June -- and got
+0.0941 c/L, opposite sign. `diagnostic_asof_provider.py` already ruled out the
provider mechanism (bit-identical result), and the TGP series itself is
bit-identical between June's xlsx and batch0's parquet column (verified
2026-08-19, 3520 overlapping dates, max|diff| = 0.0).

What's left is the baseline. This script reruns the SAME frozen batch0
snapshot / seed / fold geometry with THREE arms:

  R0_54    June's exact 54-column BASE                (expect ~189.792 if the
                                                       June vintage reproduces)
  cand_54  BASE + tgp_delta_7d, June's .asof provider (expect ~189.753)
  R0_64    batch0's 64-column baseline, as an in-process anchor
                                                      (expect 189.660)

Reading it:
  * cand_54 - R0_54 ~ -0.039  ->  June reproduces on the batch0 snapshot. The
    baseline composition IS the explanation, and data-vintage decay (lead 3) is
    ruled out in the same stroke.
  * R0_64 ~ 189.660  ->  this process reproduces the batch0 run, so any
    divergence in the pair above is about the columns, not the environment.
  * cand_54 - R0_54 still positive -> baseline composition is NOT the cause;
    move to lead 2 (fold 13) / lead 3 (vintage).

Run (heavy, ~35 min, 3 arms x 14 outer folds):
  PYTHONPATH=. uv run python experiments/candidates/batch0/tgp_delta_7d/diagnostic_baseline_composition.py \
    2>&1 | tee experiments/candidates/batch0/tgp_delta_7d/diagnostic_baseline_composition.log
"""
from __future__ import annotations

import json
import pathlib
import time

import pandas as pd

from experiments.lib.constants import SHOCK_FOLDS
from experiments.lib.realised import ArmSpec, run_paired_realised_backtest
from fuel_signal.features import (
    FEATURE_COLUMNS,
    LGA_FEATURE_COLUMNS,
    NETWORK_FEATURE_COLUMNS,
    load_features,
)

HERE = pathlib.Path(__file__).resolve().parent
BATCH_DIR = pathlib.Path("experiments/batches/batch0")
COL = "tgp_delta_7d"

# June's BASE, reconstructed from the same three production lists realised_tgp.py
# used. Asserted == batch0's baseline minus the 10 brand-trough columns below.
BASE_54 = FEATURE_COLUMNS + LGA_FEATURE_COLUMNS + NETWORK_FEATURE_COLUMNS

# Same inner-OOF geometry as both June's run and the batch0 harness.
INNER_FOLDS = {"train_min_days": 1095, "val_days": 90, "step_days": 90}


def main() -> None:
    t0 = time.perf_counter()

    frame = load_features(BATCH_DIR / "features.csv")  # no .csv here -> parquet, as runner.py
    baseline_64 = json.loads((BATCH_DIR / "baseline_columns.json").read_text())

    extra = sorted(set(baseline_64) - set(BASE_54))
    missing = sorted(set(BASE_54) - set(baseline_64))
    print(f"BASE_54 n={len(BASE_54)}  batch0 baseline n={len(baseline_64)}")
    print(f"  batch0-only ({len(extra)}): {extra}")
    print(f"  BASE_54-only ({len(missing)}): {missing}")
    assert len(BASE_54) == 54, f"expected June's 54-column BASE, got {len(BASE_54)}"
    assert not missing, "batch0 baseline must be a strict superset of June's BASE"
    assert all(c.startswith("days_since_trough_entry_") for c in extra), extra

    # June's live provider, verbatim in mechanism: one market-wide value per date,
    # looked up .asof(decision date). Confirmed market-wide before use.
    nuniq = frame.groupby("price_date")[COL].nunique(dropna=False).max()
    print(f"{COL} distinct values per date: max={nuniq} (expect 1 -- market-wide)")
    assert nuniq == 1, "tgp_delta_7d is not market-wide; the .asof provider would be wrong"

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
        ArmSpec("R0_54", frame, feature_columns=BASE_54),
        ArmSpec("cand_54", frame, feature_columns=BASE_54 + [COL],
                extra_feature_provider=asof_provider),
        ArmSpec("R0_64", frame, feature_columns=baseline_64),
    ]

    res = run_paired_realised_backtest(
        arms, BASE_54, seed=42, held_tau=None,
        inner_fold_params=INNER_FOLDS,
        db_path=BATCH_DIR / "fuel_signal.db", collect_fills=True,
    )

    res.per_window.to_csv(HERE / "baseline_composition_per_window.csv", index=False)
    res.aggregate.to_csv(HERE / "baseline_composition_aggregate.csv", index=False)
    res.deltas.to_csv(HERE / "baseline_composition_deltas.csv", index=False)
    if res.fills is not None:
        res.fills.to_csv(HERE / "baseline_composition_fills.csv", index=False)

    print("\n=== aggregate (pooled CPL per arm, own + held tau) ===")
    print(res.aggregate.to_string(index=False))
    print(f"\nasof provider hits={stats['hits']} misses={stats['misses']}")

    agg = res.aggregate.set_index("arm")
    print("\n=== the 2x2 (pooled cpl_held; June's numbers for reference) ===")
    print(f"  54-col baseline   this run {agg.loc['R0_54', 'cpl_held']:.4f}   June 189.7923")
    print(f"  54-col + tgp      this run {agg.loc['cand_54', 'cpl_held']:.4f}   June 189.7529")
    print(f"  64-col baseline   this run {agg.loc['R0_64', 'cpl_own']:.4f}   batch0 189.6600  (own tau)")
    print("  64-col + tgp      (not rerun)          batch0 189.7541")
    d54 = agg.loc["cand_54", "cpl_held"] - agg.loc["R0_54", "cpl_held"]
    print(f"\n  delta on 54-col baseline: {d54:+.4f}   (June: -0.0394)")
    print("  delta on 64-col baseline: +0.0941  (batch0, already established)")

    print("\n=== per-fold delta_cpl_held vs R0_54, tagged shock/normal ===")
    piv = res.deltas.pivot_table(index="fold", columns="arm", values="delta_cpl_held")
    piv["regime"] = ["shock" if f in SHOCK_FOLDS else "normal" for f in piv.index]
    print(piv.to_string(float_format=lambda x: f"{x:+.4f}"))

    print(f"\n[wall] {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
