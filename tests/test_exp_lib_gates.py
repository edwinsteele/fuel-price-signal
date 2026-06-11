"""Tests for experiments/lib/gates — GateSpec and evaluate_gates."""
from __future__ import annotations

import math

import pandas as pd

from experiments.lib.gates import GateSpec, evaluate_gates


def _fold_run(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


SPEC = GateSpec(
    cohort_col="delta_ll_hard25_median",
    pop_col="delta_ll_all_median",
    target_fold=7,
    target_max=-0.04,
    worst_fold_max=0.01,
    net_pop_max=0.0,
)


def _passing_rows(run: str = "R1") -> list[dict]:
    return [
        {"fold": f, "run": run, "delta_ll_hard25_median": -0.05, "delta_ll_all_median": -0.02}
        for f in range(1, 15)
        if f != 7
    ] + [{"fold": 7, "run": run, "delta_ll_hard25_median": -0.06, "delta_ll_all_median": -0.03}]


def test_all_gates_pass():
    fold_run = _fold_run(_passing_rows())
    results = evaluate_gates(fold_run, SPEC, "R1")
    assert all(g["passed"] for g in results), results


def test_gate_names_and_structure():
    fold_run = _fold_run(_passing_rows())
    results = evaluate_gates(fold_run, SPEC, "R1")
    assert len(results) == 3
    for g in results:
        assert {"name", "threshold", "value", "passed"} == set(g.keys())


def test_target_fold_gate_fails_when_improvement_insufficient():
    rows = _passing_rows()
    # fold 7 barely misses the -0.04 threshold
    for r in rows:
        if r["fold"] == 7:
            r["delta_ll_hard25_median"] = -0.03
    fold_run = _fold_run(rows)
    results = evaluate_gates(fold_run, SPEC, "R1")
    target_gate = next(g for g in results if g["name"].startswith("target_fold"))
    assert not target_gate["passed"]
    assert math.isclose(target_gate["value"], -0.03)
    assert math.isclose(target_gate["threshold"], -0.04)


def test_worst_fold_gate_fails_on_regression():
    rows = _passing_rows()
    # inject a fold with a regression beyond +0.01
    rows.append({"fold": 9, "run": "R1", "delta_ll_hard25_median": 0.05, "delta_ll_all_median": 0.02})
    fold_run = _fold_run(rows)
    results = evaluate_gates(fold_run, SPEC, "R1")
    worst_gate = next(g for g in results if g["name"].startswith("worst_fold"))
    assert not worst_gate["passed"]
    assert math.isclose(worst_gate["value"], 0.05)


def test_net_pop_gate_fails_when_mean_positive():
    rows = [
        {"fold": f, "run": "R1", "delta_ll_hard25_median": -0.05, "delta_ll_all_median": 0.01}
        for f in range(1, 15)
    ]
    fold_run = _fold_run(rows)
    results = evaluate_gates(fold_run, SPEC, "R1")
    net_gate = next(g for g in results if g["name"].startswith("net_pop"))
    assert not net_gate["passed"]


def test_sign_edge_exactly_at_threshold_passes():
    # value == threshold is a pass (<=)
    rows = _passing_rows()
    for r in rows:
        if r["fold"] == 7:
            r["delta_ll_hard25_median"] = -0.04  # exactly at target_max
    fold_run = _fold_run(rows)
    results = evaluate_gates(fold_run, SPEC, "R1")
    target_gate = next(g for g in results if g["name"].startswith("target_fold"))
    assert target_gate["passed"]


def test_run_filtered_correctly():
    rows = _passing_rows("R1") + [
        {"fold": f, "run": "R2", "delta_ll_hard25_median": 0.1, "delta_ll_all_median": 0.1}
        for f in range(1, 15)
    ]
    fold_run = _fold_run(rows)
    results = evaluate_gates(fold_run, SPEC, "R1")
    assert all(g["passed"] for g in results), "R2's bad values should not affect R1 evaluation"


def test_missing_target_fold_yields_nan_and_fails():
    rows = [
        r for r in _passing_rows() if r["fold"] != 7
    ]
    fold_run = _fold_run(rows)
    results = evaluate_gates(fold_run, SPEC, "R1")
    target_gate = next(g for g in results if g["name"].startswith("target_fold"))
    assert math.isnan(target_gate["value"])
    assert not target_gate["passed"]
