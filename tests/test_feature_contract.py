"""The feature-scope contract (fps-zci) — one symbol, checked against ground truth.

Two contract defects reached production-grade experiment results before anything
noticed: fps-sa1 (R0 resolved to 64 columns by *discovering* the rejected Phase 4b
brand troughs in the features header) and fps-zci (the right 54 columns in sorted
order, which fits a different LightGBM model). Both were invisible because nothing
compared the code's idea of the contract to the artifact that defines it, and
nothing recorded which baseline a result was measured against.
"""
from __future__ import annotations

import pathlib

import pandas as pd
import pytest

from fuel_signal.features import (
    FEATURE_COLUMNS,
    LGA_FEATURE_COLUMNS,
    LOCKED_FEATURE_COLUMNS,
    LOCKED_FEATURE_FINGERPRINT,
    NETWORK_FEATURE_COLUMNS,
    NON_MODEL_COLUMNS,
    NON_MODEL_REASON_INCONCLUSIVE,
    NON_MODEL_REASON_REJECTED,
    baseline_fingerprint,
    non_model_columns,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CALIBRATED_MODEL_PATH = REPO_ROOT / "data" / "models" / "lgbm_calibrated.joblib"


# ── the symbol ────────────────────────────────────────────────────────────────

def test_locked_feature_columns_is_the_group_composition_in_production_order():
    assert LOCKED_FEATURE_COLUMNS == (
        FEATURE_COLUMNS + LGA_FEATURE_COLUMNS + NETWORK_FEATURE_COLUMNS
    )


def test_locked_feature_columns_is_not_sorted():
    """Production order is append order, not alphabetical (fps-zci).

    LightGBM breaks equal-gain split ties by feature index, so a permutation of the
    same columns fits a different model — measured at 732/47,823 val rows' changed
    probabilities and 0.038 c/L on batch0's pooled realised delta.
    """
    assert LOCKED_FEATURE_COLUMNS != sorted(LOCKED_FEATURE_COLUMNS)


def test_locked_feature_columns_has_no_duplicates():
    assert len(LOCKED_FEATURE_COLUMNS) == len(set(LOCKED_FEATURE_COLUMNS))


def test_locked_feature_columns_excludes_every_non_model_column():
    assert not set(LOCKED_FEATURE_COLUMNS) & set(NON_MODEL_COLUMNS)


def test_locked_feature_columns_contains_no_brand_trough_column():
    """The fps-sa1 shape, asserted WITHOUT the model artifact.

    The ordered-equality check against lgbm_calibrated.joblib is the real ground
    truth, but data/models/ is gitignored, so that test always skips in CI — leaving
    a corrupted lock to be caught by nothing. Brand troughs cannot be enumerated
    statically (they are DB-derived), so this asserts the rule instead: every
    trough-prefixed column in the lock must be one of the LGA troughs.
    """
    lga = set(LGA_FEATURE_COLUMNS)
    brand_like = [
        c for c in LOCKED_FEATURE_COLUMNS
        if c.startswith("days_since_trough_entry_") and c not in lga
    ]
    assert not brand_like, (
        "Phase 4b brand troughs were evaluated and rejected on 2026-06-02; they are "
        f"computed into features.csv but must not be in the lock. Found: {brand_like}"
    )


def test_non_model_detector_does_not_defer_to_the_lock(monkeypatch):
    """Guard the guard: a column wrongly IN the lock must still be reported.

    A detector that skipped whatever the lock already claims would be blind to
    exactly the defect it exists to find — the lock was the thing that was wrong in
    fps-sa1. So non_model_columns() decides by what a column IS, not by where it
    currently appears.
    """
    import fuel_signal.features as feats

    leaked = "days_since_trough_entry_zzz_test_brand"
    monkeypatch.setattr(feats, "LOCKED_FEATURE_COLUMNS", LOCKED_FEATURE_COLUMNS + [leaked])
    found = feats.non_model_columns(_frame(leaked))
    assert found[leaked][0] == NON_MODEL_REASON_REJECTED


# ── ground truth ──────────────────────────────────────────────────────────────

@pytest.mark.skipif(
    not CALIBRATED_MODEL_PATH.exists(),
    reason=f"{CALIBRATED_MODEL_PATH} is gitignored; only present in a full local checkout",
)
def test_locked_feature_columns_equals_the_locked_model_artifact():
    """The artifact is authoritative; the constant must match it element-for-element.

    ORDERED equality, deliberately — set equality would have passed straight through
    the fps-zci defect, where the set was right and the order was not.
    """
    import joblib

    artifact = joblib.load(CALIBRATED_MODEL_PATH)
    assert artifact["feature_columns"] == LOCKED_FEATURE_COLUMNS, (
        "fuel_signal.features.LOCKED_FEATURE_COLUMNS has drifted from "
        f"{CALIBRATED_MODEL_PATH}. Either the constants moved without a re-lock, or "
        "the artifact on disk is not the locked model. Do not 'fix' this by sorting "
        "or reordering either side — see docs/CONVENTIONS.md."
    )


# ── the non-model registry ────────────────────────────────────────────────────

def _frame(*columns: str) -> pd.DataFrame:
    return pd.DataFrame({c: [0.0] for c in (list(LOCKED_FEATURE_COLUMNS) + list(columns))})


def test_non_model_columns_names_the_tgp_column_inconclusive():
    """tgp_delta_7d is INCONCLUSIVE, not held-out-pending-graduation.

    The distinction is load-bearing for readers, not for the model: the June 2026
    graduation was retracted (see the reason string), so nothing is "pending". A
    held-out code here reads as a queued retrain and is what a future session
    would act on.
    """
    found = non_model_columns(_frame("tgp_delta_7d"))
    assert found["tgp_delta_7d"][0] == NON_MODEL_REASON_INCONCLUSIVE


def test_non_model_columns_names_brand_troughs_as_rejected():
    """Brand troughs are DB-derived, so they are matched by rule, not by name."""
    found = non_model_columns(_frame("days_since_trough_entry_zzz_test_brand"))
    assert found["days_since_trough_entry_zzz_test_brand"][0] == NON_MODEL_REASON_REJECTED


def test_non_model_columns_does_not_claim_lga_troughs():
    """LGA troughs share the trough prefix but ARE in the lock — the asymmetry that
    made discovery-based scoping unable to tell 'not in scope' from 'in scope'."""
    found = non_model_columns(_frame("tgp_delta_7d"))
    assert not set(found) & set(LGA_FEATURE_COLUMNS)


def test_non_model_columns_ignores_identifier_and_label_columns():
    """Only columns that are computed FEATURES but excluded get a reason — idents and
    labels were never candidates for the contract."""
    found = non_model_columns(_frame("price_date", "station_code", "label"))
    assert found == {}


# ── the fingerprint ───────────────────────────────────────────────────────────

def test_baseline_fingerprint_leads_with_the_column_count():
    assert baseline_fingerprint(LOCKED_FEATURE_COLUMNS).startswith(
        f"{len(LOCKED_FEATURE_COLUMNS)}:"
    )


def test_baseline_fingerprint_distinguishes_a_permutation():
    """The fps-zci defect in one assertion: same set, different order, different run."""
    assert baseline_fingerprint(LOCKED_FEATURE_COLUMNS) != baseline_fingerprint(
        sorted(LOCKED_FEATURE_COLUMNS)
    )


def test_baseline_fingerprint_distinguishes_an_extra_column():
    """The fps-sa1 defect in one assertion: 54 vs 54-plus-a-rejected-group."""
    leaked = list(LOCKED_FEATURE_COLUMNS) + ["days_since_trough_entry_zzz_test_brand"]
    assert baseline_fingerprint(leaked) != baseline_fingerprint(LOCKED_FEATURE_COLUMNS)


def test_baseline_fingerprint_is_stable_across_calls():
    assert baseline_fingerprint(LOCKED_FEATURE_COLUMNS) == LOCKED_FEATURE_FINGERPRINT


def test_experiments_lib_constants_reexports_the_same_contract():
    """experiments/lib is the import surface for experiment scripts; a hand-written
    second copy there would be the drift this issue exists to remove."""
    from experiments.lib.constants import BASELINE_COLUMNS, BASELINE_FINGERPRINT

    assert BASELINE_COLUMNS == LOCKED_FEATURE_COLUMNS
    assert BASELINE_FINGERPRINT == LOCKED_FEATURE_FINGERPRINT


def test_experiments_lib_baseline_columns_is_a_copy_not_an_alias():
    """Equal in value, separate in identity — mutating the re-export must not reach
    through into fuel_signal's canonical list."""
    from experiments.lib import constants

    assert constants.BASELINE_COLUMNS is not LOCKED_FEATURE_COLUMNS
    scratch = constants.BASELINE_COLUMNS
    scratch.append("mutation_probe")
    try:
        assert "mutation_probe" not in LOCKED_FEATURE_COLUMNS
    finally:
        scratch.remove("mutation_probe")


def test_write_meta_does_not_drag_the_feature_layer_into_import_time():
    """experiments.lib.io is a leaf serialisation module and must stay one.

    fuel_signal.features transitively imports scipy.signal (via lga_leadership),
    which took io.py from ~27ms to ~700ms when the import sat at module scope. The
    import lives inside write_meta instead.
    """
    import ast

    tree = ast.parse((REPO_ROOT / "experiments" / "lib" / "io.py").read_text())
    module_level = [
        n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))
    ]
    offenders = [
        m for n in module_level for m in [getattr(n, "module", None) or ""]
        if m.startswith("fuel_signal") or m.startswith("experiments.lib.constants")
    ]
    assert not offenders, f"heavy import moved back to io.py module scope: {offenders}"


# ── the detector: no second copy of the composition ───────────────────────────

_GROUP_NAMES = {"FEATURE_COLUMNS", "LGA_FEATURE_COLUMNS", "NETWORK_FEATURE_COLUMNS"}

#: Surfaces that must go through the symbol. Dated experiment directories under
#: experiments/ are a lab book — frozen records of runs that already happened — so
#: they keep whatever composition they were run with; rewriting them would falsify
#: the record without changing a single result.
_LIVE_SURFACES = (
    REPO_ROOT / "fuel_signal",
    REPO_ROOT / "experiments" / "lib",
    REPO_ROOT / "experiments" / "pipeline",
    REPO_ROOT / "experiments" / "TEMPLATE_paired_wfcv.py",
)

#: The one place the composition is allowed to be written out — it is the definition.
_CONTRACT_DEFINITION = REPO_ROOT / "fuel_signal" / "features.py"


def _live_source_files() -> list[pathlib.Path]:
    files: list[pathlib.Path] = []
    for surface in _LIVE_SURFACES:
        if surface.is_dir():
            files.extend(sorted(surface.rglob("*.py")))
        elif surface.is_file():
            files.append(surface)
    return [f for f in files if f != _CONTRACT_DEFINITION]


def _concatenates_the_groups(node: object) -> bool:
    """True if `node` is an `A + B (+ C)` chain naming 2+ of the column-group constants."""
    import ast

    if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Add):
        return False
    names: set[str] = set()
    stack = [node]
    while stack:
        cur = stack.pop()
        if isinstance(cur, ast.BinOp) and isinstance(cur.op, ast.Add):
            stack.extend([cur.left, cur.right])
        elif isinstance(cur, ast.Name):
            names.add(cur.id)
    return len(names & _GROUP_NAMES) >= 2


def test_no_live_module_rehandrolls_the_locked_composition():
    """The composition is written out once, where it is defined — nowhere else.

    An AST check, not a grep, so prose and docstrings that *describe* the contract
    stay free while a real `FEATURE_COLUMNS + LGA_FEATURE_COLUMNS + ...` expression
    fails. Retyping it at ~20 call sites is the mechanism behind this whole issue:
    landing a feature group means editing all of them, and a script written before
    #216 silently carries a 50-column baseline (fps-zci).
    """
    import ast

    scanned = _live_source_files()
    assert len(scanned) > 20, f"scanner found almost nothing to check: {scanned}"

    offenders: list[str] = []
    for path in scanned:
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if _concatenates_the_groups(node):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")

    assert not offenders, (
        "These modules rebuild the locked feature set by hand instead of importing "
        f"LOCKED_FEATURE_COLUMNS (or experiments.lib.constants.BASELINE_COLUMNS): "
        f"{offenders}"
    )


def test_the_detector_would_actually_fire():
    """Guard the guard — a scanner that silently matches nothing is worse than none."""
    import ast

    tree = ast.parse("cols = FEATURE_COLUMNS + LGA_FEATURE_COLUMNS + NETWORK_FEATURE_COLUMNS\n")
    assert any(_concatenates_the_groups(node) for node in ast.walk(tree))
