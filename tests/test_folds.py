from __future__ import annotations

from experiments.lib.constants import SHOCK_QUANTILE
from experiments.lib.folds import compute_shock_folds


def test_compute_shock_folds_empty_input_is_empty():
    assert compute_shock_folds({}) == frozenset()


def test_compute_shock_folds_zero_quantile_is_empty():
    fold_ll = {1: 0.5, 2: 0.1, 3: 0.9}
    assert compute_shock_folds(fold_ll, quantile=0.0) == frozenset()


def test_compute_shock_folds_full_quantile_is_every_fold():
    fold_ll = {1: 0.5, 2: 0.1, 3: 0.9}
    assert compute_shock_folds(fold_ll, quantile=1.0) == frozenset({1, 2, 3})


def test_compute_shock_folds_picks_the_highest_log_loss_folds():
    # 4 folds, quantile 0.25 -> ceil(4 * 0.25) = 1 -> only the single hardest fold.
    fold_ll = {1: 0.30, 2: 0.10, 3: 0.45, 4: 0.20}
    assert compute_shock_folds(fold_ll, quantile=0.25) == frozenset({3})


def test_compute_shock_folds_ceil_rounds_up_not_down():
    # 5 folds, quantile 0.25 -> ceil(1.25) = 2 hardest folds, not 1.
    fold_ll = {1: 0.10, 2: 0.50, 3: 0.20, 4: 0.40, 5: 0.30}
    assert compute_shock_folds(fold_ll, quantile=0.25) == frozenset({2, 4})


def test_compute_shock_folds_out_of_range_quantile_is_clamped():
    fold_ll = {1: 0.5, 2: 0.1}
    assert compute_shock_folds(fold_ll, quantile=-1.0) == frozenset()
    assert compute_shock_folds(fold_ll, quantile=5.0) == frozenset({1, 2})


def test_compute_shock_folds_default_quantile_matches_the_shared_constant():
    # 14 folds (the project's real WFCV fold count) at the default quantile mirrors the
    # old fixed-index SHOCK_FOLDS's size (4/14) without pinning specific fold numbers.
    fold_ll = {i: float(i) for i in range(1, 15)}
    shock = compute_shock_folds(fold_ll)
    assert shock == compute_shock_folds(fold_ll, quantile=SHOCK_QUANTILE)
    assert len(shock) == 4
    assert shock == frozenset({11, 12, 13, 14})  # the 4 highest-ll fold indices
