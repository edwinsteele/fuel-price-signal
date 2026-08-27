"""Tests for the texture-ICC estimator in `experiments/2026-08-27_texture_icc/measure_icc.py`
(bd fps-3jj.23).

Experiment scripts in this repo are conventionally untested print-and-exit code, and that
convention is right for most of them. This one argued past it: a 6.1h run's verdict — whether
`placebo.TEXTURE_ICC_BOUND` is replaced, raised, or held — is read straight off `_anova_icc`,
and review of PR #337 found two defects in it that a known-ICC check catches directly (a
silently dropped singleton group, and a "95% CI" whose two ends were built at different
tails). Validating an estimator against data whose answer is known is the cheapest test there
is, so it is committed rather than run once in a scratch buffer.

The scripts live outside any package, so they are loaded by path.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.stats import f as f_dist

EXPERIMENT_DIR = pathlib.Path(__file__).resolve().parents[1] / "experiments" / "2026-08-27_texture_icc"


def _load(name: str):
    """Import a script from the experiment dir by path, with that dir importable so its
    `from bank_model import ...` sibling import resolves the same way it does under
    `PYTHONPATH=. uv run python experiments/2026-08-27_texture_icc/<script>.py`."""
    sys.path.insert(0, str(EXPERIMENT_DIR))
    try:
        spec = importlib.util.spec_from_file_location(f"_icc_{name}", EXPERIMENT_DIR / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(EXPERIMENT_DIR))


measure_icc = _load("measure_icc")
bank_model = _load("bank_model")


def _balanced_frame(icc: float, k: int, m: int, seed: int) -> pd.DataFrame:
    """k columns x m draws whose delta = texture_of_column + alignment, with
    Var(texture) = icc and Var(alignment) = 1 - icc. The population ICC is then exactly `icc`."""
    rng = np.random.default_rng(seed)
    texture = rng.normal(0.0, np.sqrt(icc), k)
    return pd.DataFrame(
        [
            {"source_column": f"c{i}", "delta": texture[i] + rng.normal(0.0, np.sqrt(1.0 - icc))}
            for i in range(k)
            for _ in range(m)
        ]
    )


# ── the defects review found ──────────────────────────────────────────────────

def test_a_singleton_group_is_kept_not_silently_dropped():
    """A column with one draw carries no WITHIN-column variation but does carry BETWEEN-column
    variation. The original filtered it out and reported the post-drop counts, so a 5-draw,
    3-column bank printed n=4, k=2 with nothing in the output saying a column had vanished."""
    frame = pd.DataFrame(
        [
            {"source_column": "a", "delta": 1.0}, {"source_column": "a", "delta": 1.2},
            {"source_column": "b", "delta": 3.0}, {"source_column": "b", "delta": 3.3},
            {"source_column": "c", "delta": 9.9},  # singleton, far from the rest
        ]
    )

    result = measure_icc._anova_icc(frame)

    assert result["n"] == 5, "every draw must count toward n"
    assert result["k"] == 3, "every source column must count toward k"


def test_the_singleton_actually_moves_the_estimate():
    """Guards the test above against being satisfied by bookkeeping alone — the dropped column
    was the most extreme one, so keeping it must change the ICC, not just the printed counts."""
    rows = [
        {"source_column": "a", "delta": 1.0}, {"source_column": "a", "delta": 1.2},
        {"source_column": "b", "delta": 3.0}, {"source_column": "b", "delta": 3.3},
    ]
    without = measure_icc._anova_icc(pd.DataFrame(rows))
    with_singleton = measure_icc._anova_icc(
        pd.DataFrame([*rows, {"source_column": "c", "delta": 9.9}])
    )

    assert with_singleton["icc"] != pytest.approx(without["icc"])


def test_the_two_sided_ci_is_symmetric_in_its_tails():
    """Both ends of a 95% CI must come from the same alpha. The original paired a 2.5%-tail
    lower with a 5%-tail upper and labelled the pair '95% CI'."""
    result = measure_icc._anova_icc(_balanced_frame(0.4, k=6, m=5, seed=11))
    df1, df2, k_bar, f_stat = result["df1"], result["df2"], result["k_bar"], result["f_stat"]

    def icc_at(quantile: float) -> float:
        f = f_stat / f_dist.ppf(quantile, df1, df2)
        return (f - 1.0) / (f + k_bar - 1.0)

    assert result["icc_ci_lower"] == pytest.approx(icc_at(0.975))
    assert result["icc_ci_upper"] == pytest.approx(icc_at(0.025))


def test_the_one_sided_upper_bound_matches_how_0391_was_derived():
    """`texture_channel.py` produced the superseded 0.391 as `f_stat / f.ppf(0.05, df1, df2)`
    pushed through the ICC formula. The bound this script reports has to be the same
    construction, or the comparison against the value being replaced is between two different
    alphas — which is exactly the defect an earlier revision of `power.py` shipped."""
    result = measure_icc._anova_icc(_balanced_frame(0.4, k=6, m=5, seed=12))
    f_upper = result["f_stat"] / f_dist.ppf(0.05, result["df1"], result["df2"])
    expected = (f_upper - 1.0) / (f_upper + result["k_bar"] - 1.0)

    assert result["icc_upper"] == pytest.approx(expected)
    assert result["icc_upper"] > result["icc"], "an upper bound sits above the point estimate"


# ── does it recover an ICC it is given? ───────────────────────────────────────

@pytest.mark.parametrize("true_icc", [0.0, 0.2, 0.391, 0.7])
def test_recovers_a_known_icc_on_average(true_icc):
    """The estimator is the whole verdict, so it is checked against data whose answer is known.
    Averaged over 200 balanced 8x4 banks; ICC(1) carries a small downward bias at this size,
    which is why the tolerance is 0.08 rather than something tighter."""
    estimates = [
        measure_icc._anova_icc(_balanced_frame(true_icc, k=8, m=4, seed=s))["icc"]
        for s in range(200)
    ]

    assert np.mean(estimates) == pytest.approx(true_icc, abs=0.08)


def test_the_one_sided_upper_bound_covers_the_truth():
    """A bound that does not cover is worse than no bound. ~95% of banks must have
    icc_upper >= the true ICC; asserted loosely (>= 88%) so 200 draws of sampling noise cannot
    make this flap."""
    covered = sum(
        measure_icc._anova_icc(_balanced_frame(0.391, k=8, m=4, seed=s))["icc_upper"] >= 0.391
        for s in range(200)
    )

    assert covered / 200 >= 0.88


def test_refuses_a_bank_with_no_within_column_variation():
    """Every column drawn once is a bank that cannot separate alignment from texture at all."""
    frame = pd.DataFrame([{"source_column": f"c{i}", "delta": float(i)} for i in range(5)])

    with pytest.raises(SystemExit, match="no\n *WITHIN-column variation|WITHIN-column variation"):
        measure_icc._anova_icc(frame)


# ── bank_model: the overlap reconstruction both scripts share ─────────────────

def test_model_bank_skips_unscreenable_columns_rather_than_truncating():
    """The five all-NaN batch1 columns must be absent from every draw, while the columns at the
    END of the declared list — which an earlier `[:49]` truncation dropped instead — must be
    reachable."""
    columns = [f"col_{i}" for i in range(54)]
    unscreenable = tuple(columns[10:15])

    bank = bank_model.model_bank(columns, 20, 3, unscreenable=unscreenable)

    used = {col for draw in bank for col, _seed, _corr in draw}
    assert not used & set(unscreenable)
    assert columns[53] in used, "the tail of the declared list must still be drawn from"


def test_model_bank_raises_instead_of_hanging_above_the_usable_count():
    """`candidate_sequence` is unbounded, so a group that can never fill used to spin forever
    rather than reaching the trailing assertion."""
    columns = [f"col_{i}" for i in range(54)]

    with pytest.raises(ValueError, match="usable columns"):
        bank_model.model_bank(columns, 2, 50, unscreenable=tuple(columns[:5]))


#: The ICC the published tables were transcribed at. Held as a literal rather than imported so
#: that moving `placebo.TEXTURE_ICC_BOUND` fails the two tests below LOUDLY — every one of those
#: numbers is quoted in docs/CONVENTIONS.md, docs/routines/generator.md and this experiment's
#: README, and they have to be re-transcribed in the same change. Was 0.391 (a bound on the
#: by-FAMILY ICC) until `fps-3jj.23` measured the by-COLUMN quantity on 2026-08-27.
PUBLISHED_ICC = 0.274


def test_the_published_icc_is_the_one_that_ships():
    """The pins below are only meaningful while this matches — otherwise they would quietly
    describe a ruler nothing grades against."""
    from experiments.pipeline.placebo import TEXTURE_ICC_BOUND

    assert TEXTURE_ICC_BOUND == PUBLISHED_ICC, (
        "TEXTURE_ICC_BOUND moved. Re-transcribe the effective-draw and bar tables in "
        "docs/CONVENTIONS.md, docs/routines/generator.md, placebo.py's own comment block and "
        "experiments/2026-08-27_texture_icc/README.md, then update PUBLISHED_ICC here."
    )


def test_model_bank_reproduces_the_published_bar_table():
    """The arity/ICC table in docs/CONVENTIONS.md, the PR body and this experiment's README is
    transcribed from `power.py`. It was produced under a `[:49]` truncation that dropped the
    wrong five columns; this pins that the corrected model gives the same overlap, so the
    published numbers stand rather than needing re-transcription."""
    from experiments.pipeline.placebo import effective_n_draws

    columns = [f"col_{i}" for i in range(54)]
    unscreenable = tuple(columns[10:15])
    expected_n_eff = {1: 20.0, 2: 20.0, 3: 18.17, 10: 10.81}

    for arity, n_eff in expected_n_eff.items():
        bank = bank_model.model_bank(columns, 20, arity, unscreenable=unscreenable)
        assert effective_n_draws(bank, icc=PUBLISHED_ICC) == pytest.approx(n_eff, abs=0.01), (
            f"arity {arity}"
        )


def test_overlap_is_invariant_to_WHICH_columns_are_dead():
    """`effective_n_draws` reads the pattern of sharing, not which names share — so the model
    must not depend on the identity of the unscreenable set, only its size. This is what makes
    the earlier `[:49]` truncation accidentally right, and it is worth pinning rather than
    relying on."""
    from experiments.pipeline.placebo import effective_n_draws

    columns = [f"col_{i}" for i in range(54)]
    candidates = [tuple(columns[i:i + 5]) for i in (0, 7, 20, 33, 49)]

    results = {
        tuple(
            round(effective_n_draws(bank_model.model_bank(columns, 20, arity, unscreenable=dead)), 6)
            for arity in (3, 10, 20)
        )
        for dead in candidates
    }

    assert len(results) == 1, f"overlap changed with the identity of the dead set: {results}"


def test_overlap_DOES_move_with_the_usable_count():
    """The counterpart, and the reason the published table is not self-evident: one more or one
    fewer dead column shifts every wide-arity bar. Pinned so that a future change to the assumed
    count is loud rather than a silent re-description of the batch (PR #337 review, second
    round)."""
    from experiments.pipeline.placebo import effective_n_draws

    columns = [f"col_{i}" for i in range(54)]
    n_eff = {
        n_dead: round(
            effective_n_draws(
                bank_model.model_bank(columns, 20, 20, unscreenable=tuple(columns[:n_dead])),
                icc=PUBLISHED_ICC,
            ), 2
        )
        for n_dead in (4, 5, 6)
    }

    assert n_eff == {4: 6.85, 5: 6.74, 6: 6.63}
    assert len(set(n_eff.values())) == 3, "the count must matter, or the pin above is vacuous"


def test_unscreenable_is_read_from_a_floors_own_stamp_when_present():
    """A floor computed from fps-3jj.23 onward records the all-NaN set it actually observed, so
    the model can stop trusting a hardcoded environmental fact. Batch1's two committed floors
    predate the stamp, which is why the fallback survives."""
    stamped = {"all_nan_baseline_columns": ["a", "b"]}

    assert bank_model.unscreenable_from_floor(stamped) == ("a", "b")
    assert bank_model.unscreenable_from_floor({"all_nan_baseline_columns": []}) == ()
    assert bank_model.unscreenable_from_floor({}) is None, "pre-stamp floors must be detectable"
