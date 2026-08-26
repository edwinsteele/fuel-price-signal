"""The placebo bank's OVERLAP STRUCTURE at a given (n_draws, arity), without any fits.

Shared by `power.py` and `measure_icc.py` so the bar tables they print cannot drift apart —
they did, briefly, as two copies of the same loop (PR #337 review).

**This is a MODEL of `placebo._assemble_draws`, and the difference is worth stating rather
than glossed.** The real thing screens each candidate against the frozen frame and skips what
fails; this reproduces the lap structure and the group assembly, and stands in for the screen
with a DECLARED list of columns known to be unscreenable. That is faithful for batch1, whose
only screen failures are all-NaN columns, and it is not a general substitute for the screen —
a batch where a real column failed on correlation would need the frame.

The distinction matters because an earlier version of this helper took `baseline_columns[:49]`
as "the usable columns". That is wrong twice over: it drops the last five DECLARED columns
(`days_since_trough_entry_woollahra`, both `network_*`, both `lga_phase_*`) rather than the
five all-NaN ones, and it rotates laps over a 49-column list where the shipped code rotates
over all 54 and lets the screen reject. It happened to produce the right overlap counts on
batch1 — `effective_n_draws` reads the PATTERN of sharing, not which names share — but it was
right by luck, and it claimed in its docstring to be "the real one rather than a model of it".
"""
from __future__ import annotations

from experiments.pipeline.placebo import candidate_sequence

#: batch1's unscreenable columns: entirely NaN in the frozen frame, so `screen_draw_groups`
#: correlates them to NaN and skips them on every lap (placebo.py's module docstring). Measured
#: off `experiments/batches/batch1/features.parquet` (`frame[c].notna().any()`), not assumed —
#: 49 of the 54 declared columns are usable. Declared here rather than discovered because the
#: scripts that import this run from clones where the frozen frame is gitignored.
BATCH1_UNSCREENABLE = (
    "days_since_trough_entry_bayside",
    "days_since_trough_entry_botany_bay",
    "days_since_trough_entry_hunters_hill",
    "days_since_trough_entry_lane_cove",
    "days_since_trough_entry_waverley",
)


def model_bank(
    baseline_columns: list[str],
    n_draws: int,
    arity: int,
    *,
    unscreenable: tuple[str, ...] = BATCH1_UNSCREENABLE,
) -> list[list[tuple[str, int, float]]]:
    """The bank `_assemble_draws` would assemble at this shape, in `effective_n_draws`' format.

    Pass the FULL declared column list — all 54 — exactly as `compute_noise_floor` does. Laps
    rotate over that list and the unscreenable columns are skipped where the screen would skip
    them, which is not the same as rotating over a pre-filtered list.

    Raises ValueError rather than looping forever when `arity` exceeds the usable column count.
    A draw needs `arity` distinct columns; above that, every remaining candidate is already in
    the group being built, `candidate_sequence` is unbounded, and the previous version of this
    loop simply never returned (its trailing `AssertionError` was unreachable).
    """
    usable = [c for c in baseline_columns if c not in set(unscreenable)]
    if arity > len(usable):
        raise ValueError(
            f"arity {arity} exceeds the {len(usable)} usable columns "
            f"({len(baseline_columns)} declared, {len(unscreenable)} unscreenable) — a draw "
            "needs that many DISTINCT source columns, so no bank exists at this width."
        )
    skip = set(unscreenable)
    groups: list[list[tuple[str, int, float]]] = []
    current: list[tuple[str, int, float]] = []
    used: set[str] = set()
    for column, seed in candidate_sequence(baseline_columns, n_draws * arity):
        if column in skip or column in used:
            continue
        current.append((column, seed, 0.0))
        used.add(column)
        if len(current) == arity:
            groups.append(current)
            current, used = [], set()
            if len(groups) == n_draws:
                return groups
    raise AssertionError("candidate_sequence is unbounded and arity is checked — unreachable")
