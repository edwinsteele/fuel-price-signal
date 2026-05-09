"""Tests for fuel_signal.fp_cost."""

from __future__ import annotations

import pathlib

import pandas as pd
import pytest
from click.testing import CliRunner

from fuel_signal.fp_cost import (
    CLUSTER_A_LABEL,
    CLUSTER_B_LABEL,
    compute_damage,
    format_summary,
    main,
    plot_fp_distribution,
)


def _df(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


# ---------------------------------------------------------------------------
# compute_damage
# ---------------------------------------------------------------------------

def test_cluster_b_large_drop():
    fp = compute_damage(_df({"today_price_cents": 170.0, "future_min_cents": 160.0, "label": 0}))
    assert fp["cluster"].iloc[0] == CLUSTER_B_LABEL
    assert fp["damage"].iloc[0] == pytest.approx(10.0)


def test_cluster_a_small_drop():
    fp = compute_damage(_df({"today_price_cents": 170.0, "future_min_cents": 168.5, "label": 0}))
    assert fp["cluster"].iloc[0] == CLUSTER_A_LABEL


def test_cluster_a_price_rose():
    fp = compute_damage(_df({"today_price_cents": 160.0, "future_min_cents": 165.0, "label": 0}))
    assert fp["cluster"].iloc[0] == CLUSTER_A_LABEL
    assert fp["damage"].iloc[0] == pytest.approx(-5.0)


def test_cluster_a_exact_threshold():
    # damage == threshold exactly → cluster A (not B, which requires strictly greater)
    fp = compute_damage(
        _df({"today_price_cents": 163.0, "future_min_cents": 160.0, "label": 0}),
        threshold_cents=3.0,
    )
    assert fp["cluster"].iloc[0] == CLUSTER_A_LABEL


def test_excludes_label_1_rows():
    df = _df(
        {"today_price_cents": 170.0, "future_min_cents": 155.0, "label": 1},
        {"today_price_cents": 170.0, "future_min_cents": 160.0, "label": 0},
    )
    fp = compute_damage(df)
    assert len(fp) == 1
    assert fp["label"].iloc[0] == 0


def test_custom_threshold():
    # damage=4.0 — cluster B at threshold=3.0, cluster A at threshold=5.0
    row = {"today_price_cents": 164.0, "future_min_cents": 160.0, "label": 0}
    fp3 = compute_damage(_df(row), threshold_cents=3.0)
    fp5 = compute_damage(_df(row), threshold_cents=5.0)
    assert fp3["cluster"].iloc[0] == CLUSTER_B_LABEL
    assert fp5["cluster"].iloc[0] == CLUSTER_A_LABEL


# ---------------------------------------------------------------------------
# format_summary
# ---------------------------------------------------------------------------

def _mixed_fp() -> pd.DataFrame:
    rows = (
        [{"today_price_cents": 170.0, "future_min_cents": 160.0, "label": 0}] * 10  # cluster B
        + [{"today_price_cents": 170.0, "future_min_cents": 169.0, "label": 0}] * 10  # cluster A
    )
    return compute_damage(pd.DataFrame(rows))


def test_format_summary_contains_clusters():
    out = format_summary(_mixed_fp())
    assert "Cluster A" in out
    assert "Cluster B" in out


def test_format_summary_contains_suggestion():
    out = format_summary(_mixed_fp())
    assert "Suggested FP penalty" in out


def test_format_summary_empty_cluster_b():
    # All cluster A — no suggestion should crash
    rows = [{"today_price_cents": 160.0, "future_min_cents": 159.0, "label": 0}] * 5
    fp = compute_damage(pd.DataFrame(rows))
    out = format_summary(fp)
    assert "Cluster A" in out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_missing_csv(tmp_path: pathlib.Path):
    result = CliRunner().invoke(main, ["--features-csv", str(tmp_path / "missing.csv")])
    assert result.exit_code != 0
    assert "not found" in result.output.lower()


def test_cli_missing_columns(tmp_path: pathlib.Path):
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame({"label": [0, 1]}).to_csv(bad_csv, index=False)
    result = CliRunner().invoke(main, ["--features-csv", str(bad_csv)])
    assert result.exit_code != 0
    assert "missing columns" in result.output.lower()


def test_cli_writes_plot(tmp_path: pathlib.Path):
    csv_path = tmp_path / "features.csv"
    plot_path = tmp_path / "out.png"
    rows = (
        [{"today_price_cents": 170.0, "future_min_cents": 160.0, "label": 0}] * 20
        + [{"today_price_cents": 170.0, "future_min_cents": 169.0, "label": 0}] * 20
        + [{"today_price_cents": 160.0, "future_min_cents": 160.0, "label": 1}] * 10
    )
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    result = CliRunner().invoke(main, [
        "--features-csv", str(csv_path),
        "--plot", str(plot_path),
    ])
    assert result.exit_code == 0, result.output
    assert plot_path.exists()


def test_plot_constant_damage_no_crash(tmp_path: pathlib.Path):
    # All rows have identical damage — quantile range collapses to a single value.
    rows = [{"today_price_cents": 170.0, "future_min_cents": 160.0, "label": 0}] * 20
    fp = compute_damage(pd.DataFrame(rows))
    plot_fp_distribution(fp, out_path=tmp_path / "out.png")
    assert (tmp_path / "out.png").exists()


def test_cli_custom_threshold(tmp_path: pathlib.Path):
    csv_path = tmp_path / "features.csv"
    plot_path = tmp_path / "out.png"
    rows = [{"today_price_cents": 170.0, "future_min_cents": 164.0, "label": 0}] * 30
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    # damage=6.0; at threshold=5.0 → cluster B; at threshold=7.0 → cluster A
    result = CliRunner().invoke(main, [
        "--features-csv", str(csv_path),
        "--plot", str(plot_path),
        "--threshold", "5.0",
    ])
    assert result.exit_code == 0, result.output
    assert "Label threshold: 5.0c" in result.output
