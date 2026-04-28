"""Tests for fuel_signal.compare — series comparison command."""

import pytest
from click.testing import CliRunner

from fuel_signal.compare import main as compare_cmd
from fuel_signal.db import (
    create_schema,
    open_db,
    upsert_daily_prices,
    upsert_stations,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db_with_two_stations(tmp_path):
    """DB with two stations and overlapping daily_prices for compare tests."""
    db_path = tmp_path / "test.db"
    c = open_db(db_path)
    create_schema(c)

    upsert_stations(c, [
        {
            "station_code": 101,
            "name": "Ampol Springwood",
            "address": "1 Macquarie Road, Springwood",
            "suburb": "Springwood",
            "postcode": "2777",
            "brand": "Ampol",
        },
        {
            "station_code": 102,
            "name": "Shell Blaxland",
            "address": "5 Great Western Highway, Blaxland",
            "suburb": "Blaxland",
            "postcode": "2774",
            "brand": "Shell",
        },
    ])

    # Station 101 is cheaper on days 1-3, station 102 on days 4-5, equal on day 6.
    # Day:       2024-01-01  2024-01-02  2024-01-03  2024-01-04  2024-01-05  2024-01-06
    # Ampol:        160.0       162.0       158.0       172.0       170.0       165.0
    # Shell:        165.0       165.0       165.0       165.0       162.0       165.0
    # diff (A-B):    -5.0        -3.0        -7.0        +7.0        +8.0         0.0
    upsert_daily_prices(c, [
        (101, "E10", "2024-01-01", 160.0),
        (101, "E10", "2024-01-02", 162.0),
        (101, "E10", "2024-01-03", 158.0),
        (101, "E10", "2024-01-04", 172.0),
        (101, "E10", "2024-01-05", 170.0),
        (101, "E10", "2024-01-06", 165.0),
        (102, "E10", "2024-01-01", 165.0),
        (102, "E10", "2024-01-02", 165.0),
        (102, "E10", "2024-01-03", 165.0),
        (102, "E10", "2024-01-04", 165.0),
        (102, "E10", "2024-01-05", 162.0),
        (102, "E10", "2024-01-06", 165.0),
    ])
    c.commit()
    c.close()
    return db_path


# ---------------------------------------------------------------------------
# _resolve_series error cases
# ---------------------------------------------------------------------------

class TestResolveSeriesErrors:
    def test_station_not_found(self, db_with_two_stations):
        runner = CliRunner()
        result = runner.invoke(
            compare_cmd, ["Nonexistent Station", "sydney", "--db", str(db_with_two_stations)]
        )
        assert result.exit_code != 0
        assert "No station found" in result.output

    def test_ambiguous_station_lists_matches(self, db_with_two_stations):
        # "a" matches both: "Ampol Springwood" (has 'A') and "Shell Blaxland" (has 'a')
        runner = CliRunner()
        result = runner.invoke(
            compare_cmd, ["a", "sydney", "--db", str(db_with_two_stations)]
        )
        assert result.exit_code != 0
        assert "Multiple stations" in result.output
        assert "Ampol Springwood" in result.output
        assert "Shell Blaxland" in result.output

    def test_suburb_name_does_not_cause_false_ambiguity(self, db_with_two_stations):
        # "Springwood" is a suburb shared by multiple stations; name-only matching
        # should find only "Ampol Springwood" (name contains "Springwood"), not
        # every station whose suburb happens to be Springwood.
        runner = CliRunner()
        result = runner.invoke(
            compare_cmd, ["Ampol Springwood", "sydney", "--db", str(db_with_two_stations)]
        )
        assert result.exit_code == 0
        assert "Ampol Springwood" in result.output

    def test_station_id_resolves_directly(self, db_with_two_stations):
        runner = CliRunner()
        result = runner.invoke(
            compare_cmd, ["101", "sydney", "--db", str(db_with_two_stations)]
        )
        assert result.exit_code == 0
        assert "Ampol Springwood" in result.output

    def test_unknown_station_id_errors(self, db_with_two_stations):
        runner = CliRunner()
        result = runner.invoke(
            compare_cmd, ["9999", "sydney", "--db", str(db_with_two_stations)]
        )
        assert result.exit_code != 0
        assert "No station found" in result.output

    def test_ambiguous_lga_lists_matches(self, db_with_two_stations):
        runner = CliRunner()
        result = runner.invoke(
            compare_cmd, ["lga:bay", "sydney", "--db", str(db_with_two_stations)]
        )
        assert result.exit_code != 0
        assert "Ambiguous" in result.output

    def test_unknown_lga(self, db_with_two_stations):
        runner = CliRunner()
        result = runner.invoke(
            compare_cmd, ["lga:atlantis", "sydney", "--db", str(db_with_two_stations)]
        )
        assert result.exit_code != 0
        assert "No LGA matching" in result.output

    def test_db_not_found(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(
            compare_cmd, ["Ampol", "sydney", "--db", str(tmp_path / "missing.db")]
        )
        assert result.exit_code != 0
        assert "not found" in result.output.lower()


# ---------------------------------------------------------------------------
# compare — output content tests
# ---------------------------------------------------------------------------

class TestCompare:
    def test_a_cheaper_more_often(self, db_with_two_stations):
        runner = CliRunner()
        result = runner.invoke(
            compare_cmd,
            ["Ampol Springwood", "Shell Blaxland", "--db", str(db_with_two_stations)],
        )
        assert result.exit_code == 0, result.output
        # Ampol cheaper 3 days, Shell cheaper 2 days, equal 1 day (within 0.5c)
        assert "3 days" in result.output
        assert "2 days" in result.output
        assert "1 days" in result.output

    def test_period_and_header_shown(self, db_with_two_stations):
        runner = CliRunner()
        result = runner.invoke(
            compare_cmd,
            ["Ampol Springwood", "Shell Blaxland", "--db", str(db_with_two_stations)],
        )
        assert "2024-01-01 to 2024-01-06" in result.output
        assert "6 overlapping days" in result.output
        assert "Ampol Springwood" in result.output
        assert "Shell Blaxland" in result.output

    def test_mean_prices_shown(self, db_with_two_stations):
        runner = CliRunner()
        result = runner.invoke(
            compare_cmd,
            ["Ampol Springwood", "Shell Blaxland", "--db", str(db_with_two_stations)],
        )
        # Ampol mean = (160+162+158+172+170+165)/6 = 164.5
        # Shell mean = (165+165+165+165+162+165)/6 = 164.5
        assert "164.5c" in result.output

    def test_overall_summary_negligible_when_equal_means(self, db_with_two_stations):
        runner = CliRunner()
        result = runner.invoke(
            compare_cmd,
            ["Ampol Springwood", "Shell Blaxland", "--db", str(db_with_two_stations)],
        )
        # Both means are 164.5c, diff = 0.0 ≤ within (0.5)
        assert "negligible" in result.output

    def test_within_option_changes_equal_bucket(self, db_with_two_stations):
        runner = CliRunner()
        result = runner.invoke(
            compare_cmd,
            [
                "Ampol Springwood", "Shell Blaxland",
                "--within", "10",
                "--db", str(db_with_two_stations),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "6 days" in result.output  # all 6 days equal

    def test_overall_cheaper_message(self, tmp_path):
        """When one series is consistently cheaper, overall message names it."""
        db_path = tmp_path / "test.db"
        c = open_db(db_path)
        create_schema(c)
        upsert_stations(c, [
            {
                "station_code": 201,
                "name": "Cheap Station",
                "address": "1 Low Street, Nowhere",
                "suburb": "Nowhere",
                "postcode": "2000",
                "brand": "Cheap",
            },
            {
                "station_code": 202,
                "name": "Dear Station",
                "address": "1 High Street, Nowhere",
                "suburb": "Nowhere",
                "postcode": "2000",
                "brand": "Dear",
            },
        ])
        upsert_daily_prices(c, [
            (201, "E10", "2024-01-01", 150.0),
            (201, "E10", "2024-01-02", 152.0),
            (202, "E10", "2024-01-01", 160.0),
            (202, "E10", "2024-01-02", 162.0),
        ])
        c.commit()
        c.close()

        runner = CliRunner()
        result = runner.invoke(
            compare_cmd,
            ["Cheap Station", "Dear Station", "--db", str(db_path)],
        )
        assert result.exit_code == 0, result.output
        assert "10.0c cheaper at Cheap Station" in result.output

    def test_no_overlapping_dates_error(self, tmp_path):
        db_path = tmp_path / "test.db"
        c = open_db(db_path)
        create_schema(c)
        upsert_stations(c, [
            {
                "station_code": 301,
                "name": "Alpha Station",
                "address": "1 Alpha Road, Alpha",
                "suburb": "Alpha",
                "postcode": "2000",
                "brand": "Alpha",
            },
            {
                "station_code": 302,
                "name": "Beta Station",
                "address": "2 Beta Road, Beta",
                "suburb": "Beta",
                "postcode": "2000",
                "brand": "Beta",
            },
        ])
        upsert_daily_prices(c, [
            (301, "E10", "2024-01-01", 160.0),
            (302, "E10", "2024-02-01", 160.0),
        ])
        c.commit()
        c.close()

        runner = CliRunner()
        result = runner.invoke(
            compare_cmd,
            ["Alpha Station", "Beta Station", "--db", str(db_path)],
        )
        assert result.exit_code != 0
        assert "No overlapping" in result.output

    def test_avg_delta_shown_when_one_series_cheaper(self, db_with_two_stations):
        runner = CliRunner()
        result = runner.invoke(
            compare_cmd,
            ["Ampol Springwood", "Shell Blaxland", "--db", str(db_with_two_stations)],
        )
        # Ampol cheaper on days 1-3 by 5, 3, 7c → avg 5.0c
        assert "avg 5.0c cheaper" in result.output
        # Shell cheaper on days 4-5 by 7, 8c → avg 7.5c
        assert "avg 7.5c cheaper" in result.output
