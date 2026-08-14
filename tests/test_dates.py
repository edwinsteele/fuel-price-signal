"""Tests for fuel_signal.dates — YYYYMMDD int / ISO string / date conversions."""

from datetime import date

import pytest

from fuel_signal.dates import date_from_int, date_to_int, int_to_date


@pytest.mark.parametrize(
    "s, expected",
    [
        ("2024-01-15", 20240115),
        ("2024-12-31", 20241231),
        ("2000-01-01", 20000101),
    ],
)
def test_date_to_int(s, expected):
    assert date_to_int(s) == expected


@pytest.mark.parametrize(
    "v, expected",
    [
        (20240115, "2024-01-15"),
        (20241231, "2024-12-31"),
        (20000101, "2000-01-01"),
    ],
)
def test_date_from_int(v, expected):
    assert date_from_int(v) == expected


@pytest.mark.parametrize(
    "v, expected",
    [
        (20240115, date(2024, 1, 15)),
        (20241231, date(2024, 12, 31)),
        (20000101, date(2000, 1, 1)),
    ],
)
def test_int_to_date(v, expected):
    assert int_to_date(v) == expected


@pytest.mark.parametrize("s", ["2024-01-15", "2024-12-31", "2000-01-01"])
def test_date_to_int_round_trip(s):
    v = date_to_int(s)
    assert date_from_int(v) == s
    assert int_to_date(v) == date.fromisoformat(s)
