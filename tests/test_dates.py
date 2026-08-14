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
        ("2024-02-29", 20240229),  # leap day
        ("1900-01-01", 19000101),  # century non-leap year
        ("2099-12-31", 20991231),  # far-future year
    ],
)
def test_date_to_int(s, expected):
    assert date_to_int(s) == expected


def test_date_to_int_invalid_string_raises():
    with pytest.raises(ValueError):
        date_to_int("not-a-date")


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
        (20240229, date(2024, 2, 29)),  # leap day
    ],
)
def test_int_to_date(v, expected):
    assert int_to_date(v) == expected


def test_int_to_date_invalid_calendar_date_raises():
    with pytest.raises(ValueError):
        int_to_date(20241332)  # month 13, day 32


@pytest.mark.parametrize(
    "s", ["2024-01-15", "2024-12-31", "2000-01-01", "2024-02-29", "1900-01-01"]
)
def test_date_to_int_round_trip(s):
    v = date_to_int(s)
    assert date_from_int(v) == s
    assert int_to_date(v) == date.fromisoformat(s)


@pytest.mark.parametrize(
    "v", [20240115, 20241231, 20000101, 20240229, 19000101]
)
def test_date_from_int_round_trip(v):
    s = date_from_int(v)
    assert date_to_int(s) == v
    assert int_to_date(v) == date.fromisoformat(s)
