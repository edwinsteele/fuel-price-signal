"""YYYYMMDD integer ↔ ISO date string ↔ date object conversions."""

from datetime import date


def date_to_int(s: str) -> int:
    """'2024-01-15' → 20240115"""
    return int(s[:10].replace("-", ""))


def date_from_int(v: int) -> str:
    """20240115 → '2024-01-15'"""
    s = str(v)
    return f"{s[:4]}-{s[4:6]}-{s[6:]}"


def int_to_date(v: int) -> date:
    """20240115 → date(2024, 1, 15)"""
    return date.fromisoformat(date_from_int(v))
