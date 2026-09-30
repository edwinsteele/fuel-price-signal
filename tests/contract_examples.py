"""Worked examples from docs/api-contract.md, extracted at test time.

Each full example in the contract sits in a marked fence (```json <marker>).
Golden tests build a response through the real serializer with the example's
fixed inputs and assert equality with contract_example(<marker>), so an edit to
an example that the server can't emit fails here instead of drifting from a
hand-copied literal. The consumer extracts the same fences, so both sides'
tests read one document.
"""

from __future__ import annotations

import json
import re
from functools import cache
from pathlib import Path

CONTRACT = Path(__file__).resolve().parents[1] / "docs" / "api-contract.md"


@cache
def _contract_text() -> str:
    return CONTRACT.read_text(encoding="utf-8")


def contract_example(marker: str) -> object:
    # Whole-line match on the fence: ```json <marker>-v2 is a different example.
    pattern = re.compile(
        rf"^```json {re.escape(marker)}[ \t]*\n(.*?)\n```[ \t]*$", re.MULTILINE | re.DOTALL
    )
    matches = pattern.findall(_contract_text())
    if len(matches) != 1:
        raise LookupError(
            f"expected exactly one ```json {marker} block in {CONTRACT}, found {len(matches)}"
        )
    return json.loads(matches[0])
