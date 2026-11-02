"""Assertions for memory estimate warnings."""

import re


def assert_warns(warnings: list[str], match: str = "") -> None:
    assert any(re.search(match, item) for item in warnings), warnings
