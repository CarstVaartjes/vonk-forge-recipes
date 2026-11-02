"""Assertions for memory estimate warnings."""

import re


def assert_warns(warnings: list[str], match: str = "") -> None:
    assert any(re.search(match, item) for item in warnings), warnings


def assert_not_warns(warnings: list[str], match: str) -> None:
    """The real recipe's declaration does not trip this rule; others may warn."""
    assert not any(re.search(match, item) for item in warnings), warnings
