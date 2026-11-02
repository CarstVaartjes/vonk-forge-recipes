"""Assertions for memory estimate warnings."""

import re


def assert_warns(warnings: list[str], match: str = "") -> None:
    assert any(re.search(match, item) for item in warnings), warnings


def assert_not_warns(warnings: list[str], match: str) -> None:
    """The real recipe's declaration does not trip this rule; others may warn."""
    assert not any(re.search(match, item) for item in warnings), warnings


def declare_peak(recipe, peak_bytes: int = 124_000_000_000) -> None:
    """Overwrite a real recipe's declared memory so a test judges a synthetic one.

    A test may read a real recipe for its shape; it must not gate on the peak the
    catalog happens to declare (tools/check-estimate-gates).
    """
    for role in recipe.topology.roles:
        memory = role.resources.memory
        memory.peak_bytes = peak_bytes
        memory.load_peak_bytes = None
        memory.minimum_swap_bytes = 0
