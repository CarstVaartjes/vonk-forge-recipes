"""Tests take the repository root from tests/repo_root.py, never from ``__file__``."""

from __future__ import annotations

import ast

from repo_root import TESTS

OWNERS = {"repo_root.py", "conftest.py"}


def test_only_the_root_owner_reads_file() -> None:
    offenders = [
        f"{path.relative_to(TESTS)}:{node.lineno}"
        for path in sorted(TESTS.rglob("*.py"))
        if path.name not in OWNERS
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.Name) and node.id == "__file__"
    ]
    assert not offenders, offenders
