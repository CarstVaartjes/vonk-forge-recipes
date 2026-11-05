"""One reader of a receipt's failure: the sweep's outcome comes from ``spark_sweep/receipt.py``.

Reading only a receipt's lifecycle word (or its failure beside it) is how 383 exhausted
preparations were re-requested forever. No other module may read ``failure`` off a document.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "spark_sweep"


def _reads(tree: ast.AST):
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "failure"
        ) or (
            isinstance(node, ast.Subscript)
            and isinstance(node.ctx, ast.Load)  # recording our own signature is no read
            and isinstance(node.slice, ast.Constant)
            and node.slice.value == "failure"
        ):
            yield node.lineno


def test_no_module_reads_a_receipts_failure_but_the_receipt_reader() -> None:
    offenders = [
        f"{path.name}:{line}"
        for path in sorted(ROOT.glob("*.py"))
        if path.name != "receipt.py"
        for line in _reads(ast.parse(path.read_text()))
    ]
    assert not offenders, offenders
