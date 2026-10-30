"""A modular pipeline selects its workflow once.

``ModularPipeline.from_pretrained(..., workflow=...)`` prunes the blocks to that
workflow; passing ``workflow=`` again to ``load_components`` calls ``get_workflow``
on the already-pruned blocks and raises ``NotImplementedError: workflows is not
supported because _workflow_map is not set``.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _passes_workflow(call: ast.Call) -> bool:
    return any(keyword.arg == "workflow" for keyword in call.keywords)


def _double_selections(source: str) -> list[int]:
    lines = []
    for function in ast.walk(ast.parse(source)):
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        calls = [
            (node, node.func.attr)
            for node in ast.walk(function)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        ]
        loads = [
            call
            for call, name in calls
            if name == "from_pretrained" and _passes_workflow(call)
        ]
        components = [
            call
            for call, name in calls
            if name == "load_components" and _passes_workflow(call)
        ]
        if loads and components:
            lines.extend(call.lineno for call in components)
    return lines


def test_guard_detects_the_failing_pattern() -> None:
    bad = (
        "def load():\n"
        "    pipe = P.from_pretrained(p, workflow=w)\n"
        "    pipe.load_components(workflow=w, dtype=d)\n"
    )
    assert _double_selections(bad) == [3]


def test_guard_accepts_each_single_selection() -> None:
    from_pretrained_only = (
        "def load():\n"
        "    pipe = P.from_pretrained(p, workflow=w)\n"
        "    pipe.load_components(dtype=d)\n"
    )
    load_components_only = (
        "def load():\n"
        "    pipe = P.from_pretrained(p)\n"
        "    pipe.load_components(workflow=w, dtype=d)\n"
    )
    assert _double_selections(from_pretrained_only) == []
    assert _double_selections(load_components_only) == []


def test_no_adapter_selects_a_workflow_twice() -> None:
    offenders = []
    for path in sorted((ROOT / "adapters").rglob("*.py")):
        if any(part in {"vendor", "vendored"} for part in path.parts):
            continue
        for line in _double_selections(path.read_text(encoding="utf-8")):
            offenders.append(f"{path.relative_to(ROOT)}:{line}")
    assert offenders == []
