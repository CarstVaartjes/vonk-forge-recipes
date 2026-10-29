"""No adapter forces an eager load (low_cpu_mem_usage=False) next to a quantization_config.

diffusers' ``from_pretrained`` raises ``ValueError`` for that combination; the default
(``None`` -> ``True``) is the supported quantized load.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _eager_quantized_loads(source: str) -> list[int]:
    lines = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        names = {keyword.arg: keyword.value for keyword in node.keywords}
        eager = names.get("low_cpu_mem_usage")
        if (
            "quantization_config" in names
            and isinstance(eager, ast.Constant)
            and eager.value is False
        ):
            lines.append(eager.lineno)
    return lines


def test_guard_detects_the_failing_call() -> None:
    bad = "m = M.from_pretrained(p, quantization_config=q, low_cpu_mem_usage=False)\n"
    assert _eager_quantized_loads(bad) == [1]
    assert (
        _eager_quantized_loads("m = M.from_pretrained(p, quantization_config=q)\n")
        == []
    )
    assert (
        _eager_quantized_loads("m = M.from_pretrained(p, low_cpu_mem_usage=False)\n")
        == []
    )


def test_no_adapter_pairs_quantization_with_eager_load() -> None:
    offenders = []
    for path in sorted((ROOT / "adapters").rglob("*.py")):
        if any(part in {"vendor", "vendored"} for part in path.parts):
            continue
        for line in _eager_quantized_loads(path.read_text(encoding="utf-8")):
            offenders.append(f"{path.relative_to(ROOT)}:{line}")
    assert offenders == []
