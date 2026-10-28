"""An SGLang reasoning recipe whose template opens `<think>` must honour the switch.

SGLang forces the reasoning parser (the opening tag is in the prompt, so the
output has none) from the chat template's `enable_thinking` switch. A template
that always writes `<think>` and ignores the switch leaves the parser unforced
for a request that turns thinking off, and the end tag leaks into `content`
(smoke case A391 on the GLM-5.3 EXL3 baseline). The switch is a per-request
input, never a recipe option.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GENERATION_PROMPT = re.compile(
    r"add_generation_prompt\s*-?%\}(?P<tail>.*?)\{%-?\s*endif", re.DOTALL
)
SWITCH = re.compile(r"enable_thinking\s+is\s+defined\s+and\s+not\s+enable_thinking")


def _copied_templates(build_dir: Path) -> dict[str, Path]:
    """Container path -> repository file for each `COPY src dest` in the Dockerfile."""
    copies: dict[str, Path] = {}
    for line in (build_dir / "Dockerfile").read_text().splitlines():
        parts = [p for p in line.split() if not p.startswith("--")]
        if len(parts) == 3 and parts[0] == "COPY":
            copies[parts[2]] = build_dir / parts[1]
    return copies


def _sglang_reasoning_templates() -> list[tuple[str, Path]]:
    found: list[tuple[str, Path]] = []
    for path in sorted((ROOT / "recipes").glob("*.json")):
        recipe = json.loads(path.read_text())
        runtime = recipe.get("runtime", {})
        if runtime.get("engine") != "sglang":
            continue
        args = {a["name"]: a.get("value") for a in runtime.get("arguments", [])}
        template = args.get("chat-template")
        build = recipe.get("execution", {}).get("build", {})
        context = build.get("context", {}).get("path")
        if not (args.get("reasoning-parser") and template and context):
            continue
        source = _copied_templates(ROOT / context).get(str(template))
        if source is not None:
            found.append((path.stem, source))
    return found


def test_sglang_reasoning_templates_honour_enable_thinking() -> None:
    templates = _sglang_reasoning_templates()
    assert {slug for slug, _ in templates} >= {
        "glm-5-3-flash-exl3-a0-2-05bpw-0xsero-sglang-single",
        "glm-5-3-flash-exl3-mosaic-12l-0xsero-sglang-single",
    }
    for slug, source in templates:
        tail = GENERATION_PROMPT.findall(source.read_text())[-1]
        if "<think>" in tail:
            assert SWITCH.search(tail), (
                f"{slug}: {source.relative_to(ROOT)} opens <think> in the generation "
                "prompt without the enable_thinking switch"
            )
