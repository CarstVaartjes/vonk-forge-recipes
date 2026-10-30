"""A reasoning recipe that serves a repository chat template must honour the switch.

The strength rule applies to every engine; the `<think>` rule to SGLang. SGLang
forces the reasoning parser (the opening tag is in the prompt, so the output has
none) from the chat template's `enable_thinking` switch. A template
that always writes `<think>` and ignores the switch leaves the parser unforced
for a request that turns thinking off, and the end tag leaks into `content`
(smoke case A391 on the GLM-5.3 EXL3 baseline). The switch is a per-request
input, never a recipe option.

The Muse Glimmer template has no `<think>` tag: `render_reasoning` always writes
"Reasoning strength: high.", so a request with thinking off reasoned to
finish_reason=length with null content (smoke V7). Any repository template with a
"Reasoning strength" line must derive it from the switch.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GENERATION_PROMPT = re.compile(
    r"add_generation_prompt\s*-?%\}(?P<tail>.*?)\{%-?\s*endif", re.DOTALL
)
SWITCH = re.compile(r"enable_thinking\s+is\s+defined\s+and\s+not\s+enable_thinking")
MUSE_TEMPLATES = sorted(
    (ROOT / "adapters").glob("*/*/vonk-muse-glimmer-chat-template.jinja")
)
MUSE_UPSTREAM_SHA256 = (
    "114f55ebdc1804c1af371197b9fdf2d6bb925966c9dfe46b73782a71bc07965e"
)
MUSE_SWITCHED_RS = (
    "'low' if (enable_thinking is defined and not enable_thinking) or "
    "(thinking is defined and not thinking) else (reasoning_strength if "
    "reasoning_strength is defined and reasoning_strength else 'high')"
)
MUSE_UPSTREAM_RS = "reasoning_strength if reasoning_strength is defined and reasoning_strength else 'high'"


def _copied_templates(build_dir: Path) -> dict[str, Path]:
    """Container path -> repository file for each `COPY src dest` in the Dockerfile."""
    copies: dict[str, Path] = {}
    for line in (build_dir / "Dockerfile").read_text().splitlines():
        parts = [p for p in line.split() if not p.startswith("--")]
        if len(parts) == 3 and parts[0] == "COPY":
            copies[parts[2]] = build_dir / parts[1]
    return copies


def _reasoning_templates() -> list[tuple[str, str, Path]]:
    found: list[tuple[str, str, Path]] = []
    for path in sorted((ROOT / "recipes").glob("*.json")):
        recipe = json.loads(path.read_text())
        runtime = recipe.get("runtime", {})
        args = {a["name"]: a.get("value") for a in runtime.get("arguments", [])}
        template = args.get("chat-template")
        build = recipe.get("execution", {}).get("build", {})
        context = build.get("context", {}).get("path")
        if not (args.get("reasoning-parser") and template and context):
            continue
        source = _copied_templates(ROOT / context).get(str(template))
        if source is not None:
            found.append((path.stem, runtime.get("engine", ""), source))
    return found


def test_reasoning_templates_honour_enable_thinking() -> None:
    templates = _reasoning_templates()
    assert {slug for slug, _, _ in templates} >= {
        "glm-5-3-flash-exl3-a0-2-05bpw-0xsero-sglang-single",
        "glm-5-3-flash-exl3-mosaic-12l-0xsero-sglang-single",
        "muse-glimmer-30b-nvfp4-vllm-mia-single",
        "muse-glimmer-30b-nvfp4-r0b0tlab-vllm-single",
    }
    for slug, engine, source in templates:
        text = source.read_text()
        if "Reasoning strength" in text:
            assert SWITCH.search(text), (
                f"{slug}: {source.relative_to(ROOT)} writes a reasoning strength "
                "without the enable_thinking switch"
            )
        prompts = GENERATION_PROMPT.findall(text)
        tail = prompts[-1] if prompts else ""
        if engine == "sglang" and "<think>" in tail:
            assert SWITCH.search(tail), (
                f"{slug}: {source.relative_to(ROOT)} opens <think> in the generation "
                "prompt without the enable_thinking switch"
            )


def test_reasoning_strength_templates_honour_the_switch() -> None:
    for path in sorted((ROOT / "adapters").rglob("*.jinja")):
        text = path.read_text()
        if "Reasoning strength" in text:
            assert SWITCH.search(text), (
                f"{path.relative_to(ROOT)} writes a reasoning strength without the "
                "enable_thinking switch"
            )


def test_muse_template_differs_from_upstream_only_in_the_switch() -> None:
    assert len(MUSE_TEMPLATES) == 2
    for path in MUSE_TEMPLATES:
        ours = path.read_bytes()
        assert ours.count(MUSE_SWITCHED_RS.encode()) == 1
        upstream = ours.replace(MUSE_SWITCHED_RS.encode(), MUSE_UPSTREAM_RS.encode())
        assert hashlib.sha256(upstream).hexdigest() == MUSE_UPSTREAM_SHA256
    assert len({p.read_bytes() for p in MUSE_TEMPLATES}) == 1
