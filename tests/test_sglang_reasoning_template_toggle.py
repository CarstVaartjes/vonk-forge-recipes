"""A reasoning recipe that serves a repository chat template must honour the switch.

The strength rule applies to every engine; the `<think>` rule to SGLang and to
the vLLM GLM parsers (`glm45`, `glm47`). The opening tag is in the prompt, so
the output has none; the parser is forced from the chat template's
`enable_thinking` switch. A template that always writes `<think>` and ignores
the switch leaves the parser unforced for a request that turns thinking off,
and the end tag leaks into `content` (smoke case A391 on the GLM-5.3 EXL3
baseline, SGLang and the vLLM K2 recipe). The switch is a per-request input,
never a recipe option.

Other vLLM parsers (`deepseek_r1` on the proven NVFP4 dual recipe) are not held
to the rule. A template baked into the HF weights (not a repository file)
cannot be checked statically; smoke A391 on hardware covers it. The one gap this
file closes is a baked template already proven switchless on hardware: its
12-hex digest prefix (the `chat-template-<prefix>-` model file id) is listed in
`SWITCHLESS_BAKED` and a GLM-parser recipe shipping it must serve a repository
template instead.

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
# Mia's GLM template derives `thinking_enabled` from enable_thinking / thinking.
SWITCH_IDIOMS = re.compile(SWITCH.pattern + r"|\bif\s+thinking_enabled\b")
GLM_VLLM_PARSERS = {"glm45", "glm47"}
SWITCHLESS_BAKED = {
    # 0xSero/GLM-5.3-Flash-EXL3-TR3-2.0bpw @35b4b580 chat_template.jinja:212 opens
    # `<think>` with no switch; A391 on hardware 2026-11-05 leaked `</think>`.
    "d15e56b35f44": "0xSero GLM-5.3-Flash EXL3 TR3 template",
}
BAKED_TEMPLATE_ID = re.compile(r"chat-template-([0-9a-f]{12})-")
MUSE_TEMPLATES = sorted(
    (ROOT / "adapters").glob("*/*/vonk-muse-glimmer-chat-template.jinja")
)
MUSE_SERVED_PATH = "/opt/vonk/muse-glimmer-chat-template.jinja"
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


def _reasoning_templates() -> tuple[list[tuple[str, str, str, str, Path]], list[str]]:
    found: list[tuple[str, str, str, str, Path]] = []
    baked: list[str] = []
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
            found.append(
                (
                    path.stem,
                    runtime.get("engine", ""),
                    str(args["reasoning-parser"]),
                    str(template),
                    source,
                )
            )
        elif runtime.get("engine") == "sglang":
            baked.append(f"{path.stem}: {template}")
    return found, baked


def test_reasoning_templates_honour_enable_thinking() -> None:
    templates, baked = _reasoning_templates()
    assert not baked, (
        "template must be a repository file so the thinking switch can be checked: "
        + "; ".join(baked)
    )
    for slug, engine, parser, _, source in templates:
        text = source.read_text()
        if "Reasoning strength" in text:
            assert SWITCH.search(text), (
                f"{slug}: {source.relative_to(ROOT)} writes a reasoning strength "
                "without the enable_thinking switch"
            )
        prompts = GENERATION_PROMPT.findall(text)
        tail = prompts[-1] if prompts else ""
        if (engine == "sglang" or parser in GLM_VLLM_PARSERS) and "<think>" in tail:
            assert SWITCH_IDIOMS.search(tail), (
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
    served = {
        source
        for _, _, _, template, source in _reasoning_templates()[0]
        if template == MUSE_SERVED_PATH
    }
    assert served and served == set(MUSE_TEMPLATES)
    for path in MUSE_TEMPLATES:
        ours = path.read_bytes()
        assert ours.count(MUSE_SWITCHED_RS.encode()) == 1
        upstream = ours.replace(MUSE_SWITCHED_RS.encode(), MUSE_UPSTREAM_RS.encode())
        assert hashlib.sha256(upstream).hexdigest() == MUSE_UPSTREAM_SHA256
    assert len({p.read_bytes() for p in MUSE_TEMPLATES}) == 1


def test_glm_recipes_do_not_ship_a_known_switchless_baked_template() -> None:
    served = {slug for slug, *_ in _reasoning_templates()[0]}
    for path in sorted((ROOT / "recipes").glob("*.json")):
        recipe = json.loads(path.read_text())
        args = {
            a["name"]: a.get("value")
            for a in recipe.get("runtime", {}).get("arguments", [])
        }
        if str(args.get("reasoning-parser", "")) not in GLM_VLLM_PARSERS:
            continue
        for prefix in BAKED_TEMPLATE_ID.findall(json.dumps(recipe["models"])):
            assert prefix not in SWITCHLESS_BAKED or path.stem in served, (
                f"{path.stem}: the model's own {SWITCHLESS_BAKED[prefix]} ignores "
                "enable_thinking; serve a repository template with the switch"
            )
