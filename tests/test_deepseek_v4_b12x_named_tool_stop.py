"""A thinking-off named tool call must be able to stop on the B12X vLLM build.

``eugr/spark-vllm-b12x`` carries vLLM b5f995e7. There strict tool calling is on
by default (``vllm/envs.py:2039``), so ``DelegatingParser._apply_structural_tag``
(``vllm/parser/abstract_parser.py:543-566``) replaces a named or required tool
choice with xgrammar's ``deepseek_v4`` structural tag, built with
``reasoning=self._reasoning_enabled()``. That reads ``thinking_enabled`` from
the reasoning parser (``abstract_parser.py:540``, forwarded by
``vllm/parser/engine/adapters.py:132``), and ``DeepSeekV4Parser``
(``vllm/parser/deepseek_v4.py:212-232``) computes ``thinking`` from the chat
template kwargs but never exposes it, so the default ``True`` wins even for
``thinking: false``. xgrammar's ``reasoning="enabled"`` prefix then requires a
``</think>`` the chat-mode prompt already closed: end of sequence stays masked
after the call and generation runs to ``max_tokens``. On 2026-10-27 the sfxnz
Vision-Exp dual answered T_REPORT with the one correct call, empty content, no
reasoning and ``finish_reason: length`` at 256 of 256 tokens.

``VLLM_ENFORCE_STRICT_TOOL_CALLING=0`` sends named and required calls through
the JSON-schema grammar instead, the path the hardware-proven DeepSeek V4
tonyd2wild DSpark dual takes (its vLLM overlay defaults strict tool calling
off). The automatic tool choice that T_PRODUCT exercises builds no structural
tag for non-strict tools either way.
"""

from __future__ import annotations

import json
from pathlib import Path

from qualification.definitions_loader import load_definitions

ROOT = Path(__file__).resolve().parents[1]
B12X_IMAGE = "docker.io/eugr/spark-vllm-b12x"
STRICT_TOOL_CALLING = "VLLM_ENFORCE_STRICT_TOOL_CALLING"


def _forced_tool_cases(definitions: dict) -> set[str]:
    """Smoke cases whose request forces a tool call (named or required)."""
    forced = set()
    for name, case in definitions["service_case_templates"].items():
        body = case.get("body") or {}
        choice = body.get("tool_choice")
        if body.get("tools") and (isinstance(choice, dict) or choice == "required"):
            forced.add(name)
    return forced


def _b12x_deepseek_v4_forced_tool_recipes() -> dict[str, dict]:
    definitions = load_definitions()
    forced = _forced_tool_cases(definitions)
    found = {}
    for path in sorted((ROOT / "recipes").glob("*.json")):
        recipe = json.loads(path.read_text())
        runtime = recipe.get("runtime") or {}
        if runtime.get("engine") != "vllm":
            continue
        build = (recipe.get("execution") or {}).get("build") or {}
        if (build.get("base_image") or {}).get("repository") != B12X_IMAGE:
            continue
        arguments = {
            arg["name"]: arg.get("value") for arg in runtime.get("arguments", [])
        }
        if arguments.get("tool-call-parser") != "deepseek_v4":
            continue
        spec = definitions["service_recipes"].get(f"vonk-forge/{path.stem}") or {}
        if forced & set(spec.get("smoke_cases", [])):
            found[path.stem] = recipe
    return found


def test_t_report_is_a_forced_tool_case() -> None:
    assert "T_REPORT" in _forced_tool_cases(load_definitions())


def test_b12x_deepseek_v4_forced_tool_recipes_disable_strict_tool_grammar() -> None:
    recipes = _b12x_deepseek_v4_forced_tool_recipes()
    assert "deepseek-v4-flash-vision-exp-sfxnz-vllm-dual" in recipes
    for slug, recipe in recipes.items():
        environment = {
            item["name"]: item.get("value")
            for item in recipe["runtime"].get("environment", [])
        }
        assert environment.get(STRICT_TOOL_CALLING) == "0", slug
