"""What a serving recipe's engine configuration really enables.

A recipe may claim tool calling (the ``tool-use`` tag) and the sweep may run
tool-call smoke cases only when the serving configuration turns the engine's
tool-call parser on. Engines that parse nothing unless asked (vLLM, SGLang,
TensorRT-LLM) answer a tool request with HTTP 400 or with the raw tool-call
text in ``content``; that is the creator's or the playbook's configuration, not
a failure of the model, so the recipe must not promise what it does not serve.

``serving_capabilities`` reads the flags the recipe launches the engine with:
its ``runtime.arguments``, the default choice of each option, the adapter's own
launch scripts, and, for a creator kit whose entrypoint injects flags from
inside the image, ``qualification/kit-engine-flags.json`` (each entry cites
where the kit sets them).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
KIT_FLAGS = Path(__file__).resolve().parent / "kit-engine-flags.json"

# Engines that parse tool calls and reasoning only when a flag asks them to.
FLAG_ENGINES = frozenset({"vllm", "sglang", "tensorrt-llm"})
# Engines whose server always parses (llama.cpp through its chat template,
# TensorFold, the exllamav3 OpenAI server, ds4): no flag decides.
NATIVE_ENGINES = frozenset({"llama-cpp", "tensorfold", "exllamav3", "ds4"})

TOOL_CASES = frozenset({"T_REPORT", "T_PRODUCT", "NEM_TOOL"})
VISION_CASES = frozenset({"V_RED", "V7"})
STRICT_ARITHMETIC = ("A391", "A323")
UNPARSED_ARITHMETIC = {"A391": "A391_UNPARSED", "A323": "A323_UNPARSED"}

_TOOL_PARSER = re.compile(r"(?<![\w-])--?tool[-_](?:call[-_])?parser\b")
_AUTO_TOOL = re.compile(r"(?<![\w-])--enable-auto-tool-choice\b")
_REASONING_PARSER = re.compile(r"(?<![\w-])--?reasoning[-_]parser\b")
_YAML_TOOL = re.compile(r"^\s*tool_parser\s*:", re.MULTILINE)
_YAML_REASONING = re.compile(r"^\s*reasoning_parser\s*:", re.MULTILINE)
_SKIPPED_DIRS = frozenset({"vendor", "patches", "tests", "licenses", "model-cards"})
_SKIPPED_SUFFIXES = frozenset({".md", ".patch", ".gz", ".json", ".lock", ".txt"})
_VISION_TAGS = frozenset(
    {"vision", "multimodal", "image-input", "vision-language", "native-vision"}
)
_VISION_OFF = re.compile(r"language-model-only|limit-mm-per-prompt\W+\{?\W*image\W+0\b")


@dataclass(frozen=True)
class Capabilities:
    tools: bool
    reasoning_parser: bool
    vision: bool
    flags_from: str


def _arguments(recipe: Mapping[str, Any]) -> list[tuple[str, Any]]:
    found = [
        (str(item["name"]), item.get("value"))
        for item in recipe["runtime"].get("arguments", [])
    ]
    for option in recipe.get("options", []) or []:
        for choice in option.get("choices", []):
            if choice.get("default"):
                found += [
                    (str(item["name"]), item.get("value"))
                    for item in choice.get("args", [])
                ]
    return found


def _flag_text(arguments: Iterable[tuple[str, Any]]) -> str:
    """Arguments as a command line, so one pattern reads recipes and scripts."""
    parts: list[str] = []
    for name, value in arguments:
        if value is False:
            continue
        parts.append(name if name.startswith("-") else f"--{name}")
        if value is not None and value is not True:
            parts.append(str(value))
    return " ".join(parts)


def _adapter_text(recipe: Mapping[str, Any], root: Path) -> str:
    build = recipe.get("execution", {}).get("build")
    if not build:
        return ""
    base = root / build["context"]["path"]
    text: list[str] = []
    if base.is_dir():
        for path in sorted(base.rglob("*")):
            relative = path.relative_to(base)
            if (
                not path.is_file()
                or _SKIPPED_DIRS & set(relative.parts)
                or path.name.startswith(("Dockerfile", "LICENSE"))
                or path.suffix in _SKIPPED_SUFFIXES
                or path.stat().st_size > 200_000
            ):
                continue
            try:
                text.append(path.read_text(encoding="utf-8"))
            except UnicodeDecodeError:
                continue
    return "\n".join(text)


def _kit_flags() -> dict[str, Any]:
    if not KIT_FLAGS.exists():
        return {}
    return json.loads(KIT_FLAGS.read_text(encoding="utf-8"))


def tags(recipe: Mapping[str, Any]) -> set[str]:
    return set(recipe["metadata"]["tags"])


def multimodal(recipe: Mapping[str, Any], models: Mapping[str, Any]) -> bool:
    """The recipe serves images: declared by its tags or by a model it loads."""
    arguments = _flag_text(_arguments(recipe))
    if _VISION_OFF.search(arguments):
        return False
    if tags(recipe) & _VISION_TAGS:
        return True
    for entry in recipe["models"]:
        model = models.get(entry["model"]["slug"], {})
        if "image" in model.get("modalities", []) or "image-understanding" in model.get(
            "capabilities", []
        ):
            return True
    return False


def serving_capabilities(
    recipe: Mapping[str, Any],
    slug: str,
    models: Mapping[str, Any],
    root: Path = ROOT,
) -> Capabilities:
    engine = recipe["runtime"]["engine"]
    if engine in NATIVE_ENGINES:
        jinja_off = "--no-jinja" in _flag_text(_arguments(recipe))
        return Capabilities(
            tools=not jinja_off,
            reasoning_parser=True,
            vision=multimodal(recipe, models),
            flags_from="native",
        )
    if engine not in FLAG_ENGINES:
        raise ValueError(f"{slug}: engine {engine!r} has no capability rule")
    text = _flag_text(_arguments(recipe))
    scripts = _adapter_text(recipe, root)
    source = "runtime arguments" if _TOOL_PARSER.search(text) else "adapter scripts"
    both = text + "\n" + scripts
    if engine == "vllm":
        tools = bool(_TOOL_PARSER.search(both) and _AUTO_TOOL.search(both))
    else:
        tools = bool(_TOOL_PARSER.search(both) or _YAML_TOOL.search(both))
    reasoning = bool(_REASONING_PARSER.search(both) or _YAML_REASONING.search(both))
    kit = _kit_flags().get(slug)
    if kit:
        tools = tools or bool(kit.get("tool_parser"))
        reasoning = reasoning or bool(kit.get("reasoning_parser"))
        source = f"kit: {kit['source']}"
    return Capabilities(
        tools=tools,
        reasoning_parser=reasoning,
        vision=multimodal(recipe, models),
        flags_from=source,
    )


def generic_case_problems(
    slug: str, recipe: Mapping[str, Any], smoke_cases: list[str], caps: Capabilities
) -> list[str]:
    """Every way a recipe's smoke cases or tags claim more (or less) than it serves."""
    engine = recipe["runtime"]["engine"]
    cases = set(smoke_cases)
    problems: list[str] = []
    asks_tools = bool(cases & TOOL_CASES)
    tagged = "tool-use" in tags(recipe)
    if engine in FLAG_ENGINES:
        if asks_tools != caps.tools:
            problems.append(
                f"{slug}: tool cases {sorted(cases & TOOL_CASES)} but the "
                f"{engine} launch {'enables' if caps.tools else 'does not enable'} "
                "a tool-call parser (tools need --enable-auto-tool-choice and "
                "--tool-call-parser on vLLM, --tool-call-parser on SGLang, "
                "tool_parser on TensorRT-LLM)"
            )
        if tagged != caps.tools:
            problems.append(
                f"{slug}: tag tool-use is {'set' if tagged else 'missing'} but "
                f"the launch {'enables' if caps.tools else 'does not enable'} "
                "a tool-call parser"
            )
    else:
        if asks_tools and not caps.tools:
            problems.append(f"{slug}: tool cases but {engine} runs without tools")
        if tagged and not caps.tools:
            problems.append(f"{slug}: tag tool-use but {engine} runs without tools")
    if "NEM_REASON" in cases and not caps.reasoning_parser:
        problems.append(f"{slug}: NEM_REASON needs a reasoning parser")
    arithmetic = [c for c in smoke_cases if c in STRICT_ARITHMETIC]
    unparsed = [c for c in smoke_cases if c in UNPARSED_ARITHMETIC.values()]
    if (
        engine in FLAG_ENGINES
        and not caps.reasoning_parser
        and "reasoning" in tags(recipe)
        and arithmetic
    ):
        problems.append(
            f"{slug}: a reasoning model without a reasoning parser may answer "
            f"with its thinking in content; {arithmetic} must be the _UNPARSED "
            "variant"
        )
    if (caps.reasoning_parser or engine not in FLAG_ENGINES) and unparsed:
        problems.append(f"{slug}: {unparsed} but the engine parses reasoning")
    if cases & VISION_CASES and not caps.vision:
        problems.append(f"{slug}: image cases but the recipe is not multimodal")
    return problems
