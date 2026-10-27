"""A thinking-off forced tool call must be able to stop on the b5f995e7 B12X vLLM.

Hardware, 2026-10-27: deepseek-v4-flash-vision-exp-sfxnz-vllm-dual answered
T_REPORT (a named tool choice, ``thinking: false``) with the one correct call,
empty content, no reasoning and ``finish_reason: length`` at 256 of 256 tokens.
Its base ``docker.io/eugr/spark-vllm-b12x@sha256:7dc02f16...`` carries vLLM
0.1.dev20133+gb5f995e73 (local-inference-lab/vllm ``dev/infernal-invocation``,
per the image's ``/workspace/build-metadata.yaml``). That build turns strict
tool calling on by default and builds the forced call's ``deepseek_v4``
structural tag with ``reasoning=True`` whenever the reasoning parser does not
report ``thinking_enabled``, which ``DeepSeekV4Parser`` never does. xgrammar's
enabled reasoning prefix then waits for a ``</think>`` that the chat-mode prompt
has already closed, so end of sequence stays masked until ``max_tokens``.

These tests run that pinned vLLM code (fixture) on the T_REPORT request.
``VLLM_ENFORCE_STRICT_TOOL_CALLING=0`` sends the forced call through the
JSON-schema grammar instead. The newer B12X base ``03a956...`` (vLLM 8244d1c4,
``dev/karmic-kraken``) passes ``reasoning=False`` and is not affected.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/vllm-b5f995e7-forced-tool-grammar.py.txt"
# The base image whose vLLM is b5f995e7 (content identity, not repository name).
B5F995E7_BASE = "7dc02f162929943ba2e14514066ed2a04bb7e9ed3592d4eb460ebcbb1f8376bd"
STRICT = "VLLM_ENFORCE_STRICT_TOOL_CALLING"
SFXNZ = "deepseek-v4-flash-vision-exp-sfxnz-vllm-dual"


class ChatCompletionRequest(SimpleNamespace):
    pass


class ResponsesRequest(SimpleNamespace):
    pass


class ChatCompletionNamedToolChoiceParam(SimpleNamespace):
    pass


class ToolChoiceFunction(SimpleNamespace):
    pass


class StructuredOutputsParams(SimpleNamespace):
    def __init__(self, json: Any = None, structural_tag: str | None = None) -> None:
        super().__init__(json=json, structural_tag=structural_tag)


class ParserEngine:
    """The constructor surface DeepSeekV4Parser calls; it defines no thinking_enabled."""

    def __init__(self, tokenizer, tools, *, parser_engine_config, **kwargs) -> None:
        self.parser_engine_config = parser_engine_config

    def _convert_args(self, raw_args: str, partial: bool) -> str:
        return raw_args


def _named_tool_schema(*, tool_choice, tools) -> dict | None:
    """vLLM's get_json_schema_from_tools for a named choice: that tool's parameters."""
    if isinstance(tool_choice, ChatCompletionNamedToolChoiceParam):
        name = tool_choice.function.name
        return next(t.function.parameters for t in tools if t.function.name == name)
    return None


class _Envs:
    """vllm.envs: each attribute evaluates its environment lambda on access."""

    def __init__(self, namespace: dict[str, Any]) -> None:
        self._namespace = namespace

    def __getattr__(self, name: str) -> Any:
        return self._namespace["environment_variables"][name]()


def _load_vllm(monkeypatch: pytest.MonkeyPatch, strict: str | None) -> dict:
    """Execute the pinned vLLM code with the environment a recipe launches."""
    if strict is None:
        monkeypatch.delenv(STRICT, raising=False)
    else:
        monkeypatch.setenv(STRICT, strict)
    requested: list[dict] = []

    def get_model_structural_tag(**kwargs: Any) -> Any:
        requested.append(kwargs)
        return SimpleNamespace(model_dump=lambda: {"model": kwargs["model"]})

    registry = ModuleType("vllm.tool_parsers.structural_tag_registry")
    registry.get_model_structural_tag = get_model_structural_tag  # type: ignore[attr-defined]
    for name in ("vllm", "vllm.tool_parsers"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, registry.__name__, registry)

    namespace: dict[str, Any] = {
        "os": os,
        "json": json,
        "Any": Any,
        "TokenizerLike": object,
        "Tool": object,
        "ChatCompletionRequest": ChatCompletionRequest,
        "ResponsesRequest": ResponsesRequest,
        "ChatCompletionNamedToolChoiceParam": ChatCompletionNamedToolChoiceParam,
        "ToolChoiceFunction": ToolChoiceFunction,
        "StructuredOutputsParams": StructuredOutputsParams,
        "get_json_schema_from_tools": _named_tool_schema,
        "ParserEngine": ParserEngine,
        "deepseek_v4_config": lambda thinking: SimpleNamespace(thinking=thinking),
    }
    namespace["envs"] = _Envs(namespace)
    exec(compile(FIXTURE.read_text(), str(FIXTURE), "exec"), namespace)  # noqa: S102 - pinned source.
    namespace["requested_structural_tags"] = requested
    return namespace


def _recipe(slug: str) -> dict:
    return json.loads((ROOT / "recipes" / f"{slug}.json").read_text())


def _t_report(recipe: dict) -> ChatCompletionRequest:
    """T_REPORT as this recipe's server receives it (server kwargs, then request)."""
    shared = json.loads((ROOT / "qualification/shared.json").read_text())
    body = shared["service_case_templates"]["T_REPORT"]["body"]
    arguments = {a["name"]: a.get("value") for a in recipe["runtime"]["arguments"]}
    kwargs = json.loads(arguments.get("default-chat-template-kwargs") or "{}")
    kwargs.update(body["chat_template_kwargs"])
    tools = [
        SimpleNamespace(function=SimpleNamespace(**tool["function"]))
        for tool in body["tools"]
    ]
    choice = body["tool_choice"]
    return ChatCompletionRequest(
        tools=tools,
        tool_choice=ChatCompletionNamedToolChoiceParam(
            type=choice["type"], function=SimpleNamespace(**choice["function"])
        ),
        chat_template_kwargs=kwargs,
        structured_outputs=None,
        response_format=None,
        _grammar_from_tool_parser=False,
    )


def _serve(vllm: dict, request: ChatCompletionRequest) -> ChatCompletionRequest:
    """DelegatingParser.adjust_request with the deepseek_v4 parsers, as serving runs it."""
    engine = vllm["DeepSeekV4Parser"](
        None, request.tools, chat_template_kwargs=dict(request.chat_template_kwargs)
    )
    reasoner = object.__new__(vllm["ParserEngineReasoningAdapter"])
    reasoner._parser_engine = engine
    reasoner.adjust_request = lambda r: r  # ParserEngine only unskips special tokens.
    parser = object.__new__(vllm["DelegatingParser"])
    parser._reasoning_parser = reasoner
    parser._tool_parser = object.__new__(vllm["DeepSeekV4EngineToolParser"])
    assert engine.parser_engine_config.thinking is False  # the parser knows: off
    return parser.adjust_request(request)


def _environment(recipe: dict) -> dict[str, str]:
    return {e["name"]: e.get("value") for e in recipe["runtime"].get("environment", [])}


def test_default_strict_grammar_demands_reasoning_for_a_thinking_off_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vllm = _load_vllm(monkeypatch, strict=None)
    request = _serve(vllm, _t_report(_recipe(SFXNZ)))
    assert vllm["requested_structural_tags"] == [
        {
            "model": "deepseek_v4",
            "tools": request.tools,
            "tool_choice": request.tool_choice,
            "reasoning": True,
        }
    ]
    assert request.structured_outputs.structural_tag is not None
    assert request._grammar_from_tool_parser is True
    assert vllm["DeepSeekV4EngineToolParser"].supports_required_and_named is False


def _b5f995e7_deepseek_v4_recipes() -> list[str]:
    found = []
    for path in sorted((ROOT / "recipes").glob("*.json")):
        recipe = json.loads(path.read_text())
        runtime = recipe["runtime"]
        build = recipe.get("execution", {}).get("build") or {}
        arguments = {a["name"]: a.get("value") for a in runtime["arguments"]}
        if (
            runtime["engine"] == "vllm"
            and (build.get("base_image") or {}).get("digest") == B5F995E7_BASE
            and arguments.get("tool-call-parser") == "deepseek_v4"
        ):
            found.append(path.stem)
    return found


def test_every_b5f995e7_deepseek_v4_recipe_serves_forced_calls_by_json_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slugs = _b5f995e7_deepseek_v4_recipes()
    assert SFXNZ in slugs
    for slug in slugs:
        recipe = _recipe(slug)
        vllm = _load_vllm(monkeypatch, strict=_environment(recipe).get(STRICT))
        request = _serve(vllm, _t_report(recipe))
        assert vllm["requested_structural_tags"] == [], slug
        assert request.structured_outputs.structural_tag is None, slug
        assert request.structured_outputs.json == request.tools[0].function.parameters
        assert request._grammar_from_tool_parser is False, slug
        assert vllm["DeepSeekV4EngineToolParser"].supports_required_and_named, slug
