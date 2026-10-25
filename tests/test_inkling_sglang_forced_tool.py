"""A forced Inkling tool choice on SGLang is one native call.

Hardware, 2026-10-25: inkling-small-nvfp4-sglang-mia-dual and
inkling-small-nvfp4-kv-1m-drowzeys-sglang-dual answered T_REPORT (a named tool
choice) with two identical report_temperature calls, 35 completion tokens of
which 17 were counted as reasoning. Inkling's reasoning is always on and ends at
<|end_message|>, so SGLang's reasoning wrapper left the grammar off for the
model's first (correct) call and the grammar then forced a second one. These
tests run the pinned SGLang code (fixtures) with the shared build patch applied
by the same entry point the images run.
"""

from __future__ import annotations

import ast
import enum
import json
import logging
import os
import subprocess
import sys
import types
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jsonschema
import pytest

ROOT = Path(__file__).resolve().parents[1]
PATCH = "inkling-forced-tool-grammar.py"
CANONICAL = ROOT / "adapters/inkling/small-dual" / PATCH
FIXTURES = ROOT / "tests/fixtures"
MESSAGE_MODEL = "<|message_model|>"
INVOKE_JSON = "<|content_invoke_tool_json|>"
END_MESSAGE = "<|end_message|>"
END_SAMPLING = "<|content_model_end_sampling|>"
CHAT = "srt/entrypoints/openai/serving_chat.py"
DETECTOR = "srt/function_call/inkling_detector.py"


def _inkling_sglang_contexts() -> set[Path]:
    contexts = set()
    for path in sorted((ROOT / "recipes").glob("*.json")):
        recipe = json.loads(path.read_text())
        runtime = recipe["runtime"]
        arguments = {item["name"]: item.get("value") for item in runtime["arguments"]}
        if (
            runtime["engine"] == "sglang"
            and arguments.get("reasoning-parser") == "inkling"
        ):
            build = recipe.get("execution", {}).get("build")
            assert build is not None, f"{path.name}: no execution.build to patch"
            contexts.add(ROOT / build["context"]["path"])
    return contexts


def test_every_inkling_sglang_recipe_applies_the_shared_patch() -> None:
    contexts = _inkling_sglang_contexts()
    assert CANONICAL.parent in contexts
    for context in contexts:
        assert (context / PATCH).read_bytes() == CANONICAL.read_bytes(), context
        dockerfile = (context / "Dockerfile").read_text()
        copy = f"COPY {PATCH} /opt/vonk/build/{PATCH}"
        run = f"python3 /opt/vonk/build/{PATCH}"
        assert copy in dockerfile and run in dockerfile, context
        assert dockerfile.index(copy) < dockerfile.index(run), context


def _tree(tmp_path: Path, detector: str | None = None) -> Path:
    package = tmp_path / "sglang"
    for relative, fixture in (
        (CHAT, "sglang-serving-chat-reasoning.py.txt"),
        (DETECTOR, "inkling-detector.py.txt"),
    ):
        (package / relative).parent.mkdir(parents=True, exist_ok=True)
        (package / relative).write_bytes((FIXTURES / fixture).read_bytes())
    (package / "__init__.py").write_text("")
    if detector is not None:
        (package / DETECTOR).write_text(detector)
    return package


def _apply(package: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CANONICAL)],
        cwd=package.parent,
        env={**os.environ, "PYTHONPATH": str(package.parent)},
        capture_output=True,
        text=True,
        check=False,
    )


class ToolChoice(SimpleNamespace):
    pass


class ToolStrictLevel(enum.IntEnum):
    OFF = 0
    FUNCTION = 1
    PARAMETER = 2


@pytest.fixture(autouse=True)
def strict_level(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """sglang.srt.environ as the patched detector imports it (SGLANG_TOOL_STRICT_LEVEL)."""
    level = SimpleNamespace(value=ToolStrictLevel.OFF)
    environ = types.ModuleType("sglang.srt.environ")
    environ.ToolStrictLevel = ToolStrictLevel  # type: ignore[attr-defined]
    environ.envs = SimpleNamespace(  # type: ignore[attr-defined]
        SGLANG_TOOL_STRICT_LEVEL=SimpleNamespace(get=lambda: level.value)
    )
    monkeypatch.setitem(sys.modules, "sglang.srt.environ", environ)
    return level


def _get_reasoning(source: str) -> Callable[[Any], bool]:
    namespace: dict[str, Any] = {
        "ChatCompletionRequest": SimpleNamespace,
        "ToolChoice": ToolChoice,
        "logger": logging.getLogger(__name__),
    }
    exec(compile(source, CHAT, "exec"), namespace)  # noqa: S102 - pinned source.
    server = object.__new__(namespace["OpenAIServingChat"])
    server.reasoning_parser = "inkling"
    # The Inkling model has no chat template, so the parser default decides.
    server.template_manager = SimpleNamespace(reasoning_config=None)
    server._reasoning_detector = SimpleNamespace(reasoning_default="always")
    return server._get_reasoning_from_request


def _detector(source: str) -> Any:
    """Execute the pinned detector without importing its GPU-dependent package."""

    @dataclass
    class Result:
        normal_text: str = ""
        calls: list = field(default_factory=list)

    @dataclass
    class Call:
        tool_index: int
        name: str
        parameters: str

    class Base:
        def __init__(self):
            self._buffer = ""

        def get_structural_tag(self, *args, **kwargs):
            return None

        def _get_tool_indices(self, tools):
            return {tool.function.name: i for i, tool in enumerate(tools)}

        @staticmethod
        def _ends_with_partial_token(text, token):
            return max(
                (i for i in range(1, len(token)) if text.endswith(token[:i])),
                default=0,
            )

    class Tag:
        @staticmethod
        def model_validate(value):
            return value

    tree = ast.parse(source)
    module = ast.Module(
        body=[
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        ],
        type_ignores=[],
    )
    tokens = {
        "CONTENT_INVOKE_TOOL_JSON": INVOKE_JSON,
        "CONTENT_INVOKE_TOOL_TEXT": "<|content_invoke_tool_text|>",
        "END_MESSAGE": END_MESSAGE,
        "MESSAGE_MODEL": MESSAGE_MODEL,
    }
    namespace: dict[str, Any] = {
        "BaseFormatDetector": Base,
        "StructuralTag": Tag,
        "StreamingParseResult": Result,
        "ToolCallItem": Call,
        "Mapping": dict,
        "Tool": object,
        "json": json,
        "logger": logging.getLogger(__name__),
        "INKLING_CONTROL_TOKENS": tuple(tokens.values()),
        "INKLING_SPECIAL_TOKEN_IDS": {},
        **tokens,
    }
    exec(compile(module, DETECTOR, "exec"), namespace)  # noqa: S102 - pinned source.
    return namespace["InklingDetector"]()


def _reasoner_wrapper(grammar: Any, require_reasoning: bool) -> Any:
    class BaseGrammarObject:
        def __init__(self):
            self._finished = False
            self.grammar_stats = None
            self.current_token = None

    namespace: dict[str, Any] = {
        "BaseGrammarObject": BaseGrammarObject,
        "torch": SimpleNamespace(Tensor=object),
    }
    source = (FIXTURES / "sglang-reasoner-grammar-object.py.txt").read_text()
    exec(compile(source, "reasoner_grammar_backend.py", "exec"), namespace)  # noqa: S102
    # ReasonerGrammarBackend: think_end_id is the reasoning parser's end token.
    wrapper = namespace["ReasonerGrammarObject"](grammar, think_end_id=END_MESSAGE)
    wrapper.maybe_init_reasoning(require_reasoning)
    return wrapper


class TagGrammar:
    """XGrammar's structural-tag semantics over text tokens: the begin and end
    literals are exact, the content must satisfy the tool's JSON schema, and
    after the end only the stop token is accepted."""

    def __init__(self, structural_tag: dict) -> None:
        frame = structural_tag["format"]
        self.tags = frame["elements"] if frame["type"] == "or" else [frame]
        assert all(tag["type"] == "tag" for tag in self.tags)
        self.text = ""
        self.terminated = False

    def _complete(self, text: str) -> bool:
        for tag in self.tags:
            begin, end = tag["begin"], tag["end"]
            if text.startswith(begin) and text.endswith(end):
                try:
                    payload = json.loads(text[len(begin) : -len(end)])
                    jsonschema.validate(payload, tag["content"]["json_schema"])
                except (ValueError, jsonschema.ValidationError):
                    continue
                return True
        return False

    def allows(self, token: str) -> bool:
        if self._complete(self.text):
            return token == END_SAMPLING
        text = self.text + token
        return any(
            tag["begin"].startswith(text)
            or (text.startswith(tag["begin"]) and END_MESSAGE not in text)
            or (text.startswith(tag["begin"]) and self._complete(text))
            for tag in self.tags
        )

    def fill_vocab_mask(self, vocab_mask: dict, idx: int) -> None:
        vocab_mask[idx] = self.allows

    def accept_token(self, token: str) -> None:
        assert self.allows(token), token
        if token == END_SAMPLING:
            self.terminated = True
        else:
            self.text += token

    def is_terminated(self) -> bool:
        return self.terminated


# The model's natural answer to T_REPORT, as Inkling tokens: one call, then the
# end of sampling. Inkling opens its own block; the prompt ends without
# <|message_model|> (render_inkling_messages(add_generation_prompt=False)).
CALL = [
    MESSAGE_MODEL,
    "report_temperature",
    INVOKE_JSON,
    '{"name":',
    '"report_temperature"',
    ',"args":',
    '{"city":"Amsterdam"}',
    "}",
    END_MESSAGE,
]


def _natural(output: list[str]) -> str:
    """The model finishes the call it is in and ends sampling after a call."""
    if not output:
        return CALL[0]
    if output[-1] == END_MESSAGE:
        return END_SAMPLING
    start = len(output) - 1 - output[::-1].index(MESSAGE_MODEL)
    return CALL[len(output) - start]


def _generate(wrapper: Any) -> list[str]:
    """Greedy decoding through SGLang's wrapper: the model prefers its natural
    next token, else the first allowed token of its call (temperature 0)."""
    output: list[str] = []
    while len(output) < 4 * len(CALL):
        natural = _natural(output)
        mask: dict[int, Callable[[str], bool]] = {}
        wrapper.fill_vocab_mask(mask, 0)
        allowed = mask.get(0, lambda _token: True)
        token = next(t for t in [natural, *CALL, END_SAMPLING] if allowed(t))
        wrapper.accept_token(token)
        output.append(token)
        if token == END_SAMPLING or wrapper.is_terminated():
            return output
    raise AssertionError(f"generation did not stop: {output}")


def _t_report() -> tuple[SimpleNamespace, list[SimpleNamespace]]:
    shared = json.loads((ROOT / "qualification/shared.json").read_text())
    body = shared["service_case_templates"]["T_REPORT"]["body"]
    tools = [
        SimpleNamespace(
            function=SimpleNamespace(**{"strict": False, **tool["function"]})
        )
        for tool in body["tools"]
    ]
    choice = body["tool_choice"]
    assert choice["type"] == "function"  # T_REPORT names its tool
    request = SimpleNamespace(
        tools=tools,
        tool_choice=ToolChoice(function=SimpleNamespace(**choice["function"])),
        chat_template_kwargs=body["chat_template_kwargs"],
        reasoning_effort=None,
    )
    return request, tools


def _calls(output: list[str], detector: Any, tools: list) -> list[tuple[str, dict]]:
    text = "".join(token for token in output if token != END_SAMPLING)
    result = detector.detect_and_parse(text, tools)
    return [(call.name, json.loads(call.parameters)) for call in result.calls]


def test_t_report_gets_exactly_one_call(tmp_path: Path) -> None:
    package = _tree(tmp_path)
    unpatched_reasoning = _get_reasoning((package / CHAT).read_text())
    assert _apply(package).returncode == 0
    get_reasoning = _get_reasoning((package / CHAT).read_text())
    detector = _detector((package / DETECTOR).read_text())
    request, tools = _t_report()
    # Both pins' call shapes: b7252cc6 passes three keywords, 94602c9c four.
    tag = detector.get_structural_tag(
        tools=tools, thinking_mode=False, tool_choice=request.tool_choice
    )
    assert tag == detector.get_structural_tag(
        tools=tools,
        thinking_mode=False,
        tool_choice=request.tool_choice,
        parallel_tool_calls=True,
    )
    one = [("report_temperature", {"city": "Amsterdam"})]

    # The hardware failure: reasoning required, the first call is generated
    # before the grammar applies, then the grammar forces a second call.
    assert unpatched_reasoning(request) is True
    output = _generate(_reasoner_wrapper(TagGrammar(tag), True))
    assert _calls(output, detector, tools) == one * 2

    # Patched: no reasoning phase, the grammar is one call from token 0.
    assert get_reasoning(request) is False
    wrapper = _reasoner_wrapper(TagGrammar(tag), False)
    output = _generate(wrapper)
    assert output == [*CALL, END_SAMPLING]
    assert wrapper.tokens_in_think == -1
    assert _calls(output, detector, tools) == one


def test_only_forced_tool_choices_skip_reasoning(tmp_path: Path) -> None:
    package = _tree(tmp_path)
    assert _apply(package).returncode == 0
    get_reasoning = _get_reasoning((package / CHAT).read_text())
    request, _tools = _t_report()
    assert get_reasoning(request) is False
    for choice in ("auto", "none", None):
        request.tool_choice = choice
        assert get_reasoning(request) is True, choice
    request.tool_choice = "required"
    assert get_reasoning(request) is False


def test_required_grammar_is_one_call_of_any_tool(tmp_path: Path) -> None:
    package = _tree(tmp_path)
    assert _apply(package).returncode == 0
    detector = _detector((package / DETECTOR).read_text())
    _request, tools = _t_report()
    other = SimpleNamespace(
        function=SimpleNamespace(name="other", parameters=None, strict=False)
    )
    tag = detector.get_structural_tag(tools=[*tools, other], tool_choice="required")
    frame = tag["format"]
    assert frame["type"] == "or"
    assert [element["begin"] for element in frame["elements"]] == [
        f'{MESSAGE_MODEL}{name}{INVOKE_JSON}{{"name":"{name}","args":'
        for name in ("report_temperature", "other")
    ]
    assert [element["content"]["json_schema"] for element in frame["elements"]] == [
        {},
        {},
    ]
    output = _generate(_reasoner_wrapper(TagGrammar(tag), False))
    assert _calls(output, detector, tools) == [
        ("report_temperature", {"city": "Amsterdam"})
    ]
    grammar = TagGrammar(tag)
    for token in [MESSAGE_MODEL, "report_temperature", INVOKE_JSON]:
        grammar.accept_token(token)
    assert not grammar.allows(END_SAMPLING)
    tools[0].function.strict = True
    strict = detector.get_structural_tag(tools=[*tools, other], tool_choice="required")
    bad = TagGrammar(strict)
    for token in CALL[:6]:
        bad.accept_token(token)
    assert not bad.allows('{"city":3}' + "}" + END_MESSAGE)
    # Automatic choice keeps upstream's grammar.
    assert detector.get_structural_tag(tools=tools, tool_choice="auto") is None


def test_changed_upstream_code_fails_the_build_and_writes_nothing(
    tmp_path: Path,
) -> None:
    package = _tree(tmp_path / "patched")
    assert _apply(package).returncode == 0
    patched = {path: path.read_bytes() for path in package.rglob("*.py")}
    again = _apply(package)
    assert again.returncode != 0 and "pinned SGLang code changed" in again.stderr
    assert {path: path.read_bytes() for path in package.rglob("*.py")} == patched

    detector = (FIXTURES / "inkling-detector.py.txt").read_text()
    drifted = detector.replace("def get_auto_tool_call_structural_tag(", "def x(")
    package = _tree(tmp_path / "drifted", detector=drifted)
    chat = (package / CHAT).read_bytes()
    result = _apply(package)
    assert result.returncode != 0 and "pinned SGLang code changed" in result.stderr
    assert (package / CHAT).read_bytes() == chat


def test_arguments_follow_upstream_strictness(
    tmp_path: Path, strict_level: SimpleNamespace
) -> None:
    """Upstream's get_legacy_structural_tag rule: the schema binds the arguments
    only for a strict tool or SGLANG_TOOL_STRICT_LEVEL >= PARAMETER, else {}."""
    package = _tree(tmp_path)
    assert _apply(package).returncode == 0
    detector = _detector((package / DETECTOR).read_text())
    request, tools = _t_report()
    schema = tools[0].function.parameters

    def arguments() -> object:
        tag = detector.get_structural_tag(tools=tools, tool_choice=request.tool_choice)
        return tag["format"]["content"]["json_schema"]

    assert arguments() == {}
    tools[0].function.strict = True
    assert arguments() == schema
    tools[0].function.strict = False
    strict_level.value = ToolStrictLevel.FUNCTION
    assert arguments() == {}
    strict_level.value = ToolStrictLevel.PARAMETER
    assert arguments() == schema
