"""Run the creator's Gemma4 channel stripper, with and without our unclosed-channel hunk."""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "adapters/llm/r0b0tlab-diffusiongemma-vllm-single"
SANITIZER = (
    ADAPTER
    / "third_party/spark-vllm-docker/mods/diffusiongemma"
    / "gemma4-content-channel-sanitizer.patch"
)
PATCHER = runpy.run_path(str(ADAPTER / "patch-gemma4-unclosed-channel.py"))

CONSTANTS = """
_REASONING_START = "<|channel>"
_THOUGHT_PREFIX = "thought\\n"
_REASONING_END = "<channel|>"
"""


def _creator_source() -> str:
    """Post-image of the shipped patch for the stripper and its label helper."""
    lines = SANITIZER.read_text().splitlines()
    start = next(
        i
        for i, line in enumerate(lines)
        if "+def _strip_disabled_thinking_channel" in line
    )
    body: list[str] = []
    for line in lines[start:]:
        if line.startswith((" def ", "@@", "diff ")):
            break
        if line[:1] in {"+", " "}:
            body.append(line[1:])
    return "\n".join(body)


def _stripper(patched: bool):
    source = _creator_source()
    helper = """
def _strip_thought_label(text):
    if text.startswith(_THOUGHT_PREFIX):
        return text[len(_THOUGHT_PREFIX) :]
    return text
"""
    full = "from __future__ import annotations\n" + CONSTANTS + helper + source
    if patched:
        full = PATCHER["patch_text"](full)
    namespace: dict = {}
    exec(compile(full, "<creator-gemma4-parser>", "exec"), namespace)  # noqa: S102
    return namespace["_strip_disabled_thinking_channel"]


def test_creator_function_is_extracted_from_the_shipped_patch() -> None:
    assert _stripper(False)("plain") == "plain"


def test_unclosed_channel_is_dropped_without_the_hunk() -> None:
    assert _stripper(False)("<|channel>thought\n391", wait_for_end=False) is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("<|channel>thought\n391", "391"),
        ("<|channel>thought\n…<channel|>391", "391"),
        ("Answer: <|channel>thought\n391", "Answer: 391"),
        ("<|channel>391", "391"),
        ("391", "391"),
    ],
)
def test_patched_non_streaming_returns_answer_text(text: str, expected: str) -> None:
    assert _stripper(True)(text, wait_for_end=False) == expected


def test_streaming_branch_is_unchanged() -> None:
    strip = _stripper(True)
    assert strip("<|channel>thought\n391", wait_for_end=True) is None
    assert strip("x<|channel>thought\n391", wait_for_end=True) == "x"


def test_patch_refuses_a_changed_parser() -> None:
    with pytest.raises(ValueError, match="review the patch"):
        PATCHER["patch_text"]("def other():\n    pass\n")


def test_dockerfile_applies_the_patch_to_the_installed_parser() -> None:
    dockerfile = (ADAPTER / "Dockerfile").read_text()
    assert "python3 /tmp/patch-gemma4-unclosed-channel.py" in dockerfile
    assert "gemma4_reasoning_parser.py" in dockerfile
