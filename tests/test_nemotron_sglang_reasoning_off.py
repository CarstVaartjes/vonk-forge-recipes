from __future__ import annotations

import ast
import hashlib
import runpy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from vonk_forge_contracts import RecipeDefinition

ROOT = Path(__file__).resolve().parents[1]
# Exact source from SGLang commit 6a4216edb6d25f669ea81de146f86c01eab31128,
# extracted from the pinned ARM64 image's source layer:
# sha256:24ecc38e842ad2cdee50f8704f4e01afabcc825da878ed36d9b476edf8aefc39.
FIXTURE = ROOT / "tests/fixtures/sglang-nemotron-reasoning-parser.txt"
IMAGE_DIGESTS = {
    "6fb82d4fa57425bc0df5b35fba328b08bda950ecf78ce60ea63d8830fc271eb5",
    "7b7ede416bebd9ae0ba8bd9009614ba3cc78338ec01941a4f624876ab9558007",
}
CONTEXTS = (
    ROOT / "adapters/nvidia/r0b0tlab-super120-sglang-single",
    ROOT / "adapters/nvidia/nemotron-3-nano-omni-30b-a3b-bf16-nvidia-sglang-single",
)


def load_parser(path: Path) -> Any:
    tree = ast.parse(path.read_text())
    # Only the external request and unused Harmony imports need GPU-free stubs.
    tree.body = [
        node
        for node in tree.body
        if not (
            isinstance(node, ast.ImportFrom)
            and (node.module or "").startswith("sglang.")
        )
    ]
    namespace: dict[str, Any] = {
        "ChatCompletionRequest": SimpleNamespace,
        "HarmonyParser": object,
    }
    exec(compile(tree, str(path), "exec"), namespace)  # noqa: S102
    return namespace["ReasoningParser"]


@pytest.mark.parametrize("context", CONTEXTS, ids=lambda path: path.name)
def test_explicit_off_overrides_template_heuristic(
    context: Path, tmp_path: Path
) -> None:
    parser_path = tmp_path / "reasoning_parser.py"
    parser_path.write_bytes(FIXTURE.read_bytes())
    patch = runpy.run_path(str(context / "patch-reasoning-off.py"))
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == patch["SOURCE_SHA256"]
    request = SimpleNamespace(
        chat_template_kwargs={"enable_thinking": False}, continue_final_message=False
    )
    # Both serving_chat response paths OR the static template heuristic with
    # the request flag, so the old parser receives True even for explicit OFF.
    old_parser = load_parser(parser_path)
    assert old_parser(
        "nemotron_3", force_reasoning=True, request=request
    ).parse_non_stream("391") == ("391", "")

    patch["patch_parser"](parser_path)
    parser = load_parser(parser_path)
    assert parser("nemotron_3", force_reasoning=True, request=request).parse_non_stream(
        "391"
    ) == ("", "391")
    streaming = parser("nemotron_3", force_reasoning=True, request=request)
    chunks = [streaming.parse_stream_chunk(chunk) for chunk in ("3", "9", "1")]
    assert "".join(reasoning or "" for reasoning, _ in chunks) == ""
    assert "".join(content or "" for _, content in chunks) == "391"

    # ON/default retain genuine reasoning, including a truncated reasoning turn.
    for kwargs in ({}, {"enable_thinking": True}):
        request.chat_template_kwargs = kwargs
        assert parser(
            "nemotron_3", force_reasoning=True, request=request
        ).parse_non_stream("multiply</think>391") == ("multiply", "391")
        assert parser(
            "nemotron_3", force_reasoning=True, request=request
        ).parse_non_stream("still thinking") == ("still thinking", "")
        streaming = parser("nemotron_3", force_reasoning=True, request=request)
        chunks = [
            streaming.parse_stream_chunk(chunk)
            for chunk in ("multiply", "</think>", "391")
        ]
        assert "".join(reasoning or "" for reasoning, _ in chunks) == "multiply"
        assert "".join(content or "" for _, content in chunks) == "391"

    request.chat_template_kwargs = {"enable_thinking": False}
    assert parser("qwen3", force_reasoning=True, request=request).parse_non_stream(
        "391"
    ) == ("391", "")


def test_unknown_source_fails_without_writing(tmp_path: Path) -> None:
    path = tmp_path / "reasoning_parser.py"
    path.write_text("unexpected upstream source")
    for context in CONTEXTS:
        patch = runpy.run_path(str(context / "patch-reasoning-off.py"))
        with pytest.raises(SystemExit, match="unexpected pinned parser bytes"):
            patch["patch_parser"](path)
        assert path.read_text() == "unexpected upstream source"


def test_every_recipe_using_the_affected_image_builds_the_patch() -> None:
    covered: set[Path] = set()
    for path in (ROOT / "recipes").glob("*.json"):
        recipe = RecipeDefinition.model_validate_json(path.read_text())
        build = recipe.execution.build
        if (
            recipe.runtime.engine != "sglang"
            or build.base_image.digest not in IMAGE_DIGESTS
        ):
            continue
        if not any(
            arg.name == "reasoning-parser" and arg.value == "nemotron_3"
            for arg in recipe.runtime.arguments
        ):
            continue
        context = ROOT / build.context.path
        dockerfile = (ROOT / build.dockerfile).read_text()
        assert (
            "COPY patch-reasoning-off.py /opt/vonk/patch-reasoning-off.py" in dockerfile
        )
        assert "RUN python3 /opt/vonk/patch-reasoning-off.py" in dockerfile
        assert (context / "patch-reasoning-off.py").read_bytes() == (
            CONTEXTS[0] / "patch-reasoning-off.py"
        ).read_bytes()
        covered.add(context)
    assert covered == set(CONTEXTS)
