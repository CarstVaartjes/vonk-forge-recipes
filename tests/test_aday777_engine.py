"""Keep the failed CUDA overlay out of the ADay777 recovery baseline."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from vonk_forge_contracts import RecipeDefinition, document_sha256

from qualification.memory_need import argument_value, load_inputs

ROOT = Path(__file__).resolve().parents[1]
SLUG = "qwen3-8-27b-aday777-abliterated-drowzeys-vllm-single"
BACKEND = (
    ROOT
    / "adapters/qwen/drowzeys-aday777-single/recipe/build/a4q2_backend_flashinfer.py"
)


def test_aday777_avoids_failed_cuda_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    recipe = RecipeDefinition.model_validate_json(
        (ROOT / "recipes" / f"{SLUG}.json").read_text()
    )
    assert not recipe.options, "Choices must not re-enable the unqualified paths"
    assert argument_value(recipe, "enforce-eager") is True
    assert argument_value(recipe, "speculative-config") is None
    assert argument_value(recipe, "enable-flashinfer-autotune") is None
    assert argument_value(recipe, "kv-cache-dtype") == "fp8"

    # Execute the vendored backend's actual A4Q routing expression on the CPU.
    # This is a routing regression, not a reproduction of an asynchronous GPU
    # illegal access: the retained run log lost the original kernel frames.
    tree = ast.parse(BACKEND.read_text())
    gate = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_vllm_nvfp4_a4q_enabled"
    )
    namespace: dict[str, Any] = {"__builtins__": __builtins__}
    exec(  # noqa: S102 - execute only the pinned, repository-owned routing gate
        compile(ast.Module(body=[gate], type_ignores=[]), str(BACKEND), "exec"),
        namespace,
    )
    routing = next(
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute) and target.attr == "a4q_prefill"
            for target in node.targets
        )
    )
    monkeypatch.delenv("VLLM_NVFP4_A4Q", raising=False)
    for entry in recipe.runtime.environment:
        monkeypatch.setenv(entry.name, str(entry.value))
    state = SimpleNamespace(
        use_fa2_nvfp4_kv=argument_value(recipe, "kv-cache-dtype") == "nvfp4",
        head_dim=256,
        use_dcp=False,
    )
    namespace["self"] = state
    expression = compile(ast.Expression(body=routing), str(BACKEND), "eval")
    assert eval(expression, namespace) is False
    # Inject the old settings: prove the same engine expression selects the
    # custom QK CUDA kernel with this checkpoint's geometry.
    monkeypatch.setenv("VLLM_NVFP4_A4Q", "1")
    state.use_fa2_nvfp4_kv = True
    assert eval(expression, namespace) is True


def test_proven_drowzeys_siblings_remain_at_hardware_identity() -> None:
    for slug in (
        "qwen3-8-27b-nvfp4-drowzeys-vllm-single",
        "qwen3-8-27b-aeon-abliterated-dflash2-drowzeys-vllm-single",
    ):
        document = json.loads((ROOT / "recipes" / f"{slug}.json").read_text())
        proof = load_inputs().hardware_passes[f"vonk-forge/{slug}"]
        assert document_sha256(document) == proof.content_sha256
