"""HunyuanOCR's GB10 recipe must not enter the failing graph-capture path."""

from pathlib import Path

from vonk_forge_contracts import RecipeDefinition

ROOT = Path(__file__).resolve().parents[1]


def test_hunyuanocr_selects_eager_execution_before_gpu_initialization() -> None:
    recipe = RecipeDefinition.model_validate_json(
        (ROOT / "recipes/hunyuanocr-1-5-vllm-single.json").read_text()
    )
    # vLLM's enforce_eager controls profile_cudagraph_memory/capture_model.
    # The sweep fails there before health or inference; retain all model,
    # context, concurrency and attention choices while disabling capture.
    args = {argument.name: argument.value for argument in recipe.runtime.arguments}
    assert args.get("enforce-eager") is True
