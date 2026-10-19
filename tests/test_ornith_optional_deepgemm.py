"""The runtime-only Ornith image must not initialize unused CUDA-toolkit JIT."""

from pathlib import Path

from vonk_forge_contracts import RecipeDefinition

ROOT = Path(__file__).resolve().parents[1]


def test_triton_marlin_ornith_modes_do_not_import_deepgemm() -> None:
    recipe = RecipeDefinition.model_validate_json(
        (
            ROOT / "recipes/ornith-1-5-35b-a3b-nvfp4-w4a16-r0b0tlab-sglang-single.json"
        ).read_text()
    )
    # Both signed launch choices use Triton attention and Marlin MoE; neither
    # needs DeepGEMM. Upstream's runtime-only image has no CUDA compiler tree.
    for choice in recipe.options[0].choices:
        runtime = recipe.with_option_choices({"speculation": choice.value}).runtime
        args = {item.name: item.value for item in runtime.arguments}
        assert args["attention-backend"] == "triton"
        assert args["moe-runner-backend"] == "marlin"
        env = {item.name: item.value for item in runtime.environment}
        assert env.get("SGLANG_ENABLE_JIT_DEEPGEMM") == "0"
