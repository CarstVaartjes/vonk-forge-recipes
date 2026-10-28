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


def test_runtime_only_image_drops_toolkit_extensions_and_imports_without_them() -> None:
    # The base has no CUDA toolkit: deep_ep asserts in find_cuda_home on import
    # (outside SGLang's ImportError guard) and deep_gemm asserts the same way.
    # Removing both lets SGLang's own guards apply; the build then imports the
    # dispatcher and LoRA modules with no GPU and no CUDA_HOME, so a regression
    # fails the image build instead of the hardware start.
    adapter = ROOT / "adapters/nvidia/r0b0tlab-ornith-sglang-single"
    dockerfile = (adapter / "Dockerfile").read_text()
    for removed in ("deep_ep", "deep_gemm", "sgl_deep_ep-*", "sgl_deep_gemm-*"):
        assert f"site-packages/{removed}" in dockerfile
    assert "env -u CUDA_HOME -u CUDA_PATH /opt/venv/bin/python -c" in dockerfile
    assert "sglang.srt.layers.moe.token_dispatcher.deepep" in dockerfile
    assert "sglang.srt.lora.lora_manager" in dockerfile
    assert not (adapter / "patch-deepgemm.py").exists()
    assert "CUDA_HOME=" not in dockerfile
