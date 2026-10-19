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


def test_disabled_deepgemm_never_imports_toolkit_dependent_module(
    tmp_path: Path,
) -> None:
    import subprocess
    import sys

    # Upstream e76d36214b0f4a6d1cd7af6bab6e199c97bcddfa, configurer.py;
    # its import line matches the sweep traceback. DeepGEMM asserts without nvcc.
    source = ROOT / "tests/fixtures/ornith-deepgemm-configurer.py.txt"
    target = tmp_path / "configurer.py"
    target.write_bytes(source.read_bytes())
    patch = ROOT / "adapters/nvidia/r0b0tlab-ornith-sglang-single/patch-deepgemm.py"
    script = """
import runpy, sys, types
for name in ("sglang", "sglang.srt", "sglang.srt.environ", "sglang.srt.utils"):
    sys.modules[name] = types.ModuleType(name)
sys.modules["sglang.srt.environ"].envs = types.SimpleNamespace(
    SGLANG_ENABLE_JIT_DEEPGEMM=types.SimpleNamespace(get=lambda: False))
utils = sys.modules["sglang.srt.utils"]
utils.get_device_sm = lambda: 121
utils.is_cuda = lambda: True
utils.is_musa = lambda: False
utils.is_sm100_supported = lambda: False
result = runpy.run_path(sys.argv[1])
assert result["ENABLE_JIT_DEEPGEMM"] is False
"""
    (tmp_path / "deep_gemm.py").write_text(
        "raise AssertionError('CUDA toolkit unavailable')\n"
    )
    # Reproduce the upstream fault before applying the build-time patch.
    import os

    env = {**os.environ, "PYTHONPATH": str(tmp_path)}
    before = subprocess.run(
        [sys.executable, "-c", script, str(target)],
        env=env,
        capture_output=True,
        check=False,
    )
    assert before.returncode != 0
    assert b"CUDA toolkit unavailable" in before.stderr
    subprocess.run([sys.executable, str(patch), str(target)], check=True)
    subprocess.run([sys.executable, "-c", script, str(target)], env=env, check=True)
