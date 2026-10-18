"""CPU-only regression checks for the image's CUDA build assertions."""

from __future__ import annotations

import runpy
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "adapters/three-d/step1x-3d"
GUARD = ADAPTER / "check_cuda_extensions.py"


def run_guard(
    *, with_cuda: bool, architecture: str, bad_extension: str = "pytorch3d._C"
) -> mock.Mock:
    pytorch3d = ModuleType("pytorch3d")
    native = ModuleType("pytorch3d._C")
    native.__file__ = "/site-packages/pytorch3d/_C.so"
    if with_cuda:
        native.__dict__["knn_check_version"] = lambda: None
    pytorch3d.__dict__["_C"] = native
    modules = {
        "torch": ModuleType("torch"),
        "pytorch3d": pytorch3d,
        native.__name__: native,
    }
    for name in ("custom_rasterizer_kernel", "_nvdiffrast_c"):
        module = ModuleType(name)
        module.__file__ = f"/site-packages/{name}.so"
        modules[name] = module

    def inspect_extension(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        target = modules[bad_extension].__file__
        arch = architecture if command[-1] == target else "sm_121"
        return subprocess.CompletedProcess(
            command, 0, stdout=f"ELF file: kernel.{arch}.cubin", stderr=""
        )

    with (
        mock.patch.dict(sys.modules, modules),
        mock.patch("subprocess.run", side_effect=inspect_extension) as run,
    ):
        runpy.run_path(str(GUARD), run_name="__main__")
    return run


def test_cpu_only_pytorch3d_is_rejected_before_cuda_inspection() -> None:
    with pytest.raises(SystemExit, match="pytorch3d was built without CUDA support"):
        run_guard(with_cuda=False, architecture="sm_121")


@pytest.mark.parametrize(
    "extension", ["pytorch3d._C", "custom_rasterizer_kernel", "_nvdiffrast_c"]
)
def test_cuda_extension_without_spark_kernels_is_rejected(extension: str) -> None:
    with pytest.raises(SystemExit, match=f"{extension} has no sm_121 CUDA kernels"):
        run_guard(with_cuda=True, architecture="sm_90", bad_extension=extension)


def test_spark_extensions_are_checked_without_initializing_a_gpu() -> None:
    run = run_guard(with_cuda=True, architecture="sm_121")
    assert [call.args[0] for call in run.call_args_list] == [
        ["cuobjdump", "--list-elf", f"/site-packages/{path}.so"]
        for path in ("pytorch3d/_C", "custom_rasterizer_kernel", "_nvdiffrast_c")
    ]
    assert all(call.kwargs["check"] for call in run.call_args_list)


def test_step1x_forces_cuda_and_runs_guard_after_compiling_extensions() -> None:
    dockerfile = (ADAPTER / "Dockerfile").read_text()
    assert "FORCE_CUDA=1" in dockerfile
    assert "TORCH_CUDA_ARCH_LIST=12.1" in dockerfile
    assert (
        "COPY check_cuda_extensions.py /opt/vonk/source/check_cuda_extensions.py"
        in dockerfile
    )
    assert dockerfile.index(
        "/opt/step1x/step1x3d_texture/differentiable_renderer"
    ) < dockerfile.index("&& python3 /opt/vonk/source/check_cuda_extensions.py")


@pytest.mark.parametrize("adapter", ["triposg", "skintokens"])
def test_other_simple_three_d_extension_builds_check_spark_kernels(
    adapter: str,
) -> None:
    dockerfile = (ROOT / "adapters/three-d" / adapter / "Dockerfile").read_text()
    assert "cuobjdump --list-elf" in dockerfile
    assert (
        "grep --quiet sm_121" in dockerfile
        if adapter == "triposg"
        else "sm_12[01]" in dockerfile
    )


def test_cuda_inspector_failure_fails_the_image_build() -> None:
    with (
        mock.patch(
            "subprocess.run", side_effect=subprocess.CalledProcessError(1, "cuobjdump")
        ),
        pytest.raises(subprocess.CalledProcessError),
    ):
        # The guard's subprocess mock calls this failing inspector.
        import importlib.util

        spec = importlib.util.spec_from_file_location("cuda_guard", GUARD)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        native = ModuleType("pytorch3d._C")
        native.__file__ = "/site-packages/pytorch3d/_C.so"
        native.__dict__["knn_check_version"] = lambda: None
        pytorch3d = ModuleType("pytorch3d")
        pytorch3d.__dict__["_C"] = native
        with mock.patch.dict(
            sys.modules,
            {
                "torch": ModuleType("torch"),
                "pytorch3d": pytorch3d,
                "pytorch3d._C": native,
            },
        ):
            module.__dict__["main"]()
