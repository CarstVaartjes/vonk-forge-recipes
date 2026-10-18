"""The b12x proxy takes a device name, while NCCL keeps exact device:port rails."""

import ast
import importlib.util
import os
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "adapters/glm/vcruz-full-tensorfold-four"


def load(name):
    spec = importlib.util.spec_from_file_location(name, ADAPTER / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("selector", ["=rocep1s0f1:1", "=rocep1s0f1:1,roceP2p1s0f1:1"])
def test_controller_primary_device_is_translated_without_changing_nccl(
    selector, monkeypatch
):
    monkeypatch.setenv("NCCL_IB_HCA", selector)
    assert load("fabric_binding").primary_hca(selector) == "rocep1s0f1"
    assert os.environ["NCCL_IB_HCA"] == selector


@pytest.mark.parametrize(
    "selector", ["", "=device:2", "=device:1,", "=device*:1", "device", "=device:0"]
)
def test_unusable_primary_fabric_is_not_silently_substituted(selector):
    with pytest.raises(ValueError):
        load("fabric_binding").primary_hca(selector)


def test_pinned_kernel_consumes_b12x_device_and_leaves_nccl_selector(
    tmp_path, monkeypatch
):
    relative = "src/tensorfold/families/glm_moe_dsa/cuda/roce.py"
    with tarfile.open(ADAPTER / "vendor/tensorfold.tar.gz") as archive:
        member = next(
            m for m in archive.getmembers() if m.name.endswith("/" + relative)
        )
        stream = archive.extractfile(member)
        assert stream
        original = stream.read().decode()
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_text(original)
    selector = "=rocep1s0f1:1,roceP2p1s0f1:1"
    monkeypatch.setenv("NCCL_IB_HCA", selector)
    monkeypatch.setenv("B12X_ROCE_HCA", "rocep1s0f1")

    def selected_device(source):
        statement = next(
            n
            for n in ast.walk(ast.parse(source))
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "hca" for t in n.targets)
        )
        scope = {"os": os}
        exec(  # noqa: S102 - execute one pinned binding assignment without GPU imports.
            compile(
                ast.Module(body=[statement], type_ignores=[]), "fabric-site", "exec"
            ),
            scope,
        )
        return scope["hca"]

    assert (
        selected_device(original) == selector
    )  # old kernel receives an unusable b12x device name
    load("patch_fabric").patch(tmp_path)
    assert selected_device(path.read_text()) == "rocep1s0f1"
    assert os.environ["NCCL_IB_HCA"] == selector
    with pytest.raises(ValueError, match="binding site changed"):
        load("patch_fabric").patch(tmp_path)
