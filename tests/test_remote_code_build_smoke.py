"""Every planned image checks the exact Python modules its Models select."""

from __future__ import annotations

import hashlib
import runpy
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest
from vonk_forge_contracts.recipe import RecipeDefinition

ROOT = Path(__file__).resolve().parents[1]
TOOL = runpy.run_path(str(ROOT / "tools/prepare-remote-code-smoke"))
MOSS = ROOT / "adapters/video/moss-vl-realtime"
KEY = "a" * 64


def test_moss_has_hash_locked_arm64_torchcodec_and_imports_native_decoder() -> None:
    lock = (MOSS / "requirements.lock").read_text()
    assert "torchcodec-0.17.0%2Bcpu-cp312-cp312-manylinux_2_27_aarch64" in lock
    assert (
        "--hash=sha256:187383c26847db470a22a9aa8b15e45611dda594f3fdc1a6a32b670f1d7ddd82"
        in lock
    )
    assert (
        "from torchcodec.decoders import VideoDecoder"
        in (MOSS / "Dockerfile").read_text()
    )


def test_moss_smoke_covers_pinned_model_and_processor_imports(tmp_path: Path) -> None:
    source = "recipes/moss-vl-realtime-11b-pytorch-single.json"
    plan = TOOL["Plan"].model_validate(
        {
            "images": [
                {"build_key": KEY, "recipes": [{"source_path": source}]},
                {
                    "build_key": "b" * 64,
                    "recipes": [
                        {
                            "source_path": "recipes/qwen-image-2512-lightning-diffusers-single.json"
                        }
                    ],
                },
            ]
        }
    )
    TOOL["prepare"](plan, tmp_path)
    assert not (tmp_path / ("b" * 64) / "remote-code-smoke.py").exists()
    program = (tmp_path / KEY / "remote-code-smoke.py").read_text()
    # Obtain the data without performing network I/O or importing transformers.
    files = (
        runpy.run_path
        if False
        else TOOL["model_files"](
            RecipeDefinition.model_validate_json((ROOT / source).read_bytes()), ROOT
        )
    )
    assert {item[3] for item in files} == {
        "configuration_moss_vl.py",
        "modeling_moss_vl.py",
        "processing_moss_vl.py",
        "video_processing_moss_vl.py",
    }
    assert all(
        "/resolve/d1f71a588deaf2b9b9cea7c6fff66cb10847cc86/" in item[0]
        for item in files
    )
    assert "check_imports(str(path))" in program
    assert "video_processing_moss_vl.py" in program


def test_image_workflow_checks_imports_before_platform_layer_and_push() -> None:
    workflow = (ROOT / ".github/workflows/recipe-images.yml").read_text()
    assert "python tools/prepare-remote-code-smoke" in workflow
    assert "if [ -f remote-code-smoke.py ]; then" in workflow
    assert workflow.index("python tools/prepare-remote-code-smoke") < workflow.index(
        "name: Upload the image plan"
    )
    assert workflow.index(
        "localhost/vonk/recipe-build /tmp/remote-code-smoke.py"
    ) < workflow.index("mkdir -p adapter")
    assert workflow.index("mkdir -p adapter") < workflow.index(
        "name: Push to every recipe"
    )


@pytest.mark.parametrize("corrupt", [False, True])
def test_smoke_fails_on_missing_dependency_or_changed_source(
    tmp_path: Path, corrupt: bool
) -> None:
    content = b"import absent_remote_dependency\n"
    source = tmp_path / "modeling_example.py"
    source.write_bytes(content + (b"# changed" if corrupt else b""))
    files = [
        (
            source.as_uri(),
            hashlib.sha256(content).hexdigest(),
            len(content),
            source.name,
        )
    ]
    module = types.ModuleType("transformers.dynamic_module_utils")
    checked: list[bytes] = []

    def check_imports(path: str) -> None:
        checked.append(Path(path).read_bytes())
        __import__("absent_remote_dependency")

    module.__dict__["check_imports"] = check_imports
    program = TOOL["CHECK"].format(files=files)
    script = tmp_path / "remote-code-smoke.py"
    script.write_text(program)
    with (
        patch.dict(sys.modules, {"transformers.dynamic_module_utils": module}),
        pytest.raises(RuntimeError if corrupt else ImportError),
    ):
        runpy.run_path(str(script))
    assert checked == ([] if corrupt else [content])
