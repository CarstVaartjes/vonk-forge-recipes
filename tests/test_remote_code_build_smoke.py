"""Every planned image checks the exact Python modules its Models select."""

from __future__ import annotations

import ast
import hashlib
import json
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
    files = TOOL["model_files"](
        RecipeDefinition.model_validate_json((ROOT / source).read_bytes()), ROOT
    )
    assert {Path(item[3]).name for item in files if item[3].endswith(".py")} == {
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
    assert "video_preprocessor_config.json" in program


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


def run_smoke(tmp_path: Path, sources: dict[str, bytes]) -> list[str]:
    """Run the emitted program on pinned bytes without installing model engines."""
    files = []
    for name, content in sources.items():
        path = tmp_path / "source" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        files.append(
            (path.as_uri(), hashlib.sha256(content).hexdigest(), len(content), name)
        )
    return execute_smoke(tmp_path, files)


def execute_smoke(tmp_path: Path, files: list[tuple[str, str, int, str]]) -> list[str]:
    module = types.ModuleType("transformers.dynamic_module_utils")
    checked: list[str] = []

    def check_imports(path: str) -> list[str]:
        checked.append(Path(path).name)
        # Match Transformers' contract: check external imports and return local
        # relative module names. No model code is executed by this smoke.
        tree = ast.parse(Path(path).read_bytes())
        relatives = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    __import__(alias.name)
            elif isinstance(node, ast.ImportFrom) and node.level:
                assert node.module is not None
                relatives.append(node.module)
        return relatives

    module.__dict__["check_imports"] = check_imports
    script = tmp_path / "remote-code-smoke.py"
    script.write_text(TOOL["CHECK"].format(files=files))
    with patch.dict(sys.modules, {"transformers.dynamic_module_utils": module}):
        runpy.run_path(str(script))
    return checked


@pytest.mark.parametrize(
    "parser", ["super_v3_reasoning_parser.py", "nano_v3_reasoning_parser.py"]
)
def test_nemotron_auto_map_excludes_engine_parser(tmp_path: Path, parser: str) -> None:
    checked = run_smoke(
        tmp_path,
        {
            "config.json": json.dumps(
                {
                    "auto_map": {
                        "AutoConfig": "configuration_nemotron_h.NemotronHConfig",
                        "AutoModelForCausalLM": "modeling_nemotron_h.NemotronHForCausalLM",
                    }
                }
            ).encode(),
            "configuration_nemotron_h.py": b"import json\n",
            "modeling_nemotron_h.py": b"from .configuration_nemotron_h import NemotronHConfig\n",
            parser: b"import absent_vllm_engine_dependency\n",
        },
    )
    assert set(checked) == {"configuration_nemotron_h.py", "modeling_nemotron_h.py"}
    assert len(checked) == 2


def test_moss_processor_and_video_auto_maps_are_checked(
    tmp_path: Path,
) -> None:
    sources = {
        "config.json": b'{"auto_map":{"AutoModel":"modeling_moss_vl.MossVLModel"}}',
        "preprocessor_config.json": b'{"auto_map":{"AutoProcessor":"processing_moss_vl.MossVLProcessor"}}',
        "video_preprocessor_config.json": b'{"auto_map":{"AutoVideoProcessor":"video_processing_moss_vl.MossVLVideoProcessor"}}',
        "modeling_moss_vl.py": b"from .configuration_moss_vl import MossVLConfig\n",
        "configuration_moss_vl.py": b"import json\n",
        "processing_moss_vl.py": b"import json\n",
        "video_processing_moss_vl.py": b"from .processing_moss_vl import MossVLProcessor\n",
        "unused_plugin.py": b"import absent_engine_dependency\n",
    }
    assert set(run_smoke(tmp_path, sources)) == {
        "modeling_moss_vl.py",
        "configuration_moss_vl.py",
        "processing_moss_vl.py",
        "video_processing_moss_vl.py",
    }
    sources["video_processing_moss_vl.py"] = b"import absent_torchcodec_dependency\n"
    with pytest.raises(ImportError, match="absent_torchcodec_dependency"):
        run_smoke(tmp_path, sources)


def test_all_config_roots_tokenizer_pairs_and_model_isolation(tmp_path: Path) -> None:
    sources = {
        "one/config.json": b'{"auto_map":{"AutoConfig":"configuration.Config"}}',
        "one/tokenizer_config.json": b'{"auto_map":{"AutoTokenizer":[null,"tokenization.Fast"]}}',
        "one/processor_config.json": b'{"auto_map":{"AutoProcessor":"processing.Processor"}}',
        "one/preprocessor_config.json": b'{"auto_map":{"AutoImageProcessor":"image_processing.ImageProcessor","NativeProcessor":"BuiltinProcessor"}}',
        "one/configuration.py": b"import json\n",
        "one/tokenization.py": b"import json\n",
        "one/processing.py": b"import json\n",
        "one/image_processing.py": b"import json\n",
        "two/config.json": b"{}",
        "two/configuration.py": b"import absent_engine_dependency\n",
    }
    assert set(run_smoke(tmp_path, sources)) == {
        "configuration.py",
        "tokenization.py",
        "processing.py",
        "image_processing.py",
    }


@pytest.mark.parametrize("config", [False, True])
@pytest.mark.parametrize("corrupt", [False, True])
def test_smoke_fails_on_missing_dependency_or_changed_source(
    tmp_path: Path, corrupt: bool, config: bool
) -> None:
    sources = {
        "config.json": b'{"auto_map":{"AutoModel":"modeling_example.Model"}}',
        "modeling_example.py": b"import absent_remote_dependency\n",
    }
    files = []
    for name, content in sources.items():
        source = tmp_path / name
        changed = corrupt and (name.endswith(".json") == config)
        source.write_bytes(content + (b"# changed" if changed else b""))
        files.append(
            (source.as_uri(), hashlib.sha256(content).hexdigest(), len(content), name)
        )
    with pytest.raises(RuntimeError if corrupt else ImportError):
        execute_smoke(tmp_path, files)


def test_smoke_does_not_fetch_unselected_auto_map_module(tmp_path: Path) -> None:
    assert (
        run_smoke(
            tmp_path, {"config.json": b'{"auto_map":{"AutoModel":"unselected.Model"}}'}
        )
        == []
    )


def test_smoke_requires_selected_relative_import(tmp_path: Path) -> None:
    with pytest.raises(KeyError):
        run_smoke(
            tmp_path,
            {
                "config.json": b'{"auto_map":{"AutoModel":"modeling.Model"}}',
                "modeling.py": b"from .unselected import Dependency\n",
            },
        )
