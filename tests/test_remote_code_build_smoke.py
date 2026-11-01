"""Every planned image checks the exact Python modules its Models select."""

from __future__ import annotations

import ast
import hashlib
import io
import json
import runpy
import shutil
import sys
import types
import urllib.error
from email.message import Message
from pathlib import Path
from typing import Self
from unittest.mock import patch

import pytest
from vonk_forge_contracts.recipe import RecipeDefinition

ROOT = Path(__file__).resolve().parents[1]
TOOL = runpy.run_path(str(ROOT / "tools/prepare-remote-code-smoke"))
MOSS = ROOT / "adapters/video/moss-vl-realtime"
KEY = "a" * 64


def emitted_model_code(program: str) -> list[str]:
    for node in ast.parse(program).body:
        if (
            isinstance(node, ast.Assign)
            and ast.unparse(node.targets[0]) == "MODEL_CODE"
        ):
            return ast.literal_eval(node.value)
    raise AssertionError("MODEL_CODE missing")


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
    # pytorch-pipeline loads the model through Transformers' AutoModel.
    assert emitted_model_code(program) == ["moss-vl-realtime-11b-bf16-d1f71a58"]
    assert "video_processing_moss_vl.py" in program
    assert "video_preprocessor_config.json" in program


def pinned(
    tmp_path: Path, sources: dict[str, bytes]
) -> list[tuple[str, str, int, str]]:
    files = []
    for name, content in sources.items():
        path = tmp_path / "source" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        files.append(
            (path.as_uri(), hashlib.sha256(content).hexdigest(), len(content), name)
        )
    return files


# A served Model without remote code: only its config.json is pinned.
SERVED = {"served/config.json": b'{"model_type": "laguna"}'}


def installed_vllm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, get_config: str
) -> Path:
    """A vLLM whose get_config body is ``get_config``, importable by subprocesses."""
    package = tmp_path / "site" / "vllm" / "transformers_utils"
    package.mkdir(parents=True)
    for directory in (package.parent, package):
        (directory / "__init__.py").write_text("")
    record = tmp_path / "parsed.txt"
    (package / "config.py").write_text(
        "import pathlib\n"
        "def get_config(path, trust_remote_code):\n"
        f"    record = pathlib.Path({str(record)!r})\n"
        "    assert (pathlib.Path(path) / 'config.json').is_file()\n"
        f"{get_config}"
    )
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "site"))
    monkeypatch.setenv("PATH", str(tmp_path / "empty-path"))
    return record


ACCEPTS = (
    "    record.write_text(pathlib.Path(path).name + ' ' + str(trust_remote_code))\n"
)


def test_vllm_parses_the_served_config_of_a_model_without_remote_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = installed_vllm(tmp_path, monkeypatch, ACCEPTS)
    assert (
        execute_smoke(tmp_path, pinned(tmp_path, SERVED), [], [("served", True)]) == []
    )
    assert record.read_text() == "served True"


def test_a_config_the_installed_vllm_rejects_fails_the_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # vLLM 0.30.0's LagunaConfig under Transformers 5.17 (sweep 2026-10-27).
    installed_vllm(
        tmp_path,
        monkeypatch,
        "    raise AttributeError(\"'str' object has no attribute 'get'\")\n",
    )
    with pytest.raises(SystemExit, match="vllm rejected the served config of served"):
        execute_smoke(tmp_path, pinned(tmp_path, SERVED), [], [("served", False)])


def test_the_adapter_engine_interpreter_parses_before_python3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = installed_vllm(tmp_path, monkeypatch, ACCEPTS)
    used = tmp_path / "used.txt"
    engine = tmp_path / "engine-python"
    engine.write_text(f'#!/bin/sh\necho engine >> {used}\nexec {sys.executable} "$@"\n')
    engine.chmod(0o755)
    # An interpreter that cannot run vLLM (any status but 4) yields to the next.
    without = tmp_path / "python-without-vllm"
    without.write_text(f"#!/bin/sh\necho without >> {used}\nexit 1\n")
    without.chmod(0o755)
    files = pinned(tmp_path, SERVED)
    execute_smoke(tmp_path, files, [], [("served", False)], [str(without), str(engine)])
    assert used.read_text().split() == ["without", "engine"]
    assert record.read_text() == "served False"


def script(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(0o755)
    return path


def test_no_interpreter_with_vllm_warns_on_the_image_instead_of_failing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("PATH", str(tmp_path / "empty-path"))
    monkeypatch.delenv("PYTHONPATH", raising=False)
    broken = script(tmp_path / "broken-python", "exit 1\n")
    execute_smoke(
        tmp_path, pinned(tmp_path, SERVED), [], [("served", False)], [str(broken)]
    )
    out = capsys.readouterr().out
    assert (
        f"::warning::image {KEY}: no interpreter ran vllm get_config for served" in out
    )
    # Each interpreter tried is named with its status; none counts as a rejection.
    assert f"{broken}: exit 1" in out
    assert f"{sys.executable}: exit 1" in out


def test_a_missing_or_unrunnable_interpreter_is_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = installed_vllm(tmp_path, monkeypatch, ACCEPTS)
    unrunnable = tmp_path / "not-executable-python"
    unrunnable.write_text("#!/bin/sh\nexit 4\n")
    pythons = [str(tmp_path / "absent" / "python"), str(unrunnable)]
    execute_smoke(tmp_path, pinned(tmp_path, SERVED), [], [("served", False)], pythons)
    assert record.read_text() == "served False"


def test_the_vllm_scripts_env_shebang_interpreter_is_tried_before_python3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = installed_vllm(tmp_path, monkeypatch, ACCEPTS)
    used = tmp_path / "used.txt"
    bin_dir = tmp_path / "bin"
    script(
        bin_dir / "enginepy", f'echo enginepy >> {used}\nexec {sys.executable} "$@"\n'
    )
    (bin_dir / "vllm").write_text("#!/usr/bin/env enginepy\n")
    (bin_dir / "vllm").chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir))
    execute_smoke(tmp_path, pinned(tmp_path, SERVED), [], [("served", False)])
    assert used.read_text().split() == ["enginepy"]
    assert record.read_text() == "served False"


# Resolved before any test empties PATH.
SLEEP = shutil.which("sleep")


def hung_python(tmp_path: Path) -> Path:
    assert SLEEP is not None
    return script(tmp_path / "hung-python", f"{SLEEP} 30\n")


def test_a_hung_interpreter_yields_to_the_next(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = installed_vllm(tmp_path, monkeypatch, ACCEPTS)
    hung = hung_python(tmp_path)
    files = pinned(tmp_path, SERVED)
    execute_smoke(
        tmp_path, files, [], [("served", False)], [str(hung)], parse_timeout_seconds=1
    )
    assert record.read_text() == "served False"


def test_a_hung_interpreter_and_no_other_warns_instead_of_failing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("PATH", str(tmp_path / "empty-path"))
    monkeypatch.delenv("PYTHONPATH", raising=False)
    hung = hung_python(tmp_path)
    files = pinned(tmp_path, SERVED)
    execute_smoke(
        tmp_path, files, [], [("served", False)], [str(hung)], parse_timeout_seconds=1
    )
    out = capsys.readouterr().out
    assert f"::warning::image {KEY}: no interpreter ran vllm get_config" in out
    assert f"{hung}: did not finish within 1 s" in out


def test_a_get_config_that_hangs_fails_with_its_stack(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    installed_vllm(tmp_path, monkeypatch, "    import time\n    time.sleep(30)\n")
    files = pinned(tmp_path, SERVED)
    with pytest.raises(SystemExit, match=r"did not finish within 1 s \(its stack"):
        execute_smoke(tmp_path, files, [], [("served", False)], parse_timeout_seconds=1)
    assert "in get_config" in capfd.readouterr().err


class Response(io.BytesIO):
    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def test_transient_download_errors_are_retried_and_client_errors_are_not(
    tmp_path: Path,
) -> None:
    content = SERVED["served/config.json"]
    url = "https://huggingface.co/x/resolve/r/config.json"
    files = [
        (url, hashlib.sha256(content).hexdigest(), len(content), "served/config.json")
    ]
    outcomes: list[object] = [
        urllib.error.HTTPError(url, 503, "unavailable", Message(), None),
        urllib.error.HTTPError(url, 429, "slow down", Message(), None),
        urllib.error.URLError("connection reset"),
        TimeoutError("read timed out"),
    ]
    calls: list[str] = []

    def urlopen(requested: str, timeout: int) -> Response:
        calls.append(requested)
        if outcomes:
            outcome = outcomes.pop(0)
            assert isinstance(outcome, BaseException)
            raise outcome
        return Response(content)

    with patch("urllib.request.urlopen", urlopen), patch("time.sleep") as sleep:
        assert execute_smoke(tmp_path, files, []) == []
    assert len(calls) == 5 and sleep.call_count == 4

    def missing(requested: str, timeout: int) -> Response:
        calls.append(requested)
        raise urllib.error.HTTPError(url, 404, "not found", Message(), None)

    calls.clear()
    with (
        patch("urllib.request.urlopen", missing),
        patch("time.sleep"),
        pytest.raises(urllib.error.HTTPError),
    ):
        execute_smoke(tmp_path, files, [])
    assert len(calls) == 1


def emitted(program: str, name: str) -> object:
    for node in ast.parse(program).body:
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == name:
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} missing")


@pytest.mark.parametrize(
    ("source", "configs", "pythons"),
    [
        # No remote code: main generated no smoke for it at all.
        (
            "recipes/gemma-4-26b-a4b-vllm-single.json",
            [("gemma-4-26b-a4b-it-4d7ae498", False)],
            [],
        ),
        # vLLM 0.30.0 rejected this config at start on hardware (2026-10-27).
        (
            "recipes/laguna-s-2-1-nvfp4-vllm-single.json",
            [("laguna-s-2-1-nvfp4-07614121", False)],
            [],
        ),
        # The served target, never its drafter; the recipe's trust flag.
        (
            "recipes/laguna-s-2-1-nvfp4-r0b0tlab-vllm-single.json",
            [("laguna-s-2-1-nvfp4-07614121", True)],
            [],
        ),
        # The Controller rewrites argv /models to the endpoint role's first
        # mount (/models/target), never to the drafter.
        (
            "recipes/nemotron-3-5-bolt-lilicorr-nvidia-vllm-single.json",
            [("nemotron-3-5-bolt-24gb-c6cc2e8f", True)],
            [],
        ),
        # A bare launcher serves /models; its adapter names the engine Python.
        (
            "recipes/mimo-v2-6-pro-mopd-arvq-drowzeys-vllm-four.json",
            [("mimo-v2-6-pro-mopd-arvq-hybrid21-bb83094e", True)],
            ["/opt/arvq/.venv/bin/python"],
        ),
    ],
)
def test_every_vllm_recipe_plans_its_served_config_parse(
    tmp_path: Path,
    source: str,
    configs: list[tuple[str, bool]],
    pythons: list[str],
) -> None:
    plan = TOOL["Plan"].model_validate(
        {"images": [{"build_key": KEY, "recipes": [{"source_path": source}]}]}
    )
    TOOL["prepare"](plan, tmp_path)
    program = (tmp_path / KEY / "remote-code-smoke.py").read_text()
    assert emitted(program, "VLLM_CONFIGS") == configs
    assert emitted(program, "ENGINE_PYTHONS") == pythons
    files = emitted(program, "FILES")
    assert isinstance(files, list)
    assert f"{configs[0][0]}/config.json" in [name for *_, name in files]


def test_a_trust_remote_code_choice_also_parses_with_remote_code() -> None:
    source = "recipes/deepseek-v4-1-flash-exl3-mia-dual.json"
    assert TOOL["trust_remote_code"](recipe(source)) == (False,)
    changed = with_argument(source, "trust-remote-code", True, True)
    assert TOOL["trust_remote_code"](changed) == (False, True)


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


def run_smoke(
    tmp_path: Path, sources: dict[str, bytes], model_code: list[str]
) -> list[str]:
    """Run the emitted program on pinned bytes without installing model engines."""
    files = []
    for name, content in sources.items():
        path = tmp_path / "source" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        files.append(
            (path.as_uri(), hashlib.sha256(content).hexdigest(), len(content), name)
        )
    return execute_smoke(tmp_path, files, model_code)


def execute_smoke(
    tmp_path: Path,
    files: list[tuple[str, str, int, str]],
    model_code: list[str],
    vllm_configs: list[tuple[str, bool]] | None = None,
    engine_pythons: list[str] | None = None,
    parse_timeout_seconds: int | None = None,
) -> list[str]:
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
    script.write_text(
        TOOL["CHECK"].format(
            files=files,
            model_code=model_code,
            vllm_configs=vllm_configs or [],
            engine_pythons=engine_pythons or [],
            image=KEY,
        )
    )
    if parse_timeout_seconds is not None:
        program = script.read_text()
        assert "PARSE_TIMEOUT_SECONDS = 600" in program
        script.write_text(
            program.replace(
                "PARSE_TIMEOUT_SECONDS = 600",
                f"PARSE_TIMEOUT_SECONDS = {parse_timeout_seconds}",
            )
        )
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
        ["."],
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
    assert set(run_smoke(tmp_path, sources, ["."])) == {
        "modeling_moss_vl.py",
        "configuration_moss_vl.py",
        "processing_moss_vl.py",
        "video_processing_moss_vl.py",
    }
    sources["video_processing_moss_vl.py"] = b"import absent_torchcodec_dependency\n"
    with pytest.raises(ImportError, match="absent_torchcodec_dependency"):
        run_smoke(tmp_path, sources, ["."])


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
    assert set(run_smoke(tmp_path, sources, [])) == {
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
        execute_smoke(tmp_path, files, ["."])


def test_smoke_does_not_fetch_unselected_auto_map_module(tmp_path: Path) -> None:
    assert (
        run_smoke(
            tmp_path,
            {"config.json": b'{"auto_map":{"AutoModel":"unselected.Model"}}'},
            ["."],
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
            ["."],
        )


# DSpark drafters (RadixArk/Inkling-Small-DSpark-Preview, Qwen3.8-27B-DSpark)
# map only AutoModel to a module that needs specforge. vLLM and SGLang serve
# the registered architecture natively and never import it.
DSPARK = {
    "config.json": b'{"architectures":["DSparkDraftModel"],"auto_map":{"AutoModel":"dspark.DSparkDraftModel"}}',
    "dspark.py": b"import absent_specforge_dependency\n",
}


def test_native_engine_skips_auto_model_only_auto_map(tmp_path: Path) -> None:
    assert run_smoke(tmp_path, DSPARK, []) == []
    with pytest.raises(ImportError, match="absent_specforge_dependency"):
        run_smoke(tmp_path / "transformers", DSPARK, ["."])


def test_native_engine_still_refuses_unpinned_external_auto_model(
    tmp_path: Path,
) -> None:
    config = b'{"auto_map":{"AutoModel":"other/repo--dspark.DSparkDraftModel"}}'
    with pytest.raises(ValueError, match="unpinned external auto_map"):
        run_smoke(tmp_path, {"config.json": config}, [])


def test_native_engine_still_verifies_config_bytes(tmp_path: Path) -> None:
    files = []
    for name, content in DSPARK.items():
        source = tmp_path / name
        source.write_bytes(content + b" ")
        files.append(
            (source.as_uri(), hashlib.sha256(content).hexdigest(), len(content), name)
        )
    with pytest.raises(RuntimeError, match="content mismatch"):
        execute_smoke(tmp_path, files, [])


def test_native_engine_still_checks_auto_config(tmp_path: Path) -> None:
    sources = {
        "config.json": b'{"auto_map":{"AutoConfig":"configuration.Config","AutoModelForCausalLM":"modeling.Model"}}',
        "configuration.py": b"import json\n",
        "modeling.py": b"import absent_engine_unused_dependency\n",
    }
    assert run_smoke(tmp_path, sources, []) == ["configuration.py"]
    sources["configuration.py"] = b"import absent_config_dependency\n"
    with pytest.raises(ImportError, match="absent_config_dependency"):
        run_smoke(tmp_path / "missing", sources, [])


def recipe(source: str) -> RecipeDefinition:
    return RecipeDefinition.model_validate_json((ROOT / source).read_bytes())


@pytest.mark.parametrize(
    "source",
    [
        "recipes/inkling-small-nvfp4-kv-1m-drowzeys-sglang-dual.json",
        "recipes/inkling-small-nvfp4-sglang-mia-dual.json",
        "recipes/inkling-small-nvfp4-dspark-tonyd2wild-vllm-dual.json",
        "recipes/qwen3-8-27b-nvfp4-dspark-sglang-single.json",
    ],
)
def test_dspark_recipes_do_not_import_drafter_model_code(
    tmp_path: Path, source: str
) -> None:
    plan = TOOL["Plan"].model_validate(
        {"images": [{"build_key": KEY, "recipes": [{"source_path": source}]}]}
    )
    TOOL["prepare"](plan, tmp_path)
    assert (
        emitted_model_code((tmp_path / KEY / "remote-code-smoke.py").read_text()) == []
    )


def with_argument(
    source: str, name: str, value: object, choice: bool
) -> RecipeDefinition:
    document = recipe(source).model_dump(mode="json")
    argument = {"name": name, "value": value}
    if choice:
        document["options"][0]["choices"][-1]["args"].append(argument)
    else:
        document["runtime"]["arguments"].append(argument)
    return RecipeDefinition.model_validate(document)


VLLM_OPTIONS = "recipes/step-3-7-flash-nvfp4-vllm-mia-dual.json"
SGLANG_OPTIONS = "recipes/qwen3-8-27b-nvfp4-mtp-r0b0tlab-sglang-single.json"


@pytest.mark.parametrize("choice", [False, True])
@pytest.mark.parametrize(
    ("source", "name", "value", "imports"),
    [
        (VLLM_OPTIONS, "model-impl", "auto", False),
        (VLLM_OPTIONS, "model-impl", "vllm", False),
        (VLLM_OPTIONS, "model-impl", "transformers", True),
        # vLLM serves ModelOpt checkpoints natively (this recipe already does).
        (VLLM_OPTIONS, "quantization", "modelopt_fp8", False),
        (SGLANG_OPTIONS, "model-impl", "transformers", True),
        (SGLANG_OPTIONS, "quantization", "modelopt_fp4", False),
        (SGLANG_OPTIONS, "quantization", "modelopt_fp8", True),
        (SGLANG_OPTIONS, "modelopt-quant", "fp8", True),
        (SGLANG_OPTIONS, "modelopt-export-path", "/tmp/export", True),
        # The draft's quantization drives its ModelConfig the same way.
        (SGLANG_OPTIONS, "speculative-draft-model-quantization", "modelopt_fp8", True),
        (SGLANG_OPTIONS, "speculative-draft-model-quantization", "unquant", False),
    ],
)
def test_transformers_backend_imports_remote_model(
    source: str, name: str, value: str, imports: bool, choice: bool
) -> None:
    # Base arguments and option-choice arguments both reach the engine.
    assert not TOOL["imports_remote_model"](recipe(source))
    changed = with_argument(source, name, value, choice)
    assert TOOL["imports_remote_model"](changed) is imports


def test_transformers_engines_import_remote_model() -> None:
    assert TOOL["imports_remote_model"](
        recipe("recipes/moss-vl-realtime-11b-pytorch-single.json")
    )


def test_the_native_tensorfold_engine_does_not_import_remote_model() -> None:
    # Its image has no `transformers`; it implements the architectures itself.
    assert not TOOL["imports_remote_model"](
        recipe("recipes/qwen3-8-flash-next-taussoe-tensorfold-single.json")
    )


def test_a_list_valued_quantization_does_not_crash_the_plan() -> None:
    changed = with_argument(SGLANG_OPTIONS, "quantization", ["a", "b"], False)
    assert TOOL["imports_remote_model"](changed) is False
