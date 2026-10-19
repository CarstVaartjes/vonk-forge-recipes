"""Driverless builds retain strict engine-schema validation before serving."""

import os
import runpy
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = (
    ROOT
    / "adapters/nvidia/llama-3-3-70b-instruct-nvfp4-draft-target-nvidia-trtllm-single"
)


@pytest.fixture
def runtime(monkeypatch: pytest.MonkeyPatch):
    config = {"speculative_config": {"engine_owned": "unchanged"}}
    yaml = types.ModuleType("yaml")
    yaml.__dict__["safe_load"] = lambda _: config
    monkeypatch.setitem(sys.modules, "yaml", yaml)

    def load():
        source = runpy.run_path(str(ADAPTER / "verify-runtime.py"))
        source["main"].__globals__["CONFIG"] = ADAPTER / "llm-api-config.yml"
        monkeypatch.setattr(source["os"], "access", lambda *_: True)
        monkeypatch.setattr(source["shutil"], "which", lambda _: "trtllm-serve")
        return source["main"]

    return config, load


def test_driverless_build_does_not_import_cuda_schema(runtime, monkeypatch):
    engine = types.ModuleType("tensorrt_llm.llmapi.llm_args")

    def unavailable(name):
        raise ImportError("libcuda.so.1 is unavailable on this build host")

    engine.__dict__["__getattr__"] = unavailable
    monkeypatch.setitem(sys.modules, engine.__name__, engine)
    _, load = runtime
    load()()


@pytest.mark.parametrize("invalid", [False, True])
def test_serving_uses_installed_schema_and_propagates_refusal(
    runtime, monkeypatch, invalid
):
    schema = object()
    validated = Mock(side_effect=ValueError("invalid speculation") if invalid else None)
    validator = Mock(return_value=types.SimpleNamespace(validate_python=validated))
    pydantic = types.ModuleType("pydantic")
    pydantic.__dict__["TypeAdapter"] = validator
    engine = types.ModuleType("tensorrt_llm.llmapi.llm_args")
    engine.__dict__["SpeculativeConfig"] = schema
    for module in (pydantic, engine):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    config, load = runtime
    if invalid:
        with pytest.raises(ValueError, match="invalid speculation"):
            load()(engine_schema=True)
    else:
        load()(engine_schema=True)
    validator.assert_called_once_with(schema)
    validated.assert_called_once_with(config["speculative_config"])


@pytest.mark.parametrize("config", [{}, {"speculative_config": []}])
def test_build_rejects_malformed_config(runtime, config):
    fixture, load = runtime
    fixture.clear()
    if isinstance(config, dict):
        fixture.update(config)
    with pytest.raises(ValueError, match="speculative_config mapping"):
        load()()


@pytest.mark.parametrize("validation_status", [0, 19])
def test_launcher_validates_before_exec_and_preserves_arguments(
    tmp_path, validation_status
):
    python = tmp_path / "python3"
    python.write_text(
        '#!/bin/sh\n[ "$2" = "--engine-schema" ] || exit 97\n'
        f"exit {validation_status}\n"
    )
    server = tmp_path / "trtllm-serve"
    server.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\"\n")
    for executable in (python, server):
        executable.chmod(0o755)
    result = subprocess.run(
        ["/bin/sh", str(ADAPTER / "trtllm-serve"), "--model", "model with spaces"],
        env={**os.environ, "PATH": str(tmp_path)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == validation_status
    assert result.stdout.splitlines() == (
        ["--model", "model with spaces"] if validation_status == 0 else []
    )
