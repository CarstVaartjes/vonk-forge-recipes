"""The image build validates speculation with the installed engine's schema."""

import runpy
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


def test_build_checks_speculation_with_installed_tensorrt_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema = object()
    validated = Mock()
    validator = Mock(return_value=types.SimpleNamespace(validate_python=validated))
    pydantic = types.ModuleType("pydantic")
    pydantic.__dict__["TypeAdapter"] = validator
    engine = types.ModuleType("tensorrt_llm.llmapi.llm_args")
    engine.__dict__["SpeculativeConfig"] = schema
    config = {"speculative_config": {"engine_owned": "unchanged"}}
    yaml = types.ModuleType("yaml")
    yaml.__dict__["safe_load"] = Mock(return_value=config)
    for module in (pydantic, engine, yaml):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    source = runpy.run_path(str(ADAPTER / "verify-runtime.py"))
    source["main"].__globals__["CONFIG"] = ADAPTER / "llm-api-config.yml"
    monkeypatch.setattr(source["os"], "access", lambda *_: True)
    monkeypatch.setattr(source["shutil"], "which", lambda _: "trtllm-serve")
    source["main"]()
    validator.assert_called_once_with(schema)
    validated.assert_called_once_with(config["speculative_config"])
