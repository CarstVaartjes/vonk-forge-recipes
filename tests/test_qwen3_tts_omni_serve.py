"""The Qwen3-TTS vLLM-Omni launcher scopes the recipe's context to the talker stage."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

ADAPTER = Path(__file__).resolve().parents[1] / (
    "adapters/qwen/qwen3-tts-customvoice-vllm-omni-single"
)


def _launcher() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "qwen3_tts_omni_serve", ADAPTER / "qwen3-tts-omni-serve.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_context_becomes_a_talker_stage_override() -> None:
    arguments = _launcher().talker_context(
        ["serve", "/models", "--max-model-len", "4096", "--omni"]
    )
    assert "--max-model-len" not in arguments
    assert arguments[:2] == ["serve", "/models"]
    overrides = json.loads(arguments[arguments.index("--stage-overrides") + 1])
    assert overrides == {"0": {"max_model_len": 4096}}
    assert "--omni" in arguments


def test_equals_form_is_scoped_the_same_way() -> None:
    arguments = _launcher().talker_context(["serve", "/models", "--max-model-len=2048"])
    assert json.loads(arguments[arguments.index("--stage-overrides") + 1]) == {
        "0": {"max_model_len": 2048}
    }


def test_other_arguments_pass_through_unchanged() -> None:
    arguments = ["serve", "/models", "--gpu-memory-utilization", "0.197", "--omni"]
    assert _launcher().talker_context(list(arguments)) == arguments


@pytest.mark.parametrize(
    "arguments",
    [
        ["serve", "--max-model-len"],
        ["serve", "--max-model-len", "0"],
        ["serve", "--max-model-len", "four"],
        ["serve", "--max-model-len", "4096", "--stage-overrides", "{}"],
    ],
)
def test_unusable_context_or_a_second_override_is_refused(arguments: list[str]) -> None:
    with pytest.raises(SystemExit):
        _launcher().talker_context(arguments)


def test_the_deploy_config_is_the_creators_two_stage_file() -> None:
    text = (ADAPTER / "qwen3_tts.yaml").read_text(encoding="utf-8")
    assert "max_model_len: 4096" in text
    assert "max_model_len: 65536" in text
