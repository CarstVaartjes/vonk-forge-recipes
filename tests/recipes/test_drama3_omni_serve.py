"""The Drama 3 vLLM-Omni launcher gives each stage its own context and memory budget."""

from __future__ import annotations

import importlib.util
import json
from types import ModuleType

import pytest
from repo_root import ROOT

ADAPTER = ROOT / "adapters/fishaudio/drama-3-vllm-omni-single"


def _launcher() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "drama3_omni_serve", ADAPTER / "drama3-omni-serve.py"
    )
    assert spec
    assert spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _overrides(arguments: list[str]) -> dict[str, dict[str, float]]:
    return json.loads(arguments[arguments.index("--stage-overrides") + 1])


def test_backbone_and_codec_budgets_become_separate_stage_overrides() -> None:
    arguments = _launcher().stage_scoped(
        [
            "serve",
            "/models",
            "--gpu-memory-utilization",
            "0.513",
            "--codec-gpu-memory-utilization=0.066",
            "--max-model-len",
            "8192",
            "--omni",
        ]
    )
    assert arguments[:2] == ["serve", "/models"]
    assert "--omni" in arguments
    assert not any("memory-utilization" in a or "max-model-len" in a for a in arguments)
    assert _overrides(arguments) == {
        "0": {"max_model_len": 8192, "gpu_memory_utilization": 0.513},
        "1": {"gpu_memory_utilization": 0.066},
    }


def test_other_arguments_pass_through_unchanged() -> None:
    arguments = ["serve", "/models", "--omni", "--port", "8000"]
    assert _launcher().stage_scoped(list(arguments)) == arguments


@pytest.mark.parametrize(
    "arguments",
    [
        ["serve", "--codec-gpu-memory-utilization"],
        ["serve", "--gpu-memory-utilization", "most"],
        ["serve", "--max-model-len", "8192", "--stage-overrides", "{}"],
    ],
)
def test_a_missing_or_bad_value_or_a_second_override_is_refused(
    arguments: list[str],
) -> None:
    with pytest.raises(SystemExit):
        _launcher().stage_scoped(arguments)
