from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


def _wrapper() -> ModuleType:
    path = Path(__file__).with_name("qwen38-single-serve.py")
    spec = importlib.util.spec_from_file_location("qwen38_single_serve", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_native_context_adds_no_yarn_override() -> None:
    arguments = ["--max-model-len", "262144"]
    _wrapper().yarn_override(arguments)
    assert arguments == ["--max-model-len", "262144"]


def test_longer_context_adds_upstream_yarn_factor() -> None:
    arguments = ["--max-model-len", "524288"]
    _wrapper().yarn_override(arguments)
    overrides = json.loads(arguments[arguments.index("--hf-overrides") + 1])
    assert overrides == {
        "text_config": {
            "rope_parameters": {
                "rope_type": "yarn",
                "factor": 2.0,
                "original_max_position_embeddings": 262144,
            }
        }
    }


def test_explicit_overrides_are_preserved() -> None:
    arguments = ["--max-model-len", "400000", "--hf-overrides", "{}"]
    _wrapper().yarn_override(arguments)
    assert arguments[-1] == "{}"
    assert arguments.count("--hf-overrides") == 1


def test_context_above_validated_ceiling_is_rejected() -> None:
    with pytest.raises(SystemExit):
        _wrapper().yarn_override(["--max-model-len", "524289"])


def test_capture_sizes_cover_every_verify_width() -> None:
    arguments = [
        "--max-num-seqs",
        "4",
        "--speculative-config",
        '{"method":"mtp","num_speculative_tokens":3}',
        "--compilation-config",
        '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}',
    ]
    _wrapper().capture_sizes(arguments)
    config = json.loads(arguments[arguments.index("--compilation-config") + 1])
    assert config == {
        "mode": 0,
        "cudagraph_mode": "FULL_DECODE_ONLY",
        "cudagraph_capture_sizes": [4, 8, 12, 16],
    }


def test_explicit_capture_sizes_are_preserved() -> None:
    raw = '{"cudagraph_capture_sizes":[1,2]}'
    arguments = ["--max-num-seqs", "4", "--compilation-config", raw]
    _wrapper().capture_sizes(arguments)
    assert arguments[-1] == raw
