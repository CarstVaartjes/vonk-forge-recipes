"""Producer-discovered Qwen3.8 adapter tests.

The adapter source directory contains the implementation fixture, while this
module is the CI discovery entrypoint. Loading by source path keeps the
hyphenated adapter directory importable without changing the package layout.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_SOURCE_TEST = (
    Path(__file__).parents[1]
    / "adapters/qwen/flash-next-vllm-dual/test_qwen38_wrapper.py"
)
_SPEC = importlib.util.spec_from_file_location("qwen38_adapter_source_tests", _SOURCE_TEST)
assert _SPEC and _SPEC.loader
_SOURCE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SOURCE)


def test_qwen38_preparation_reruns_for_source_change_without_writing_source(tmp_path: Path) -> None:
    _SOURCE.test_preparation_reruns_for_source_change_without_writing_source(tmp_path)


def test_qwen38_hf_overrides_preserve_explicit_values_and_apply_defaults(tmp_path: Path) -> None:
    _SOURCE.test_hf_overrides_merges_safe_options_and_enforces_yarn_guard(tmp_path)


def test_qwen38_oci_runtime_metadata_and_entrypoint_are_declared() -> None:
    _SOURCE.test_oci_runtime_metadata_and_entrypoint_are_declared()
