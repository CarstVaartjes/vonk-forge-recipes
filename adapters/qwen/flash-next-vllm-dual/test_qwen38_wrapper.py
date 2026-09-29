from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Protocol, cast


class _Wrapper(Protocol):
    def _merged_hf_overrides(self, arguments: list[str]) -> str | None: ...

    def _set_option(self, arguments: list[str], option: str, value: str) -> None: ...


def _wrapper() -> _Wrapper:
    path = Path(__file__).with_name("qwen38-vllm-wrapper.py")
    spec = importlib.util.spec_from_file_location("qwen38_vllm_wrapper", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return cast(_Wrapper, module)


def test_hf_overrides_merges_safe_options_and_enforces_yarn_guard() -> None:
    wrapper = _wrapper()
    arguments = [
        "--max-model-len",
        "1000000",
        "--hf-overrides",
        '{"text_config":{"custom_flag":true}}',
    ]
    merged_json = wrapper._merged_hf_overrides(arguments)
    assert merged_json is not None
    wrapper._set_option(arguments, "--hf-overrides", merged_json)
    merged = json.loads(arguments[arguments.index("--hf-overrides") + 1])
    assert arguments.count("--hf-overrides") == 1
    assert merged["text_config"]["custom_flag"] is True
    assert merged["text_config"]["rope_parameters"]["rope_type"] == "yarn"

    explicit = '{ "text_config": { "rope_parameters": {"rope_type":"custom"}, "opaque": "x;$✓" } }'
    explicit_args = ["--max-model-len", "1000000", "--hf-overrides", explicit]
    assert wrapper._merged_hf_overrides(explicit_args) == explicit

    native_args = ["--max-model-len", "262144"]
    assert wrapper._merged_hf_overrides(native_args) is None

    duplicate = ["--hf-overrides", explicit, "--hf-overrides", '{"other":"$;✓"}']
    duplicate.extend(("--max-model-len", "1000000"))
    assert wrapper._merged_hf_overrides(duplicate) is None


def test_oci_runtime_metadata_and_entrypoint_are_declared() -> None:
    adapter = Path(__file__).parent
    dockerfile = (adapter / "Dockerfile").read_text()
    assert 'ai.vonkforge.runtime-interface="v1"' in dockerfile
    recipe = json.loads(
        (
            adapter.parents[2] / "recipes/qwen3-8-flash-next-nvfp4-vllm-dual.json"
        ).read_text()
    )
    assert recipe["runtime"]["entrypoint"] == ["/opt/vonk/bin/qwen38-vllm-serve"]
