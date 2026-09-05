from __future__ import annotations

import importlib.util
import json
import stat
from pathlib import Path


def _wrapper():
    path = Path(__file__).with_name("qwen38-vllm-wrapper.py")
    spec = importlib.util.spec_from_file_location("qwen38_vllm_wrapper", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preparation_reruns_for_source_change_without_writing_source(tmp_path: Path) -> None:
    wrapper = _wrapper()
    source = tmp_path / "models"
    source.mkdir()
    config = source / "config.json"
    config.write_text(json.dumps({"text_config": {"ple_embedding_dtype": "float8_e4m3fn"}}))
    (source / "weights.index.json").write_text('{"weight_map": {"x": "x.safetensors"}}')
    (source / "x.safetensors").write_bytes(b"immutable model bytes")
    patcher = tmp_path / "patcher.py"
    patcher.write_text("import sys\n")
    cache = tmp_path / "cache" / "qwen38-model"
    wrapper.SOURCE = source
    wrapper.PREPARED_ROOT = cache
    wrapper.PREPARED = cache / "current"
    wrapper.PATCHER = patcher

    first = wrapper.prepare_model()
    first_target = wrapper.PREPARED.resolve()
    assert wrapper.prepare_model() == first
    assert wrapper.PREPARED.resolve() == first_target

    config.write_text(json.dumps({"text_config": {"ple_embedding_dtype": "nvfp4"}}))
    config.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    source.chmod(stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
    second = wrapper.prepare_model()
    assert second != first
    assert wrapper.PREPARED.resolve() != first_target
    assert json.loads(config.read_text())["text_config"]["ple_embedding_dtype"] == "nvfp4"
    assert json.loads((wrapper.PREPARED / ".vonk-prepared.json").read_text())["source_fingerprint"] == second


def test_hf_overrides_merges_safe_options_and_enforces_yarn_guard(tmp_path: Path) -> None:
    wrapper = _wrapper()
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "config.json").write_text(json.dumps({"text_config": {"ple_embedding_dtype": "float8_e4m3fn"}}))
    wrapper.PREPARED = prepared
    arguments = [
        "--max-model-len",
        "1000000",
        "--hf-overrides",
        '{"text_config":{"custom_flag":true}}',
    ]
    merged_json = wrapper._merged_hf_overrides(arguments)
    wrapper._set_option(arguments, "--hf-overrides", merged_json)
    merged = json.loads(arguments[arguments.index("--hf-overrides") + 1])
    assert arguments.count("--hf-overrides") == 1
    assert merged["text_config"]["custom_flag"] is True
    assert merged["text_config"]["ple_embedding_dtype"] == "float8_e4m3fn"
    assert merged["text_config"]["rope_parameters"]["rope_type"] == "yarn"

    explicit = '{ "text_config": { "ple_embedding_dtype": "nvfp4", "rope_parameters": {"rope_type":"custom"}, "opaque": "x;$✓" } }'
    explicit_args = ["--max-model-len", "262144", "--hf-overrides", explicit]
    assert wrapper._merged_hf_overrides(explicit_args) == explicit

    default_args = ["--max-model-len", "262144"]
    default = json.loads(wrapper._merged_hf_overrides(default_args))
    assert default["text_config"]["ple_embedding_dtype"] == "float8_e4m3fn"
    assert "rope_parameters" not in default["text_config"]

    duplicate = ["--hf-overrides", explicit, "--hf-overrides", '{"other":"$;✓"}']
    assert wrapper._merged_hf_overrides(duplicate) is None
