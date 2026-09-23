from __future__ import annotations

import importlib.util
import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest


def _wrapper():
    path = Path(__file__).with_name("qwen38-vllm-wrapper.py")
    spec = importlib.util.spec_from_file_location("qwen38_vllm_wrapper", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preparation_reruns_for_source_change_without_writing_source(
    tmp_path: Path,
) -> None:
    wrapper = _wrapper()
    source = tmp_path / "models"
    source.mkdir()
    config = source / "config.json"
    config.write_text(
        json.dumps({"text_config": {"ple_embedding_dtype": "float8_e4m3fn"}})
    )
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
    source.chmod(
        stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH
    )
    second = wrapper.prepare_model()
    assert second != first
    assert wrapper.PREPARED.resolve() != first_target
    assert (
        json.loads(config.read_text())["text_config"]["ple_embedding_dtype"] == "nvfp4"
    )
    assert (
        json.loads((wrapper.PREPARED / ".vonk-prepared.json").read_text())[
            "source_fingerprint"
        ]
        == second
    )


def _start_lock_holder(
    root: Path, ready: Path, duration_seconds: float = 60.0
) -> subprocess.Popen[bytes]:
    code = """\
import fcntl
import json
import os
from pathlib import Path
import sys
import time
root = Path(sys.argv[1])
root.mkdir(parents=True, exist_ok=True)
descriptor = os.open(root / '.prepare.lock', os.O_CREAT | os.O_RDWR, 0o600)
fcntl.flock(descriptor, fcntl.LOCK_EX)
metadata = json.dumps({'pid': os.getpid(), 'started_at_utc': 'test'})
os.ftruncate(descriptor, 0)
os.write(descriptor, metadata.encode())
Path(sys.argv[2]).write_text('ready')
time.sleep(float(sys.argv[3]))
"""
    return subprocess.Popen(
        [sys.executable, "-c", code, str(root), str(ready), str(duration_seconds)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _wait_for_lock_holder(holder: subprocess.Popen[bytes], ready: Path) -> None:
    deadline = time.monotonic() + 5.0
    while not ready.exists() and time.monotonic() < deadline:
        if holder.poll() is not None:
            raise AssertionError("preparation-lock holder exited before acquiring")
        time.sleep(0.01)
    assert ready.exists(), "preparation-lock holder did not become ready"


def _stop_lock_holder(holder: subprocess.Popen[bytes]) -> None:
    if holder.poll() is not None:
        return
    holder.terminate()
    try:
        holder.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        holder.kill()
        holder.wait(timeout=5.0)


def test_preparation_lock_is_bounded_and_reports_owner(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    wrapper = _wrapper()
    root = tmp_path / "cache"
    ready = tmp_path / "holder.ready"
    wrapper.PREPARATION_LOCK_TIMEOUT_SECONDS = 0.04
    wrapper.PREPARATION_LOCK_POLL_SECONDS = 0.01
    wrapper.PREPARATION_LOCK_STATUS_INTERVAL_SECONDS = 0.01
    holder = _start_lock_holder(root, ready, duration_seconds=1.0)
    try:
        _wait_for_lock_holder(holder, ready)
        with pytest.raises(SystemExit) as error, wrapper._preparation_lock(root):
            raise AssertionError("a contended lock must not be acquired")
        message = str(error.value)
        assert "dependency=qwen38-model-preparation" in message
        assert f"owner_pid={holder.pid}" in message
        assert "wait_limit_seconds=0.04" in message
        assert "deadline_utc=" in message
        assert "resume_condition=retry_after_owner_releases_lock" in message
        telemetry = capsys.readouterr().err
        assert f"owner_pid={holder.pid}" in telemetry
        assert "next_check_utc=" in telemetry
    finally:
        _stop_lock_holder(holder)


def test_preparation_lock_recovers_after_owner_process_dies(tmp_path: Path) -> None:
    wrapper = _wrapper()
    root = tmp_path / "cache"
    ready = tmp_path / "holder.ready"
    holder = _start_lock_holder(root, ready)
    try:
        _wait_for_lock_holder(holder, ready)
        holder.terminate()
        holder.wait(timeout=5.0)
        started = time.monotonic()
        with wrapper._preparation_lock(root):
            pass
        assert time.monotonic() - started < 2.0
        owner = json.loads((root / ".prepare.lock").read_text())
        assert owner["pid"] == os.getpid()
    finally:
        _stop_lock_holder(holder)


def test_hung_preparation_helper_releases_lock_for_retry(tmp_path: Path) -> None:
    wrapper = _wrapper()
    source = tmp_path / "models"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps({"text_config": {"ple_embedding_dtype": "float8_e4m3fn"}})
    )
    patcher = tmp_path / "patcher.py"
    patcher.write_text("import time\ntime.sleep(1)\n")
    cache = tmp_path / "cache" / "qwen38-model"
    wrapper.SOURCE = source
    wrapper.PREPARED_ROOT = cache
    wrapper.PREPARED = cache / "current"
    wrapper.PATCHER = patcher
    wrapper.PREPARATION_HELPER_TIMEOUT_SECONDS = 0.05

    started = time.monotonic()
    with pytest.raises(SystemExit) as error:
        wrapper.prepare_model()
    assert time.monotonic() - started < 2.0
    assert "preparation helper" in str(error.value)
    assert "exceeded 0.05s" in str(error.value)
    assert "resume_condition=retry_after_helper_is_repaired" in str(error.value)

    patcher.write_text("import sys\n")
    fingerprint = wrapper.prepare_model()
    assert (
        json.loads((wrapper.PREPARED / ".vonk-prepared.json").read_text())[
            "source_fingerprint"
        ]
        == fingerprint
    )
    assert json.loads((cache / ".prepare.lock").read_text())["pid"] == os.getpid()


def test_hf_overrides_merges_safe_options_and_enforces_yarn_guard(
    tmp_path: Path,
) -> None:
    wrapper = _wrapper()
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "config.json").write_text(
        json.dumps({"text_config": {"ple_embedding_dtype": "float8_e4m3fn"}})
    )
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
