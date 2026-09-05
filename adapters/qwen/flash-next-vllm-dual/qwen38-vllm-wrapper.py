#!/usr/bin/env python3
"""Run Qwen3.8 vLLM through Controller-managed topology and cache paths."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from ipaddress import ip_address
from pathlib import Path

SOURCE = Path("/models")
PREPARED_ROOT = Path("/outputs/cache/qwen38-model")
PREPARED = PREPARED_ROOT / "current"
PATCHER = Path("/opt/vonk/build/patch_checkpoint_config.py")
DETECT = Path("/opt/vonk/build/detect_ple_dtype.py")
UPSTREAM_REVISION = "c2325b22602b51a5faf55fc2bebccc34f3f80b9f"
MODEL_DOCUMENT_SHA256 = "0f2c8617255df59583d6def4f71cb20ec63709aaf7c801ae8ce71f6a18e5edc4"
METADATA_HASH_LIMIT = 16 * 1024 * 1024


def value(arguments: list[str], option: str) -> str | None:
    if option not in arguments:
        return None
    index = arguments.index(option)
    if index + 1 >= len(arguments):
        raise SystemExit(f"{option} requires a value")
    return arguments[index + 1]


def _source_fingerprint(source: Path) -> str:
    """Hash the model manifest and all small source files.

    Large weight payloads are Controller-selected immutable artifacts. Their
    relative paths and byte sizes are included here while configs, indexes,
    tokenizers, and other small files are content-hashed. The canonical Model
    document digest is included so a different selected model cannot reuse a
    prepared configuration directory.
    """
    digest = hashlib.sha256()
    digest.update(f"model-document:{MODEL_DOCUMENT_SHA256}\n".encode())
    digest.update(f"upstream:{UPSTREAM_REVISION}\n".encode())
    entries = sorted((item for item in source.rglob("*") if item.is_file()), key=lambda item: item.relative_to(source).as_posix())
    for path in entries:
        relative = path.relative_to(source).as_posix()
        size = path.stat().st_size
        digest.update(f"path:{relative}\nsize:{size}\n".encode())
        if size <= METADATA_HASH_LIMIT:
            content_digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    content_digest.update(chunk)
            digest.update(f"content:{content_digest.hexdigest()}\n".encode())
    return digest.hexdigest()


@contextmanager
def _preparation_lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(root / ".prepare.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _marker(path: Path) -> dict[str, str] | None:
    try:
        document = json.loads((path / ".vonk-prepared.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    return document if isinstance(document, dict) else None


def _publish_prepared(source_fingerprint: str) -> None:
    published = PREPARED_ROOT / f"model-{source_fingerprint}"
    expected_marker = {
        "model_document_sha256": MODEL_DOCUMENT_SHA256,
        "source_fingerprint": source_fingerprint,
        "upstream_revision": UPSTREAM_REVISION,
    }
    if _marker(published) != expected_marker:
        staging = PREPARED_ROOT / f".staging-{source_fingerprint}-{os.getpid()}"
        staging.mkdir()
        try:
            for entry in SOURCE.iterdir():
                (staging / entry.name).symlink_to(entry)
            subprocess.run([sys.executable, str(PATCHER), str(staging), str(staging)], check=True)
            for name in ("config", "hf_quant_config"):
                patched = staging / f"{name}_patched.json"
                if patched.is_file():
                    target = staging / f"{name}.json"
                    target.unlink(missing_ok=True)
                    patched.replace(target)
            marker_tmp = staging / ".vonk-prepared.json.tmp"
            marker_tmp.write_text(json.dumps(expected_marker, sort_keys=True) + "\n", encoding="utf-8")
            os.replace(marker_tmp, staging / ".vonk-prepared.json")
            if published.exists():
                os.replace(published, PREPARED_ROOT / f".stale-{published.name}-{os.getpid()}")
            os.replace(staging, published)
        except BaseException:
            if staging.exists():
                for child in staging.iterdir():
                    if child.is_file() or child.is_symlink():
                        child.unlink(missing_ok=True)
                staging.rmdir()
            raise
    current_tmp = PREPARED_ROOT / f".current-{os.getpid()}"
    current_tmp.unlink(missing_ok=True)
    current_tmp.symlink_to(published.name)
    if PREPARED.exists() and not PREPARED.is_symlink():
        os.replace(PREPARED, PREPARED_ROOT / f".stale-current-{os.getpid()}")
    else:
        PREPARED.unlink(missing_ok=True)
    os.replace(current_tmp, PREPARED)


def prepare_model() -> str:
    """Prepare a content-bound cache view and return its source fingerprint."""
    source_fingerprint = _source_fingerprint(SOURCE)
    with _preparation_lock(PREPARED_ROOT):
        expected = {
            "model_document_sha256": MODEL_DOCUMENT_SHA256,
            "source_fingerprint": source_fingerprint,
            "upstream_revision": UPSTREAM_REVISION,
        }
        if _marker(PREPARED) != expected:
            _publish_prepared(source_fingerprint)
    return source_fingerprint


def _set_option(arguments: list[str], option: str, option_value: str) -> None:
    indexes = [index for index, item in enumerate(arguments) if item == option]
    if indexes:
        index = indexes[0]
        if index + 1 >= len(arguments):
            raise SystemExit(f"{option} requires a value")
        arguments[index + 1] = option_value
    else:
        arguments.extend((option, option_value))


def _merged_hf_overrides(arguments: list[str]) -> str | None:
    option_count = arguments.count("--hf-overrides")
    if option_count > 1:
        # Preserve repeated engine options exactly; there is no safe way to
        # merge opaque JSON values without changing the engine's semantics.
        return None
    existing = value(arguments, "--hf-overrides")
    if existing is None:
        overrides: dict[str, object] = {}
    else:
        try:
            parsed = json.loads(existing)
        except json.JSONDecodeError as error:
            raise SystemExit(f"--hf-overrides must be a JSON object: {error}") from None
        if not isinstance(parsed, dict):
            raise SystemExit("--hf-overrides must be a JSON object")
        overrides = parsed
    text_config = overrides.get("text_config", {})
    if not isinstance(text_config, dict):
        raise SystemExit("--hf-overrides.text_config must be a JSON object")
    text_config = dict(text_config)
    config = json.loads((PREPARED / "config.json").read_text(encoding="utf-8"))
    source_text_config = config.get("text_config", config)
    ple_dtype = str(source_text_config.get("ple_embedding_dtype") or "")
    if not ple_dtype:
        ple_dtype = subprocess.run(
            [sys.executable, str(DETECT), str(PREPARED)], check=True, capture_output=True, text=True
        ).stdout.strip()
    inject_ple_dtype = bool(ple_dtype and "ple_embedding_dtype" not in text_config)
    if inject_ple_dtype:
        text_config["ple_embedding_dtype"] = ple_dtype
    max_len = value(arguments, "--max-model-len")
    if max_len is None or not max_len.isascii() or not max_len.isdigit() or int(max_len) <= 0:
        raise SystemExit("--max-model-len must be a positive integer")
    inject_yarn = int(max_len) > 262144 and "rope_parameters" not in text_config
    if inject_yarn:
        required_rope = {
            "rope_type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 262144,
        }
        text_config["rope_parameters"] = required_rope
    if existing is not None and not inject_ple_dtype and not inject_yarn:
        return existing
    if text_config:
        overrides["text_config"] = text_config
    return json.dumps(overrides, separators=(",", ":")) if overrides else None


def main() -> None:
    arguments = sys.argv[1:]
    node_count = value(arguments, "--nnodes")
    node_rank = value(arguments, "--node-rank")
    backend = value(arguments, "--distributed-executor-backend")
    headless = "--headless" in arguments
    local = os.environ.get("VONK_LOCAL_ADDR")
    master = os.environ.get("VONK_MASTER_ADDR")
    port = os.environ.get("VONK_MASTER_PORT")
    fabric = ("NCCL_SOCKET_IFNAME", "NCCL_IB_HCA", "NCCL_IB_GID_INDEX", "TP_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME")
    try:
        if backend != "mp" or node_count != "2" or node_rank not in {"0", "1"}:
            raise ValueError
        if headless != (node_rank == "1") or not local or not master or not port:
            raise ValueError
        if any(not os.environ.get(name) for name in fabric) or not port.isascii() or not port.isdigit():
            raise ValueError
        if not 1024 <= int(port) <= 65535:
            raise ValueError
        ip_address(local)
        ip_address(master)
    except ValueError:
        raise SystemExit("Qwen3.8 TP2 requires complete Controller rendezvous and fabric") from None

    prepare_model()
    override = _merged_hf_overrides(arguments)
    if override:
        _set_option(arguments, "--hf-overrides", override)
    arguments.extend(("--master-addr", master, "--master-port", port))
    os.environ["VLLM_HOST_IP"] = local
    os.environ["MASTER_ADDR"] = master
    os.environ["MASTER_PORT"] = port
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    vllm = next((candidate for candidate in ("/usr/local/bin/vllm", "/opt/vllm/.venv/bin/vllm") if Path(candidate).is_file()), None)
    if vllm is None:
        raise SystemExit("the pinned Qwen vLLM executable is missing")
    os.execv(vllm, (vllm, "serve", str(PREPARED), *arguments))


if __name__ == "__main__":
    main()
