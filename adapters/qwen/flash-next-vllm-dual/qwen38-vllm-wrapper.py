#!/usr/bin/env python3
"""Run Qwen3.8 Flash Next vLLM through Controller-managed topology."""

from __future__ import annotations

import json
import os
import sys
from ipaddress import ip_address
from pathlib import Path

SOURCE = Path("/models")
NATIVE_CONTEXT = 262144
YARN_ROPE = {
    "rope_type": "yarn",
    "factor": 4.0,
    "original_max_position_embeddings": NATIVE_CONTEXT,
}


def value(arguments: list[str], option: str) -> str | None:
    if option not in arguments:
        return None
    index = arguments.index(option)
    if index + 1 >= len(arguments):
        raise SystemExit(f"{option} requires a value")
    return arguments[index + 1]


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
    """Add the YaRN override above the native context; keep explicit values.

    At or below the native 262,144-token context no rope override is added
    (upstream force-disables YaRN there). Repeated or explicit rope overrides
    are preserved untouched.
    """
    option_count = arguments.count("--hf-overrides")
    if option_count > 1:
        # Preserve repeated engine options exactly; there is no safe way to
        # merge opaque JSON values without changing the engine's semantics.
        return None
    existing = value(arguments, "--hf-overrides")
    max_len = value(arguments, "--max-model-len")
    if (
        max_len is None
        or not max_len.isascii()
        or not max_len.isdigit()
        or int(max_len) <= 0
    ):
        raise SystemExit("--max-model-len must be a positive integer")
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
    if int(max_len) <= NATIVE_CONTEXT or "rope_parameters" in text_config:
        return existing
    overrides["text_config"] = {**text_config, "rope_parameters": dict(YARN_ROPE)}
    return json.dumps(overrides, separators=(",", ":"))


def main() -> None:
    arguments = sys.argv[1:]
    node_count = value(arguments, "--nnodes")
    node_rank = value(arguments, "--node-rank")
    backend = value(arguments, "--distributed-executor-backend")
    headless = "--headless" in arguments
    local = os.environ.get("VONK_LOCAL_ADDR")
    master = os.environ.get("VONK_MASTER_ADDR")
    port = os.environ.get("VONK_MASTER_PORT")
    fabric = (
        "NCCL_SOCKET_IFNAME",
        "NCCL_IB_HCA",
        "NCCL_IB_GID_INDEX",
        "TP_SOCKET_IFNAME",
        "GLOO_SOCKET_IFNAME",
    )
    try:
        if backend != "mp" or node_count != "2" or node_rank not in {"0", "1"}:
            raise ValueError
        if headless != (node_rank == "1") or not local or not master or not port:
            raise ValueError
        if (
            any(not os.environ.get(name) for name in fabric)
            or not port.isascii()
            or not port.isdigit()
        ):
            raise ValueError
        if not 1024 <= int(port) <= 65535:
            raise ValueError
        ip_address(local)
        ip_address(master)
    except ValueError:
        raise SystemExit(
            "Qwen3.8 TP2 requires complete Controller rendezvous and fabric"
        ) from None

    override = _merged_hf_overrides(arguments)
    if override:
        _set_option(arguments, "--hf-overrides", override)
    arguments.extend(("--master-addr", master, "--master-port", port))
    os.environ["VLLM_HOST_IP"] = local
    os.environ["MASTER_ADDR"] = master
    os.environ["MASTER_PORT"] = port
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    vllm = next(
        (
            candidate
            for candidate in ("/usr/local/bin/vllm", "/opt/vllm/.venv/bin/vllm")
            if Path(candidate).is_file()
        ),
        None,
    )
    if vllm is None:
        raise SystemExit("the pinned Qwen vLLM executable is missing")
    os.execv(vllm, (vllm, "serve", str(SOURCE), *arguments))


if __name__ == "__main__":
    main()
