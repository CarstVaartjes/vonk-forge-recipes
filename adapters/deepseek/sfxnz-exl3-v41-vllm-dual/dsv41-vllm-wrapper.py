#!/usr/bin/env python3
"""Run sfxnz DeepSeek-V4.1-Flash EXL3 vLLM through Controller-managed topology."""

from __future__ import annotations

import os
import sys
from ipaddress import ip_address
from pathlib import Path

SOURCE = Path("/models")


def value(arguments: list[str], option: str) -> str | None:
    if option not in arguments:
        return None
    index = arguments.index(option)
    if index + 1 >= len(arguments):
        raise SystemExit(f"{option} requires a value")
    return arguments[index + 1]


def drop(arguments: list[str], option: str, takes_value: bool) -> None:
    """Remove every occurrence of an option (and its value)."""
    while option in arguments:
        index = arguments.index(option)
        del arguments[index : index + (2 if takes_value else 1)]


def apply_switches(arguments: list[str]) -> None:
    """Map the upstream run.sh switches that omit an argument rather than change it.

    SPEC=none omits --speculative-config; ENFORCE_EAGER=1 passes --enforce-eager
    instead of --compilation-config; LANGUAGE_MODEL_ONLY=1 passes
    --language-model-only instead of --mm-encoder-tp-mode.
    """
    if os.environ.get("DSV41_SPECULATION") == "off":
        drop(arguments, "--speculative-config", True)
    if os.environ.get("ENFORCE_EAGER") == "1":
        drop(arguments, "--compilation-config", True)
        arguments.append("--enforce-eager")
    if os.environ.get("LANGUAGE_MODEL_ONLY") == "1":
        drop(arguments, "--mm-encoder-tp-mode", True)
        arguments.append("--language-model-only")


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
            "DeepSeek V4.1 Flash EXL3 TP2 requires complete Controller rendezvous and fabric"
        ) from None

    for path in (
        SOURCE / "config.json",
        SOURCE / "model.safetensors.index.json",
        SOURCE / "model-00047-of-00048.safetensors",
        SOURCE / "model-00048-of-00048.safetensors",
    ):
        if not path.is_file():
            raise SystemExit(f"immutable model artifact is missing: {path}")

    apply_switches(arguments)
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
        raise SystemExit("the pinned DeepSeek V4.1 vLLM executable is missing")
    os.execv(vllm, (vllm, "serve", str(SOURCE), *arguments))


if __name__ == "__main__":
    main()
