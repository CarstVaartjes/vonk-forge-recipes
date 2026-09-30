#!/usr/bin/env python3
"""Run sfxnz DeepSeek-V4-Flash-Vision-Exp vLLM through Controller-managed topology.

Binds the Controller rendezvous to vLLM's multi-node flags and honours the
refuse-guards of upstream run.sh (without its FORCE_UNSAFE_CTX escape hatch).
"""

from __future__ import annotations

import os
import shutil
import sys
from ipaddress import ip_address

SOURCE = "/models"
# Guard constants copied from upstream run.sh (12 GiB KV pin, 1M window, 2 seqs).
KV_PIN_BYTES = 12884901888
MAX_WINDOW = 1048576
SMALL_WINDOW = 289024
MAX_SEQS = 2
SPEC_DIVISOR = 3


def value(arguments: list[str], option: str) -> str | None:
    if option not in arguments:
        return None
    index = arguments.index(option)
    if index + 1 >= len(arguments):
        raise SystemExit(f"{option} requires a value")
    return arguments[index + 1]


def integer(arguments: list[str], option: str) -> int | None:
    text = value(arguments, option)
    if text is None:
        return None
    try:
        return int(text)
    except ValueError:
        raise SystemExit(f"{option} requires an integer") from None


def refuse_guards(arguments: list[str]) -> None:
    window = integer(arguments, "--max-model-len")
    seqs = integer(arguments, "--max-num-seqs")
    pin = integer(arguments, "--kv-cache-memory")
    dtype = value(arguments, "--kv-cache-dtype")
    if window is None or seqs is None or pin is None:
        raise SystemExit(
            "--max-model-len, --max-num-seqs and --kv-cache-memory are required"
        )
    if dtype in {"fp8", "fp8_e4m3"} and window > MAX_WINDOW:
        raise SystemExit(
            f"fp8 KV pin cannot hold --max-model-len {window}; "
            f"the 12 GiB pin holds {MAX_WINDOW}"
        )
    if pin < KV_PIN_BYTES and window > SMALL_WINDOW:
        raise SystemExit(
            f"KV pin {pin} cannot hold --max-model-len {window}; "
            f"a pin below 12 GiB holds at most {SMALL_WINDOW}"
        )
    if pin > KV_PIN_BYTES:
        raise SystemExit(f"--kv-cache-memory {pin} exceeds the 12 GiB pin")
    if seqs > MAX_SEQS:
        raise SystemExit(f"--max-num-seqs {seqs} exceeds {MAX_SEQS} on this 12 GiB pin")
    speculative = value(arguments, "--speculative-config")
    if speculative is not None:
        import json

        try:
            tokens = int(json.loads(speculative)["num_speculative_tokens"])
        except (ValueError, KeyError, TypeError):
            raise SystemExit("--speculative-config needs num_speculative_tokens") from None
        if tokens % SPEC_DIVISOR:
            raise SystemExit(
                f"num_speculative_tokens {tokens} is not divisible by {SPEC_DIVISOR}"
            )


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
            "DeepSeek V4 Flash Vision TP2 requires complete Controller "
            "rendezvous and fabric"
        ) from None

    refuse_guards(arguments)
    arguments.extend(("--master-addr", master, "--master-port", port))
    os.environ["VLLM_HOST_IP"] = local
    os.environ["MASTER_ADDR"] = master
    os.environ["MASTER_PORT"] = port
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    vllm = shutil.which("vllm")
    if vllm is None:
        raise SystemExit("the pinned B12X vLLM executable is missing")
    os.execv(vllm, (vllm, "serve", SOURCE, *arguments))


if __name__ == "__main__":
    main()
