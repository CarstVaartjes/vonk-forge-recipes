#!/usr/bin/env python3
"""Run tonyd2wild's Qwen3.8 Flash Next vLLM stack through Controller-managed topology.

Placement (ranks, rendezvous, fabric) is the wrapper's only launch policy on a
multi-node topology. Everything else comes from the authored recipe. Two
derived values follow upstream's launchers:

* decode CUDA graph sizes: upstream pins the capture list to (1 + K) * s for
  every scheduler width s up to --max-num-seqs (e.g. 4,8,12,16,20,24 for MTP 3
  and six sequences) whenever it captures FULL_DECODE_ONLY graphs;
* the recipe's speculative-decoding option cannot remove an argument, so
  VONK_QWEN_SPEC=off drops --speculative-config here;
* upstream's CONTEXT profiles run vLLM's default prefill chunk (its launchers
  leave --max-num-batched-tokens out), so VONK_QWEN_CHUNK=default drops it here.
"""

from __future__ import annotations

import json
import os
import sys
from ipaddress import ip_address
from pathlib import Path

SOURCE = Path("/models")
FABRIC = (
    "NCCL_SOCKET_IFNAME",
    "NCCL_IB_HCA",
    "NCCL_IB_GID_INDEX",
    "TP_SOCKET_IFNAME",
    "GLOO_SOCKET_IFNAME",
)


def value(arguments: list[str], option: str) -> str | None:
    if option not in arguments:
        return None
    index = arguments.index(option)
    if index + 1 >= len(arguments):
        raise SystemExit(f"{option} requires a value")
    return arguments[index + 1]


def set_option(arguments: list[str], option: str, option_value: str) -> None:
    if option in arguments:
        arguments[arguments.index(option) + 1] = option_value
    else:
        arguments.extend((option, option_value))


def placement(arguments: list[str]) -> None:
    """Validate and append the Controller's multi-node rendezvous, if any."""
    nodes = value(arguments, "--nnodes")
    if nodes is None:
        return
    rank = value(arguments, "--node-rank")
    backend = value(arguments, "--distributed-executor-backend")
    local = os.environ.get("VONK_LOCAL_ADDR")
    master = os.environ.get("VONK_MASTER_ADDR")
    port = os.environ.get("VONK_MASTER_PORT")
    try:
        if backend != "mp" or nodes not in {"2", "4"} or rank is None:
            raise ValueError
        if not rank.isdigit() or not 0 <= int(rank) < int(nodes):
            raise ValueError
        if ("--headless" in arguments) != (rank != "0"):
            raise ValueError
        if not local or not master or not port:
            raise ValueError
        if any(not os.environ.get(name) for name in FABRIC):
            raise ValueError
        if not os.environ["NCCL_IB_GID_INDEX"].isdigit():
            raise ValueError
        if not port.isascii() or not port.isdigit() or not 1024 <= int(port) <= 65535:
            raise ValueError
        ip_address(local)
        ip_address(master)
    except ValueError:
        raise SystemExit(
            "multi-node Qwen3.8 requires exact Controller rendezvous, ranks and fabric"
        ) from None
    arguments.extend(("--master-addr", master, "--master-port", port))
    os.environ["VLLM_HOST_IP"] = local
    os.environ["MASTER_ADDR"] = master
    os.environ["MASTER_PORT"] = port


def speculative(arguments: list[str]) -> int:
    """Apply the off switch and return the number of drafted tokens."""
    raw = value(arguments, "--speculative-config")
    if raw is None:
        return 0
    if os.environ.get("VONK_QWEN_SPEC") == "off":
        index = arguments.index("--speculative-config")
        del arguments[index : index + 2]
        return 0
    return int(json.loads(raw).get("num_speculative_tokens", 0))


def default_chunk(arguments: list[str]) -> None:
    if os.environ.get("VONK_QWEN_CHUNK") == "default" and (
        "--max-num-batched-tokens" in arguments
    ):
        index = arguments.index("--max-num-batched-tokens")
        del arguments[index : index + 2]


def capture_sizes(arguments: list[str], depth: int) -> None:
    raw = value(arguments, "--compilation-config")
    config = json.loads(raw) if raw else {}
    if not isinstance(config, dict):
        raise SystemExit("--compilation-config must be a JSON object")
    if config.get("cudagraph_mode") != "FULL_DECODE_ONLY":
        return
    if "cudagraph_capture_sizes" in config:
        return
    seqs = value(arguments, "--max-num-seqs")
    if seqs is None or not seqs.isascii() or not seqs.isdigit() or int(seqs) < 1:
        raise SystemExit("--max-num-seqs must be a positive integer")
    config["cudagraph_capture_sizes"] = [(1 + depth) * s for s in range(1, int(seqs) + 1)]
    set_option(
        arguments, "--compilation-config", json.dumps(config, separators=(",", ":"))
    )


def main() -> None:
    arguments = sys.argv[1:]
    for name in ("config.json", "model.safetensors.index.json"):
        if not (SOURCE / name).is_file():
            raise SystemExit(f"immutable model artifact is missing: {SOURCE / name}")
    placement(arguments)
    capture_sizes(arguments, speculative(arguments))
    default_chunk(arguments)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    vllm = next(
        (
            candidate
            for candidate in ("/usr/local/bin/vllm", "/opt/vllm/.venv/bin/vllm")
            if Path(candidate).is_file() and os.access(candidate, os.X_OK)
        ),
        None,
    )
    if vllm is None:
        raise SystemExit("the pinned Qwen vLLM executable is missing")
    os.execv(vllm, (vllm, "serve", str(SOURCE), *arguments))


if __name__ == "__main__":
    main()
