#!/usr/bin/env python3
"""Bind Controller placement to the torch-distributed launch of MiaAI-Lab's Step 3.7 kit."""

from __future__ import annotations

import os
import sys
from ipaddress import ip_address
from pathlib import Path

MODEL = Path("/models")


def _value(arguments: list[str], option: str) -> str | None:
    if option not in arguments:
        return None
    index = arguments.index(option)
    if index + 1 >= len(arguments):
        raise SystemExit(f"{option} requires a value")
    return arguments[index + 1]


arguments = sys.argv[1:]
node_count = _value(arguments, "--nnodes")
node_rank = _value(arguments, "--node-rank")
mechanism = _value(arguments, "--distributed-executor-backend")
headless = "--headless" in arguments
local_address = os.environ.get("VONK_LOCAL_ADDR")
master_address = os.environ.get("VONK_MASTER_ADDR")
master_port = os.environ.get("VONK_MASTER_PORT")
fabric_names = (
    "NCCL_SOCKET_IFNAME",
    "NCCL_IB_HCA",
    "NCCL_IB_GID_INDEX",
    "TP_SOCKET_IFNAME",
    "GLOO_SOCKET_IFNAME",
)
try:
    if (
        mechanism != "mp"
        or node_count != "2"
        or node_rank not in {"0", "1"}
        or headless != (node_rank == "1")
        or not local_address
        or not master_address
        or not master_port
        or any(not os.environ.get(name) for name in fabric_names)
        or not master_port.isascii()
        or not master_port.isdigit()
        or not 1024 <= int(master_port) <= 65535
    ):
        raise ValueError
    ip_address(local_address)
    ip_address(master_address)
except ValueError:
    raise SystemExit(
        "Step 3.7 TP2 requires exact Controller rendezvous, ranks, and fabric"
    ) from None

for path in (
    MODEL / "config.json",
    MODEL / "model.safetensors.index.json",
    MODEL / "hf_quant_config.json",
    MODEL / "chat_template.jinja",
):
    if not path.is_file():
        raise SystemExit(f"immutable model artifact is missing: {path}")

# Placement transport is the only wrapper-owned launch policy. Every engine
# option comes from the authored Recipe and its chosen options.
arguments.extend(("--master-addr", master_address, "--master-port", master_port))
os.environ["VLLM_HOST_IP"] = local_address
os.environ["MASTER_ADDR"] = master_address
os.environ["MASTER_PORT"] = master_port

vllm = "/usr/local/bin/vllm"
if not (Path(vllm).is_file() and os.access(vllm, os.X_OK)):
    raise SystemExit("the pinned vLLM executable is missing")
os.execv(vllm, (vllm, *arguments))
