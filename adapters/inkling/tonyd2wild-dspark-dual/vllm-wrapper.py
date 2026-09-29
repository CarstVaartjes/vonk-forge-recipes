#!/usr/bin/env python3
"""Bind Controller two-phase placement to the Inkling Small NVFP4 DSpark TP2 profile."""

from __future__ import annotations

import os
import shutil
import sys
from ipaddress import ip_address
from pathlib import Path

TARGET = Path("/models/target")
NODES = "2"
VLLM_PATH = "/usr/local/bin:/usr/bin:/workspace/vllm"


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
        or node_count != NODES
        or node_rank is None
        or not node_rank.isascii()
        or not node_rank.isdigit()
        or not 0 <= int(node_rank) < int(NODES)
        or headless != (node_rank != "0")
        or not local_address
        or not master_address
        or not master_port
        or any(not os.environ.get(name) for name in fabric_names)
        or not os.environ["NCCL_IB_GID_INDEX"].isascii()
        or not os.environ["NCCL_IB_GID_INDEX"].isdigit()
        or not master_port.isascii()
        or not master_port.isdigit()
        or not 1024 <= int(master_port) <= 65535
    ):
        raise ValueError
    ip_address(local_address)
    ip_address(master_address)
except ValueError:
    raise SystemExit(
        "Inkling Small NVFP4 DSpark TP2 requires exact Controller rendezvous, ranks, and fabric"
    ) from None

for path in (
    TARGET / "config.json",
    TARGET / "tokenizer.json",
    Path("/models/draft/model.safetensors"),
    Path("/models/draft/dspark.py"),
):
    if not path.is_file():
        raise SystemExit(f"immutable model artifact is missing: {path}")
if str(TARGET) not in arguments:
    raise SystemExit("the immutable /models/target checkpoint argument is required")

# Placement transport is the only wrapper-owned launch policy; every engine option
# comes from the authored Recipe. The parent image bakes one host's addresses.
os.environ.pop("NODE_IP", None)
os.environ["VLLM_HOST_IP"] = local_address
os.environ["MASTER_ADDR"] = master_address
os.environ["MASTER_PORT"] = master_port
arguments.extend(("--master-addr", master_address, "--master-port", master_port))

vllm = shutil.which("vllm", path=VLLM_PATH)
if vllm is None:
    raise SystemExit("the pinned vLLM executable is missing")
os.execv(vllm, (vllm, *arguments))
