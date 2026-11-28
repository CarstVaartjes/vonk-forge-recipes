#!/usr/bin/env python3
"""Bind Controller two-phase placement to KindlingAI's Nemotron 3.5 Super VL TP2 launch (compose.tp2.yaml)."""

from __future__ import annotations

import os
import shutil
import sys
from ipaddress import ip_address
from pathlib import Path

MAIN = Path("/models/main")
DRAFT = Path("/models/draft")
OWN_DIRECTORY = "/opt/vonk/bin"


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
        "Nemotron 3.5 Super VL TP2 requires exact Controller rendezvous, ranks, and fabric"
    ) from None

for path in (MAIN / "config.json", MAIN / "model.safetensors.index.json", DRAFT / "config.json"):
    if not path.is_file():
        raise SystemExit(f"immutable model artifact is missing: {path}")

if str(MAIN) not in arguments:
    raise SystemExit("the immutable /models/main checkpoint argument is required")

# Placement transport is the only wrapper-owned launch policy: the kit's compose.tp2.yaml passes
# --master-addr and --master-port and exports VLLM_HOST_IP. Every engine option, including the
# drafter, the kernels and the parsers, comes from the authored Recipe.
arguments.extend(("--master-addr", master_address, "--master-port", master_port))
os.environ["VLLM_HOST_IP"] = local_address

search_path = os.pathsep.join(
    entry
    for entry in os.environ.get("PATH", "").split(os.pathsep)
    if entry and entry != OWN_DIRECTORY
)
vllm = shutil.which("vllm", path=search_path)
if vllm is None:
    raise SystemExit("the pinned vLLM executable is missing")
os.execv(vllm, (vllm, *arguments))
