#!/usr/bin/env python3
"""Bind Controller placement to christopherowen's spark-ds41f ``vllm serve`` launch on three Sparks.

The Controller renders vLLM's native multi-node launch (``--distributed-executor-backend mp``, ``--nnodes 3``, ``--node-rank``,
``--headless`` on the workers), which is spark-ds41f's own (``bin/spark``'s ``expected_command``). The wrapper checks the Controller
rendezvous and fabric, pins the node address, makes the engine's cache directories and execs vLLM. Engine options come from the Recipe;
checkpoint I/O (``--load-format direct``) is the creator's, served by the image's POSIX stand-in for io_uring.
"""

from __future__ import annotations

import os
import shutil
import sys
from ipaddress import ip_address
from pathlib import Path

VLLM_PATH = "/usr/local/bin:/usr/bin:/opt/spark-ds41f/candidate/vllm"
NODES = "3"
CACHE_VARIABLES = (
    "HOME",
    "XDG_CACHE_HOME",
    "VLLM_CACHE_ROOT",
    "CUDA_CACHE_PATH",
    "B12X_COMPILE_CACHE_DIR",
)


def _value(arguments: list[str], option: str) -> str | None:
    if option not in arguments:
        return None
    index = arguments.index(option)
    if index + 1 >= len(arguments):
        raise SystemExit(f"{option} requires a value")
    return arguments[index + 1]


def main() -> None:
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
            "the three-Spark launch requires exact Controller rendezvous, ranks, and fabric"
        ) from None
    for name in CACHE_VARIABLES:
        if os.environ.get(name):
            Path(os.environ[name]).mkdir(parents=True, exist_ok=True)
    # Placement transport is the only wrapper-owned launch policy.
    os.environ["VLLM_HOST_IP"] = local_address
    os.environ["MASTER_ADDR"] = master_address
    os.environ["MASTER_PORT"] = master_port
    arguments.extend(("--master-addr", master_address, "--master-port", master_port))
    vllm = shutil.which("vllm", path=VLLM_PATH)
    if vllm is None:
        raise SystemExit("the pinned vLLM executable is missing")
    os.execv(vllm, (vllm, *arguments))


if __name__ == "__main__":
    main()
