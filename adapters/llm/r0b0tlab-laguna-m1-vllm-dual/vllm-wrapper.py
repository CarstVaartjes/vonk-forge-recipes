#!/usr/bin/env python3
"""Bind Vonk placement to the r0b0tlab Laguna M.1 dual-GB10 Ray launch."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from ipaddress import ip_address


def _value(arguments: list[str], option: str) -> str | None:
    if option not in arguments:
        return None
    index = arguments.index(option)
    if index + 1 >= len(arguments):
        raise SystemExit(f"{option} requires a value")
    return arguments[index + 1]


def _remove_option(arguments: list[str], option: str) -> None:
    if option in arguments:
        index = arguments.index(option)
        del arguments[index : index + 2]


def _executable(name: str) -> str:
    path = shutil.which(name)
    if path is None or not os.access(path, os.X_OK):
        raise SystemExit(f"runtime executable is missing: {name}")
    return path


def _alive_ray_nodes(address: str) -> int:
    program = (
        "import ray; "
        f"ray.init(address={address!r}, logging_level='ERROR'); "
        "print('VONK_ALIVE=' + str(sum(n.get('Alive') is True for n in ray.nodes()))); "
        "ray.shutdown()"
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    for line in reversed(result.stdout.splitlines()):
        if line.startswith("VONK_ALIVE="):
            return int(line.split("=", 1)[1])
    return 0


arguments = sys.argv[1:]
# The pinned image's default env omits CUDA_HOME and torch/lib (upstream launch-script fix).
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
_torch_lib = "/usr/local/lib/python3.12/dist-packages/torch/lib"
os.environ["LD_LIBRARY_PATH"] = ":".join(
    part for part in (_torch_lib, os.environ.get("LD_LIBRARY_PATH", "")) if part
)
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
        mechanism != "ray"
        or node_count is None
        or node_rank is None
        or not node_count.isascii()
        or not node_count.isdigit()
        or int(node_count) < 2
        or not node_rank.isascii()
        or not node_rank.isdigit()
        or not 0 <= int(node_rank) < int(node_count)
        or headless != (int(node_rank) > 0)
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
        "distributed vLLM requires complete placement rendezvous and fabric"
    ) from None

os.environ["VLLM_HOST_IP"] = local_address
os.environ["MASTER_ADDR"] = master_address
os.environ["MASTER_PORT"] = master_port

_remove_option(arguments, "--nnodes")
_remove_option(arguments, "--node-rank")
if "--headless" in arguments:
    arguments.remove("--headless")
ray = _executable("ray")
address = f"{master_address}:{master_port}"
if headless:
    os.execv(
        ray,
        (
            ray,
            "start",
            "--address",
            address,
            "--node-ip-address",
            local_address,
            "--disable-usage-stats",
            "--block",
        ),
    )
subprocess.run(
    [
        ray,
        "start",
        "--head",
        "--node-ip-address",
        local_address,
        "--port",
        master_port,
        "--include-dashboard=false",
        "--disable-usage-stats",
        "--object-store-memory=4294967296",
    ],
    check=True,
    timeout=120,
)
deadline = time.monotonic() + 900
while _alive_ray_nodes(address) != int(node_count):
    if time.monotonic() >= deadline:
        raise SystemExit("Ray cluster did not reach the declared node count")
    time.sleep(2)

vllm = _executable("vllm")
os.execv(vllm, (vllm, *arguments))
