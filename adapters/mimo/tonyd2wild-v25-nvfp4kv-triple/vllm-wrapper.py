#!/usr/bin/env python3
"""Bind Controller three-node placement to the MiMo-V2.5 Ray TP3 profile."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from ipaddress import ip_address
from pathlib import Path

TARGET = Path("/models/target")
# Upstream caps the Ray object store at 1 GiB: an uncapped store takes unified
# memory that the 1M-token KV pool needs.
OBJECT_STORE_BYTES = "1073741824"


def _value(arguments: list[str], option: str) -> str | None:
    if option not in arguments:
        return None
    index = arguments.index(option)
    if index + 1 >= len(arguments):
        raise SystemExit(f"{option} requires a value")
    return arguments[index + 1]


def _remove_option(arguments: list[str], option: str) -> None:
    index = arguments.index(option)
    del arguments[index : index + 2]


def _executable(*candidates: str) -> str:
    for candidate in candidates:
        if Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return candidate
    raise SystemExit(f"runtime executable is missing: {candidates}")


def _alive_ray_nodes(python: str, address: str) -> int:
    program = (
        "import ray; "
        f"ray.init(address={address!r}, logging_level='ERROR'); "
        "print('VONK_ALIVE=' + str(sum(n.get('Alive') is True for n in ray.nodes()))); "
        "ray.shutdown()"
    )
    result = subprocess.run(
        [python, "-c", program], capture_output=True, text=True, timeout=60
    )
    for line in result.stdout.splitlines():
        if line.startswith("VONK_ALIVE="):
            return int(line.removeprefix("VONK_ALIVE="))
    return 0


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
        mechanism != "ray"
        or node_count != "3"
        or node_rank not in {"0", "1", "2"}
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
        "MiMo V2.5 TP3 requires exact Controller rendezvous, ranks, and fabric"
    ) from None

for path in (
    TARGET / "config.json",
    TARGET / "model.safetensors.index.json",
    TARGET / "model-mtp.safetensors",
    TARGET / "model-inputscales.safetensors",
    TARGET / "tokenizer.json",
):
    if not path.is_file():
        raise SystemExit(f"immutable model artifact is missing: {path}")

if str(TARGET) not in arguments:
    raise SystemExit("the immutable /models/target checkpoint argument is required")

# Placement transport is the only wrapper-owned launch policy. All engine
# options come from the authored Recipe and remain overrideable by the runtime
# argv. The one exception is the recipe's speculative-decoding option, which
# cannot remove an argument, so VONK_MIMO_SPEC=off drops --speculative-config.
if os.environ.get("VONK_MIMO_SPEC") == "off" and "--speculative-config" in arguments:
    index = arguments.index("--speculative-config")
    del arguments[index : index + 2]

# The published upstream image bakes one host's addresses into its environment.
for stale in ("RAY_NODE_IP_ADDRESS", "RAY_OVERRIDE_NODE_IP_ADDRESS"):
    os.environ.pop(stale, None)
os.environ["VLLM_HOST_IP"] = local_address
os.environ["MASTER_ADDR"] = master_address
os.environ["MASTER_PORT"] = master_port

_remove_option(arguments, "--nnodes")
_remove_option(arguments, "--node-rank")
if headless:
    arguments.remove("--headless")
ray = _executable("/usr/local/bin/ray")
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
            "--num-gpus=1",
            f"--object-store-memory={OBJECT_STORE_BYTES}",
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
        "--num-gpus=1",
        f"--object-store-memory={OBJECT_STORE_BYTES}",
        "--disable-usage-stats",
    ],
    check=True,
    timeout=120,
)
python = _executable("/usr/bin/python3", "/usr/local/bin/python3")
deadline = time.monotonic() + 900
while _alive_ray_nodes(python, address) != int(node_count):
    if time.monotonic() >= deadline:
        raise SystemExit("Ray cluster did not reach the declared node count")
    time.sleep(2)

vllm = _executable("/usr/local/bin/vllm")
os.execv(vllm, (vllm, *arguments))
