#!/usr/bin/env python3
"""Bind Controller placement to the M3 TP3 Ray launch of the chthonic image."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from ipaddress import ip_address
from pathlib import Path

NCCL = "/opt/nccl230/build/lib/libnccl.so.2"
TARGET = Path("/models/target")
OBJECT_STORE_BYTES = "1073741824"


def _pop_value(arguments: list[str], option: str) -> str | None:
    if option not in arguments:
        return None
    index = arguments.index(option)
    if index + 1 >= len(arguments):
        raise SystemExit(f"{option} requires a value")
    value = arguments[index + 1]
    del arguments[index : index + 2]
    return value


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
        [python, "-c", program],
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
node_count = _pop_value(arguments, "--nnodes")
node_rank = _pop_value(arguments, "--node-rank")
_pop_value(arguments, "--master-addr")
_pop_value(arguments, "--master-port")
headless = "--headless" in arguments
if headless:
    arguments.remove("--headless")

# The image bakes an older NCCL through LD_PRELOAD and three path variables. A
# baked LD_PRELOAD overrides a symlink swap, so upstream re-points all of them
# to the host-built v2.30u1 before starting (README "Fix 2").
if not os.path.isfile(NCCL):
    raise SystemExit("the pinned NCCL v2.30u1 runtime is missing from the image")
for name in ("VLLM_NCCL_SO_PATH", "NCCL_LOCAL_INFERENCE_PATH", "NCCL_PR2127_PATH"):
    os.environ.pop(name, None)
os.environ["LD_PRELOAD"] = NCCL
os.environ["LD_LIBRARY_PATH"] = "/opt/nccl230/build/lib:" + os.environ.get(
    "LD_LIBRARY_PATH", ""
)

if node_count != "3":
    raise SystemExit("M3 TP3 requires exactly three nodes")
local_address = os.environ.get("VONK_LOCAL_ADDR")
master_address = os.environ.get("VONK_MASTER_ADDR")
master_port = os.environ.get("VONK_MASTER_PORT")
fabric_names = (
    "NCCL_SOCKET_IFNAME",
    "NCCL_IB_HCA",
    "TP_SOCKET_IFNAME",
    "GLOO_SOCKET_IFNAME",
)
try:
    if (
        node_rank is None
        or not node_rank.isascii()
        or not node_rank.isdigit()
        or not 0 <= int(node_rank) < 3
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
        "M3 TP3 requires exact Controller rendezvous, ranks, and fabric"
    ) from None

try:
    backend_index = arguments.index("--distributed-executor-backend")
    if arguments[backend_index + 1] != "ray":
        raise ValueError
except (ValueError, IndexError):
    raise SystemExit("the M3 TP3 adapter requires the Ray launch mechanism") from None

for path in (TARGET / "config.json", TARGET / "model.safetensors.index.json"):
    if not path.is_file():
        raise SystemExit(f"immutable model artifact is missing: {path}")
if str(TARGET) not in arguments:
    raise SystemExit("the immutable /models/target checkpoint argument is required")

os.environ["VLLM_HOST_IP"] = local_address
os.environ["MASTER_ADDR"] = master_address
os.environ["MASTER_PORT"] = master_port
ray = _executable("/opt/venv/bin/ray")
address = f"{master_address}:{master_port}"
if headless:
    # Upstream caps the plasma store on every node (README fix 2).
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
            "--object-store-memory",
            OBJECT_STORE_BYTES,
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
        "--num-gpus=1",
        "--object-store-memory",
        OBJECT_STORE_BYTES,
        "--include-dashboard=false",
        "--disable-usage-stats",
    ],
    check=True,
    timeout=120,
)
python = _executable("/opt/venv/bin/python")
deadline = time.monotonic() + 900
while _alive_ray_nodes(python, address) != 3:
    if time.monotonic() >= deadline:
        raise SystemExit("Ray cluster did not reach the declared node count")
    time.sleep(2)

vllm = _executable("/opt/venv/bin/vllm")
os.execv(vllm, (vllm, *arguments))
