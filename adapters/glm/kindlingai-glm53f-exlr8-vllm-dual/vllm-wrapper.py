#!/usr/bin/env python3
"""Bind Vonk placement to the kindlingai GLM 5.3 Flash launch: vLLM on mentat in place of Ray.

mentat is the creator's Ray replacement: ``ray`` is a symlink to ``mentatd``, whose daemon runs in the
rank 0 container on the rendezvous port, every rank's agent registers with it, and vLLM's Ray executor
talks to the shim. Nothing here needs a host daemon.
"""

from __future__ import annotations

import os
import re
import resource
import subprocess
import sys
import time
from ipaddress import ip_address
from pathlib import Path

GROUP = "glm53"
# What the creator's entrypoint exports for every rank (and the preflight's headroom beyond weights and KV).
OTHER_GIB = 30
# Platform-owned names a recipe cannot set; the platform's own value wins. The autotune cache stays in the
# run's fresh /tmp, as the creator wipes it at every start (a persisted one deadlocks the next boot at TP>1).
PLATFORM_STATE = {
    "TILELANG_CACHE_DIR": "/outputs/cache/tilelang",
    "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR": "/tmp/flashinfer_autotune",
}
STATE_DIRECTORIES = (
    *PLATFORM_STATE,
    "VLLM_WEIGHT_SNAPSHOT_DIR",
    "VLLM_ADAPTIVE_K_CONTROL",
    "VLLM_DRAFT_TRUNC_CONTROL",
)
GPU_TOTAL = re.compile(r"[0-9.]+/([0-9.]+) GPU")


def _value(arguments: list[str], option: str) -> str | None:
    if option not in arguments:
        return None
    index = arguments.index(option)
    if index + 1 >= len(arguments):
        raise SystemExit(f"{option} requires a value")
    return arguments[index + 1]


def _remove_option(arguments: list[str], option: str) -> str | None:
    value = _value(arguments, option)
    if value is None:
        return None
    index = arguments.index(option)
    del arguments[index : index + 2]
    return value


def _executable(*candidates: str) -> str:
    for candidate in candidates:
        if Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return candidate
    raise SystemExit(f"runtime executable is missing: {candidates}")


def _gpu_total(ray: str, address: str) -> int:
    """GPUs the group's agents registered: the figure after the slash of ``Resources: U/T GPU``."""
    try:
        result = subprocess.run(
            [ray, "status", "--address", address, "--group", GROUP],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return 0
    found = GPU_TOTAL.search(result.stdout)
    return int(float(found.group(1))) if found else 0


def _meminfo_gib(field: str) -> int | None:
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, _, rest = line.partition(":")
        if name == field and rest.split():
            return int(rest.split()[0]) >> 20
    return None


def _release_model_pages(tensor_parallel: int, kv_cache_bytes: int) -> None:
    """Drop the model files' cached pages when the free memory is short of the rank's need.

    MemAvailable counts the page cache as free, but on GB10 a CUDA allocation can fail before the kernel
    reclaims it (NCCL init died out of memory with ~113 GB cached after a weight copy). The files are
    read-only mounts; dropping their own pages needs no privilege.
    """
    free = _meminfo_gib("MemFree")
    if free is None:
        return
    models = [path for path in Path("/models").rglob("*") if path.is_file()]
    snapshots = Path(os.environ.get("VLLM_WEIGHT_SNAPSHOT_DIR", "/nonexistent"))
    weights_gib = sum(path.stat().st_size for path in models) >> 30
    if free >= -(-weights_gib // tensor_parallel) + (kv_cache_bytes >> 30) + OTHER_GIB:
        return
    files = [*models, *(path for path in snapshots.rglob("*") if path.is_file())]
    for path in files:
        try:
            descriptor = os.open(path, os.O_RDONLY)
        except OSError:
            continue
        try:
            os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(descriptor)


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

_remove_option(arguments, "--nnodes")
_remove_option(arguments, "--node-rank")
if "--headless" in arguments:
    arguments.remove("--headless")

# A crash must not dump core: the engine maps ~90 GiB and a dump once wedged a box until a power cycle.
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

# Every rank's agent is a node of its own, identified by its fabric address.
os.environ["VLLM_HOST_IP"] = local_address
os.environ["MENTAT_NODE_IP"] = local_address
os.environ["MENTAT_GROUP"] = GROUP
os.environ["CONTAINER_NAME"] = GROUP
os.environ["MENTAT_ANNOUNCE_PORT"] = "0"
os.environ["MENTAT_ANNOUNCE_ADDRS"] = local_address

for name, default in PLATFORM_STATE.items():
    os.environ.setdefault(name, default)
for name in STATE_DIRECTORIES:
    if os.environ.get(name):
        Path(os.environ[name]).mkdir(parents=True, exist_ok=True)

_release_model_pages(
    int(_value(arguments, "--tensor-parallel-size") or "1"),
    int(_value(arguments, "--kv-cache-memory-bytes") or "0"),
)

ray = _executable("/usr/local/bin/ray")
address = f"{master_address}:{master_port}"
if headless:
    os.environ["RAY_ADDRESS"] = address
    os.execv(ray, (ray, "start", "--block", f"--address={address}"))

local = f"127.0.0.1:{master_port}"
os.environ["RAY_ADDRESS"] = local
daemon = subprocess.Popen(
    [
        _executable("/usr/local/bin/mentatd"),
        "daemon",
        "--port",
        master_port,
        "--node-ip",
        master_address,
    ]
)
subprocess.run(
    [ray, "start", "--head", f"--address={local}", f"--node-ip-address={local_address}"],
    check=True,
    timeout=120,
)
deadline = time.monotonic() + 900
while _gpu_total(ray, local) < int(node_count):
    if daemon.poll() is not None:
        raise SystemExit("the mentat daemon exited")
    if time.monotonic() >= deadline:
        raise SystemExit("the mentat group did not reach the declared GPU count")
    time.sleep(2)

vllm = _executable("/usr/local/bin/vllm")
os.execv(vllm, (vllm, *arguments))
