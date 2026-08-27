#!/usr/bin/env python3
"""Bind Vonk placement to Drowzeys' native-mp 1M GLM 5.3 launch."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from ipaddress import ip_address
from pathlib import Path


MODEL_SOURCE = Path("/models")
MODEL_VIEW = Path("/outputs/glm53-index-topk-2044")


def _prepare_model_view(source: Path, target: Path) -> Path:
    """Mirror the checkpoint with only index_topk patched for B12X."""
    source_config = source / "config.json"
    if not source_config.is_file():
        raise SystemExit(f"model config is missing: {source_config}")
    config = json.loads(source_config.read_text(encoding="utf-8"))
    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        raise SystemExit("GLM 5.3 text_config is missing")
    original = text_config.get("index_topk")
    if original not in {2044, 2048}:
        raise SystemExit(f"unexpected GLM 5.3 index_topk: {original!r}")
    text_config["index_topk"] = 2044

    target.mkdir(parents=True, exist_ok=True)
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        destination = target / relative
        if relative == Path("config.json"):
            continue
        if path.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
            continue
        if destination.exists() or destination.is_symlink():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(path)
    temporary = target / ".config.json.tmp"
    temporary.write_text(
        json.dumps(config, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target / "config.json")
    return target


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
        mechanism not in {"mp", "ray"}
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

nccl = "/opt/vonk/lib/libnccl.so.2"
if not os.path.isfile(nccl):
    raise SystemExit("the pinned NCCL runtime is missing from the adapter image")
os.environ["LD_PRELOAD"] = nccl
os.environ["VLLM_HOST_IP"] = local_address
os.environ["MASTER_ADDR"] = master_address
os.environ["MASTER_PORT"] = master_port
os.environ["KV_FP8_ROPE"] = "1"
os.environ["GLM53_PAD_ROPE"] = "auto"
os.environ["VLLM_PLUGINS"] = ""
os.environ["VLLM_ENABLE_CUDA_COMPATIBILITY"] = "0"
os.environ["VLLM_ENGINE_READY_TIMEOUT_S"] = "3600"
os.environ["DG_JIT_CACHE_DIR"] = "/outputs/cache/deep_gemm_nvdev"

if str(MODEL_SOURCE) not in arguments:
    raise SystemExit("the immutable /models checkpoint argument is required")
model_index = arguments.index(str(MODEL_SOURCE))
arguments[model_index] = str(_prepare_model_view(MODEL_SOURCE, MODEL_VIEW))
if "--attention-backend" not in arguments:
    arguments.extend(("--attention-backend", "B12X_MLA_SPARSE"))

if mechanism == "ray":
    _remove_option(arguments, "--nnodes")
    _remove_option(arguments, "--node-rank")
    if "--headless" in arguments:
        arguments.remove("--headless")
    ray = _executable("/opt/vllm/.venv/bin/ray", "/usr/local/bin/ray")
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
    python = _executable("/opt/vllm/.venv/bin/python", "/usr/local/bin/python3")
    deadline = time.monotonic() + 900
    while _alive_ray_nodes(python, address) != int(node_count):
        if time.monotonic() >= deadline:
            raise SystemExit("Ray cluster did not reach the declared node count")
        time.sleep(2)
else:
    arguments.extend(("--master-addr", master_address, "--master-port", master_port))

vllm = _executable("/opt/vllm/.venv/bin/vllm", "/usr/local/bin/vllm")
os.execv(vllm, (vllm, *arguments))
