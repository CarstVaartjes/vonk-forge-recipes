#!/usr/bin/env python3
"""Run 0xSero's DeepSeek-V4.1-Flash EXL3 vLLM through Controller-managed topology."""

from __future__ import annotations

import json
import os
import sys
from ipaddress import ip_address
from pathlib import Path

SOURCE = Path("/models")
PLAN_NAME = "J268-ho-v31-K5n.json"
CACHE = Path("/outputs/cache")


def value(arguments: list[str], option: str) -> str | None:
    if option not in arguments:
        return None
    index = arguments.index(option)
    if index + 1 >= len(arguments):
        raise SystemExit(f"{option} requires a value")
    return arguments[index + 1]


def main() -> None:
    arguments = sys.argv[1:]
    node_count = value(arguments, "--nnodes")
    node_rank = value(arguments, "--node-rank")
    backend = value(arguments, "--distributed-executor-backend")
    headless = "--headless" in arguments
    local = os.environ.get("VONK_LOCAL_ADDR")
    master = os.environ.get("VONK_MASTER_ADDR")
    port = os.environ.get("VONK_MASTER_PORT")
    fabric = (
        "NCCL_SOCKET_IFNAME",
        "NCCL_IB_HCA",
        "NCCL_IB_GID_INDEX",
        "TP_SOCKET_IFNAME",
        "GLOO_SOCKET_IFNAME",
    )
    try:
        if backend != "mp" or node_count != "2" or node_rank not in {"0", "1"}:
            raise ValueError
        if headless != (node_rank == "1") or not local or not master or not port:
            raise ValueError
        if (
            any(not os.environ.get(name) for name in fabric)
            or not port.isascii()
            or not port.isdigit()
        ):
            raise ValueError
        if not 1024 <= int(port) <= 65535:
            raise ValueError
        ip_address(local)
        ip_address(master)
    except ValueError:
        raise SystemExit(
            "DeepSeek V4.1 Flash EXL3 TP2 requires complete Controller rendezvous and fabric"
        ) from None

    # upstream launch.sh preflight: the assembled Engram shards and the plan.
    for path in (
        SOURCE / "config.json",
        SOURCE / "model.safetensors.index.json",
        SOURCE / "model-00005-of-00006.safetensors",
        SOURCE / "model-00006-of-00006.safetensors",
        SOURCE / "exl3" / "plan" / PLAN_NAME,
    ):
        if not path.is_file():
            raise SystemExit(f"immutable model artifact is missing: {path}")

    # upstream launch.sh: the plan names bank directories relative to exl3/; the
    # checkpoint is mounted read-only, so the rewritten copy lives in the cache.
    plan = json.loads((SOURCE / "exl3" / "plan" / PLAN_NAME).read_text())
    plan["qdir"] = ":".join(
        f"{SOURCE}/exl3/{name.split('/')[-1]}" for name in plan["qdir"].split(":")
    )
    plans = CACHE / "plans"
    plans.mkdir(parents=True, exist_ok=True)
    (plans / PLAN_NAME).write_text(json.dumps(plan))
    for directory in ("b12x", "b12x-roce"):
        (CACHE / directory).mkdir(parents=True, exist_ok=True)

    arguments.extend(("--master-addr", master, "--master-port", port))
    os.environ["VLLM_HOST_IP"] = local
    os.environ["MASTER_ADDR"] = master
    os.environ["MASTER_PORT"] = port
    os.environ["ST_EXL3_PLAN"] = str(plans / PLAN_NAME)
    os.environ.setdefault("B12X_COMPILE_CACHE_DIR", str(CACHE / "b12x"))
    os.environ.setdefault("B12X_ROCE_CACHE_DIR", str(CACHE / "b12x-roce"))
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
        raise SystemExit("the pinned DeepSeek V4.1 vLLM executable is missing")
    os.execv(vllm, (vllm, "serve", str(SOURCE), *arguments))


if __name__ == "__main__":
    main()
