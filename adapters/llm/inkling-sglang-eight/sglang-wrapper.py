#!/usr/bin/env python3
"""Fail closed around SGLang's native multi-node Inkling launcher."""

from __future__ import annotations

import os
import sys
from ipaddress import ip_address


def _argument(arguments: list[str], name: str) -> str:
    try:
        value = arguments[arguments.index(name) + 1]
    except (ValueError, IndexError):
        raise SystemExit(f"distributed Inkling requires {name}") from None
    return value


def main() -> None:
    arguments = sys.argv[1:]
    if arguments[:2] != ["--model-path", "/models"]:
        raise SystemExit("this adapter only serves the immutable /models Inkling snapshot")
    nnodes = _argument(arguments, "--nnodes")
    node_rank = _argument(arguments, "--node-rank")
    dist_init = _argument(arguments, "--dist-init-addr")
    if nnodes != "8" or not node_rank.isdigit() or not 0 <= int(node_rank) < 8:
        raise SystemExit("the flagship Inkling topology requires exactly eight Spark ranks")

    required = (
        "VONK_LOCAL_ADDR",
        "VONK_MASTER_ADDR",
        "VONK_MASTER_PORT",
        "NCCL_SOCKET_IFNAME",
        "NCCL_IB_HCA",
        "NCCL_IB_GID_INDEX",
        "TP_SOCKET_IFNAME",
        "GLOO_SOCKET_IFNAME",
    )
    if any(not os.environ.get(name) for name in required):
        raise SystemExit("distributed Inkling requires complete rendezvous and RoCE settings")
    try:
        ip_address(os.environ["VONK_LOCAL_ADDR"])
        ip_address(os.environ["VONK_MASTER_ADDR"])
        port = int(os.environ["VONK_MASTER_PORT"])
        if not 1024 <= port <= 65535:
            raise ValueError
    except ValueError:
        raise SystemExit("distributed Inkling received an invalid rendezvous address") from None
    expected_dist_init = f'{os.environ["VONK_MASTER_ADDR"]}:{os.environ["VONK_MASTER_PORT"]}'
    if dist_init == "VONK_MASTER_ADDR:VONK_MASTER_PORT":
        arguments[arguments.index("--dist-init-addr") + 1] = expected_dist_init
    elif dist_init != expected_dist_init:
        raise SystemExit("SGLang dist-init address does not match Vonk placement authority")

    os.environ["SGLANG_HOST_IP"] = os.environ["VONK_LOCAL_ADDR"]
    os.execv(
        sys.executable,
        (sys.executable, "-m", "sglang.launch_server", *arguments),
    )


if __name__ == "__main__":
    main()
