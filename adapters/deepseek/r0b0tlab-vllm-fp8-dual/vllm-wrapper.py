#!/usr/bin/env python3
"""Native exec wrapper binding Vonk placement rendezvous to vLLM multi-node flags."""

from __future__ import annotations

import os
import shutil
import sys
from ipaddress import ip_address

arguments = sys.argv[1:]
# The pinned image's default env omits CUDA_HOME and torch/lib (upstream launch-script fix).
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
_torch_lib = "/usr/local/lib/python3.12/dist-packages/torch/lib"
os.environ["LD_LIBRARY_PATH"] = ":".join(
    part for part in (_torch_lib, os.environ.get("LD_LIBRARY_PATH", "")) if part
)
if "--nnodes" in arguments:
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
            not local_address
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
    arguments.extend(("--master-addr", master_address, "--master-port", master_port))
vllm = shutil.which("vllm")
if vllm is None:
    raise SystemExit("the vllm executable is missing from the image")
os.execv(vllm, (vllm, *arguments))
