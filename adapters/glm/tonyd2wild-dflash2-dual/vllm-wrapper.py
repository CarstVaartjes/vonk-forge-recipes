#!/usr/bin/env python3
"""Bind Controller two-phase placement to the audited DFlash2 TP2 profile."""

from __future__ import annotations

import json
import os
import sys
from ipaddress import ip_address
from pathlib import Path


TARGET = Path("/models/target")
DRAFTER = Path("/models/drafter")


def _value(arguments: list[str], option: str) -> str | None:
    if option not in arguments:
        return None
    index = arguments.index(option)
    if index + 1 >= len(arguments):
        raise SystemExit(f"{option} requires a value")
    return arguments[index + 1]


def _require_value(arguments: list[str], option: str, expected: str) -> None:
    value = _value(arguments, option)
    if value != expected:
        raise SystemExit(f"{option} must be {expected!r}, got {value!r}")


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
        "DFlash2 TP2 requires exact Controller rendezvous, ranks, and fabric"
    ) from None

for path in (
    TARGET / "config.json",
    TARGET / "model.safetensors.index.json",
    TARGET / "model_mtp.safetensors",
    TARGET / "chat_template.thinking-off.jinja",
    DRAFTER / "config.json",
    DRAFTER / "model.safetensors",
):
    if not path.is_file():
        raise SystemExit(f"immutable model artifact is missing: {path}")

if str(TARGET) not in arguments:
    raise SystemExit("the immutable /models/target checkpoint argument is required")
_require_value(arguments, "--gpu-memory-utilization", "0.85")
_require_value(arguments, "--max-model-len", "262144")
_require_value(arguments, "--max-num-seqs", "6")
_require_value(arguments, "--max-num-batched-tokens", "8192")
_require_value(arguments, "--block-size", "2304")
_require_value(arguments, "--moe-backend", "marlin")
_require_value(arguments, "--kv-cache-dtype", "fp8_e4m3")
_require_value(arguments, "--kv-cache-memory", "3221225472")
_require_value(
    arguments,
    "--chat-template",
    "/models/target/chat_template.thinking-off.jinja",
)
_require_value(arguments, "--tool-call-parser", "glm47")
_require_value(arguments, "--reasoning-parser", "glm45")
if "--enforce-eager" not in arguments or "--enable-auto-tool-choice" not in arguments:
    raise SystemExit("the audited eager tool-use profile is required")

try:
    speculative = json.loads(_value(arguments, "--speculative-config") or "")
    template_kwargs = json.loads(
        _value(arguments, "--default-chat-template-kwargs") or ""
    )
except json.JSONDecodeError as error:
    raise SystemExit("DFlash2 and thinking-off arguments must contain JSON") from error
if speculative != {
    "method": "dflash",
    "model": str(DRAFTER),
    "num_speculative_tokens": 7,
}:
    raise SystemExit("the exact DFlash2 K7 specification is required")
if template_kwargs != {"enable_thinking": False}:
    raise SystemExit("the thinking-off template contract is required")

arguments.extend(("--master-addr", master_address, "--master-port", master_port))
os.environ["VLLM_HOST_IP"] = local_address
os.environ["MASTER_ADDR"] = master_address
os.environ["MASTER_PORT"] = master_port

vllm = next(
    (
        candidate
        for candidate in ("/usr/local/bin/vllm", "/opt/vllm/.venv/bin/vllm")
        if Path(candidate).is_file() and os.access(candidate, os.X_OK)
    ),
    None,
)
if vllm is None:
    raise SystemExit("the pinned vLLM executable is missing")
os.execv(vllm, (vllm, *arguments))
