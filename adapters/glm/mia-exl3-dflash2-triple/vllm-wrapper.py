#!/usr/bin/env python3
"""Bind Controller placement to MiaAI-Lab's native-mp EXL3 TP3 runtime."""

from __future__ import annotations

import os
import sys
from ipaddress import ip_address
from pathlib import Path

TARGET = Path("/models/target")
DRAFTER = Path("/models/drafter")
# The DFlash2 drafter has 32 attention / 8 KV heads; neither divides by three and
# the drafter process reads the world TP even with draft_tensor_parallel_size=1,
# so upstream's start-tp3.sh serves a copy padded to 36 / 9 heads (its
# overlay/tp3/pad-tp3-config.py --tp 3, applied to the pinned drafter config).
# drafter-config-tp3.json is that result, committed; the recipe's
# speculative-config points at this working view. Nothing under /models is written.
DRAFTER_VIEW = Path("/outputs/glm53-drafter-tp3")
DRAFTER_PADDED_CONFIG = Path("/opt/glm53/drafter-config-tp3.json")


def _prepare_drafter_view(source: Path, target: Path, padded: Path) -> None:
    """Mirror the immutable drafter with only config.json replaced by the padded one."""
    text = padded.read_text(encoding="utf-8")
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
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(target / "config.json")


def _value(arguments: list[str], option: str) -> str | None:
    if option not in arguments:
        return None
    index = arguments.index(option)
    if index + 1 >= len(arguments):
        raise SystemExit(f"{option} requires a value")
    return arguments[index + 1]


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
        "EXL3 TP3 requires exact Controller rendezvous, ranks, and fabric"
    ) from None

if str(TARGET) not in arguments:
    raise SystemExit("the immutable /models/target checkpoint argument is required")
for path in (
    TARGET / "config.json",
    DRAFTER / "config.json",
    DRAFTER / "model.safetensors",
    DRAFTER_PADDED_CONFIG,
):
    if not path.is_file():
        raise SystemExit(f"immutable model artifact is missing: {path}")

speculative = _value(arguments, "--speculative-config")
if speculative is None or str(DRAFTER_VIEW) not in speculative:
    raise SystemExit(f"the speculative-config must name the drafter view {DRAFTER_VIEW}")
_prepare_drafter_view(DRAFTER, DRAFTER_VIEW, DRAFTER_PADDED_CONFIG)

# Controller owns placement transport. Recipe-authored engine arguments, including
# ordinary vLLM flags and opaque runtime extensions, pass through unchanged.
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
