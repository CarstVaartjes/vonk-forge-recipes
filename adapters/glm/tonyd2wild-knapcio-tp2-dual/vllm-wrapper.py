#!/usr/bin/env python3
"""Bind Controller two-phase placement to knapcio's GLM 5.3 Flash NVFP4 DFlash2 stack on two Sparks (tonyd2wild's port).

The stack serves a derived target ("lossless8": the nvidia NVFP4 checkpoint with its dense projections re-encoded in
8 bit) and a derived FP8 drafter, both built on CPU from the immutable Models by the kit's own scripts
(docs/weights.md of knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4 at 770d115). This wrapper builds the two views once per
input set into the persistent cache, reuses a complete view and rebuilds a malformed or interrupted one, then hands
off to vLLM with the placement the Controller supplies.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from ipaddress import ip_address
from pathlib import Path

TARGET = Path("/models/target")
DRAFTER = Path("/models/drafter")
SCRIPTS = Path("/overlay/scripts")
CACHE = Path("/outputs/cache/knapcio")
VIEW_KEY_FILES = (
    (TARGET, "config.json"),
    (TARGET, "model.safetensors.index.json"),
    (TARGET, "hf_quant_config.json"),
    (DRAFTER, "config.json"),
)
# The kit's scripts/build_lossless8.sh uses hardlinks for its zero-copy layout. The Models are read-only mounts that
# hardlinks cannot cross, so a source under /models is linked with a symlink to the real shard file; every other link
# (the overlay shard, which lives beside the view) stays a real hardlink, so it survives the staging directory's rename.
MODELS = Path("/models")
SYMLINK_RUNNER = (
    "import os, runpy, sys\n"
    "link = os.link\n"
    "def link_or_symlink(src, dst, **kw):\n"
    "    real = os.path.realpath(src)\n"
    "    if real.startswith({models!r}):\n"
    "        os.symlink(real, dst)\n"
    "    else:\n"
    "        link(src, dst, **kw)\n"
    "os.link = link_or_symlink\n"
    "sys.argv = sys.argv[1:]\n"
    "runpy.run_path(sys.argv[0], run_name='__main__')\n"
)


def _value(arguments: list[str], option: str) -> str | None:
    if option not in arguments:
        return None
    index = arguments.index(option)
    if index + 1 >= len(arguments):
        raise SystemExit(f"{option} requires a value")
    return arguments[index + 1]


def _view_key() -> str:
    digest = hashlib.sha256()
    for directory, name in VIEW_KEY_FILES:
        digest.update(name.encode())
        digest.update((directory / name).read_bytes())
    digest.update(str((DRAFTER / "model.safetensors").stat().st_size).encode())
    for script in ("glm_quant_mix.py", "drafter_fp8.py"):
        digest.update((SCRIPTS / script).read_bytes())
    return digest.hexdigest()[:16]


def _run(*command: str) -> None:
    print("+", *command, flush=True)
    subprocess.run(command, check=True)


def _complete(view: Path) -> bool:
    names = [
        "lossless8/model.safetensors.index.json",
        "lossless8/config.json",
        "lossless8/qmix-manifest.json",
        "draft/model.safetensors",
        "draft/config.json",
    ]
    try:
        # The overlay shard carries the re-encoded tensors; a view whose link to it is missing or dangling is broken.
        names.append("lossless8/" + json.loads((view / names[2]).read_bytes())["overlay"])
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return all((view / name).is_file() for name in names)


def _publish_link(name: str, destination: Path) -> None:
    link = CACHE / name
    temporary = CACHE / f".{name}.{os.getpid()}"
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(destination)
    os.replace(temporary, link)


def _prepare_proxy() -> None:
    # The RDMA proxy library is prebuilt in the image; the loader wants a directory it may also compile into.
    prebuilt = Path("/opt/glm-roce-b12x/proxy-cache")
    proxy = CACHE / "roce-proxy"
    if not proxy.is_dir() or not any(proxy.iterdir()):
        staging = CACHE / f".tmp-roce-proxy-{os.getpid()}"
        shutil.copytree(prebuilt, staging)
        shutil.rmtree(proxy, ignore_errors=True)
        os.replace(staging, proxy)
    os.environ["B12X_ROCE_CACHE_DIR"] = str(proxy)


def _prepare_views() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    for stale in CACHE.glob(".tmp-*"):
        shutil.rmtree(stale, ignore_errors=True)
    view = CACHE / _view_key()
    if view.exists() and not _complete(view):
        shutil.rmtree(view)
    if not view.exists():
        staging = CACHE / f".tmp-{view.name}-{os.getpid()}"
        python = sys.executable
        quant_mix = str(SCRIPTS / "glm_quant_mix.py")
        staging.mkdir()
        plan = staging / "plan-lossless8.json"
        _run(python, quant_mix, "plan", "--src", str(TARGET), "--arm", "lossless8", "--out", str(plan))
        _run(
            python, quant_mix, "quant", "--src", str(TARGET), "--plan", str(plan), "--arm", "lossless8",
            "--out", str(staging / "overlay"), "--skip-bf16", "--threads", "4",
        )
        runner = SYMLINK_RUNNER.format(models=str(MODELS) + os.sep)
        _run(
            python, "-c", runner, quant_mix, "assemble", "--base", str(TARGET),
            "--overlay", str(staging / "overlay"), "--dst", str(staging / "lossless8"),
        )
        _run(python, str(SCRIPTS / "drafter_fp8.py"), str(DRAFTER), str(staging / "draft"))
        if not _complete(staging):
            raise SystemExit("the derived target and drafter views are incomplete")
        os.replace(staging, view)
    _publish_link("target", view / "lossless8")
    _publish_link("draft", view / "draft")


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
        "GLM 5.3 Flash knapcio TP2 requires exact Controller rendezvous, ranks, and fabric"
    ) from None

for path in (
    TARGET / "config.json",
    TARGET / "model.safetensors.index.json",
    TARGET / "hf_quant_config.json",
    TARGET / "chat_template.jinja",
    DRAFTER / "config.json",
    DRAFTER / "model.safetensors",
):
    if not path.is_file():
        raise SystemExit(f"immutable model artifact is missing: {path}")

if str(TARGET) not in arguments:
    raise SystemExit("the immutable /models/target checkpoint argument is required")

_prepare_views()
_prepare_proxy()

# Placement transport is the wrapper-owned launch policy. The kit keeps RoCEnante on the same HCA set as NCCL
# (profiles/current.env: B12X_ROCE_HCA=$IB_HCA); NCCL_IB_HCA carries the Controller's names, with the exact-match "=".
arguments[arguments.index(str(TARGET))] = str(CACHE / "target")
arguments.extend(("--master-addr", master_address, "--master-port", master_port))
os.environ["VLLM_HOST_IP"] = local_address
os.environ["MASTER_ADDR"] = master_address
os.environ["MASTER_PORT"] = master_port
os.environ["B12X_ROCE_HCA"] = os.environ["NCCL_IB_HCA"].lstrip("=")
# The kit puts its overlay tree on PYTHONPATH (sitecustomize.py installs the hooks in every Python process).
os.environ["PYTHONPATH"] = os.pathsep.join(
    ["/overlay/overlay", *filter(None, [os.environ.get("PYTHONPATH")])]
)

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
