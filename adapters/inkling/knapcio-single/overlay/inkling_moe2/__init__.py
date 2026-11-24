"""Inkling-2-Mini routed-expert GEMM tiles for DSpark verify batches on one GB10 (SGLang overlay).

    INKLING_MOE2=tiles        (default: empty -> install() does nothing)

tiles   per-token-count tile tables for the expert grouped GEMMs (tiles.py): NVFP4 layers 0-19
        (inkling2_grouped_sm120._tiles, extending inkling_kernels.moe_tiles) and MXFP8 layers 20-23
        (inkling2_grouped_sm120.mxfp8_expert_matmul -> FlashInfer's tuned CuTe entry).
        Tables: INKLING_MOE2_NVFP4 / INKLING_MOE2_MXFP8 (grammar in tiles.py); unset = the defaults below,
        "none" = that half off.

Install after inkling_kernels (sitecustomize order), so the NVFP4 wrapper sits outside moe_tiles' wrapper.
install() parses the env, checks the patch targets in the installed sglang sources without importing sglang or
torch, and attaches the patch through a post-import hook. Any problem raises RuntimeError (the combined loader
then exits with code 78).
"""

import importlib.util
import os
import sys

from . import _hook, tiles

LEVERS = ("tiles",)
STATE = {"installed": False, "levers": (), "applied": []}

# Defaults: bit-identical configurations that beat the shipped choice on GB10 by the bench (README.md).
DEFAULT_NVFP4 = "9-64@2304:swap_ab=1,tile_n=32,tile_k=256;65-128:swap_ab=1,tile_n=32,tile_k=256"
DEFAULT_MXFP8 = "512-1536@2304x1152:tile_m=32,tile_n=128;1024-1536@2304x2304:tile_m=32,tile_n=128"

_SOURCE_CHECKS = {
    "tiles": (
        ("kernels/ops/gemm/inkling2_grouped_sm120.py", tiles.SOURCE_NEEDLES),
        (
            "srt/models/inkling_common/inkling2_moe.py",
            ("grouped_sm120.nvfp4_expert_matmul(", "grouped_sm120.mxfp8_expert_matmul(", "tokens=tokens,"),
        ),
    ),
}


def _log(msg):
    sys.stderr.write(f"[inkling_moe2 pid={os.getpid()}] {msg}\n")
    sys.stderr.flush()


def parse_levers(value):
    levers = tuple(dict.fromkeys(v.strip() for v in value.split(",") if v.strip()))
    unknown = [v for v in levers if v not in LEVERS]
    if unknown:
        raise RuntimeError(f"inkling_moe2: unknown INKLING_MOE2 entries {unknown}; known: {', '.join(LEVERS)}")
    return levers


def tables_from_env():
    nv = os.environ.get("INKLING_MOE2_NVFP4", DEFAULT_NVFP4).strip()
    mx = os.environ.get("INKLING_MOE2_MXFP8", DEFAULT_MXFP8).strip()
    nv = "" if nv.lower() == "none" else nv
    mx = "" if mx.lower() == "none" else mx
    try:
        return tiles.parse_nvfp4(nv), tiles.parse_mxfp8(mx)
    except ValueError as exc:
        raise RuntimeError(f"inkling_moe2: {exc}") from exc


def _sglang_root():
    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("inkling_moe2: the sglang package is not importable")
    return list(spec.submodule_search_locations)[0]


def check_sources(levers, root=None):
    root = root or _sglang_root()
    for rel, needles in (item for lever in levers for item in _SOURCE_CHECKS[lever]):
        path = os.path.join(root, rel)
        try:
            with open(path, encoding="utf-8") as f:
                text = f.read()
        except OSError as exc:
            raise RuntimeError(f"inkling_moe2: patch target {path} is missing") from exc
        missing = [n for n in needles if n not in text]
        if missing:
            raise RuntimeError(f"inkling_moe2: patch target {path} lacks {missing}")


def install(levers=None, root=None):
    """Activate the levers named in INKLING_MOE2 (or `levers`). Idempotent."""
    if STATE["installed"]:
        return STATE["levers"]
    levers = parse_levers(os.environ.get("INKLING_MOE2", "") if levers is None else ",".join(levers))
    if not levers:
        STATE["installed"] = True
        return ()
    details = []
    if "tiles" in levers:
        nv, mx = tables_from_env()
        if not nv and not mx:
            raise RuntimeError("inkling_moe2: tiles is on but both tables are empty")
        details.append(f"tiles({tiles.describe(nv, mx)})")
    check_sources(levers, root)
    verbose = os.environ.get("INKLING_MOE2_VERBOSE")

    if "tiles" in levers:
        def on_grouped(module):
            done = tiles.patch(module, nv, mx)
            STATE["applied"].extend(done)
            if verbose:
                _log(f"applied {', '.join(done) or 'nothing (already patched)'} to {module.__name__}")

        _hook.when_imported(tiles.TARGET_MODULE, on_grouped)
    STATE.update(installed=True, levers=levers)
    _log("enabled " + " ".join(details))
    return levers
