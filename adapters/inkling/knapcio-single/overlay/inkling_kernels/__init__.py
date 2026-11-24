"""Inkling-2-Mini target-side kernel overlay for SGLang on DGX Spark (GB10, sm_121).

    INKLING_KERNELS=qkvr_gemv,moe_tiles   (default: empty -> install() does nothing)

qkvr_gemv  Inkling2Attention._project_sources: per-source column-split GEMV
           instead of a 3M-row stacked projection (see qkvr_gemv.py).
           Env: INKLING_QKVR_MAX_M (16), INKLING_QKVR_TILE ("32,256,4").
moe_tiles  inkling2_grouped_sm120._tiles from INKLING_MOE_TILES (see moe_tiles.py).

install() parses the env, checks that the patch targets exist in the installed
sglang sources (without importing sglang), and attaches each patch to its target
module through a post-import hook, so it costs nothing in processes that never
import the model. It prints one log line. Any problem raises RuntimeError; a patch
whose target attribute is missing at import time makes that import fail.
"""

import importlib.util
import os
import sys

from . import _hook, moe_tiles

LEVERS = ("qkvr_gemv", "moe_tiles")

# Files (relative to the sglang package) and strings each patch relies on.
_SOURCE_CHECKS = {
    "qkvr_gemv": (
        (
            "srt/models/inkling_common/inkling2_attn.py",
            ("class Inkling2Attention(", "def _project_sources(", "def _stacked("),
        ),
        (
            "srt/layers/quantization/unquant.py",
            ("def _sm12x_bf16_gemv_eligible(",),
        ),
    ),
    "moe_tiles": (
        (
            "kernels/ops/gemm/inkling2_grouped_sm120.py",
            ("def _tiles(", "_tiles(tokens=", "group_gemm_nvfp4_nt_groupwise"),
        ),
    ),
}

_QKVR_MAX_M_LIMIT = 64

STATE = {"installed": False, "levers": (), "applied": []}


def _log(msg):
    sys.stderr.write(f"[inkling_kernels pid={os.getpid()}] {msg}\n")
    sys.stderr.flush()


def qkvr_config_from_env():
    max_m = int(os.environ.get("INKLING_QKVR_MAX_M", "16"))
    if not 1 <= max_m <= _QKVR_MAX_M_LIMIT:
        raise ValueError(f"INKLING_QKVR_MAX_M must be in [1, {_QKVR_MAX_M_LIMIT}]")
    tile = os.environ.get("INKLING_QKVR_TILE", "32,256,4")
    try:
        bn, bk, warps = (int(v) for v in tile.split(","))
    except ValueError as exc:
        raise ValueError(f"INKLING_QKVR_TILE {tile!r} is not 'BN,BK,WARPS'") from exc
    for v, name in ((bn, "BN"), (bk, "BK"), (warps, "WARPS")):
        if v <= 0 or v & (v - 1):
            raise ValueError(f"INKLING_QKVR_TILE {name}={v} must be a power of two")
    if bn < 16 or bk < 16:
        raise ValueError("INKLING_QKVR_TILE BN and BK must be >= 16")
    return {"max_m": max_m, "bn": bn, "bk": bk, "warps": warps}


def _sglang_root():
    spec = importlib.util.find_spec("sglang")  # does not execute sglang
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("inkling_kernels: the sglang package is not importable")
    return list(spec.submodule_search_locations)[0]


def _check_sources(levers):
    root = _sglang_root()
    for lever in levers:
        for rel, needles in _SOURCE_CHECKS[lever]:
            path = os.path.join(root, rel)
            try:
                with open(path, encoding="utf-8") as f:
                    text = f.read()
            except OSError as exc:
                raise RuntimeError(
                    f"inkling_kernels: {lever} patch target {path} is missing"
                ) from exc
            missing = [n for n in needles if n not in text]
            if missing:
                raise RuntimeError(
                    f"inkling_kernels: {lever} patch target {path} lacks {missing}"
                )


def _apply_qkvr(cfg):
    def callback(module):
        from . import qkvr_gemv

        qkvr_gemv.patch(module, cfg)
        STATE["applied"].append("qkvr_gemv")
        if os.environ.get("INKLING_KERNELS_VERBOSE"):
            _log(f"applied qkvr_gemv to {module.__name__}")

    _hook.when_imported("sglang.srt.models.inkling_common.inkling2_attn", callback)


def _apply_moe_tiles(rules):
    def callback(module):
        moe_tiles.patch(module, rules)
        STATE["applied"].append("moe_tiles")
        if os.environ.get("INKLING_KERNELS_VERBOSE"):
            _log(f"applied moe_tiles to {module.__name__}")

    _hook.when_imported(moe_tiles.TARGET_MODULE, callback)


def parse_levers(value):
    levers = tuple(dict.fromkeys(v.strip() for v in value.split(",") if v.strip()))
    unknown = [v for v in levers if v not in LEVERS]
    if unknown:
        raise RuntimeError(
            f"inkling_kernels: unknown INKLING_KERNELS entries {unknown}; "
            f"known: {', '.join(LEVERS)}"
        )
    return levers


def install(levers=None):
    """Activate the levers named in INKLING_KERNELS (or `levers`). Idempotent."""
    if STATE["installed"]:
        return STATE["levers"]
    if levers is None:
        levers = parse_levers(os.environ.get("INKLING_KERNELS", ""))
    else:
        levers = parse_levers(",".join(levers))
    if not levers:
        STATE["installed"] = True
        return ()
    details = []
    qkvr_cfg = rules = None
    try:
        if "qkvr_gemv" in levers:
            qkvr_cfg = qkvr_config_from_env()
            details.append(
                "qkvr_gemv(max_m={max_m},tile={bn}x{bk},warps={warps})".format(
                    **qkvr_cfg
                )
            )
        if "moe_tiles" in levers:
            spec = os.environ.get("INKLING_MOE_TILES", "")
            if not spec.strip():
                raise ValueError("moe_tiles needs INKLING_MOE_TILES")
            rules = moe_tiles.parse(spec)
            details.append(f"moe_tiles({moe_tiles.describe(rules)})")
    except ValueError as exc:
        raise RuntimeError(f"inkling_kernels: {exc}") from exc
    _check_sources(levers)
    if qkvr_cfg is not None:
        _apply_qkvr(qkvr_cfg)
    if rules is not None:
        _apply_moe_tiles(rules)
    STATE.update(installed=True, levers=levers)
    _log("enabled " + " ".join(details))
    return levers
