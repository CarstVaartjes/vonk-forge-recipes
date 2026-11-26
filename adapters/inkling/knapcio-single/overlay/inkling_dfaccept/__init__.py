"""Sparse-q fused sampled accept for the DFlash2 selector draft (SPEC=dflash2), as an import-time overlay.

    INKLING_DFLASH_FUSED_ACCEPT=1          (default: unset -> install() does nothing)
    INKLING_DFLASH_FUSED_MIN_BS=16         smallest batch (requests) served fused (default 16)

DFlashWorkerV2._selector_sampling_accept is served by core.accept_core_sparse (no dense [bs, 7, V] q, no dense
[bs, 8, V] p) on eligible calls: plain temperature sampling for every request (no top-k / top-p / min-p, every
temperature > 0), no verify cutoff (the verify-cap port's calls take the stock path), CUDA, TP1, not under CUDA-graph
capture. Anything else takes the stock path. See core.py for the
algorithm and the exactness argument (same law as the stock accept).
"""

import os
import sys

from . import _hook

ENV = "INKLING_DFLASH_FUSED_ACCEPT"
WORKER = "sglang.srt.speculative.dflash_worker_v2"
STATE = {"installed": False, "on": False, "calls": 0, "fused": 0}


class DfacceptFatal(BaseException):
    """A patch target is missing or changed."""


def _log(msg):
    sys.stderr.write(f"[inkling_dfaccept pid={os.getpid()}] {msg}\n")
    sys.stderr.flush()


def enabled(env=None):
    env = os.environ if env is None else env
    v = env.get(ENV, "").strip().lower()
    if v in ("", "0", "off", "false", "no", "none"):
        return False
    if v in ("1", "on", "true", "yes"):
        return True
    raise RuntimeError(f"inkling_dfaccept: {ENV}={v!r} (use 1 or 0)")


def install(env=None):
    if STATE["installed"]:
        return STATE["on"]
    STATE["installed"] = True
    if not enabled(env):
        return False
    from . import patches

    _hook.when_imported(WORKER, patches.patch_worker)
    STATE["on"] = True
    _log("armed: DFLASH selector sampled accept from the logits with sparse q (eligible calls)")
    return True
