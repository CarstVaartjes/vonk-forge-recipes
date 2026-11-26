"""Draft-side byte cuts for the DFlash2 drafter (SGLang DFLASH worker), as an import-time overlay.

    INKLING_DFLASH_BYTES=mx8,hot8                (default: unset -> install() does nothing)

Options (comma list):
  mx8      the drafter's five decoder layers (qkv, o, gate_up, down linears and the convolutions' kernel
           projections) read MXFP8 copies of their BF16 weights through inkling_draftbytes' W8 kernels (one E8M0
           scale per 32 inputs) at <= 16 rows, its CUDA weight-streaming GEMV; larger batches keep the BF16 linears
           (the Triton W8 path does not beat them there; INKLING_DRAFT_W8_MAX_M overrides the 16)
  hot8     the selector's candidate projection reads an FP8 (row-scaled) copy of the target head restricted to the
           first INKLING_DFLASH_HOT_N (default 65536) ids of INKLING_DRAFT_HOTVOCAB_FILE (frequency-ranked), instead
           of the full BF16 head (200,058 x 2,304): the top-16 candidates come from those ids only
  hot16    as hot8 with a BF16 copy of the hot rows (exact hot-row logits; twice the bytes of hot8)
  mx8w     with mx8: every row count up to INKLING_DFLASH_MX8W_MAX_M (default 256) runs FlashInfer's
           stock b12x MXFP8 GEMM (flashinfer.mm_mxfp8, backend "b12x", SM12x) on the same MXFP8 bytes (mx8's FP8
           weights, its E8M0 scales swizzled once by flashinfer.block_scale_interleave) with the activations quantized
           to MXFP8 per call (flashinfer.mxfp8_quantize): W8A8 on the linears with K and N multiples of 128 (not the
           576-row conv kernel projections, which keep mx8). GB10 microbench 2026-11-25, 30 layer linears, cold L2:
           BF16 3.33 / 3.42 / 3.48 / 4.05 ms at 8 / 32 / 64 / 192 rows -> b12x 1.90 / 1.96 / 2.00 / 2.34 ms; mx8's GEMV
           2.23 ms at 8 rows

Every option changes only what the drafter proposes: the selector's q is computed from the same candidates the
path is drawn from, and the target forward, verify and accept are untouched (draft-only, like inkling_draftbytes).
Proposals change, so acceptance (tokens per step) has to be measured against the stock drafter.
"""

import os
import sys

from . import _hook

ENV = "INKLING_DFLASH_BYTES"
PARTS = ("mx8", "hot8", "hot16", "mx8w")
DFLASH_MODULE = "sglang.srt.models.dflash"
WORKER_MODULE = "sglang.srt.speculative.dflash_worker_v2"
STATE = {"installed": False, "opts": (), "hot_n": 65536, "hot_file": None, "prepared": 0, "hot_built": 0,
         "mx8w_max_m": 256, "mx8w_linears": 0, "mx8w_calls": 0}


class DfbytesFatal(BaseException):
    """A patch target is missing or changed (BaseException: SGLang's model registry swallows Exception)."""


def _log(msg):
    sys.stderr.write(f"[inkling_dfbytes pid={os.getpid()}] {msg}\n")
    sys.stderr.flush()


def parse(env=None):
    env = os.environ if env is None else env
    raw = env.get(ENV, "").strip().lower()
    if raw in ("", "0", "off", "none", "false", "no"):
        return ()
    parts = tuple(p.strip() for p in raw.split(",") if p.strip())
    bad = [p for p in parts if p not in PARTS]
    if bad:
        raise RuntimeError(f"inkling_dfbytes: {ENV}={raw!r}: unknown {bad} (use {','.join(PARTS)})")
    if "hot8" in parts and "hot16" in parts:
        raise RuntimeError(f"inkling_dfbytes: {ENV}: hot8 and hot16 are alternatives")
    if "mx8w" in parts and "mx8" not in parts:
        raise RuntimeError(f"inkling_dfbytes: {ENV}: mx8w needs mx8 (it reuses mx8's MXFP8 bytes)")
    if "mx8w" in parts:
        raw_m = env.get("INKLING_DFLASH_MX8W_MAX_M", "").strip()
        if raw_m and (not raw_m.isdigit() or not 1 <= int(raw_m) <= 4096):
            raise RuntimeError(f"inkling_dfbytes: INKLING_DFLASH_MX8W_MAX_M={raw_m!r} (1..4096)")
    return parts


def install(env=None):
    if STATE["installed"]:
        return STATE["opts"]
    env = os.environ if env is None else env
    opts = parse(env)
    STATE["installed"] = True
    STATE["opts"] = opts
    if not opts:
        return opts
    if any(p.startswith("hot") for p in opts):
        path = env.get("INKLING_DRAFT_HOTVOCAB_FILE", "").strip()
        if not path:
            raise RuntimeError("inkling_dfbytes: hot8/hot16 need INKLING_DRAFT_HOTVOCAB_FILE (ranked token ids)")
        raw_n = env.get("INKLING_DFLASH_HOT_N", "").strip()
        n = int(raw_n) if raw_n else 65536
        if not 1024 <= n <= 200058:
            raise RuntimeError(f"inkling_dfbytes: INKLING_DFLASH_HOT_N={n} outside 1024..200058")
        STATE.update(hot_file=path, hot_n=n)
    if "mx8w" in opts:
        raw_m = env.get("INKLING_DFLASH_MX8W_MAX_M", "").strip()
        STATE["mx8w_max_m"] = int(raw_m) if raw_m else 256
    from . import patches

    patches.configure_draftbytes(opts)
    _hook.when_imported(DFLASH_MODULE, patches.patch_dflash)
    _hook.when_imported(WORKER_MODULE, patches.patch_worker)
    _log(f"armed: {','.join(opts)}" + (f" (hot set: first {STATE['hot_n']} ids)" if STATE["hot_file"] else ""))
    return opts
