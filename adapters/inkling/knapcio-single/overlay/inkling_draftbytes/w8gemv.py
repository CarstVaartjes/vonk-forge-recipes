"""`w8gemv` (EXPERIMENTAL, not recommended: on GB10 it streams at 140-200 GB/s, below `w8tile`'s Triton launch on
the narrow-N shapes; kept as a bit-exact CUDA base for later work): the draftmx8 (MXFP8) draft linears at 1-16 rows
through csrc/w8gemv.cu, a CUDA weight-streaming GEMV
with the Triton kernel's per-output arithmetic (one ascending-k MMA chain per output, Triton's kWidth-4 operand
order inside each MMA, exact dequant to BF16, FP32 accumulation, + bias, one rounding): the same bits as _w8_mm_kernel, at a higher streaming rate on the narrow-N
shapes. Built with torch.utils.cpp_extension at draft prepare (before any CUDA graph capture) into
INKLING_DRAFT_W8GEMV_CACHE (default ~/.cache/inkling-draftbytes-w8gemv/<tag>); a later start with the same
inputs only imports the built .so.

Variant (cfg = MT * 1000 + WPC * 100 + D * 10 + XD): MT n-tiles of 8 weight rows per warp, WPC warps per CTA, D
steps of 64 bytes of weights a lane keeps in flight, XD steps of activations. INKLING_DRAFT_W8GEMV_CFG forces one variant for every shape; by default
choose_cfg() picks per shape (the GB10 table below).
"""

from __future__ import annotations

import hashlib
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, "csrc", "w8gemv.cu")
NAME = "inkling_draftbytes_w8gemv"
VARIANTS = (1142, 1144, 1182, 1184, 1242, 1244, 1282, 1284, 1442, 1444, 1482, 1484)
MAX_ROWS = 16

_EXT = None
INFO: dict = {}

# (N, K) -> cfg, from the GB10 microbenchmark (c1leak track, 2026-11-22); other shapes: DEFAULT_CFG
TABLE: dict = {}
DEFAULT_CFG = 1242  # best or within 3 % of best on every draft shape at 1-16 rows (GB10, 2026-11-22)


def cuda_flags(arch: str = "121a"):
    return [f"-gencode=arch=compute_{arch},code=sm_{arch}", "-O3", "-std=c++20", "-lineinfo",
            "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__"]


def load(cache_dir: str | None = None, verbose: bool = False):
    global _EXT
    if _EXT is not None:
        return _EXT
    import torch
    from torch.utils import cpp_extension

    flags = cuda_flags()
    h = hashlib.sha256()
    with open(SOURCE, "rb") as f:
        h.update(f.read())
    h.update("\0".join(flags).encode() + f"\0{torch.__version__}\0{torch.version.cuda}".encode())
    cache_dir = cache_dir or os.environ.get("INKLING_DRAFT_W8GEMV_CACHE", "").strip() or os.path.join(
        os.path.expanduser("~"), ".cache", "inkling-draftbytes-w8gemv")
    build_dir = os.path.join(cache_dir, h.hexdigest()[:16])
    os.makedirs(build_dir, exist_ok=True)
    cached = any(f.endswith(".so") and f.startswith(NAME) for f in os.listdir(build_dir))
    t0 = time.monotonic()
    ext = cpp_extension.load(name=NAME, sources=[SOURCE], extra_cuda_cflags=flags, build_directory=build_dir,
                             verbose=verbose)
    if not callable(getattr(ext, "w8gemv", None)):
        raise RuntimeError(f"inkling_draftbytes: w8gemv extension in {build_dir} lacks w8gemv()")
    INFO.update(build_dir=build_dir, cached=cached, seconds=time.monotonic() - t0)
    _EXT = ext
    return ext


def choose_cfg(n: int, k: int, forced: int = 0) -> int:
    if forced:
        return forced
    return TABLE.get((n, k), DEFAULT_CFG)


def eligible_pack(pack) -> bool:
    """MX packs (e8m0 scales per 32) whose shape the kernel takes."""
    from .quant import SCALE_E8M0

    return (
        pack.scale_kind == SCALE_E8M0
        and pack.group == 32
        and pack.q.element_size() == 1
        and pack.q.is_contiguous()
        and pack.scale is not None
        and pack.scale.is_contiguous()
        and pack.k % 64 == 0
        and pack.n % 16 == 0
    )


def bias_ok(bias, n: int) -> bool:
    """The kernel adds a BF16 bias in FP32, as the Triton kernel does with a BF16 bias; anything else: Triton."""
    import torch

    return bias is None or (bias.dtype == torch.bfloat16 and bias.dim() == 1 and bias.shape[0] == n
                            and bias.is_contiguous())


def linear(x, pack, bias, cfg: int):
    """x[..., K] (BF16, 1-16 rows) @ dequant(pack)^T (+ bias) -> BF16 [..., N]."""
    import torch

    lead = x.shape[:-1]
    x2 = x.reshape(-1, x.shape[-1])
    if x2.stride(-1) != 1 or (x2.shape[0] > 1 and (x2.stride(0) % 4 or x2.stride(0) < x2.shape[1])):
        x2 = x2.contiguous()
    if x2.data_ptr() % 8:
        x2 = x2.clone()
    out = torch.empty((x2.shape[0], pack.n), dtype=torch.bfloat16, device=x2.device)
    _EXT.w8gemv(x2, pack.q, pack.scale, bias, out, cfg)
    return out.view(*lead, pack.n)
