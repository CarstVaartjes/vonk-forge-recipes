"""JIT build and call of the scale-code NVFP4 grouped GEMM (csrc/), through FlashInfer's own JIT (same nvcc flags,
include paths and TVM-FFI binding as its gemm_sm120 module; output cached under FlashInfer's cache dir)."""

import hashlib
import os
import threading

CSRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")
TILES = ((32, 128), (32, 256), (64, 128), (64, 256), (128, 128))  # 128x256 needs more smem than stock
WORKSPACE_BYTES = 8 << 20

_lock = threading.Lock()
_module = None
_workspaces = {}


def sources():
    names = sorted(f for f in os.listdir(CSRC) if f.endswith((".cu", ".cuh", ".hpp", ".h")))
    return [os.path.join(CSRC, f) for f in names]


def source_hash():
    h = hashlib.sha256()
    for path in sources():
        h.update(os.path.basename(path).encode())
        with open(path, "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:12]


def module():
    """The loaded TVM-FFI module (built on first use; the build is cached by source hash)."""
    global _module
    if _module is not None:
        return _module
    with _lock:
        if _module is None:
            from flashinfer.jit import core as jit_core

            ctx = jit_core.current_compilation_context
            flags = ctx.get_nvcc_flags_list(supported_major_versions=[12])
            spec = jit_core.gen_jit_spec(
                "inkling_scodes_" + source_hash(),
                [p for p in sources() if p.endswith(".cu")],
                extra_cuda_cflags=flags + ["-DCUTLASS_ENABLE_GDC_FOR_SM100=1"],
                extra_include_paths=[CSRC],
            )
            _module = spec.build_and_load()
    return _module


def _workspace(torch, device):
    # Same policy as FlashInfer's own grouped GEMM (_get_stream_cache_buf): one argument/tensormap workspace per
    # (device, stream). Calls on one stream (and graphs replayed on one stream) are ordered; two graphs captured on
    # the same stream must not be replayed concurrently on different streams (true for SGLang and for stock).
    stream = torch.cuda.current_stream(device).cuda_stream
    key = (device.index, stream)
    ws = _workspaces.get(key)
    if ws is None:
        ws = (torch.empty(WORKSPACE_BYTES, dtype=torch.uint8, device=device),
              torch.empty(WORKSPACE_BYTES, dtype=torch.uint8, device=device))
        _workspaces[key] = ws
    return ws


def prepare(torch, device):
    """Allocate the current stream's workspace outside graph capture (called at weight-load time)."""
    _workspace(torch, device)


def grouped_gemm(torch, values, scales, weight, weight_scales, offsets, codes, bases, expert_stride,
                 tile_n, tile_k, out):
    """out[rows, n] (BF16) = raw accumulators of the swap_ab grouped NVFP4 GEMM, weight scales from the codes.

    values [rows, k/2] u8, scales: activation scales as for FlashInfer, weight [E, n, k/2] u8, weight_scales: the
    stock swizzled scales (read only for escape words), offsets int32 [E+1]."""
    if (tile_n, tile_k) not in TILES:
        raise ValueError(f"inkling_scodes: no instantiation for tile {tile_n}x{tile_k}")
    experts, n, half_k = weight.shape
    k = half_k * 2
    dev = values.device
    for name, t, dtype in (("values", values, torch.uint8), ("weight", weight, torch.uint8),
                           ("codes", codes, torch.uint8), ("bases", bases, torch.uint8),
                           ("offsets", offsets, torch.int32), ("out", out, torch.bfloat16)):
        if t.device != dev or t.dtype != dtype or not t.is_contiguous():
            raise ValueError(f"inkling_scodes: {name} must be a contiguous {dtype} tensor on {dev}")
    if (tuple(bases.shape) != (experts, n) or codes.numel() < experts * expert_stride + 640
            or expert_stride != (n // 128) * (k // 128) * 640 or offsets.numel() != experts + 1
            or tuple(out.shape) != (values.shape[0], n)):
        raise ValueError("inkling_scodes: inconsistent shapes for the scale-code GEMM")
    # The stock swizzled scales are read for escape words: same device, contiguous bytes, whole and aligned.
    if (weight_scales.device != dev or weight_scales.element_size() != 1 or not weight_scales.is_contiguous()
            or weight_scales.numel() != experts * n * (k // 16) or weight_scales.data_ptr() % 16):
        raise ValueError("inkling_scodes: weight_scales must be the contiguous swizzled [E, n * k/16] scale bytes")
    int_ws, float_ws = _workspace(torch, dev)
    module().scodes_group_gemm_nvfp4(
        int_ws, float_ws, values, weight, scales, weight_scales.view(torch.uint8), out, offsets,
        codes, bases, n, k, tile_n, tile_k, expert_stride)
    return out
