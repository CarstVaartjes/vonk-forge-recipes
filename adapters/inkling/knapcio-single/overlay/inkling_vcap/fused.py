"""One-launch dead-row remap: out[r] = in[src[r]] for the gate's weights and ids.

Replaces two torch index_select launches per MoE visit (96 -> 48 kernels per verify
graph). Built lazily; ``selfcheck`` compares it with the torch reference on the
serving device at startup and the overlay falls back to torch if it is unavailable.
"""

from __future__ import annotations

from typing import Optional

import torch

_KERNEL = None
_IMPORT_ERROR: Optional[str] = None


def _build():
    global _KERNEL, _IMPORT_ERROR
    if _KERNEL is not None or _IMPORT_ERROR is not None:
        return _KERNEL
    try:
        import triton
        import triton.language as tl
    except Exception as exc:  # pragma: no cover - depends on the image
        _IMPORT_ERROR = repr(exc)
        return None

    @triton.jit
    def _remap_rows_kernel(
        w_in, id_in, src, w_out, id_out, w_stride, id_stride,
        K: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        s = tl.load(src + row).to(tl.int64)
        k = tl.arange(0, BLOCK_K)
        mask = k < K
        w = tl.load(w_in + s * w_stride + k, mask=mask)
        i = tl.load(id_in + s * id_stride + k, mask=mask)
        tl.store(w_out + row * K + k, w, mask=mask)
        tl.store(id_out + row * K + k, i, mask=mask)

    _KERNEL = (_remap_rows_kernel, triton.next_power_of_2)
    return _KERNEL


def available() -> bool:
    return _build() is not None


def import_error() -> Optional[str]:
    _build()
    return _IMPORT_ERROR


def supports(weights: torch.Tensor, ids: torch.Tensor) -> bool:
    return (
        weights.is_cuda
        and ids.is_cuda
        and weights.dim() == 2
        and ids.dim() == 2
        and weights.shape == ids.shape
        and weights.stride(1) == 1
        and ids.stride(1) == 1
    )


def remap_rows_fused(weights: torch.Tensor, ids: torch.Tensor, src_index: torch.Tensor):
    kernel, next_pow2 = _build()
    m, k = ids.shape
    w_out = torch.empty((m, k), dtype=weights.dtype, device=weights.device)
    id_out = torch.empty((m, k), dtype=ids.dtype, device=ids.device)
    if m:
        kernel[(m,)](
            weights, ids, src_index, w_out, id_out,
            weights.stride(0), ids.stride(0),
            K=k, BLOCK_K=next_pow2(k), num_warps=1,
        )
    return w_out, id_out


def selfcheck(device) -> Optional[str]:
    """None when the fused kernel matches index_select bit for bit on ``device``."""
    if not available():
        return f"triton unavailable: {_IMPORT_ERROR}"
    g = torch.Generator(device="cpu").manual_seed(0)
    m, k = 24, 8
    weights = torch.rand((m, k), generator=g).to(device)
    ids = torch.randint(0, 128, (m, k), generator=g, dtype=torch.int32).to(device)
    src = torch.randint(0, m, (m,), generator=g, dtype=torch.int64).to(device)
    w_ref, id_ref = weights.index_select(0, src), ids.index_select(0, src)
    w_out, id_out = remap_rows_fused(weights, ids, src)
    torch.cuda.synchronize(device)
    if not (torch.equal(w_out, w_ref) and torch.equal(id_out, id_ref)):
        return "fused remap mismatch vs index_select"
    return None
