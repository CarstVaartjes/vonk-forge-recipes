"""SM121 fallback for Qwen sparse attention's one-query varlen contract."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _varlen_one_q_attn_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    cu_seqlens_q_ptr, cu_seqlens_k_ptr,
    sm_scale,
    HQ: tl.constexpr, HKV: tl.constexpr,
    D: tl.constexpr, D_PAD: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    q_stride_t: tl.constexpr, q_stride_h: tl.constexpr,
    k_stride_t: tl.constexpr, k_stride_h: tl.constexpr,
    v_stride_t: tl.constexpr, v_stride_h: tl.constexpr,
    o_stride_t: tl.constexpr, o_stride_h: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    q_start = tl.load(cu_seqlens_q_ptr + row)
    k_start = tl.load(cu_seqlens_k_ptr + row)
    k_end = tl.load(cu_seqlens_k_ptr + row + 1)
    offs_d = tl.arange(0, D_PAD)
    mask_d = offs_d < D
    q = tl.load(
        q_ptr + (q_start * q_stride_t) + (head * q_stride_h) + offs_d,
        mask=mask_d,
        other=0.0,
    ).to(tl.float32)
    kh = head // (HQ // HKV)
    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([D_PAD], dtype=tl.float32)
    for k0 in range(k_start, k_end, BLOCK_KV):
        offs_kv = k0 + tl.arange(0, BLOCK_KV)
        mask_kv = offs_kv < k_end
        kv_ptrs = (offs_kv * k_stride_t) + (kh * k_stride_h)
        k_blk = tl.load(
            k_ptr + kv_ptrs[:, None] + offs_d[None, :],
            mask=mask_kv[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(q[None, :] * k_blk, axis=1) * sm_scale
        scores = tl.where(mask_kv, scores, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        acc = acc * alpha
        v_ptrs = (offs_kv * v_stride_t) + (kh * v_stride_h)
        v_blk = tl.load(
            v_ptr + v_ptrs[:, None] + offs_d[None, :],
            mask=mask_kv[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(p[:, None] * v_blk, axis=0)
        m_i = m_new
    out = acc / tl.where(l_i > 0.0, l_i, 1.0)
    tl.store(
        o_ptr + (row * o_stride_t) + (head * o_stride_h) + offs_d,
        out.to(o_ptr.dtype.element_ty),
        mask=mask_d,
    )


def _next_pow2(n: int) -> int:
    return triton.next_power_of_2(max(n, 16))


def triton_varlen_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int = 1,
    max_seqlen_k: int = 0,
    softmax_scale: float = 1.0,
    causal: bool = True,
    **kwargs,
) -> torch.Tensor:
    del max_seqlen_q, max_seqlen_k, causal, kwargs
    if not q.is_cuda:
        raise RuntimeError("qsa_fa_fallback requires CUDA tensors")
    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
        raise RuntimeError(f"expected 3D q/k/v, got {q.shape}/{k.shape}/{v.shape}")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise RuntimeError(f"q/k/v dtypes must match ({q.dtype}/{k.dtype}/{v.dtype})")
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise RuntimeError(f"unsupported dtype {q.dtype}")
    total_q, hq, d = q.shape
    _, hkv, dk = k.shape
    if dk != d or v.shape[2] != d:
        raise RuntimeError("head_dim mismatch")
    if hq % hkv != 0:
        raise RuntimeError(f"GQA mismatch: {hq} q heads vs {hkv} kv heads")
    if cu_seqlens_k.numel() != total_q + 1:
        raise RuntimeError("cu_seqlens_k size mismatch")
    if not torch.cuda.is_current_stream_capturing():
        expected = torch.ones_like(cu_seqlens_q[1:])
        if not bool(torch.equal(cu_seqlens_q[1:] - cu_seqlens_q[:-1], expected)):
            raise RuntimeError("qsa_fa_fallback only supports q_len==1 per sequence")
    if cu_seqlens_q.dtype != torch.int32:
        cu_seqlens_q = cu_seqlens_q.to(torch.int32)
    if cu_seqlens_k.dtype != torch.int32:
        cu_seqlens_k = cu_seqlens_k.to(torch.int32)
    q_c = q if q.is_contiguous() else q.contiguous()
    k_c = k if k.is_contiguous() else k.contiguous()
    v_c = v if v.is_contiguous() else v.contiguous()
    out = torch.empty_like(q_c)
    _varlen_one_q_attn_kernel[(total_q, hq)](
        q_c, k_c, v_c, out,
        cu_seqlens_q, cu_seqlens_k, softmax_scale,
        HQ=hq, HKV=hkv, D=d, D_PAD=_next_pow2(d), BLOCK_KV=64,
        q_stride_t=q_c.stride(0), q_stride_h=q_c.stride(1),
        k_stride_t=k_c.stride(0), k_stride_h=k_c.stride(1),
        v_stride_t=v_c.stride(0), v_stride_h=v_c.stride(1),
        o_stride_t=out.stride(0), o_stride_h=out.stride(1),
        num_warps=4,
    )
    return out
