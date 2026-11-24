"""Lever A: keep Inkling2's per-source QKVR projections on a weight-streaming GEMV.

Stock behaviour (sglang/srt/models/inkling_common/inkling2_attn.py,
Inkling2Attention._project_sources): when the q, k and v sources are distinct
tensors (global layers, and every looped core layer in passes 2-3), the three
[M, H] sources are stacked into [3M, H] and pushed through the whole qkvr weight
[N, H] (N = 8600 at TP1). The q chunk keeps the q/r/gate/sink columns, the k
chunk only the k columns, the v chunk only the v columns. At c1 verify M = 8, so
3M = 24 rows exceeds sm12x_bf16_gemv's MAX_M = 16 and the call falls to cuBLAS;
two thirds of the FLOPs are thrown away.

This module computes only the kept outputs: one Triton kernel over the N output
columns, in 32-column CTAs, where each CTA reads its weight columns once and
multiplies them by the rows of the one source that owns those columns (k columns
-> k source, v columns -> v source, everything else -> q source). The output is
written into a [3M, N] buffer at exactly the places the stock code reads, and is
then split with the stock split sizes, so every returned view has the stock shape,
stride and storage offset. Rows of a chunk for columns it does not own are left
unwritten (never read).

Math: BF16 operands, FP32 accumulation over K in BK-wide tl.dot steps, BF16 out;
the per-element K order is the stock sm12x_bf16_gemv's (same BN/BK by default),
so the result equals the stock GEMV run per source. Against cuBLAS (the stock
path at 3M > 16) it differs only by FP32 summation order.

CUDA graphs: no host sync, no data-dependent shapes, a static grid; the output is
allocated with torch.empty like the stock F.linear. First use of a (row bucket,
bias) variant compiles a Triton kernel, which happens in SGLang's eager warm-up
before capture, as for the stock GEMV.

Env:
  INKLING_QKVR_MAX_M    rows per source taken by the kernel (default 16, the stock
                        GEMV bound; at most 64). Larger M keeps the stock path.
  INKLING_QKVR_TILE     "BN,BK,WARPS" (default "32,256,4" = stock GEMV tiles).
                        A different BK changes the FP32 summation order.
"""

import importlib

import torch
import triton
import triton.language as tl

TARGET_MODULE = "sglang.srt.models.inkling_common.inkling2_attn"
TARGET_CLASS = "Inkling2Attention"
TARGET_ATTR = "_project_sources"
UNQUANT_MODULE = "sglang.srt.layers.quantization.unquant"


@triton.jit
def _qkvr_split_gemv(
    X,
    W,
    B,
    Y,
    M,
    N,
    K,
    SX,
    SW,
    SY,
    K_LO,
    K_HI,
    V_LO,
    V_HI,
    HAS_BIAS: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    MP: tl.constexpr,
):
    # X: [3M, K] stacked q/k/v sources. Y: [3M, N]. Columns [K_LO, K_HI) take the
    # k rows, [V_LO, V_HI) the v rows, all others the q rows. Region bounds are
    # multiples of BN, so one CTA serves exactly one source.
    pid = tl.program_id(0)
    c0 = pid * BN
    in_k = ((c0 >= K_LO) & (c0 < K_HI)).to(tl.int32)
    in_v = ((c0 >= V_LO) & (c0 < V_HI)).to(tl.int32)
    src = in_k + 2 * in_v
    row0 = src * M
    cols = c0 + tl.arange(0, BN)
    rows = tl.arange(0, MP)
    xrows = (row0 + rows).to(tl.int64)
    acc = tl.zeros((MP, BN), tl.float32)
    for k0 in range(0, K, BK):
        ks = k0 + tl.arange(0, BK)
        w = tl.load(
            W + cols[None, :].to(tl.int64) * SW + ks[:, None],
            (cols[None, :] < N) & (ks[:, None] < K),
            other=0,
        )
        x = tl.load(
            X + xrows[:, None] * SX + ks[None, :],
            (rows[:, None] < M) & (ks[None, :] < K),
            other=0,
        )
        acc += tl.dot(x, w)
    if HAS_BIAS:
        acc += tl.load(B + cols, cols < N, other=0).to(tl.float32)[None, :]
    tl.store(
        Y + xrows[:, None] * SY + cols[None, :],
        acc.to(Y.dtype.element_ty),
        (rows[:, None] < M) & (cols[None, :] < N),
    )


def _row_bucket(m):
    mp = 16
    while mp < m:
        mp *= 2
    return mp


def split_gemv(stacked, weight, bias, m, k_range, v_range, cfg, out=None):
    """out[3M, N]: rows [0, M) hold x_q @ W^T on every column outside k_range and
    v_range, rows [M, 2M) x_k @ W^T on k_range, rows [2M, 3M) x_v @ W^T on v_range.
    stacked is [3M, K] BF16 with unit column stride; weight [N, K] BF16."""
    n, k = weight.shape
    if out is None:
        out = torch.empty((3 * m, n), dtype=stacked.dtype, device=stacked.device)
    bn = cfg["bn"]
    _qkvr_split_gemv[(triton.cdiv(n, bn),)](
        stacked,
        weight,
        bias,
        out,
        m,
        n,
        k,
        stacked.stride(0),
        weight.stride(0),
        out.stride(0),
        k_range[0],
        k_range[1],
        v_range[0],
        v_range[1],
        HAS_BIAS=bias is not None,
        BN=bn,
        BK=cfg["bk"],
        MP=_row_bucket(m),
        num_warps=cfg["warps"],
    )
    return out


def regions(split_sizes, bn):
    """(k_range, v_range) column ranges of the qkvr output, or None if a bound is
    not a multiple of bn (the CTA-to-source map needs aligned bounds)."""
    q, kk, v = split_sizes[0], split_sizes[1], split_sizes[2]
    k_range = (q, q + kk)
    v_range = (q + kk, q + kk + v)
    if any(b % bn for b in (*k_range, *v_range)):
        return None
    return k_range, v_range


def make_project_sources(stock, attn_module, unquant, cfg):
    stacked_fn = attn_module._stacked
    eligible = unquant._sm12x_bf16_gemv_eligible
    max_m = cfg["max_m"]
    bn = cfg["bn"]

    def _project_sources(self, sources):
        q_src, k_src, v_src = sources.q, sources.k, sources.v
        # Only the stock stacked case (three distinct tensors); shared sources keep
        # the stock per-source projections.
        if len({id(q_src), id(k_src), id(v_src)}) != 3:
            return stock(self, sources)
        m = q_src.shape[0]
        plan = self.__dict__.get("_inkling_qkvr_plan", False)
        if plan is False:
            plan = regions(self.split_sizes, bn)
            self.__dict__["_inkling_qkvr_plan"] = plan
        weight = self.qkvr.weight
        bias = self.qkvr.bias
        if (
            plan is None
            or not 0 < m <= max_m
            or q_src.ndim != 2
            or not (q_src.shape == k_src.shape == v_src.shape)
            or not (q_src.dtype == k_src.dtype == v_src.dtype == torch.bfloat16)
            or weight.dim() != 2
            or weight.shape[0] != sum(self.split_sizes)
            # Same operand and mode checks as the stock sm12x GEMV route (enabled
            # backend, not compiling, no batch-invariant mode, BF16, unit strides),
            # on a one-row view so that only the row bound is ours.
            or not eligible(q_src[:1], weight, bias)
        ):
            return stock(self, sources)
        stacked = stacked_fn(sources)
        if stacked.stride(-1) != 1:
            stacked = stacked.contiguous()
        out = split_gemv(stacked, weight, bias, m, plan[0], plan[1], cfg)
        q_proj, k_proj, v_proj = (
            chunk.split(self.split_sizes, dim=-1)[:6] for chunk in out.chunk(3)
        )
        q, _, _, r, gate, sinks = q_proj
        return q, k_proj[1], v_proj[2], r, gate, sinks

    _project_sources.__wrapped__ = stock
    _project_sources.inkling_cfg = cfg
    return _project_sources


def patch(attn_module, cfg):
    cls = getattr(attn_module, TARGET_CLASS, None)
    stock = getattr(cls, TARGET_ATTR, None) if cls is not None else None
    if stock is None or getattr(attn_module, "_stacked", None) is None:
        raise RuntimeError(
            f"inkling_kernels.qkvr_gemv: {TARGET_MODULE}.{TARGET_CLASS}."
            f"{TARGET_ATTR} or _stacked is missing"
        )
    if getattr(stock, "inkling_cfg", None) is not None:
        return
    unquant = importlib.import_module(UNQUANT_MODULE)
    if not callable(getattr(unquant, "_sm12x_bf16_gemv_eligible", None)):
        raise RuntimeError(
            f"inkling_kernels.qkvr_gemv: {UNQUANT_MODULE}._sm12x_bf16_gemv_eligible "
            "is missing"
        )
    setattr(cls, TARGET_ATTR, make_project_sources(stock, attn_module, unquant, cfg))
