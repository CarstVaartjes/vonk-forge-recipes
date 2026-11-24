"""Host side of the fused attention-residual / loop-message kernels (see _triton_mix.py).

resnorm(sources, addend, queries, weights, ...)            == stock rms_norm(attn_res(...)[q], weights[q]) per query
loop_mix(sources, addend, queries, bank, bank_queries, ...) == stock message_controllers(attn_res(state), attn_res(bank))
                                                               followed by the layer's RMSNorm (norm_weights given),
                                                               or without it (FP32 out, norm_weights=None)

The controller's down projection is the stock _controller_down kernel launched exactly like
sglang.kernels.ops.elementwise.inkling2_loop.message_controllers does, on the same BF16 normed input.
Shapes are static per call, nothing syncs with the host, all buffers come from the torch allocator: CUDA-graph safe.
"""

import torch
import triton

D = 2304
CH = 256
QP = 4  # queries padded
NSP = 8  # state sources padded (the model has at most 6)
NBP = 2  # loop message bank sources (blocks after the prelude + partial)
EPS_ATTN_RES = 1e-6  # sglang attn_res() hard-codes 1e-6
EPS_CTRL = 1e-6  # message_controllers default
MIX = {"bm": 32, "bn": 64, "warps": 4, "stages": 1, "split_down": True,
       # _loop_mix1_kernel launch, NORM (two passes) and FP32-out modes. GPU-checked config (check4); the 8192-row
       # sweep (tools/sweep_mix.py) found NORM 16/64/4 warps/2 stages 0.55 ms faster and FP32 32/128/8/2 0.07 ms
       # faster (FP32 output bit-equal across configs; NORM differs only in the row sum-of-squares order)
       "per_ctrl": True, "bm1": 32, "bn1": 64, "warps1": 8, "stages1": 2,
       "bm1f": 32, "bn1f": 64, "warps1f": 8, "stages1f": 2}

_K = {}


def _qp(n):
    """Padded query count used inside a kernel (power of two >= n)."""
    return 1 if n == 1 else 4


def _kernels():
    if not _K:
        from sglang.kernels.ops.elementwise import inkling2_loop

        from . import _triton_mix

        _K.update(
            resnorm=_triton_mix._resnorm_kernel,
            stats=_triton_mix._loop_stats_kernel,
            mix=_triton_mix._loop_mix_kernel,
            down_split=_triton_mix._controller_down_split,
            mix1=_triton_mix._loop_mix1_kernel,
            down=inkling2_loop._controller_down,
            max_smem=inkling2_loop._max_shared_mem,
            pdl=None,
        )
        from sglang.kernels.jit.utils import is_arch_support_pdl

        _K["pdl"] = is_arch_support_pdl()
    return _K


def _check_sources(sources, extra=()):
    base = sources[0]
    tokens = base.shape[0]
    for x in (*sources, *extra):
        if x is None:
            continue
        if not (x.dtype == torch.bfloat16 and x.is_contiguous() and x.shape == (tokens, D) and x.device == base.device):
            return False
    return True


def resnorm(sources, addend, queries, weights, eps_norm, normalize_values, out=None):
    """attn_res over `sources` (addend added to the last one) with queries [NQ, D] (FP32, already scaled), then
    RMSNorm with weights[q] -> BF16 [NQ, T, D] (into `out` if given)."""
    k = _kernels()
    nq = queries.shape[0]
    tokens = sources[0].shape[0]
    assert 1 <= nq <= QP and len(sources) <= NSP and len(weights) == nq
    assert queries.dtype == torch.float32 and queries.is_contiguous() and queries.shape[1] == D
    if out is not None:
        assert out.shape == (nq, tokens, D) and out.dtype == torch.bfloat16 and out.is_contiguous()
    y = out if out is not None else torch.empty((nq, tokens, D), dtype=torch.bfloat16, device=sources[0].device)
    if tokens:
        k["resnorm"][(tokens,)](
            tuple(sources), addend if addend is not None else sources[0], queries, tuple(weights), y,
            tokens, EPS_ATTN_RES, float(eps_norm),
            nq, _qp(nq), NSP, addend is not None, bool(normalize_values), D, CH,
            num_warps=4, enable_fp_fusion=False,
        )
    return y


def loop_mix(sources, addend, queries, normalize_values, bank, bank_queries, down, gain, forget,
             norm_weights=None, eps_norm=1e-6, fuse_norm=True):
    """Loop message mix of one side of a layer visit.

    sources/addend/queries: the attention-residual state mix (dest), NQS = queries.shape[0] in {1, 3};
    bank/bank_queries: the message mix (NM = bank_queries.shape[0] in {1, 3} controllers);
    down/gain/forget: the stacked controller weights rows of this side.
    NQS == 1 with NM == 3 shares the single dest across the three controllers (stock passes streams * 3).
    Returns the stock message_controllers output [NM, T, D] FP32 (norm_weights None) or its RMSNorm [NM, T, D] BF16.
    """
    k = _kernels()
    nqs, nm = queries.shape[0], bank_queries.shape[0]
    assert nqs in (1, nm) and nm in (1, 3) and len(bank) == NBP and len(sources) <= NSP
    num, rank, kdim = down.shape
    assert num == nm and kdim == 2 * D and gain.shape == forget.shape == (nm, D, rank)
    assert down.is_contiguous() and gain.is_contiguous() and forget.is_contiguous()
    assert down.dtype == gain.dtype == forget.dtype == torch.bfloat16
    assert queries.dtype == bank_queries.dtype == torch.float32 and queries.is_contiguous() and bank_queries.is_contiguous()
    dev = sources[0].device
    tokens = sources[0].shape[0]
    shared = nqs == 1 and nm > 1
    if norm_weights is not None and not fuse_norm:
        mixed = loop_mix(sources, addend, queries, normalize_values, bank, bank_queries, down, gain, forget)
        from sglang.kernels.ops.layernorm.inkling2_rms_norm import rms_norm

        if all(w is norm_weights[0] for w in norm_weights):
            return rms_norm(mixed, norm_weights[0], eps_norm, output_dtype=torch.bfloat16)
        y = torch.empty((nm, tokens, D), dtype=torch.bfloat16, device=dev)
        for m, w, yy in zip(mixed.unbind(0), norm_weights, y):
            rms_norm(m, w, eps_norm, output_dtype=torch.bfloat16, out=yy)
        return y
    if norm_weights is not None:
        assert len(norm_weights) == nm
        y = torch.empty((nm, tokens, D), dtype=torch.bfloat16, device=dev)
    else:
        y = torch.empty((nm, tokens, D), dtype=torch.float32, device=dev)
    if not tokens:
        return y
    sp = torch.empty((tokens, NSP, QP), dtype=torch.float32, device=dev)
    si = torch.empty((tokens, QP), dtype=torch.float32, device=dev)
    bp = torch.empty((tokens, NBP, QP), dtype=torch.float32, device=dev)
    bi = torch.empty((tokens, QP), dtype=torch.float32, device=dev)
    split_normed = MIX["split_down"]
    nd = nqs  # 1 (one dest, shared by the controllers when NM = 3) or NM
    if split_normed:
        normed = torch.empty((nd + nm, tokens, D), dtype=torch.bfloat16, device=dev)
    else:
        normed = torch.empty((nm, tokens, kdim), dtype=torch.bfloat16, device=dev)
    add = addend if addend is not None else sources[0]
    k["stats"][(tokens,)](
        tuple(sources), add, queries, tuple(bank), bank_queries, normed, sp, si, bp, bi,
        tokens, EPS_ATTN_RES, EPS_CTRL,
        nqs, nm, QP, _qp(nqs), _qp(nm), NSP, NBP, addend is not None, bool(normalize_values), shared, D, CH,
        split_normed, nd,
        num_warps=4, enable_fp_fusion=False,
    )
    # stock message_controllers, prefill branch (tokens > 16): split-K down projection
    groups = triton.cdiv(kdim, 1024)
    split = tokens <= 4096
    block_m = block_n = 64
    block_k = 256 if tokens <= 1024 else 64
    if block_k == 256 and k["max_smem"](dev.index) < 1 << 17:
        block_k = 128
    partial_groups = groups if split else 1
    partials = torch.empty((nm, partial_groups, tokens, rank), dtype=torch.float32, device=dev)
    if split_normed:
        assert D % block_k == 0 and 1024 % block_k == 0
        k["down_split"][(nm * partial_groups, triton.cdiv(tokens, block_m), triton.cdiv(rank, block_n))](
            normed, down, partials, tokens, rank, kdim, split, block_m, block_n, block_k, nm, nd,
            num_warps=4, num_stages=3, enable_fp_fusion=False,
        )
    else:
        launch = {"launch_pdl": True} if k["pdl"] else {}
        k["down"][(triton.cdiv(tokens, block_m), triton.cdiv(rank, block_n), nm * partial_groups)](
            normed, down, partials, tokens, rank, kdim, split, block_m, block_n, block_k, k["pdl"],
            num_warps=4, num_stages=3, enable_fp_fusion=False, **launch,
        )
    if MIX["per_ctrl"]:
        # [NM, D] norm weights (a 14 KB copy; the kernel reads row i)
        wn = gain if norm_weights is None else torch.stack([w.reshape(D) for w in norm_weights])
        sfx = "" if norm_weights is not None else "f"
        bm1, bn1, warps1, stages1 = (MIX[n + sfx] for n in ("bm1", "bn1", "warps1", "stages1"))
        k["mix1"][(nm, triton.cdiv(tokens, bm1))](
            tuple(sources), add, tuple(bank), sp, si, bp, bi, partials, gain, forget, wn, y,
            tokens, float(eps_norm),
            nm, QP, NSP, NBP, addend is not None, shared, norm_weights is not None, partial_groups, rank,
            max(16, triton.next_power_of_2(rank)), D, bm1, bn1, stages1,
            num_warps=warps1, enable_fp_fusion=False,
        )
        return y
    weights = tuple(norm_weights) if norm_weights is not None else (gain,)
    k["mix"][(triton.cdiv(tokens, MIX["bm"]),)](
        tuple(sources), add, tuple(bank), sp, si, bp, bi, partials, gain, forget, weights, y,
        tokens, float(eps_norm),
        nqs, nm, QP, NSP, NBP, addend is not None, shared, norm_weights is not None, partial_groups, rank,
        max(16, triton.next_power_of_2(rank)), D, MIX["bm"], MIX["bn"],
        num_warps=MIX["warps"], num_stages=MIX["stages"], enable_fp_fusion=False,
    )
    return y


SCONV_MIN_ROWS = 2048  # stock short_convolution uses _sconv_prefill_kernel from PREFILL_MIN_ROWS rows on


def sconv_min_rows():
    """The stock threshold itself (a changed module constant must not leave this copy stale)."""
    from sglang.kernels.ops.mamba.inkling2_sconv import PREFILL_MIN_ROWS

    return max(PREFILL_MIN_ROWS, SCONV_MIN_ROWS)


def sconv_residual(x, weight, cache, metadata, attention, partial):
    """Stock short_convolution(x, weight, cache, metadata) (static weights, residual, extend, >= 2048 rows)
    followed by residual_add(partial, attention, conv) (or attention + conv when partial is None), one kernel."""
    from . import _triton_sconv

    rows, width = x.shape
    kernel_width = weight.shape[1]
    assert rows >= sconv_min_rows() and x.stride(-1) == 1 and x.dtype == torch.bfloat16
    assert cache.ndim == 3 and cache.shape[1:] == (kernel_width - 1, width)
    for t in (attention, partial):
        assert t is None or (t.shape == x.shape and t.dtype == torch.bfloat16 and t.is_contiguous())
    out = torch.empty_like(attention)
    _triton_sconv._sconv_residual_kernel[(triton.cdiv(rows, 16), triton.cdiv(width, 128))](
        x, weight, cache, metadata["safe_idx"], metadata["cache_mask"], metadata["cu"], metadata["si"],
        attention, partial if partial is not None else attention, out,
        rows, x.stride(0), *weight.stride(), *cache.stride(), width, kernel_width,
        (kernel_width + 1) ** -0.5, partial is not None, 16, 128,
        enable_fp_fusion=False,
    )
    return out
