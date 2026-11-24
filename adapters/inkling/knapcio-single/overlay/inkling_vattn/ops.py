"""DSpark target-verify attention kernels for Inkling-2 on GB10 (sm_121).

Both produce the split-KV partials (o, m, l) of the stock `_decode_partials`
(sglang.kernels.ops.attention.inkling2_decode_attention_sm90), shaped
(splits, tokens, 1, heads[, d]), so the stock `combine_gate_sink` reduces them.

- `stock_bias_partials`: the stock `_decode_partials` kernel itself (bias gathered from the compact
  (tokens, heads, extent) BF16 rel bias) with an explicit launch config (benchmarks only; the overlay's cap lever
  keeps the stock launcher, whose heuristics measured as good at the whole-visit level).
- `inkernel_bias_partials`: the same kernel body with the relative bias computed in-kernel from the
  rel-clip output r (tokens, heads, >= 16; r in [..., :16]) and the layer's rel_logits_proj (16, extent):
  per key tile, Bw = BF16(dot(r, proj[:, dbase : dbase + WB])) for the tile's window of distances
  (one m16n8k16 K-step, the same FP32 dot the stock prologue stores as BF16), then each row takes its
  diagonal slice with tl.gather. No (tokens, heads, extent) tensor is written or read.

Math as stock: s = q.k * scale + bias (FP32), bias = 0 beyond the extent, causal + left window, FP32
online softmax with exp2, P rounded to the V dtype for PV, FP32 partial o / max / sum per split.
"""

import torch
import triton
import triton.language as tl

MAX_SPLITS = 64


@triton.jit
def _inkernel_bias_partials(
    Q,
    K,
    V,
    PageTable,
    SeqLens,
    R,
    Proj,
    OPartial,
    MPartial,
    LPartial,
    scale,
    batch,
    SQ0: tl.constexpr,
    SQ1: tl.constexpr,
    SK0: tl.constexpr,
    SK1: tl.constexpr,
    SK2: tl.constexpr,
    SP: tl.constexpr,
    SR0: tl.constexpr,
    SR1: tl.constexpr,
    HEADS_K: tl.constexpr,
    QLEN: tl.constexpr,
    GROUP: tl.constexpr,
    ROWS: tl.constexpr,
    D: tl.constexpr,
    D_REL: tl.constexpr,
    PAGE: tl.constexpr,
    EXTENT: tl.constexpr,
    WINDOW: tl.constexpr,
    BLOCK_N: tl.constexpr,
    WB: tl.constexpr,
):
    tl.extra.cuda.gdc_wait()
    tl.extra.cuda.gdc_launch_dependents()
    b = tl.program_id(0) // HEADS_K
    hk = tl.program_id(0) % HEADS_K
    split = tl.program_id(1)
    splits = tl.num_programs(1)
    seqlen = tl.load(SeqLens + b)
    first = 0
    if WINDOW >= 0:
        first = tl.maximum(seqlen - QLEN - WINDOW, 0)
    tiles = tl.cdiv(seqlen - first // BLOCK_N * BLOCK_N, BLOCK_N)
    per_split = tl.cdiv(tiles, splits)
    tile_lo = first // BLOCK_N + split * per_split
    tile_hi = tl.minimum(tile_lo + per_split, first // BLOCK_N + tiles)

    rows = tl.arange(0, ROWS)
    head_mask = rows < QLEN * GROUP
    qi = rows // GROUP
    token = b * QLEN + qi
    qh = hk * GROUP + rows % GROUP
    q_pos = seqlen - QLEN + qi
    d = tl.arange(0, D)
    q = tl.load(
        Q + token[:, None] * SQ0 + qh[:, None] * SQ1 + d[None, :],
        head_mask[:, None],
        other=0,
    )
    dr = tl.arange(0, D_REL)
    r = tl.load(
        R + token[:, None] * SR0 + qh[:, None] * SR1 + dr[None, :],
        head_mask[:, None],
        other=0,
    )
    wcol = tl.arange(0, WB)
    cols = tl.arange(0, BLOCK_N)
    # Row (token qi) at key column c of a tile has distance dbase + qi + (BLOCK_N - 1 - c).
    gidx = qi[:, None] + (BLOCK_N - 1 - cols)[None, :]
    gidx = tl.minimum(gidx, WB - 1)  # padding rows (qi >= QLEN) only
    m = tl.full((ROWS,), float("-inf"), tl.float32)
    l = tl.zeros((ROWS,), tl.float32)
    acc = tl.zeros((ROWS, D), tl.float32)
    for tile in range(tile_lo, tile_hi):
        pos = tile * BLOCK_N + cols
        valid = (pos < seqlen) & (pos >= first)
        if PAGE % BLOCK_N == 0:
            page = tl.load(PageTable + b * SP + (tile * BLOCK_N) // PAGE).to(tl.int64)
            slot = (tile * BLOCK_N) % PAGE + cols
            kv_offset = page * SK0 + slot[:, None] * SK1 + hk * SK2 + d[None, :]
        else:
            page = tl.load(PageTable + b * SP + pos // PAGE, valid, other=0)
            slot = pos % PAGE
            kv_offset = (
                page.to(tl.int64)[:, None] * SK0
                + slot[:, None] * SK1
                + hk * SK2
                + d[None, :]
            )
        k = tl.load(K + kv_offset, valid[:, None], other=0)
        s = tl.dot(q, tl.trans(k)) * scale
        dist = q_pos[:, None] - pos[None, :]
        keep = valid[None, :] & (dist >= 0)
        if WINDOW >= 0:
            keep &= dist <= WINDOW
        dbase = seqlen - QLEN - (tile * BLOCK_N + BLOCK_N - 1)
        if dbase < EXTENT:
            dw = dbase + wcol
            pw = tl.load(
                Proj + dr[:, None] * EXTENT + dw[None, :],
                ((dw >= 0) & (dw < EXTENT))[None, :],
                other=0,
            )
            bw = tl.dot(r, pw).to(Proj.dtype.element_ty).to(tl.float32)
            s = s + tl.gather(bw, gidx, axis=1)
        # else: every distance of the tile is >= EXTENT (old keys of a global layer): bias 0, as stock
        s = tl.where(keep, s, float("-inf"))
        m_new = tl.maximum(m, tl.max(s, 1))
        m_ref = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2((m - m_ref) * 1.4426950408889634)
        p = tl.exp2((s - m_ref[:, None]) * 1.4426950408889634)
        l = l * alpha + tl.sum(p, 1)
        v = tl.load(V + kv_offset, valid[:, None], other=0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m = m_new
    stat = (split * batch * QLEN + token) * (HEADS_K * GROUP) + qh
    tl.store(MPartial + stat, m, head_mask)
    tl.store(LPartial + stat, l, head_mask)
    tl.store(OPartial + stat[:, None] * D + d[None, :], acc, head_mask[:, None])


def _shape(q, k_cache, seqused_k):
    tokens, heads_q, head_dim = q.shape
    batch = seqused_k.shape[0]
    qlen = tokens // max(batch, 1)
    assert qlen * batch == tokens, (tokens, batch)
    _, page_size, heads_k, _ = k_cache.shape
    return tokens, heads_q, head_dim, batch, qlen, page_size, heads_k


def launch_config(batch, heads_k, qlen, span, sms):
    """(splits, block_n, num_stages) for the in-kernel-bias verify kernel.

    span: keys a request can attend at most (window + qlen, or the page-table width in keys).
    From the GB10 sweep (results/kbench/round2/vattn/vattn_bench.py, cold KV, ctx 1K..128K, bs 1..48): aim for ~4 CTAs per
    SM (splits = ceil(4 * sms / ctas), at most 12), 64-key tiles once the grid alone has >= 128 CTAs, 2 stages.
    The stock heuristic (<= 2 CTAs per SM, 32-key tiles, 3 stages) keeps one CTA per (request, KV head) from
    7 requests up, which leaves 8-16 requests at 64-128 CTAs: 10-25 % slower there."""
    ctas = batch * heads_k
    block_n = 64 if ctas >= 128 else 32
    tiles = triton.cdiv(span, block_n)
    splits = max(1, min(triton.cdiv(4 * sms, max(1, ctas)), 12, tiles))
    return splits, block_n, 2


def _alloc(splits, tokens, heads_q, head_dim, device):
    o = torch.empty((splits, tokens, 1, heads_q, head_dim), dtype=torch.float32, device=device)
    m = torch.empty((splits, tokens, 1, heads_q), dtype=torch.float32, device=device)
    return o, m, torch.empty_like(m)


def stock_bias_partials(q, k_cache, v_cache, page_table, seqused_k, rel_bias, softmax_scale, *,
                        window_size_left=None, config=None):
    """The stock _decode_partials kernel with `config` (splits, block_n, num_stages) or launch_config's."""
    from sglang.kernels.ops.attention.inkling2_decode_attention_sm90 import _decode_partials

    tokens, heads_q, head_dim, batch, qlen, page_size, heads_k = _shape(q, k_cache, seqused_k)
    group = heads_q // heads_k
    span = window_size_left + qlen if window_size_left is not None else page_table.shape[1] * page_size
    if config is None:
        sms = torch.cuda.get_device_properties(q.device).multi_processor_count
        config = launch_config(batch, heads_k, qlen, span, sms)
    splits, block_n, stages = config
    assert page_size % block_n == 0
    o, m, l = _alloc(splits, tokens, heads_q, head_dim, q.device)
    _decode_partials[(batch * heads_k, splits)](
        q, k_cache, v_cache, page_table, seqused_k, rel_bias, o, m, l, softmax_scale, batch,
        q.stride(0), q.stride(1), k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        page_table.stride(0), rel_bias.stride(0), rel_bias.stride(1),
        heads_k, qlen, group, max(16, triton.next_power_of_2(qlen * group)), head_dim, page_size,
        rel_bias.shape[-1], -1 if window_size_left is None else window_size_left, block_n,
        num_warps=4, num_stages=stages, launch_pdl=True,
    )
    return o, m, l


def inkernel_bias_partials(q, k_cache, v_cache, page_table, seqused_k, r, proj, softmax_scale, *,
                           window_size_left=None, config=None):
    """Partials with the bias computed in-kernel. r: (tokens, heads_q, >= d_rel) with the rel-clip output
    in [..., :d_rel] (BF16, unit stride in the last dim); proj: (d_rel, extent) contiguous, r's dtype."""
    tokens, heads_q, head_dim, batch, qlen, page_size, heads_k = _shape(q, k_cache, seqused_k)
    group = heads_q // heads_k
    d_rel, extent = proj.shape
    assert proj.is_contiguous() and proj.dtype == r.dtype and r.stride(-1) == 1
    assert r.shape[0] >= tokens and r.shape[1] == heads_q and r.shape[2] >= d_rel
    assert d_rel >= 16 and d_rel & (d_rel - 1) == 0
    span = window_size_left + qlen if window_size_left is not None else page_table.shape[1] * page_size
    if config is None:
        sms = torch.cuda.get_device_properties(q.device).multi_processor_count
        config = launch_config(batch, heads_k, qlen, span, sms)
    splits, block_n, stages = config
    wb = max(16, triton.next_power_of_2(block_n + qlen - 1))
    o, m, l = _alloc(splits, tokens, heads_q, head_dim, q.device)
    _inkernel_bias_partials[(batch * heads_k, splits)](
        q, k_cache, v_cache, page_table, seqused_k, r, proj, o, m, l, softmax_scale, batch,
        q.stride(0), q.stride(1), k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        page_table.stride(0), r.stride(0), r.stride(1),
        heads_k, qlen, group, max(16, triton.next_power_of_2(qlen * group)), head_dim, d_rel, page_size,
        extent, -1 if window_size_left is None else window_size_left, block_n, wb,
        num_warps=4, num_stages=stages, launch_pdl=True,
    )
    return o, m, l
