"""Python entry point of the inkling_attn kernel (imported on the GPU path only)."""

from . import build

D_REL = 16
HEAD_DIM = 128


import os

WARPS = int(os.environ.get("INKLING_ATTN_WARPS", "4"))  # 4: BM 64, 2 CTAs / SM; 8: BM 128, 1 CTA / SM
PACKED_PROJ = True  # repack proj per call (one tiny launch) so each lane loads its B fragment with one 8-byte load
# Global layers with WARPS 4 (BF16 and packed KV): a request whose prefix holds >= W8_FROM keys runs on 8 warps
# (BM 128), the others on 4 (two launches, requests picked on the device). BM 64 holds ~71 TF/s up to ~256K prefix
# keys in isolation (~150K in serving) and falls to ~47 TF/s at 512K-1M; BM 128 holds ~69 TF/s at every prefix and
# is ~3-5 % slower below 256K. Bit-identical rows either way. 0 = off.
W8_FROM = int(os.environ.get("INKLING_ATTN_W8_FROM", "131072"))


def prefill_attention(q, k_cache, v_cache, page_table, cu_seqlens_q, seqused_k, r, proj, softmax_scale,
                      window_left, out=None, packed=None, warps=None, w8_from=None):
    """FA4-compatible (out, lse) of the Inkling prefill attention with the relative bias computed in-kernel.

    q [T, H, 128] bf16; k_cache / v_cache [pages, page_size, H_kv, 128] bf16; page_table [B, max_pages] int32;
    cu_seqlens_q [n_req + 1] int32; seqused_k [n_req] int32 (prefix + new tokens); r [T, H, 16] bf16 (after the
    rel L2 clip, the RelLogitsProj input); proj [16, rel_extent] bf16; window_left: the layer's left window
    (sliding-window layers, rel_extent == window_left + 1) or None / negative (global causal layers).
    Returns out [T, H, 128] fp32 and lse [H, T] fp32 (FA4's varlen layout). Rows past cu_seqlens_q[-1] are not
    written.
    """
    import torch

    ext = build.load()
    t, heads, _ = q.shape
    if out is None:
        out = torch.empty((t, heads, HEAD_DIM), dtype=torch.float32, device=q.device)
    lse = torch.empty((heads, t), dtype=torch.float32, device=q.device)
    wl = -1 if window_left is None or window_left < 0 else int(window_left)
    if r.stride(-1) != 1:
        r = r.contiguous()
    packed = PACKED_PROJ if packed is None else packed
    projp = torch.empty(((proj.shape[1] + 1) * D_REL,), dtype=torch.bfloat16, device=q.device) if packed else None
    ext.fwd(q, k_cache, v_cache, page_table, cu_seqlens_q, seqused_k, r, proj, out, lse, float(softmax_scale), wl,
            projp, WARPS if warps is None else int(warps),
            w8_from_keys=W8_FROM if w8_from is None else int(w8_from))
    return out, lse


def packed_prefill_attention(q, layer, page_table, cu_seqlens_q, seqused_k, softmax_scale, *,
                             r, proj, out=None, warps=None, w8_from=None):
    """Run fused projected-bias prefill with exact KV reconstructed in shared memory."""
    import torch

    from inkling_kv12s.storage import validate_paged

    validate_paged(q, layer, page_table, seqused_k, cu_seqlens_q)
    t, heads, _ = q.shape
    if out is None:
        out = torch.empty((t, heads, HEAD_DIM), dtype=torch.float32, device=q.device)
    lse = torch.empty((heads, t), dtype=torch.float32, device=q.device)
    if r.stride(-1) != 1:
        r = r.contiguous()
    projp = torch.empty(((proj.shape[1] + 1) * D_REL,), dtype=torch.bfloat16, device=q.device) if PACKED_PROJ else None
    planes = [*layer.planes, layer.stage, layer.overflow]
    ext = build.load()
    ext.fwd(q, layer.raw, layer.raw, page_table, cu_seqlens_q, seqused_k, r, proj, out, lse,
            float(softmax_scale), -1, projp, WARPS if warps is None else int(warps), planes,
            W8_FROM if w8_from is None else int(w8_from))
    return out, lse
