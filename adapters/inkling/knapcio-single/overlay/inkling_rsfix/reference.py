"""Pure-torch model of the final-token CDF pass (stock and fixed lane rules), FP32
throughout, tile by tile like the kernels. Used by the CPU tests and by the GPU check's
host-side validation. No sglang / triton imports.

``scan(tile_vals, tile_index)`` and ``tile_sum`` stand in for tl.cumsum and tl.sum of one
tile; the defaults are a sequential FP32 cumsum and torch.sum. A test can pass any scan (non-monotone, perturbed)
to model what a parallel scan may return: the fixed rule must stay valid for every one.
"""

import torch

F32 = torch.float32


def _seq_cumsum(v, tile=0):
    return torch.cumsum(v, dim=0)  # sequential on CPU


def _sum(v):
    return v.sum()


def residual(p, q=None, scale=1.0):
    """The tile values both kernels scan: p (all drafts accepted) or max(scale * p - q, 0)
    with q outside [0, 1] treated as 0."""
    p = p.to(F32)
    if q is None:
        return p.clone()
    q = q.to(F32)
    q = torch.where((q >= 0) & (q <= 1), q, torch.zeros_like(q))
    d = torch.tensor(scale, dtype=F32) * p - q
    return torch.where(d > 0, d, torch.zeros_like(d))


def pass1(val, block_v, tile_sum=_sum):
    """norm_sum as the kernels accumulate it: FP32 running sum of per-tile sums."""
    acc = torch.zeros((), dtype=F32)
    for s in range(0, val.numel(), block_v):
        acc = acc + tile_sum(val[s:s + block_v]).to(F32)
    return acc


def final_token(val, p, coin, norm_sum, *, vocab, block_v, fixed, scan=_seq_cumsum,
                tile_sum=_sum):
    """The token the CDF pass emits. ``val`` / ``p``: [vocab] residual and target row;
    lanes vocab..(tiles * block_v - 1) are the masked tail (0, as the masked loads give).
    ``scan(tile_vals, tile_index)`` may return any FP32 prefix values for the tile."""
    tiles = -(-vocab // block_v)
    lanes = tiles * block_v
    v_all = torch.zeros(lanes, dtype=F32)
    p_all = torch.zeros(lanes, dtype=F32)
    v_all[:vocab] = val.to(F32)
    p_all[:vocab] = p.to(F32)
    target_u = torch.tensor(coin, dtype=F32) * torch.as_tensor(norm_sum, dtype=F32)
    cum = torch.zeros((), dtype=F32)
    fallback = -1
    for t in range(tiles):
        s = t * block_v
        v = v_all[s:s + block_v]
        pv = p_all[s:s + block_v]
        lane = s + torch.arange(block_v)
        in_range = lane < vocab
        prefix = scan(v, t)
        total = cum + prefix.to(F32)
        cand = total > target_u
        if fixed:
            positive = in_range & (v > 0)
            cand = cand & positive
        if bool(cand.any()):
            return s + int(torch.nonzero(cand)[0])
        if fixed:
            key = torch.where(positive, vocab + lane,
                              torch.where(in_range & (pv > 0), lane, torch.full_like(lane, -1)))
            fallback = max(fallback, int(key.max()))
        cum = cum + tile_sum(v).to(F32)
    if fixed and fallback >= vocab:
        return fallback - vocab
    if fixed and fallback >= 0:
        return fallback
    return vocab - 1


def lane_is_valid(token, val, p, vocab):
    """A legal emission: in range with positive residual mass, or, when the residual has
    no positive lane, in range with positive target mass."""
    if not 0 <= token < vocab:
        return False
    if bool((val[:vocab] > 0).any()):
        return bool(val[token] > 0)
    return bool(p[token] > 0)
