"""Pure-Python index math shared by the Triton kernels and the CPU tests (no torch).

1. The sheared relative-bias row layout that FA4's ShearingBias writes
   (flash_attn/cute/shearing_bias.py) for the SM120 forward (attention tile_m 64,
   tile_n 128, pack_gqa off). `shear_params` gives, per query row, the half-open
   output column range [left, right) that holds pre-bias values: column i holds
   pre[right - 1 - i]; columns >= right hold -inf and columns < left hold the left
   pad (-inf for a sliding window, 0.0 for causal-only). `stock_shear_row` is a
   literal transcription of the CuTe kernel (lane loop, pair loads, odd/even edge
   fix-ups) used by the tests to prove that element-wise description equal.

2. The NVFP4 routed scale layout of routed_quant.cuh (scale_base / scale_offset):
   `scale_byte` and `scale_word`.
"""

import math

NEG_INF = float("-inf")


def ceil_div(a, b):
    return -(-a // b)


def n_idx_left_right(m, seqlen_q, seqlen_k, local, window_left):
    """BlockInfo.get_n_idx_left_right for pack_gqa off, causal or local with
    window_size_right 0. Inclusive left, exclusive right."""
    right = m + 1 + seqlen_k - seqlen_q
    left = max(m + seqlen_k - seqlen_q - window_left, 0) if local else 0
    return left, right


def n_block_max(m_block, seqlen_q, seqlen_k, tile_m, tile_n=128):
    """BlockInfo.get_n_block_min_max (right edge), causal or local with right 0."""
    nmax = ceil_div(seqlen_k, tile_n)
    n_idx = (m_block + 1) * tile_m + seqlen_k - seqlen_q
    return min(nmax, ceil_div(n_idx, tile_n))


def shear_params(m, seqlen_q, seqlen_k, local, window_left, rel_extent=1024, tile_m=64):
    """(left, right, num_bias_vals) of the sheared row for query row m."""
    padded_blocks = (rel_extent + 256) // 128
    n_left, n_right = n_idx_left_right(m, seqlen_q, seqlen_k, local, window_left)
    nbv = n_right - max(n_left, n_right - rel_extent)
    attn_n_block_max = n_block_max(m // tile_m, seqlen_q, seqlen_k, tile_m)
    n_block_rel0 = (n_right - 1) // 128
    block_right = 1 + max(padded_blocks - (attn_n_block_max - n_block_rel0), 0)
    right = (block_right - 1) * 128 + ((n_right - 1) % 128) + 1
    left = max(0, right - nbv)
    return left, right, nbv


def simple_shear_row(pre, m, seqlen_q, seqlen_k, local, window_left, rel_extent=1024, tile_m=64):
    """The element-wise description the Triton kernel implements."""
    left, right, nbv = shear_params(m, seqlen_q, seqlen_k, local, window_left, rel_extent, tile_m)
    left_pad = NEG_INF if local else 0.0
    out = []
    for i in range(rel_extent + 256):
        if i >= right:
            out.append(NEG_INF)
        elif i < left:
            out.append(left_pad)
        else:
            out.append(pre[right - 1 - i])
    return out


def stock_shear_row(pre, m, seqlen_q, seqlen_k, local, window_left, rel_extent=1024, tile_m=64):
    """Literal transcription of ShearingBias.kernel for one row (bf16: vec 2,
    64 columns per warp iteration, 32 lanes). Out-of-range pre reads return NaN
    so the tests catch any use of them."""
    pad = rel_extent + 256
    nbp = pad // 128
    vec, cols_per_iter = 2, 64
    right_pad = NEG_INF
    left_pad = NEG_INF if local else 0.0
    n_left, n_right = n_idx_left_right(m, seqlen_q, seqlen_k, local, window_left)
    nbv = n_right - max(n_left, n_right - rel_extent)
    is_even = n_right % 2 == 0
    attn_n_block_max = n_block_max(m // tile_m, seqlen_q, seqlen_k, tile_m)
    n_block_rel0 = (n_right - 1) // 128
    bias_block_idx_right = 1 + max(nbp - (attn_n_block_max - n_block_rel0), 0)
    bias_idx_right = (bias_block_idx_right - 1) * 128 + ((n_right - 1) % 128) + 1
    bias_idx_left = max(0, bias_idx_right - nbv)
    bias_block_idx_left = bias_idx_left // 128
    num_bias_blocks = bias_block_idx_right - bias_block_idx_left if nbv > 0 else 0
    num_right = nbp - bias_block_idx_right if nbv > 0 else nbp
    num_bias_blocks = min(num_bias_blocks, nbp)
    num_right = min(num_right, nbp)

    def p_at(i):
        return pre[i] if 0 <= i < len(pre) else float("nan")

    s = [None] * pad
    for lane in range(32):
        bias_idx = pad + lane * vec - cols_per_iter
        for i in range(num_right):
            idx4 = (nbp - 1 - i) * 32 + lane
            for j in range(4):
                s[idx4 * 4 + j] = right_pad
            bias_idx -= 128
        for _ in range(num_bias_blocks):
            for _ in range(128 // cols_per_iter):
                p = bias_idx_right - 1 - bias_idx
                in_bounds = p >= 0 and p - vec + 1 < nbv
                lower = p - 1 if is_even else max(p - 2, 0)
                upper = p - 1 if is_even else min(p, rel_extent - 2)
                frg = [left_pad, left_pad]
                if in_bounds:
                    lo = (p_at(lower // 2 * 2), p_at(lower // 2 * 2 + 1))
                    up = (p_at(upper // 2 * 2), p_at(upper // 2 * 2 + 1))
                    if is_even:
                        frg = [lo[1], lo[0]]
                    else:
                        frg = [up[0], lo[1]]
                elif p < 0:
                    frg = [right_pad, right_pad]
                s[bias_idx], s[bias_idx + 1] = frg
                bias_idx -= cols_per_iter
    if not is_even and nbv > 0:
        s[bias_idx_right] = right_pad
        if bias_idx_left - 1 >= 0:
            s[bias_idx_left - 1] = left_pad
    num_left = min(nbp - num_bias_blocks - num_right, nbp)
    for lane in range(32):
        for i in range(num_left):
            idx4 = i * 32 + lane
            for j in range(4):
                s[idx4 * 4 + j] = left_pad
    return s


# ---------------------------------------------------------------- NVFP4 scales


def scale_k_of(width):
    return (width + 63) // 64 * 4


def scale_base(offsets, expert, scale_k):
    """routed_quant.cuh scale_base (bytes)."""
    return (offsets[expert] - offsets[0] + expert * 127) // 128 * 128 * scale_k


def scale_offset(local, col, scale_k):
    """routed_quant.cuh scale_offset (bytes)."""
    return (
        ((local // 128) * (scale_k // 4) + col // 4) * 512
        + (local % 32) * 16
        + ((local // 32) % 4) * 4
        + col % 4
    )


def scale_word(local, col4, scale_k):
    """32-bit word index of scale columns [4 col4, 4 col4 + 4) of a local row."""
    return ((local // 128) * (scale_k // 4) + col4) * 128 + (local % 32) * 4 + (local // 32) % 4


def scale_rows(rows, experts):
    """_nvfp4_outputs scale row count."""
    return math.ceil((rows + 127 * experts) / 128) * 128
