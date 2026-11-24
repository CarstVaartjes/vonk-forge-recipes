"""Scale codes for NVFP4 expert weight scales (inkling_scodes).

Input: the expert weight scales in the layout SGLang's SM120 path hands to the grouped GEMM, i.e. UE4M3 bytes
swizzled into CUTLASS 128x4 atoms by swizzle_blockscale ([experts, n * k/16] or [experts, n, k/16], n % 128 == 0,
k % 128 == 0). Per expert the swizzled bytes are [n/128 row blocks][k/64 k-groups][32 r32][4 r4][4 kk] with
weight row = block * 128 + r4 * 32 + r32 and scale column = kgroup * 4 + kk.

Format (what csrc/scodes_sm120_mma_array.hpp decodes):
  bases  uint8 [experts, n]          per weight row: the row minimum when the row spans <= 31 codes; otherwise
                                    (an "escape row", ~0.02 % of the real rows) 0x80 | b with b the start of the
                                    31-value window [b, b+30] that holds most of the row's bytes
  codes  uint8 [experts * stride + 640]   stride = (n/128) * (k/128) * 640
         per expert, per row block, per 128-K chunk (= 2 k-groups = 256 SFA words of 4 bytes, word
         w = kg_local * 128 + r32 * 4 + r4): 5 bit-planes of 128 bytes; plane b, byte w // 2, nibble w % 2 (low
         first) holds bit b of the 4 codes (byte - row base) of word w, bit i = byte kk == i.
         The 640-byte zero tail keeps the kernel's whole-stage bulk copies inside the buffer when the last k tile
         of the last row block runs past K (tile_k 256 with k/128 odd); those words are never used.
Decode: byte = base + code; in an escape row code 31 marks a byte outside the window and the kernel takes the whole
4-byte SFA word from the stock (resident) swizzled scales. Exact for any UE4M3 bytes <= 0x7E.
"""

CHUNK_BYTES = 640
PLANES = 5
MAX_RANGE = (1 << PLANES) - 1


class ScodesError(ValueError):
    """Scales that the 5-bit code cannot represent (or a bad layout)."""


def _shapes(scale, n, k):
    if n % 128 or k % 128:
        raise ScodesError(f"n={n} and k={k} must be multiples of 128")
    sfk = k // 16
    experts = scale.shape[0]
    if scale.numel() != experts * n * sfk:
        raise ScodesError(f"scale has {scale.numel()} bytes, expected {experts} x {n} x {sfk}")
    return experts, n // 128, sfk // 4, k // 128


def logical_from_swizzled(torch, swz, n, k):
    """[E, n, k/16] uint8 view of swizzled scales (inverse of swizzle_blockscale)."""
    e, mt, kg, _ = _shapes(swz, n, k)
    s = swz.reshape(e, mt, kg, 32, 4, 4)
    return s.permute(0, 1, 4, 3, 2, 5).reshape(e, n, k // 16)


def encode(torch, swz, n, k, experts_per_pass=16):
    """(codes, bases, expert_stride) of swizzled UE4M3 scales (any 1-byte dtype, on any device)."""
    swz = swz.contiguous().view(torch.uint8)
    e, mt, kg, kt = _shapes(swz, n, k)
    swz = swz.reshape(e, mt, kg, 32, 4, 4)
    stride = mt * kt * CHUNK_BYTES
    codes = torch.zeros(e * stride + CHUNK_BYTES, dtype=torch.uint8, device=swz.device)
    bases = torch.empty((e, n), dtype=torch.uint8, device=swz.device)
    weights = torch.tensor([1, 2, 4, 8], dtype=torch.int16, device=swz.device)
    for e0 in range(0, e, experts_per_pass):
        e1 = min(e, e0 + experts_per_pass)
        s = swz[e0:e1].to(torch.int16)                                   # [P, mt, kg, 32, 4, 4]
        logical = s.permute(0, 1, 4, 3, 2, 5).reshape(e1 - e0, n, k // 16)
        if bool((logical > 0x7E).any()):
            raise ScodesError("scale byte with the sign bit or the E4M3 NaN code")
        lo = logical.amin(dim=2)                                          # [P, n]
        esc = (logical.amax(dim=2) - lo) > MAX_RANGE
        base = lo.clone()
        if bool(esc.any()):
            rows = logical[esc]                                           # [X, k/16]
            hist = torch.zeros((rows.shape[0], 128), dtype=torch.int32, device=rows.device)
            hist.scatter_add_(1, rows.long(), torch.ones_like(rows, dtype=torch.int32))
            win = hist.cumsum(dim=1)                                      # count in [b, b+30] = C[b+30] - C[b-1]
            upper = win[:, MAX_RANGE - 1:]                                # b = 0 .. 128-31
            lower = torch.cat([torch.zeros_like(win[:, :1]), win[:, :128 - MAX_RANGE]], dim=1)
            base[esc] = (upper - lower).argmax(dim=1).to(base.dtype)
        bases[e0:e1] = (base + esc.to(base.dtype) * 0x80).to(torch.uint8)
        base_sw = base.reshape(e1 - e0, mt, 4, 32).permute(0, 1, 3, 2)    # [P, mt, r32, r4]
        esc_sw = esc.reshape(e1 - e0, mt, 4, 32).permute(0, 1, 3, 2)
        c = s - base_sw[:, :, None, :, :, None]                           # [P, mt, kg, 32, 4, 4]
        out = (c < 0) | (c > MAX_RANGE - 1)
        c = torch.where(out & esc_sw[:, :, None, :, :, None], torch.full_like(c, MAX_RANGE), c)
        c = c.reshape(e1 - e0, mt, kt, 256, 4)                            # words of each 128-K chunk
        planes = []
        for b in range(PLANES):
            nib = (((c >> b) & 1) * weights).sum(dim=-1)                  # [P, mt, kt, 256]
            planes.append((nib[..., 0::2] | (nib[..., 1::2] << 4)).to(torch.uint8))
        chunk = torch.stack(planes, dim=3)                                # [P, mt, kt, 5, 128]
        codes[e0 * stride:e1 * stride] = chunk.reshape(-1)
    return codes, bases, stride


def decode(torch, codes, bases, n, k, stock=None):
    """Swizzled scale bytes [E, n * k/16] from (codes, bases), with the kernel's arithmetic (reference). Escape
    words (escape row and some code == 31) are taken from `stock` (the swizzled scales), as the kernel does."""
    e = bases.shape[0]
    mt, kt = n // 128, k // 128
    stride = mt * kt * CHUNK_BYTES
    ch = codes[: e * stride].reshape(e, mt, kt, PLANES, 128).to(torch.int64)
    nib = torch.stack([ch & 0xF, ch >> 4], dim=-1).reshape(e, mt, kt, PLANES, 256)  # nibble of word w
    row = (torch.arange(256, device=codes.device) % 128)
    row = (row & 3) * 32 + (row >> 2)                                                # row inside the block
    braw = bases.to(torch.int64).reshape(e, mt, 128)[:, :, row]                      # [E, mt, 256]
    base, esc = braw & 0x7F, (braw & 0x80) != 0
    v = base[:, :, None, :] * 0x01010101                                             # [E, mt, 1, 256]
    ones = torch.full_like(nib[:, :, :, 0, :], 0xF)
    for b in range(PLANES):
        v = v + (((nib[:, :, :, b, :] * 0x00204081) & 0x01010101) << b)
        ones = ones & nib[:, :, :, b, :]
    out = torch.stack([(v >> (8 * i)) & 0xFF for i in range(4)], dim=-1)             # little-endian bytes
    use_stock = (esc[:, :, None, :] & (ones != 0))                                   # [E, mt, kt, 256]
    if bool(use_stock.any()):
        if stock is None:
            raise ScodesError("escape words need the stock scales")
        st = stock.contiguous().view(torch.uint8).reshape(e, mt, kt, 256, 4).to(torch.int64)
        out = torch.where(use_stock[..., None], st, out)
    return out.to(torch.uint8).reshape(e, n * (k // 16))


def stats(torch, swz, n, k):
    """Byte budget of one tensor: raw scale bytes, coded bytes (codes + bases), rows over the 5-bit range."""
    logical = logical_from_swizzled(torch, swz.contiguous().view(torch.uint8), n, k).to(torch.int16)
    rng = logical.amax(dim=2) - logical.amin(dim=2)
    e = logical.shape[0]
    return dict(raw=int(logical.numel()), coded=e * (n // 128) * (k // 128) * CHUNK_BYTES + e * n,
                escape_rows=int((rng > MAX_RANGE).sum()), range_max=int(rng.max()))
