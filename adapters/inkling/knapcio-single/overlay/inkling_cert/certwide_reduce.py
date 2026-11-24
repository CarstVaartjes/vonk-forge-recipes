"""Device-gated softcap and argmax over the compact or fallback wide head."""

import triton as tr
import triton.language as tl
from triton.language.extra import libdevice


@tr.jit
def _key(value, col):
    # Match the stock CUDA argmax. NaNs never win and both signs of zero tie.
    value = tl.where(value == value, value, -float("inf"))
    bits = tl.where(value == 0.0, 0, value.to(tl.uint32, bitcast=True)).to(tl.uint32)
    ordered = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000)
    return (ordered.to(tl.uint64) << 32) | (~col.to(tl.uint32)).to(tl.uint64)


@tr.jit
def _partials(Dense, Sparse, State, Tiles, Groups, Partial, Capped, cap,
              N: tl.constexpr, REAL: tl.constexpr, SLOTS: tl.constexpr,
              PARTS: tl.constexpr, BLOCK: tl.constexpr, EMIT: tl.constexpr):
    row = tl.program_id(0)
    row64 = row.to(tl.int64)
    part = tl.program_id(1)
    best = tl.full((BLOCK,), 0, tl.uint64)
    if tl.load(State) == 0:
        count = tl.load(State + 1) * 16
        for base in range(part * BLOCK, count, PARTS * BLOCK):
            idx = base + tl.arange(0, BLOCK)
            tile = tl.load(Tiles + idx // 16, idx < count, other=0)
            gr = tl.load(Groups + idx // 16, idx < count, other=0)
            col = tile * 16 + idx % 16
            valid = ((gr >> (row // 16)) & 1) != 0
            valid = valid & (idx < count) & (col < REAL)
            x = tl.load(Sparse + row64 * (SLOTS * 16) + idx, valid, other=0)
            # Same Triton FP32 division, libdevice tanh, and multiply as
            # sglang.kernels.ops.activation.softcap (dynamic FP32 cap).
            y = libdevice.tanh(x / cap) * cap
            if EMIT:
                tl.store(Capped + row64 * (SLOTS * 16) + idx, y, valid)
            best = tl.maximum(best, tl.where(valid, _key(y, col), 0))
    else:
        # Ineligible replays can have consumers of logits after the graph.
        # Publish the normal capped and padded output for every fallback.
        for base in range(part * BLOCK, N, PARTS * BLOCK):
            col = base.to(tl.int64) + tl.arange(0, BLOCK)
            ptr = Dense + row64 * N + col
            x = tl.load(ptr, col < N, other=0)
            y = libdevice.tanh(x / cap) * cap
            y = tl.where(col < REAL, y, -float("inf"))
            tl.store(ptr, y, col < N)
            best = tl.maximum(best, tl.where(col < REAL, _key(y, col), 0))
    tl.store(Partial + row * PARTS + part, tl.max(best, 0))


@tr.jit
def _finish(Partial, Out, PARTS: tl.constexpr):
    row = tl.program_id(0)
    bitkey = tl.max(tl.load(Partial + row * PARTS + tl.arange(0, PARTS)).to(tl.uint64), 0)
    tl.store(Out + row, (~bitkey.to(tl.uint32)).to(tl.int64))


def reduce(dense, sparse, state, tiles, groups, partial, ids, v_real, cap, capped=None):
    parts = partial.shape[1]
    _partials[(ids.numel(), parts)](
        dense, sparse, state, tiles, groups, partial, sparse if capped is None else capped, cap,
        dense.shape[1], v_real, tiles.numel(), parts, 512, capped is not None, num_warps=4,
    )
    _finish[(ids.numel(),)](partial, ids, parts, num_warps=1)
