"""Per-lane segment sums of the staging / fallback / overflow lock words in one Triton launch (pressure guard).

out[lane] = (sum(locks[lane, :live]), sum(locks[lane, live:live + fallback]), sum(locks[lane, live + fallback:])) as
FP32, summed in int32 (exact). Replaces torch.mm(locks.float(), segments), which gives the same values while every sum
is < 2**24 (lock words are 0 / 1 and a lane has far fewer than 2**24 slots).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _segment_sums_kernel(Locks, Out, N, LIVE, FALLBACK, BLOCK: tl.constexpr):
    lane = tl.program_id(0).to(tl.int64)
    a = tl.zeros([BLOCK], tl.int32)
    b = tl.zeros([BLOCK], tl.int32)
    c = tl.zeros([BLOCK], tl.int32)
    for start in range(0, N, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        x = tl.load(Locks + lane * N + cols, cols < N, other=0)
        a += tl.where(cols < LIVE, x, 0)
        b += tl.where((cols >= LIVE) & (cols < LIVE + FALLBACK), x, 0)
        c += tl.where(cols >= LIVE + FALLBACK, x, 0)
    tl.store(Out + lane * 3 + 0, tl.sum(a, 0).to(tl.float32))
    tl.store(Out + lane * 3 + 1, tl.sum(b, 0).to(tl.float32))
    tl.store(Out + lane * 3 + 2, tl.sum(c, 0).to(tl.float32))


def segment_sums(locks, live, fallback):
    lanes, n = locks.shape
    out = torch.empty((lanes, 3), dtype=torch.float32, device=locks.device)
    if lanes:
        _segment_sums_kernel[(lanes,)](locks, out, n, live, fallback, BLOCK=1024, num_warps=4)
    return out
