"""CUDA-only helpers. Import lazily so the overlay's CPU paths need no Triton."""

import torch
import triton
import triton.language as tl


@triton.jit
def _confidence_parts(logits, temps, parts, V: tl.constexpr, GAMMA: tl.constexpr,
                      STRIDE: tl.constexpr, SPLITS: tl.constexpr, TILE: tl.constexpr):
    row, split = tl.program_id(0), tl.program_id(1)
    col = split * TILE + tl.arange(0, TILE)
    x = tl.load(logits + row * STRIDE + col, col < V, other=-float("inf")).to(tl.float32)
    x = x / tl.load(temps + row // GAMMA)
    best = tl.max(x, 0)
    first = tl.min(tl.where(x == best, col, 2147483647), 0)
    second = tl.max(tl.where(col != first, x, -float("inf")), 0)
    total = tl.sum(tl.exp(x - tl.where(best == -float("inf"), 0.0, best)), 0)
    base = parts + 3 * (row * SPLITS + split)
    tl.store(base, best)
    tl.store(base + 1, second)
    tl.store(base + 2, total)


@triton.jit
def _confidence_combine(parts, out, SPLITS: tl.constexpr, TILE: tl.constexpr):
    row = tl.program_id(0)
    split = tl.arange(0, TILE)
    base = parts + 3 * (row * SPLITS + split)
    bests = tl.load(base, split < SPLITS, other=-float("inf"))
    best = tl.max(bests, 0)
    first = tl.min(tl.where(bests == best, split, 2147483647), 0)
    seconds = tl.load(base + 1, split < SPLITS, other=-float("inf"))
    second = tl.max(tl.where(split == first, seconds, bests), 0)
    totals = tl.load(base + 2, split < SPLITS, other=0.0)
    origin = tl.where(best == -float("inf"), 0.0, best)
    total = tl.sum(totals * tl.exp(bests - origin), 0)
    tl.store(out + row * 2, tl.where(total > 0, 1.0 / total, 0.0))
    tl.store(out + row * 2 + 1, tl.where(total > 0, tl.exp(second - origin) / total, 0.0))


def confidence(flat, temps, gamma):
    if flat.stride(1) != 1:
        flat = flat.contiguous()
    rows, vocab = flat.shape
    tile = 2048
    splits = triton.cdiv(vocab, tile)
    parts = torch.empty((rows, splits, 3), device=flat.device, dtype=torch.float32)
    result = torch.empty((rows, 2), device=flat.device, dtype=torch.float32)
    _confidence_parts[(rows, splits)](
        flat, temps, parts, vocab, gamma, flat.stride(0), splits, tile,
    )
    _confidence_combine[(rows,)](parts, result, splits, triton.next_power_of_2(splits), num_warps=4)
    return result


@triton.jit(do_not_specialize=["rows"])
def _route_visit(ids, journal, control, rows, STRIDE: tl.constexpr,
                 TOPK: tl.constexpr, ID_ROW: tl.constexpr, ID_COL: tl.constexpr,
                 LANES: tl.constexpr, VISITS: tl.constexpr, TILE: tl.constexpr):
    # Writes this visit's routed ids of the real rows into slab `cursor` of the journal. It only READS the cursor:
    # the advance is a separate launch (_route_advance), stream-ordered after every warp of this one (the
    # policy3 kernel advanced the cursor in the same multi-warp program, a load/store race across warps).
    slot = tl.load(control)
    real_rows = tl.load(control + 1) * STRIDE
    col = tl.arange(0, TILE)
    row, lane = col // LANES, col % LANES
    valid = (row < real_rows) & (row < rows) & (slot < VISITS)
    expert = tl.load(ids + row * ID_ROW + lane * ID_COL, valid & (lane < TOPK), other=-1)
    tl.store(journal + slot * real_rows * LANES + col, expert, valid)


@triton.jit(do_not_specialize=["rows"])
def _route_advance(control, rows, STRIDE: tl.constexpr, VISITS: tl.constexpr):
    # One program, one warp. A target path that shards the real rows marks the journal invalid (host rejects it).
    slot = tl.load(control)
    real_rows = tl.load(control + 1) * STRIDE
    tl.store(control, tl.where(real_rows > rows, VISITS + 1, slot + 1))


def record_route(ids, journal, control, stride, lanes, visits):
    rows, topk = ids.shape
    tile = triton.next_power_of_2(rows * lanes)
    _route_visit[(1,)](
        ids, journal, control, rows, stride, topk, *ids.stride(), lanes, visits, tile,
        num_warps=4 if tile <= 2048 else 8,
    )
    _route_advance[(1,)](control, rows, stride, visits, num_warps=1)
