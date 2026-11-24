"""Packed loader for SGLang's split-KV Inkling relative-bias partials."""

from functools import partial
from typing import NamedTuple

import torch
import triton
import triton.language as tl

from inkling_kv12.ops import _finish_e, stock_config
from .storage import CAP, CHUNK, DIM, EXP_LO, PAGE, STAGE_MASK, STAGE_SHIFT, validate_paged, validate_paged_geometry

_ESCAPE_CAP = tl.constexpr(CAP)
_ESCAPE_CHUNK = tl.constexpr(CHUNK)
_EXPONENT_BASE = tl.constexpr(EXP_LO)
_STAGE_MASK = tl.constexpr(STAGE_MASK)
_STAGE_SHIFT = tl.constexpr(STAGE_SHIFT)


class CompactConfig(NamedTuple):
    query_tile: int
    key_load_tile: int
    num_warps: int
    num_stages: int


def _compact_config(qlen, group, *, ragged):
    # Ragged offsets and wide verify accumulators both add live registers.
    # Keep the four-warp dot and bound their query-head rows per CTA instead.
    if ragged or qlen == 1 or qlen > 8:
        return CompactConfig(16, CHUNK, 4, 1)
    rows = max(16, triton.next_power_of_2(qlen * group))
    return CompactConfig(min(rows, 32), CHUNK, 4, 2)


@triton.jit
def _unpack_tile(SM, EX, ESC, OFF, base, head_lane, slot0, cols, dims, valid, row_base,
                 PAGE: tl.constexpr, D: tl.constexpr, BLOCK_N: tl.constexpr):
    sm = tl.load(SM + base + row_base * D + dims[None, :], valid[:, None], other=0)
    half = tl.arange(0, D // 2)
    ex = tl.load(EX + base + row_base * (D // 2) + half[None, :], valid[:, None], other=0)
    offsets = OFF + base // 4 + head_lane * (PAGE // _ESCAPE_CHUNK + 1) + slot0 // _ESCAPE_CHUNK
    start = tl.load(offsets)
    end = tl.load(offsets + BLOCK_N // _ESCAPE_CHUNK)
    good_offsets = (start >= 0) & (end >= start) & (end <= _ESCAPE_CAP)
    start = tl.where(good_offsets, start, 0)
    end = tl.where(good_offsets, end, 0)
    stream = ESC + base // 4 + head_lane * _ESCAPE_CAP
    entries = tl.arange(0, 16)
    first = tl.load(stream + start + entries, start + entries < end, other=0)
    tile = _finish_e(sm, ex, first, stream, 0, start, end, slot0, cols, dims,
                     _EXPONENT_BASE, BLOCK_N, D, 16)
    return tl.where(valid[:, None], tile, tl.full((), 0, tl.bfloat16))


@triton.jit
def _join_rows(first, second):
    rows: tl.constexpr = first.shape[0]
    dims: tl.constexpr = first.shape[1]
    return tl.reshape(tl.permute(tl.join(first, second), (2, 0, 1)), (2 * rows, dims))


@triton.jit
def _unpack_pair(SM, EX, ESC, OFF, base, head_lane, slot0, cols, dims, valid_rows, row_base,
                 HEADS: tl.constexpr, PAGE: tl.constexpr, D: tl.constexpr, LOAD_N: tl.constexpr):
    first = _unpack_tile(SM, EX, ESC, OFF, base, head_lane, slot0, cols, dims,
                         cols < valid_rows, row_base, PAGE, D, LOAD_N)
    second = _unpack_tile(SM, EX, ESC, OFF, base, head_lane, slot0 + LOAD_N, cols, dims,
                          LOAD_N + cols < valid_rows, row_base + LOAD_N * HEADS, PAGE, D, LOAD_N)
    return _join_rows(first, second)


@triton.jit
def _load_tile(SM, EX, ESC, OFF, Refs, Flags, Staging, Overflow, page, head, slot0,
               cols, dims, valid_rows, KIND: tl.constexpr, STRIDE: tl.constexpr, NPAGES: tl.constexpr,
               HEADS: tl.constexpr, PAGE: tl.constexpr, D: tl.constexpr, BLOCK_N: tl.constexpr,
               NSTAGE: tl.constexpr, NOVERFLOW: tl.constexpr, LOAD_N: tl.constexpr):
    live = (page > 0) & (page < NPAGES)
    base = tl.where(live, page, 0).to(tl.int64) * STRIDE
    flags = tl.load(Flags + base // 4, live, other=0)
    raw_width = PAGE * HEADS * D
    rows = slot0 + cols
    valid = cols < valid_rows
    row_base = (KIND * PAGE + rows[:, None]) * HEADS + head
    tile = tl.full((BLOCK_N, D), 0, tl.bfloat16)
    if flags == 1:
        head_lane = head * 2 + KIND
        ref = tl.load(Refs + base // 4 + head_lane)
        if ref == 0:
            if LOAD_N == BLOCK_N:
                tile = _unpack_tile(SM, EX, ESC, OFF, base, head_lane, slot0, cols, dims,
                                    valid, row_base, PAGE, D, BLOCK_N)
            else:
                # Decode escapes in 32-key chunks. Join the BF16 words in key
                # order so both dots and softmax still see the full stock tile.
                tl.static_assert(BLOCK_N == 2 * LOAD_N or BLOCK_N == 4 * LOAD_N)
                sub_cols = tl.arange(0, LOAD_N)
                sub_rows = (KIND * PAGE + slot0 + sub_cols[:, None]) * HEADS + head
                first = _unpack_pair(SM, EX, ESC, OFF, base, head_lane, slot0, sub_cols, dims,
                                     valid_rows, sub_rows, HEADS, PAGE, D, LOAD_N)
                if BLOCK_N == 4 * LOAD_N:
                    second = _unpack_pair(SM, EX, ESC, OFF, base, head_lane, slot0 + 2 * LOAD_N, sub_cols, dims,
                                          valid_rows - 2 * LOAD_N, sub_rows + 2 * LOAD_N * HEADS,
                                          HEADS, PAGE, D, LOAD_N)
                    tile = _join_rows(first, second)
                else:
                    tile = first
        elif (ref > 0) & (ref <= NOVERFLOW):
            words = tl.load(Overflow + (ref - 1).to(tl.int64) * PAGE * D
                            + rows[:, None] * D + dims[None, :], valid[:, None], other=0)
            tile = words.to(tl.bfloat16, bitcast=True)
    elif flags >= 2:
        slot = flags >> _STAGE_SHIFT
        count = (flags & _STAGE_MASK) - 1
        if (slot < NSTAGE) & (count > 0) & (count <= PAGE):
            words = tl.load(Staging + slot.to(tl.int64) * (2 * raw_width)
                            + row_base * D + dims[None, :], (valid & (rows < count))[:, None], other=0)
            tile = words.to(tl.bfloat16, bitcast=True)
    return tile


@triton.jit
def _load_rows(SM, EX, ESC, OFF, Refs, Flags, Staging, Overflow, Pages, Ends, Out,
               STRIDE: tl.constexpr, NPAGES: tl.constexpr, HEADS: tl.constexpr,
               PAGE: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr,
               NSTAGE: tl.constexpr, NOVERFLOW: tl.constexpr, SKIP_EMPTY: tl.constexpr = False):
    dst_page, head, chunk = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    page = tl.load(Pages + dst_page)
    end = tl.load(Ends + dst_page)
    if SKIP_EMPTY:
        # Captured gathers launch a fixed grid: slots no request uses (end 0) are
        # neither read nor written, and the readers never reference them.
        if end == 0:
            return
    rows, dims = tl.arange(0, BLOCK), tl.arange(0, D)
    slot = chunk * BLOCK
    for kind in tl.static_range(2):
        tile = tl.full((BLOCK, D), 0, tl.bfloat16)
        if slot < end:
            tile = _load_tile(SM, EX, ESC, OFF, Refs, Flags, Staging, Overflow,
                              page, head, slot, rows, dims, end - slot, kind,
                              STRIDE, NPAGES, HEADS, PAGE, D, BLOCK, NSTAGE, NOVERFLOW, _ESCAPE_CHUNK)
        dst = (((kind * tl.num_programs(0) + dst_page).to(tl.int64) * PAGE + slot + rows[:, None])
               * HEADS + head) * D
        tl.store(Out + dst + dims[None, :], tile.to(tl.int16, bitcast=True))


@triton.jit
def _packed_partials(
    Q,
    SM, EX, ESC, OFF, Refs, Flags, Staging, Overflow,
    PageTable,
    SeqLens,
    CuQ,
    Bias,
    OPartial,
    MPartial,
    LPartial,
    scale,
    tokens,
    SQ0: tl.constexpr,
    SQ1: tl.constexpr,
    STRIDE: tl.constexpr,
    NPAGES: tl.constexpr,
    SP: tl.constexpr,
    PT_WIDTH: tl.constexpr,
    NSTAGE: tl.constexpr,
    NOVERFLOW: tl.constexpr,
    SB0: tl.constexpr,
    SB1: tl.constexpr,
    HEADS_K: tl.constexpr,
    QLEN: tl.constexpr,
    GROUP: tl.constexpr,
    ROWS: tl.constexpr,
    D: tl.constexpr,
    PAGE: tl.constexpr,
    EXTENT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    LOAD_N: tl.constexpr,
    PDL: tl.constexpr,
    RAGGED: tl.constexpr,
):
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    b = tl.program_id(0) // HEADS_K
    hk = tl.program_id(0) % HEADS_K
    split = tl.program_id(1)
    splits = tl.num_programs(1)
    seqlen = tl.minimum(tl.maximum(tl.load(SeqLens + b), 0), PT_WIDTH * PAGE)
    tiles = tl.cdiv(seqlen, BLOCK_N)
    per_split = tl.cdiv(tiles, splits)
    tile_lo = split * per_split
    tile_hi = tl.minimum(tile_lo + per_split, tiles)

    rows = tl.arange(0, ROWS)
    if QLEN * GROUP > ROWS:
        rows += tl.program_id(2) * ROWS
    if RAGGED:
        start = tl.minimum(tl.maximum(tl.load(CuQ + b), 0), tokens)
        end = tl.minimum(tl.maximum(tl.load(CuQ + b + 1), start), tokens)
        length = tl.minimum(end - start, QLEN)
    else:
        start = b * QLEN
        length = QLEN
    head_mask = rows < length * GROUP
    token = (start + rows // GROUP).to(tl.int64)
    qh = hk * GROUP + rows % GROUP
    q_pos = seqlen - length + rows // GROUP
    d = tl.arange(0, D)
    q = tl.load(
        Q + token[:, None] * SQ0 + qh[:, None] * SQ1 + d[None, :],
        head_mask[:, None],
        other=0,
    )
    m = tl.full((ROWS,), float("-inf"), tl.float32)
    l = tl.zeros((ROWS,), tl.float32)
    acc = tl.zeros((ROWS, D), tl.float32)
    cols = tl.arange(0, BLOCK_N)
    for tile in range(tile_lo, tile_hi):
        key_start = tile * BLOCK_N
        pos = key_start + cols
        valid_rows = seqlen - key_start
        valid = cols < valid_rows
        col = key_start // PAGE
        page = tl.load(PageTable + b.to(tl.int64) * SP + col, col < PT_WIDTH, other=0)
        slot0 = key_start % PAGE
        k = _load_tile(SM, EX, ESC, OFF, Refs, Flags, Staging, Overflow, page, hk, slot0,
                       cols, d, valid_rows, 0, STRIDE, NPAGES, HEADS_K, PAGE, D, BLOCK_N, NSTAGE, NOVERFLOW, LOAD_N)
        s = tl.dot(q, tl.trans(k)) * scale
        dist = q_pos[:, None] - pos[None, :]
        keep = valid[None, :] & (dist >= 0)
        bias = tl.load(
            Bias + token[:, None] * SB0 + qh[:, None] * SB1 + dist,
            head_mask[:, None] & keep & (dist < EXTENT),
            other=0,
        ).to(tl.float32)
        s = tl.where(keep, s + bias, float("-inf"))
        m_new = tl.maximum(m, tl.max(s, 1))
        m_ref = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2((m - m_ref) * 1.4426950408889634)
        p = tl.exp2((s - m_ref[:, None]) * 1.4426950408889634)
        l = l * alpha + tl.sum(p, 1)
        v = _load_tile(SM, EX, ESC, OFF, Refs, Flags, Staging, Overflow, page, hk, slot0,
                       cols, d, valid_rows, 1, STRIDE, NPAGES, HEADS_K, PAGE, D, BLOCK_N, NSTAGE, NOVERFLOW, LOAD_N)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m = m_new
    stat = (split * tokens + token) * (HEADS_K * GROUP) + qh
    tl.store(MPartial + stat, m, head_mask)
    tl.store(LPartial + stat, l, head_mask)
    tl.store(OPartial + stat[:, None] * D + d[None, :], acc, head_mask[:, None])


def _planes(layer):
    return (*layer.planes, layer.stage, layer.overflow)


def _gather_pages(layer, pages, ends, block):
    out = torch.empty((2, len(pages), PAGE, layer.heads, DIM), dtype=torch.int16, device=layer.device)
    ids = torch.tensor(pages, dtype=torch.int32, device=layer.device)
    counts = torch.tensor(ends, dtype=torch.int32, device=layer.device)
    _load_rows[(len(pages), layer.heads, PAGE // block)](
        *_planes(layer), ids, counts, out, layer.page_stride, layer.pages, layer.heads,
        PAGE, DIM, block, layer.stage.shape[0], layer.overflow.shape[0],
    )
    return out


def load_test_pages(layer, pages, block=32):
    if not 0 < pages < layer.pages or block not in (32, 64, 128):
        raise ValueError("inkling_kv12s: invalid diagnostic page gather")
    return _gather_pages(layer, list(range(1, pages + 1)), [PAGE] * pages, block).transpose(0, 1)


def gather_for_fa4(layer, q, page_table, seq_k, cu_q=None, qmax=None):
    """Decode the distinct referenced pages and remap FA4's page table in place order."""
    lengths, _ = validate_paged(q, layer, page_table, seq_k, cu_q, qmax)
    width = max(((length + PAGE - 1) // PAGE for length in lengths), default=0)
    rows = page_table[:len(lengths), :width].tolist()
    # Stock FA4 uses page zero for inactive table lookups.
    pages = {0: 0}
    ends = [0]
    remapped = [[0] * width for _ in lengths]
    for row, (length, ids) in enumerate(zip(lengths, rows)):
        for i, page in enumerate(ids[:(length + PAGE - 1) // PAGE]):
            if not 0 <= page < layer.pages:
                raise ValueError("inkling_kv12s: invalid physical page in FA4 gather")
            if page not in pages:
                pages[page] = len(ends)
                ends.append(0)
            index = pages[page]
            if page:
                ends[index] = max(ends[index], min(PAGE, length - i * PAGE))
            remapped[row][i] = index
    table = torch.zeros_like(page_table)
    if width:
        table[:len(lengths), :width] = torch.tensor(remapped, dtype=torch.int32, device=layer.device)
    kv = _gather_pages(layer, list(pages), ends, 32).view(torch.bfloat16)
    return kv[0], kv[1], table


class PrefixScratch:
    """Bounded BF16 copy of the pages one eager prefill visit reads.

    The fused prefill kernel reconstructs packed tiles per warp at ~15 TF/s, against
    ~64 TF/s on BF16 tiles. Decoding the referenced pages once per visit with the
    bitwise loader (_load_rows) and running the BF16 kernel on the copy gives the
    BF16 kernel's exact rows. One fixed buffer is allocated at startup and charged
    to the pool budget (growing it per call let the caching allocator keep every
    smaller copy); a larger batch keeps the packed in-kernel loader."""

    def __init__(self, max_pages, heads, device):
        self.max_pages = max_pages
        self.buffer = torch.empty(2 * max_pages * PAGE * heads * DIM, dtype=torch.int16, device=device)

    @staticmethod
    def nbytes(max_pages, heads):
        return 2 * max_pages * PAGE * heads * DIM * 2

    def gather(self, layer, q, page_table, seq_k, cu_q):
        lengths, _ = validate_paged(q, layer, page_table, seq_k, cu_q)
        width = max(((length + PAGE - 1) // PAGE for length in lengths), default=0)
        rows = page_table[:len(lengths), :width].tolist()
        index = {0: 0}
        ends = [0]
        remapped = [[0] * width for _ in lengths]
        for row, (length, ids) in enumerate(zip(lengths, rows)):
            for i, page in enumerate(ids[:(length + PAGE - 1) // PAGE]):
                if not 0 <= page < layer.pages:
                    raise ValueError("inkling_kv12s: invalid physical page in prefill gather")
                slot = index.get(page)
                if slot is None:
                    slot = index[page] = len(ends)
                    ends.append(0)
                if page:
                    ends[slot] = max(ends[slot], min(PAGE, length - i * PAGE))
                remapped[row][i] = slot
        count = len(ends)
        if count > self.max_pages:
            return None
        need = 2 * count * PAGE * layer.heads * DIM
        if need > self.buffer.numel():
            return None
        out = self.buffer[:need].view(2, count, PAGE, layer.heads, DIM)
        table = torch.zeros_like(page_table)
        if width:
            table[:len(lengths), :width] = torch.tensor(remapped, dtype=torch.int32, device=layer.device)
        ids = torch.tensor(list(index), dtype=torch.int32, device=layer.device)
        counts = torch.tensor(ends, dtype=torch.int32, device=layer.device)
        _load_rows[(count, layer.heads, PAGE // CHUNK)](
            *_planes(layer), ids, counts, out, layer.page_stride, layer.pages, layer.heads,
            PAGE, DIM, CHUNK, layer.stage.shape[0], layer.overflow.shape[0],
        )
        kv = out.view(torch.bfloat16)
        return kv[0], kv[1], table

    def gather_captured(self, layer, page_table, seq_k):
        """The same BF16 copy with device-side metadata only (fixed shapes, no host
        reads), for prefill CUDA graphs. Pages are not deduplicated: request b's
        page column c goes to slot 1 + (pages of requests < b) + c, slot 0 being
        the sink. Returns (k, v, table, fits); `fits` is a device bool that is false
        when the batch needs more slots than the scratch holds, and the caller then
        reads the packed store directly (nothing was gathered)."""
        cap = self.max_pages
        if cap < 2:
            raise ValueError("inkling_kv12s: a captured prefix scratch needs at least two pages")
        batch, width = seq_k.numel(), page_table.shape[1]
        device = layer.device
        seq = seq_k.to(torch.int64).clamp(0, width * PAGE)
        pages = (seq + PAGE - 1) // PAGE
        first = torch.cumsum(pages, 0) - pages + 1
        fits = pages.sum() + 1 <= cap
        col = torch.arange(width, device=device)
        use = (col[None, :] < pages[:, None]) & fits
        slot = first[:, None] + col[None, :]
        table = torch.zeros_like(page_table)
        table[:batch, :width] = torch.where(use, slot, 0).to(page_table.dtype)
        dest = torch.where(use, slot, cap).reshape(-1)
        ids = torch.zeros(cap + 1, dtype=torch.int32, device=device)
        ids.scatter_(0, dest, page_table[:batch, :width].reshape(-1).to(torch.int32))
        rows = (seq[:, None] - col[None, :] * PAGE).clamp(0, PAGE)
        ends = torch.zeros(cap + 1, dtype=torch.int32, device=device)
        ends.scatter_(0, dest, torch.where(use, rows, 0).reshape(-1).to(torch.int32))
        out = self.buffer[:2 * cap * PAGE * layer.heads * DIM].view(2, cap, PAGE, layer.heads, DIM)
        _load_rows[(cap, layer.heads, PAGE // CHUNK)](
            *_planes(layer), ids, ends, out, layer.page_stride, layer.pages, layer.heads,
            PAGE, DIM, CHUNK, layer.stage.shape[0], layer.overflow.shape[0], True,
        )
        kv = out.view(torch.bfloat16)
        return kv[0], kv[1], table, fits


@triton.jit(do_not_specialize=["stride", "width"])
def _mark_pages(Table, Seq, Ends, stride, width,
                PAGES: tl.constexpr, PAGE: tl.constexpr, BLOCK: tl.constexpr):
    request = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    seq = tl.load(Seq + request)
    end = tl.minimum(seq - col * PAGE, PAGE)
    page = tl.load(Table + request.to(tl.int64) * stride + col,
                   (col < width) & (end > 0), other=0)
    tl.atomic_max(Ends + page, end, sem="relaxed", mask=(page > 0) & (page < PAGES))


@triton.jit
def _collect_pages(Ends, Ids, Count, PAGES: tl.constexpr, BLOCK: tl.constexpr):
    page = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = tl.load(Ends + page, page < PAGES, other=0) > 0
    n = tl.sum(active.to(tl.int32), 0)
    if n > 0:
        start = tl.atomic_add(Count, n, sem="relaxed")
        index = start + tl.cumsum(active.to(tl.int32)) - 1
        tl.store(Ids + index, page, active)


@triton.jit
def _gather_active(SM, EX, ESC, OFF, Refs, Flags, Stage, Overflow,
                   Pages, Ends, Count, Key, Value, STRIDE: tl.constexpr, NPAGES: tl.constexpr,
                   HEADS: tl.constexpr, PAGE: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr,
                   STAGES: tl.constexpr, OVERFLOWS: tl.constexpr, CTAS: tl.constexpr):
    worker, head, chunk = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    slot = chunk * BLOCK
    rows, dims = tl.arange(0, BLOCK), tl.arange(0, D)
    count = tl.load(Count)
    for idx in range(worker, count, CTAS):
        page = tl.load(Pages + idx)
        end = tl.load(Ends + page)
        dst = ((page.to(tl.int64) * PAGE + slot + rows[:, None]) * HEADS + head) * D + dims[None, :]
        for kind in tl.static_range(2):
            tile = tl.full((BLOCK, D), 0, tl.bfloat16)
            if slot < end:
                tile = _load_tile(SM, EX, ESC, OFF, Refs, Flags, Stage, Overflow,
                                  page, head, slot, rows, dims, end - slot, kind,
                                  STRIDE, NPAGES, HEADS, PAGE, D, BLOCK, STAGES, OVERFLOWS, _ESCAPE_CHUNK)
            if kind == 0:
                tl.store(Key + dst, tile)
            else:
                tl.store(Value + dst, tile)


class FA4Workspace:
    """One reusable raw lane at its original physical IDs for serial FA4 calls."""

    def __init__(self, pages, heads, device):
        self.key, self.value = (
            torch.empty((pages, PAGE, heads, DIM), dtype=torch.bfloat16, device=device) for _ in range(2)
        )
        self.ends = torch.empty(pages, dtype=torch.int32, device=device)
        self.active = torch.empty_like(self.ends)
        self.count = torch.empty((), dtype=torch.int32, device=device)

    def gather(self, layer, table, seq):
        if (table.ndim != 2 or seq.ndim != 1 or table.shape[0] < seq.numel()
                or table.stride(-1) != 1 or not seq.is_contiguous()
                or table.dtype != torch.int32 or seq.dtype != torch.int32
                or table.device != layer.device or seq.device != layer.device):
            raise ValueError("inkling_kv12s: invalid FA4 gather metadata")
        # Shapes/addresses are fixed. The device deduplicates physical IDs and
        # selects the longest tail when several requests share the same page.
        self.ends.zero_()
        self.count.zero_()
        shape = (layer.pages, PAGE, layer.heads, DIM)
        elements = layer.pages * PAGE * layer.heads * DIM
        k = self.key.flatten()[:elements].view(shape)
        v = self.value.flatten()[:elements].view(shape)
        k[0].zero_()
        v[0].zero_()
        if seq.numel() and table.shape[1] and layer.pages > 1:
            _mark_pages[(seq.numel(), triton.cdiv(table.shape[1], 128))](
                table, seq, self.ends, table.stride(0), table.shape[1], layer.pages, PAGE, 128,
            )
            _collect_pages[(triton.cdiv(layer.pages, 128),)](
                self.ends, self.active, self.count, layer.pages, 128,
            )
            workers = min(16, layer.pages - 1)
            _gather_active[(workers, layer.heads, PAGE // CHUNK)](
                *_planes(layer), self.active, self.ends, self.count, k, v,
                layer.page_stride, layer.pages, layer.heads, PAGE, DIM, CHUNK,
                layer.stage.shape[0], layer.overflow.shape[0], workers, num_warps=4,
            )
        return k, v, table


def prepare_partials(layer, q, page_table, seqused_k, rel_bias, scale, *, config=None, pdl=True,
                     cu_q=None, qmax=None, resources=None, device_metadata=False):
    """Validate metadata and prepare a reusable kernel launch and its outputs."""
    if device_metadata:
        validate_paged_geometry(q, layer, page_table, seqused_k, cu_q)
        if cu_q is not None and (qmax is None or qmax < 1):
            raise ValueError("inkling_kv12s: ragged capture requires a static query bound")
        max_q = qmax
    else:
        _, lengths = validate_paged(q, layer, page_table, seqused_k, cu_q, qmax)
        max_q = max(lengths, default=0)
    tokens, heads_q, d = q.shape
    batch = seqused_k.numel()
    heads_k = layer.heads
    ragged = cu_q is not None
    if (not batch or not tokens or q.stride(-1) != 1 or rel_bias.ndim != 3
            or rel_bias.shape[:2] != q.shape[:2] or not rel_bias.shape[-1]
            or rel_bias.stride(-1) != 1 or rel_bias.device != q.device):
        raise ValueError("inkling_kv12s: unsupported compact attention layout")
    qlen = max(1, max_q) if ragged else tokens // batch
    group = heads_q // heads_k
    if config is None:
        sms = torch.cuda.get_device_properties(q.device).multi_processor_count
        config = stock_config(batch, heads_k, qlen, page_table.shape[1] * PAGE, sms)
    splits, block_n = config
    if splits < 1 or block_n not in (32, 64, 128):
        raise ValueError("inkling_kv12s: unsupported compact split or tile width")
    cfg = _compact_config(qlen, group, ragged=ragged)
    # Independent query tiles share the stock split-KV boundaries. They never
    # split one query head's softmax or change the key tile it sees.
    query_tiles = triton.cdiv(qlen * group, cfg.query_tile)
    o = torch.empty((splits, tokens, 1, heads_q, d), dtype=torch.float32, device=q.device)
    m = torch.empty(o.shape[:-1], dtype=o.dtype, device=q.device)
    l = torch.empty_like(m)
    if ragged:
        o.zero_()
        m.fill_(float("-inf"))
        l.zero_()
    launch = partial(
        _packed_partials[(batch * heads_k, splits, query_tiles)],
        q, *_planes(layer), page_table, seqused_k, cu_q if ragged else seqused_k,
        rel_bias, o, m, l, scale, tokens,
        q.stride(0), q.stride(1), layer.page_stride, layer.pages, page_table.stride(0),
        page_table.shape[1], layer.stage.shape[0], layer.overflow.shape[0],
        rel_bias.stride(0), rel_bias.stride(1), heads_k, qlen, group,
        cfg.query_tile, d, PAGE, rel_bias.shape[-1], block_n, cfg.key_load_tile, pdl, ragged,
        num_warps=cfg.num_warps, num_stages=cfg.num_stages,
        **({"launch_pdl": True} if pdl else {}),
    )
    if resources is not None:
        resources.update(cfg._asdict(), key_tile=block_n, splits=splits, query_tiles=query_tiles)
    return launch, (o, m, l)


def partials(layer, q, page_table, seqused_k, rel_bias, scale, *, config=None, pdl=True, cu_q=None, qmax=None,
             resources=None, device_metadata=False):
    """Run compact attention with the stock split-KV arithmetic and a packed tile loader."""
    launch, outputs = prepare_partials(layer, q, page_table, seqused_k, rel_bias, scale,
                                       config=config, pdl=pdl, cu_q=cu_q, qmax=qmax, resources=resources,
                                       device_metadata=device_metadata)
    kernel = launch()
    if resources is not None:
        resources.update(registers=kernel.n_regs, spills=kernel.n_spills, shared_bytes=kernel.metadata.shared)
    return outputs
