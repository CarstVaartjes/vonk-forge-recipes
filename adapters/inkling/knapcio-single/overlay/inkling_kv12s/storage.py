"""Packed page layout and the reference encoder, decoder and ownership model."""

from collections import defaultdict
from dataclasses import dataclass
import heapq
import os

import torch

from . import side_count

PAGE = 128
DIM = 128
CHUNK = 32
CAP = 64
EXP_LO = 116
PACK_BATCH = 8
EMPTY = 0
PACKED = 1
STAGED = 2
# Stage flags store slot in the high bits and valid rows plus one in the low byte.
STAGE_SHIFT = 8
STAGE_MASK = (1 << STAGE_SHIFT) - 1


class CapacityError(RuntimeError):
    """The eager store cannot retain the requested exact values."""


@dataclass(frozen=True)
class Layout:
    heads: int

    @property
    def shapes(self):
        h = self.heads
        return ((2, PAGE, h, DIM), (2, PAGE, h, DIM // 2), (h, 2, CAP),
                (h, 2, PAGE // CHUNK + 1), (h, 2), (1,))

    @property
    def offsets(self):
        from math import prod

        offsets = [0]
        for shape, width in zip(self.shapes, (1, 1, 4, 4, 4, 4)):
            offsets.append(offsets[-1] + prod(shape) * width)
        return tuple(offsets)

    @property
    def size(self):
        return (self.offsets[-1] + 127) // 128 * 128


# Dedicated slots (beyond the live staging slots) for full pages whose page-heads
# found the overflow arena exhausted: such a page stays raw instead of being sealed.
FALLBACK_PAGES = side_count("INKLING_KV12S_FALLBACK_PAGES", "64", 0)

# Per-lane event counters of a device lane: pages kept raw in a fallback slot, pages
# kept raw in a live staging slot (both side pools exhausted), and row losses
# (staging exhausted too).
EVENT_FALLBACK, EVENT_KEPT, EVENT_LOST = 0, 1, 2
EVENT_WORDS = 3


def stage_slots(running, verify_rows, *, overlap=False):
    # A partial prefix can end at row 127 before a tentative run starts.
    # Chunked prefill can also need old and new tails during one publication.
    pages_per_request = max(2, (PAGE - 1 + max(1, verify_rows) + PAGE - 1) // PAGE)
    return (running * (2 if overlap else 1) + 1) * pages_per_request


def raw_lane_bytes(heads):
    """Page-major BF16 K and V of one raw lane inside a page envelope."""
    return 2 * PAGE * heads * DIM * 2


def arenas_bytes(heads, staging_pages, overflow_heads):
    return 2 * PAGE * DIM * (2 * heads * staging_pages + overflow_heads)


def workspace_bytes(heads):
    # The eager Torch encoder and snapshot reader process one batch at a time.
    # Charge 64 B per input word for integer temporaries, indices and staging.
    return PACK_BATCH * 2 * PAGE * heads * DIM * 64


def encode(words):
    """Pack int16 bit patterns [page, K/V, row, head, dim]."""
    bits = words.to(torch.int32) & 0xFFFF
    exponent = (bits >> 7) & 255
    sm = (((bits >> 8) & 128) | (bits & 127)).to(torch.uint8)
    in_window = (exponent >= EXP_LO) & (exponent < EXP_LO + 15)
    code = torch.where(in_window, exponent - EXP_LO, 15)
    ex = (code[..., ::2] | (code[..., 1::2] << 4)).to(torch.uint8)

    n, _, _, h, _ = words.shape
    escaped = ((exponent != 0) & ~in_window).permute(0, 3, 1, 2, 4).reshape(n, h, 2, -1)
    rank = escaped.cumsum(-1, dtype=torch.int32)
    off = torch.cat((rank.new_zeros((n, h, 2, 1)), rank[..., CHUNK * DIM - 1::CHUNK * DIM]), -1)
    keep = (escaped & (rank <= CAP)).nonzero(as_tuple=True)
    page, head, kind, dim = keep
    words = bits[page, kind, dim // DIM, head, dim % DIM]
    entries = (((dim // DIM) << 23) | ((dim % DIM) << 16)).int() | words
    stream = rank.new_zeros((n, h, 2, CAP))
    stream[page, head, kind, (rank[keep] - 1).long()] = entries
    return sm, ex, stream, off


def decode(sm, ex, stream, offsets):
    """Torch reference decoder for diagnostics and bounded CPU snapshots."""
    code = torch.stack((ex & 15, ex >> 4), dim=-1).flatten(-2)
    exponent = torch.where(code == 15, 0, code.to(torch.int32) + EXP_LO)
    count, _, _, heads, _ = sm.shape
    flat = exponent.permute(0, 3, 1, 2, 4).reshape(count, heads, 2, -1).contiguous()
    live = torch.arange(CAP, device=sm.device) < offsets[..., -1, None]
    position = ((stream >> 23) & 127) * DIM + ((stream >> 16) & 127)
    flat.scatter_add_(-1, position.long(), torch.where(live, (stream >> 7) & 255, 0))
    exponent = flat.view(count, heads, 2, PAGE, DIM).permute(0, 2, 3, 1, 4)
    sign = sm.to(torch.int32)
    return (((sign & 128) << 8) | (exponent << 7) | (sign & 127)).to(torch.int16)


@dataclass
class OpenPage:
    slot: int
    end: int


class Layer:
    """One visit over a band. The allocator serializes writes, moves and frees."""

    def __init__(self, raw, heads, page_stride, offset, staging_pages, overflow_heads):
        layout = Layout(heads)
        self.raw = raw
        self.heads = heads
        self.page_stride = page_stride
        self.offset = offset
        self.layout = layout
        self.device = raw.device
        self.pages = raw.numel() // page_stride
        self.stage = torch.empty((staging_pages, 2, PAGE, heads, DIM), dtype=torch.int16, device=self.device)
        self.overflow = torch.empty((overflow_heads, PAGE, DIM), dtype=torch.int16, device=self.device)
        self._free_stage = list(range(staging_pages))
        self._free_overflow = list(range(overflow_heads))
        self.open = {}
        self.spilled = {}
        self.planes = tuple(
            self._plane(start, stop, shape, dtype)
            for start, stop, shape, dtype in zip(
                layout.offsets, layout.offsets[1:], layout.shapes,
                (torch.uint8, torch.uint8, torch.int32, torch.int32, torch.int32, torch.int32),
            )
        )
        self.flags = self.planes[-1][:, 0]

    @classmethod
    def standalone(cls, pages, heads, staging_pages=4, overflow_heads=32, device="cpu"):
        stride = Layout(heads).size
        raw = torch.zeros(pages * stride, dtype=torch.uint8, device=device)
        return cls(raw, heads, stride, 0, staging_pages, overflow_heads)

    def _plane(self, start, stop, shape, dtype):
        # Slice before reinterpreting so every byte range is aligned to dtype.
        raw = self.raw[:self.pages * self.page_stride].view(self.pages, self.page_stride)
        return raw[:, self.offset + start:self.offset + stop].view(dtype).unflatten(-1, shape)

    def _ids(self, values):
        return torch.tensor(values, dtype=torch.long, device=self.device)

    def _check_overflow(self, batches):
        needed = 0
        for _, words in batches():
            exponent = (words >> 7) & 255
            escapes = (exponent != 0) & ((exponent < EXP_LO) | (exponent >= EXP_LO + 15))
            needed += (escapes.sum(dim=(2, 4), dtype=torch.int32) > CAP).count_nonzero().item()
            if needed > len(self._free_overflow):
                raise CapacityError("inkling_kv12s: exact page-head overflow arena exhausted")

    def _pack(self, pages, words):
        sm, ex, stream, off = encode(words)
        overflow_at = (off[..., -1] > CAP).nonzero().tolist()
        refs = torch.zeros((len(pages), self.heads, 2), dtype=torch.int32)
        for i, head, kind in overflow_at:
            slot = heapq.heappop(self._free_overflow)
            self.overflow[slot].copy_(words[i, kind, :, head])
            self.spilled.setdefault(pages[i], []).append(slot)
            refs[i, head, kind] = slot + 1
        ids = self._ids(pages)
        for plane, values in zip(self.planes, (sm, ex, stream, off.clamp_max(CAP), refs.to(self.device))):
            plane.index_copy_(0, ids, values)
        self.flags.index_fill_(0, ids, PACKED)
        for page in pages:
            staged = self.open.pop(page, None)
            if staged is not None:
                heapq.heappush(self._free_stage, staged.slot)

    def prepare_commit(self, pages):
        """Preflight promotion before this or another visit publishes any pages."""
        ready = [page for page in dict.fromkeys(pages) if page in self.open]
        if any(self.open[page].end != PAGE for page in ready):
            raise RuntimeError("inkling_kv12s: accepted prefix contains an unwritten row")

        def batches():
            for start in range(0, len(ready), PACK_BATCH):
                batch = ready[start:start + PACK_BATCH]
                slots = self._ids([self.open[page].slot for page in batch])
                yield batch, self.stage.index_select(0, slots)

        self._check_overflow(batches)

        def publish():
            for batch, words in batches():
                self._pack(batch, words)

        return publish

    def commit(self, pages):
        self.prepare_commit(pages)()

    def write(self, locations, k, v, *, tentative=False):
        self.prepare_write(locations, k, v, tentative=tentative)()

    def prepare_write(self, locations, k, v, *, tentative=False):
        """Count the whole scatter under the eager allocator's single-writer ordering.

        CPU snapshot inputs are copied to the device in page batches on both passes.
        Run the returned publication before changing this visit's arena ownership.
        """
        if (k.ndim != 3 or len(locations) != k.shape[0] or k.shape != v.shape
                or k.shape[1:] != (self.heads, DIM) or k.device != v.device):
            raise ValueError("inkling_kv12s: unsupported KV scatter shape")
        groups = defaultdict(list)
        for source, loc in enumerate(locations):
            if loc <= 0:
                continue
            page, row = divmod(loc, PAGE)
            if page <= 0 or page >= self.pages:
                raise ValueError(f"inkling_kv12s: write location {loc} is outside the usable band")
            groups[page].append((row, source))
        if groups:
            flags = self.flags.index_select(0, self._ids(list(groups))).tolist()
            if any(flag == PACKED for flag in flags):
                raise RuntimeError("inkling_kv12s: write to an immutable committed page")

        staged = []
        full = []
        for page, entries in groups.items():
            entries.sort()
            start, end = entries[0][0], entries[-1][0] + 1
            old = self.open.get(page)
            if [row for row, _ in entries] != list(range(start, end)) or start > (old.end if old else 0):
                raise RuntimeError("inkling_kv12s: noncontiguous KV append")
            sources = [src for _, src in entries]
            pieces = full if not tentative and end == PAGE else staged
            pieces.append((page, start, end, sources))

        stages_needed = sum(page not in self.open for page, *_ in staged)
        stages_released = sum(page in self.open for page, *_ in full)
        if stages_needed > len(self._free_stage) + stages_released:
            raise CapacityError("inkling_kv12s: open-page staging arena exhausted")

        def gather(sources):
            ids = torch.tensor(sources, dtype=torch.long, device=k.device)
            return torch.stack(tuple(t.index_select(0, ids).to(device=self.device, dtype=torch.bfloat16)
                                     .view(torch.int16) for t in (k, v)))

        def batches():
            for begin in range(0, len(full), PACK_BATCH):
                batch = full[begin:begin + PACK_BATCH]
                pages = [page for page, *_ in batch]
                if all(start == 0 for _, start, _, _ in batch):
                    rows = gather([source for _, _, _, sources in batch for source in sources])
                    words = rows.view(2, len(batch), PAGE, self.heads, DIM).transpose(0, 1)
                else:
                    words = self.stage.new_empty((len(batch), 2, PAGE, self.heads, DIM))
                    for i, (page, start, _, sources) in enumerate(batch):
                        if start:
                            words[i, :, :start].copy_(self.stage[self.open[page].slot, :, :start])
                        words[i, :, start:].copy_(gather(sources))
                yield pages, words

        self._check_overflow(batches)

        def publish():
            for pages, words in batches():
                self._pack(pages, words)
            for page, start, end, sources in staged:
                old = self.open.get(page)
                slot = old.slot if old is not None else heapq.heappop(self._free_stage)
                self.stage[slot, :, start:end].copy_(gather(sources))
                self.open[page] = OpenPage(slot, end)
                self.flags[page] = (slot << STAGE_SHIFT) | (end + 1)

        return publish

    def invalidate(self, pages):
        if pages:
            self.flags.index_fill_(0, self._ids(pages), EMPTY)

    def _recycle(self, page):
        staged = self.open.pop(page, None)
        if staged is not None:
            heapq.heappush(self._free_stage, staged.slot)
        for slot in self.spilled.pop(page, ()):
            heapq.heappush(self._free_overflow, slot)

    def release(self, pages):
        self.invalidate(pages)
        for page in pages:
            self._recycle(page)

    def clear(self):
        self.flags.zero_()
        for page in self.open.keys() | self.spilled.keys():
            self._recycle(page)

    def transfer(self, sources, destinations):
        """Transfer side ownership after a copy between disjoint physical pages."""
        moves = [(src, dst) for src, dst in zip(sources, destinations) if src > 0 and dst > 0 and src != dst]
        self.invalidate([src for src, _ in moves])
        for source, dest in moves:
            self._recycle(dest)
            for table in (self.open, self.spilled):
                if source in table:
                    table[dest] = table.pop(source)

    def read_rows(self, locations):
        """Gather requested rows for tests and CPU snapshots, with bounded device workspace."""
        result = torch.empty((len(locations), 2, self.heads, DIM), dtype=torch.int16, device=self.device)
        groups = defaultdict(list)
        for i, loc in enumerate(locations):
            page, row = divmod(loc, PAGE)
            if not 0 <= page < self.pages:
                raise ValueError("inkling_kv12s: read location is outside the band")
            groups[page].append((i, row))
        pages = list(groups)
        for start in range(0, len(pages), PACK_BATCH):
            batch = pages[start:start + PACK_BATCH]
            ids = self._ids(batch)
            flags = self.flags.index_select(0, ids).tolist()
            packed = [page for page, flag in zip(batch, flags) if page > 0 and flag == PACKED]
            packed_rows = {}
            if packed:
                ids = self._ids(packed)
                raw = decode(*(plane.index_select(0, ids) for plane in self.planes[:4]))
                refs = self.planes[4].index_select(0, ids).tolist()
                for i, page in enumerate(packed):
                    for head, pair in enumerate(refs[i]):
                        for kind, ref in enumerate(pair):
                            if ref:
                                if not 0 < ref <= self.overflow.shape[0]:
                                    raise RuntimeError("inkling_kv12s: invalid overflow reference")
                                raw[i, kind, :, head].copy_(self.overflow[ref - 1])
                    packed_rows[page] = raw[i]
            for page, flag in zip(batch, flags):
                dest, rows = zip(*groups[page])
                if page == 0:
                    result.index_fill_(0, self._ids(dest), 0)
                    continue
                if flag == PACKED:
                    words = packed_rows[page]
                elif flag >= STAGED and page in self.open and max(rows) < self.open[page].end:
                    staged = self.open[page]
                    if flag != ((staged.slot << STAGE_SHIFT) | (staged.end + 1)):
                        raise RuntimeError("inkling_kv12s: invalid staging descriptor")
                    words = self.stage[staged.slot]
                else:
                    raise RuntimeError("inkling_kv12s: read of an unwritten KV row")
                values = words.index_select(1, self._ids(rows)).transpose(0, 1)
                result.index_copy_(0, self._ids(dest), values)
        return result.view(torch.bfloat16)


def validate_paged_geometry(q, layer, table, seq_k, cu_q=None):
    batch = seq_k.numel()
    if (q.ndim != 3 or q.dtype != torch.bfloat16 or q.shape[-1] != DIM or not q.shape[1]
            or q.shape[1] % layer.heads or q.device != layer.device
            or table.ndim != 2 or table.shape[0] < batch or table.stride(-1) != 1
            or table.shape[1] * PAGE >= (1 << 31)
            or seq_k.ndim != 1 or not seq_k.is_contiguous()
            or any(t.dtype != torch.int32 or t.device != q.device for t in (table, seq_k))):
        raise ValueError("inkling_kv12s: invalid paged attention tensor geometry")
    if cu_q is None:
        if not batch or q.shape[0] % batch:
            raise ValueError("inkling_kv12s: query rows do not match the cache batch")
    elif (cu_q.shape != (batch + 1,) or not cu_q.is_contiguous()
          or cu_q.dtype != torch.int32 or cu_q.device != q.device):
        raise ValueError("inkling_kv12s: invalid query offset tensor")


def validate_paged(q, layer, table, seq_k, cu_q=None, qmax=None):
    """Validate eager reader geometry before launch and return host sequence lengths."""
    validate_paged_geometry(q, layer, table, seq_k, cu_q)
    batch = seq_k.numel()
    lengths = seq_k.tolist()
    if any(length < 0 or length > table.shape[1] * PAGE for length in lengths):
        raise ValueError("inkling_kv12s: cache sequence length exceeds page-table bounds")
    if cu_q is None:
        query_lengths = [q.shape[0] // batch] * batch
    else:
        offsets = cu_q.tolist()
        if offsets[0] != 0 or offsets[-1] > q.shape[0]:
            raise ValueError("inkling_kv12s: query offsets exceed query bounds")
        query_lengths = [end - start for start, end in zip(offsets, offsets[1:])]
    if any(q_len < 0 or q_len > k_len or (qmax is not None and q_len > qmax)
           for q_len, k_len in zip(query_lengths, lengths)):
        raise ValueError("inkling_kv12s: invalid per-request query length")
    return lengths, query_lengths
