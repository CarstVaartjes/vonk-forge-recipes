"""Per-request token history on the device (int32), with its coverage.

hist [rows, cap] is a view of one flat buffer whose tail holds SINKS * SINK_WIDTH scratch cells; cov [rows + SINKS]
int64: the end of the contiguous written history of each slot, then one sink per batch row. Every masked write goes to
a DISTINCT sink (token (b, j) of a masked row to scratch cell b * width + j; a masked row's coverage to cov[rows + b]),
so no index_put has duplicate destinations. Host bookkeeping per slot: the allocation generation the history belongs to and
the host-known valid end of its prefill part.

Prefill (seed): an extend [prefix, end) of a request is written as fill_ids[0:end] when the slot's generation changed
or the extend starts past the host valid end (a gap), else as fill_ids[prefix:end] (a later chunk, or a rewind). On
CUDA the tokens and their flat positions go through pinned staging kept alive until an event recorded after the copy
has completed (reaped without waiting, on seeds and steps). The host ownership is published after the writes are
enqueued.

Decode (step): with anchor position p0 (the slot's committed length), a row is covered when 0 <= p0 < cap and
cov[slot] >= p0. Writes are bounds-checked (0 <= slot < rows, 0 <= pos < cap); covered rows get
cov = min(cap, p0 + offset + width).
"""

from array import array

import torch

SINKS = 256       # batch rows a step may hold
SINK_WIDTH = 16   # tokens per row a step writes at most (1 + G)


class History:
    def __init__(self, rows, cap, device):
        self.rows, self.cap = int(rows), int(cap)
        self.device = device
        self.flat = torch.zeros((self.rows * self.cap + SINKS * SINK_WIDTH,), dtype=torch.int32, device=device)
        self.hist = self.flat[: self.rows * self.cap].view(self.rows, self.cap)
        self.cov = torch.zeros((self.rows + SINKS,), dtype=torch.int64, device=device)
        self.gen = [-1] * self.rows          # host: allocation generation of each slot's history
        self.valid_end = [0] * self.rows     # host: prefill-part end written for that generation
        self._inflight = []                  # (event, pinned buffers) kept alive until the copy has run

    @staticmethod
    def bytes_for(rows, cap):
        return (int(rows) * int(cap) + SINKS * SINK_WIDTH) * 4 + (int(rows) + SINKS) * 8

    @property
    def nbytes(self):
        return self.flat.numel() * self.flat.element_size() + self.cov.numel() * self.cov.element_size()

    def reap(self):
        if self._inflight:
            self._inflight = [(ev, bufs) for ev, bufs in self._inflight if not ev.query()]

    def _h2d(self, t, keep):
        if not self.hist.is_cuda:
            return t
        pinned = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
        pinned.copy_(t)
        keep.append(pinned)
        d = pinned.to(self.device, non_blocking=True)
        keep.append(d)
        return d

    def seed(self, entries):
        """entries: [(slot, generation, prefix, end, tokens)] (tokens: the request's fill ids). Returns the slots
        written. A span past the request's fill ids or the capacity invalidates the slot (it is never matched until a
        later extend reseeds it); an empty rewind moves the slot's coverage back to its prefix."""
        pos_parts, tok_parts, cov_parts, owners = [], [], [], []
        for slot, generation, prefix, end, tokens in entries:
            slot, generation, prefix, end = int(slot), int(generation), int(prefix), int(end)
            if not (0 <= slot < self.rows):
                continue
            if prefix < 0 or end < prefix or end > len(tokens) or end > self.cap:
                owners.append((slot, -1, 0))                 # unusable span: the slot is not matched
                cov_parts.append((slot, 0))
                continue
            fresh = self.gen[slot] != generation or prefix > self.valid_end[slot]
            start = 0 if fresh else prefix
            if start >= end:
                if not fresh and end < self.valid_end[slot]:  # an empty rewind
                    owners.append((slot, generation, end))
                    cov_parts.append((slot, end))
                continue
            if isinstance(tokens, array) and tokens.typecode == "q":
                t = torch.frombuffer(memoryview(tokens)[start:end].cast("B"), dtype=torch.int64)
            else:
                t = torch.tensor(list(tokens[start:end]), dtype=torch.int64)
            tok_parts.append(t.to(torch.int32))
            pos_parts.append(torch.arange(start, end, dtype=torch.int64) + slot * self.cap)
            cov_parts.append((slot, end))
            owners.append((slot, generation, end))
        if not owners:
            return []
        keep = []
        if tok_parts:
            d_toks = self._h2d(torch.cat(tok_parts), keep)
            d_pos = self._h2d(torch.cat(pos_parts), keep)
            self.flat.index_copy_(0, d_pos, d_toks)
        d_cov = self._h2d(torch.tensor(cov_parts, dtype=torch.int64), keep)
        self.cov.index_copy_(0, d_cov[:, 0], d_cov[:, 1])
        if self.hist.is_cuda:
            ev = torch.cuda.Event()
            ev.record()
            self._inflight.append((ev, keep))
        for slot, generation, end in owners:   # host ownership after the writes are enqueued
            self.gen[slot], self.valid_end[slot] = generation, end
        self.reap()
        return [s for s, g, _ in owners if g >= 0]

    def covered(self, slots, p0, host_ok):
        """[bs] bool: rows whose history may be read and continued this step."""
        s = slots.to(torch.int64)
        p = p0.to(torch.int64)
        in_range = (s >= 0) & (s < self.rows)
        safe = torch.where(in_range, s, torch.zeros_like(s))
        return host_ok & in_range & (p >= 0) & (p < self.cap) & (self.cov[safe] >= p)

    def write_block(self, slots, p0, tokens, ok, offset=0):
        """Write tokens [bs, W] at p0 + offset .. for ok rows (positions inside [0, cap)); ok rows get
        cov = min(cap, p0 + offset + W). A step writes the anchor (offset 0, W 1) before matching and the drafted
        slots (offset 1, W G) after."""
        bs, width = tokens.shape
        if bs > SINKS or width > SINK_WIDTH:
            raise ValueError(f"block {bs} x {width} above the history sinks ({SINKS} x {SINK_WIDTH})")
        dev = tokens.device
        s = slots.to(torch.int64)
        p = p0.to(torch.int64)
        b = torch.arange(bs, device=dev, dtype=torch.int64)
        j = torch.arange(width, device=dev, dtype=torch.int64)
        pos = p[:, None] + int(offset) + j[None, :]
        row_ok = ok & (s >= 0) & (s < self.rows)
        keep = row_ok[:, None] & (pos >= 0) & (pos < self.cap)
        sink = self.rows * self.cap + b[:, None] * width + j[None, :]   # distinct scratch cells
        flat_idx = torch.where(keep, s[:, None] * self.cap + pos, sink)
        self.flat[flat_idx.reshape(-1)] = tokens.to(torch.int32).reshape(-1)
        crow = torch.where(row_ok, s, self.rows + b)            # distinct coverage sinks
        newcov = torch.clamp(p + int(offset) + width, max=self.cap)
        self.cov[crow] = torch.where(row_ok, newcov, torch.zeros_like(newcov))
        self.reap()
