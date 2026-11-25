"""(B) Draft-tail snapshots for the DFlash2 draft KV ring (draft-only state).

    INKLING_DFLASH_TAIL_SNAPSHOTS=8     slots (0 / unset: off); each holds one window of draft KV (40 MiB on Inkling)

When a DFLASH request's extend completes its prompt (end E >= MIN_POSITIONS), the ring rows of positions
[max(E - L, floor), E) are copied into a snapshot slot (L = the draft window, floor = the slot's ring floor), keyed by
a digest of (cache salt, extra key, tokens[:E]) and E. When a later request's first extend starts at a prefix-cache
hit P with a snapshot of the same prefix that ends at E >= P, the ring floor of its fresh slot is lowered to
lo = max(snapshot lo, P - L) (page-aligned up) and the snapshot rows of [lo, P) are copied into the ring before the
extend writes [P, ...): the drafter sees the window an unbroken session would hold. The target path is never touched.

Ordering (one CUDA stream): restores are enqueued inside record_floors (prefill branch, after _join_aux), BEFORE the
stock call publishes the lowered floor (their locs are computed for that floor), and before extend_locs and the
append; a request whose copies did not all enqueue keeps its stock floor. Captures run when forward_batch_generation
returns, after this forward's appends and before any later
forward can reuse the slot. Every copy is stream-ordered with the ring writes; nothing syncs the host.
"""

import hashlib
import os
import sys
import weakref
from array import array

ENV_SLOTS = "INKLING_DFLASH_TAIL_SNAPSHOTS"
MAX_SLOTS = 64
MIN_POSITIONS = 1024  # capture only prompts of at least this many positions
MIN_RESTORE = 256     # restore only when it adds at least this many positions below the hit
TAIL_CHECK = 32       # cheap pre-filter: the last tokens before E must match before hashing the prefix

STATE = {"slots": 0, "inert": "", "charged": None, "stores": weakref.WeakSet(),
         "stats": {"captures": 0, "restores": 0, "restored_positions": 0, "refreshes": 0, "misses": 0}}


def _log(msg):
    sys.stderr.write(f"[inkling_dftail pid={os.getpid()}] {msg}\n")
    sys.stderr.flush()


def slots_from_env(env=None):
    env = os.environ if env is None else env
    raw = env.get(ENV_SLOTS, "").strip()
    if raw in ("", "0"):
        return 0
    try:
        n = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"inkling_dftail: {ENV_SLOTS}={raw!r} is not an integer") from exc
    if not 0 <= n <= MAX_SLOTS:
        raise RuntimeError(f"inkling_dftail: {ENV_SLOTS}={n} out of range (0..{MAX_SLOTS})")
    return n


def _tokens_bytes(tokens, end):
    if isinstance(tokens, array) and tokens.typecode == "q":
        return memoryview(tokens)[:end].tobytes()
    return array("q", tokens[:end]).tobytes()


def prefix_digest(tokens, end, salt=None, extra=None):
    """Digest of a token prefix under its cache isolation keys (a salted prefix never matches another salt)."""
    h = hashlib.blake2b(digest_size=16)
    h.update(repr((salt, extra, int(end))).encode())
    h.update(_tokens_bytes(tokens, end))
    return h.digest()


def _tail(tokens, end):
    return tuple(int(t) for t in tokens[max(0, end - TAIL_CHECK): end])


class Entry:
    __slots__ = ("digest", "end", "lo", "tail", "salt", "extra", "used", "version")

    def __init__(self, digest, end, lo, tail, salt, extra, used, version):
        self.digest, self.end, self.lo, self.tail = digest, end, lo, tail
        self.salt, self.extra, self.used, self.version = salt, extra, used, version


class Index:
    """Host-side bookkeeping of the slots (pure Python, unit-tested without a GPU)."""

    def __init__(self, slots, length, page):
        self.slots, self.length, self.page = int(slots), int(length), max(1, int(page))
        self.entries = [None] * self.slots
        self.clock = 0

    def clear(self):
        self.entries = [None] * self.slots

    def _tick(self):
        self.clock += 1
        return self.clock

    def find(self, tokens, prefix_len, salt=None, extra=None, extend_end=None):
        """(slot, entry, lo) of the best snapshot for a request whose ring starts at the hit prefix_len and whose
        extend ends at extend_end: same prefix (salt, extra key, tokens[:E]), E >= prefix_len, and at least
        MIN_RESTORE positions in [lo, prefix_len) that are still inside the window after the extend."""
        P = int(prefix_len)
        floor_end = max(P, int(extend_end)) if extend_end is not None else P
        n = len(tokens)
        best = None
        cache = {}
        for slot, e in enumerate(self.entries):
            if e is None or e.end < P or e.end > n or e.salt != salt or e.extra != extra:
                continue
            lo = max(e.lo, floor_end - self.length)
            lo = -(-lo // self.page) * self.page  # page-aligned up, like a stock floor
            if P - lo < MIN_RESTORE:
                continue
            if _tail(tokens, e.end) != e.tail:
                continue
            d = cache.get(e.end)
            if d is None:
                d = cache[e.end] = prefix_digest(tokens, e.end, salt, extra)
            if d != e.digest:
                continue
            if best is None or lo < best[2]:
                best = (slot, e, lo)
        if best is not None:
            best[1].used = self._tick()
        return best

    def place(self, digest, end, lo, tail, salt, extra):
        """Slot for a new capture: None when the same snapshot is already held (refreshed instead), else an empty slot
        or the least recently used one. The entry is written now; the caller copies the rows."""
        for slot, e in enumerate(self.entries):
            if e is not None and e.digest == digest and e.end == end and e.lo <= lo:
                e.used = self._tick()
                return None
        empty = [s for s, e in enumerate(self.entries) if e is None]
        slot = empty[0] if empty else min(range(self.slots), key=lambda s: self.entries[s].used)
        old = self.entries[slot]
        version = 0 if old is None else old.version + 1
        self.entries[slot] = Entry(digest, end, lo, tail, salt, extra, self._tick(), version)
        return slot


class Store:
    """The device copies of the slots, next to one DFlashDraftKVRing."""

    def __init__(self, ring, k_bufs, v_bufs, slots):
        import torch

        self.torch = torch
        self.ring = ring
        self.k_bufs, self.v_bufs = list(k_bufs), list(v_bufs)
        self.length = int(ring.window_left) + 1
        page = ring.sink_rows if getattr(ring, "paged", False) else 1
        self.index = Index(slots, self.length, page)
        self.k_snap = [torch.empty((slots, self.length) + tuple(b.shape[1:]), dtype=b.dtype, device=b.device)
                       for b in self.k_bufs]
        self.v_snap = [torch.empty((slots, self.length) + tuple(b.shape[1:]), dtype=b.dtype, device=b.device)
                       for b in self.v_bufs]
        self.nbytes = sum(t.numel() * t.element_size() for t in self.k_snap + self.v_snap)
        self.floors = {}    # req pool slot -> (allocation generation, ring floor) as record_floors set it
        self.pending = []   # captures to copy when the forward returns
        self.device = ring.min_pos.device

    def clear(self):
        self.index.clear()
        self.pending = []

    def release(self):
        """Inert for the rest of the process: drop the device buffers (no cycle keeps them alive)."""
        self.clear()
        self.k_snap = self.v_snap = []
        self.k_bufs = self.v_bufs = []
        self.ring = None

    def _locs(self, req_pool_idx, start, end, floor=None):
        """Ring locs of positions [start, end) of req_pool_idx. A paged ring's locs are relative to the slot's floor:
        with ``floor`` given they are computed for that floor (DFlashDraftKVRing.locs' arithmetic, DCP 1, pinned)
        instead of reading min_pos, so a restore can be enqueued before the floor is published."""
        torch = self.torch
        positions = torch.arange(start, end, dtype=torch.int64, device=self.device)
        if floor is not None and self.ring.paged:
            ring = self.ring
            return (ring.sink_rows + int(req_pool_idx) * ring.rows_per_req
                    + torch.remainder(positions - int(floor), ring.rows_per_req))
        reqs = torch.full_like(positions, int(req_pool_idx))
        return self.ring.locs(reqs, positions)

    def restore(self, req_pool_idx, slot, entry, lo, prefix_len):
        """Copy snapshot rows [lo, prefix_len) into the ring of req_pool_idx as it will be with its floor at lo."""
        dst = self._locs(req_pool_idx, lo, prefix_len, floor=lo)
        a, b = lo - entry.lo, prefix_len - entry.lo
        for buf, snap in zip(self.k_bufs + self.v_bufs, self.k_snap + self.v_snap):
            buf.index_copy_(0, dst, snap[slot, a:b])

    def capture(self, req_pool_idx, slot, lo, end):
        src = self._locs(req_pool_idx, lo, end)
        n = end - lo
        for buf, snap in zip(self.k_bufs + self.v_bufs, self.k_snap + self.v_snap):
            self.torch.index_select(buf, 0, src, out=snap[slot, :n])


def plan_floors(store, reqs, prefix_lens, req_generation):
    """Before the stock record_floors: the restores for fresh slots (the slot's first extend since its allocation,
    exactly the slots record_floors sets). Returns [(i, req pool slot, generation, snapshot slot, entry, lo, P)] and
    the generation of every fresh request; nothing is mutated."""
    restores, fresh = [], {}
    ring = store.ring
    for i, req in enumerate(reqs):
        slot_idx = int(req.kv.req_pool_idx)
        generation = int(req_generation[slot_idx])
        if int(ring._floor_generation[slot_idx]) == generation:
            continue  # not fresh: a later chunk, or a running request in a mixed batch
        fresh[i] = (slot_idx, generation)
        P = int(prefix_lens[i])
        if P <= 0:
            continue
        hit = store.index.find(req.full_untruncated_fill_ids, P, getattr(req, "cache_salt", None),
                               getattr(req, "extra_key", None), getattr(req, "extend_end", None))
        if hit is None:
            STATE["stats"]["misses"] += 1
            continue
        slot, entry, lo = hit
        restores.append((i, slot_idx, generation, slot, entry, lo, P))
    return restores, fresh


def record_mirror(store, fresh, floors):
    """After the stock record_floors: the floor each fresh slot now holds, per allocation generation."""
    for i, (slot_idx, generation) in fresh.items():
        store.floors[slot_idx] = (generation, int(floors[i]))


def plan_captures(store, reqs, prefix_lens, req_generation):
    """The extends of this forward that complete a prompt (prefill rows only: a running request in a mixed batch
    starts at or past its prompt end): queued, copied when the forward returns."""
    for i, req in enumerate(reqs):
        end = getattr(req, "extend_end", None)
        n_prompt = len(req.origin_input_ids)
        if end is None or int(prefix_lens[i]) >= n_prompt or int(end) < n_prompt or int(end) < MIN_POSITIONS:
            continue
        if getattr(req, "positional_embed_overrides", None) is not None:
            continue  # the token ids do not determine its hidden states (the radix match is off for it too)
        end = int(end)
        slot_idx = int(req.kv.req_pool_idx)
        generation = int(req_generation[slot_idx])
        rec = store.floors.get(slot_idx)
        if rec is None or rec[0] != generation:
            continue  # floor unknown for this allocation: never capture rows the ring may not hold
        lo = max(end - store.length, rec[1])
        if end - lo < MIN_RESTORE:
            continue
        tokens = req.full_untruncated_fill_ids
        salt, extra = getattr(req, "cache_salt", None), getattr(req, "extra_key", None)
        store.pending.append((slot_idx, generation, lo, end, prefix_digest(tokens, end, salt, extra),
                              _tail(tokens, end), salt, extra))
