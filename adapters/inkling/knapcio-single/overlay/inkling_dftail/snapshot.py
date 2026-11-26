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

(A defers to B) INKLING_DFLASH_TAIL_DEFER=1, with the prefix tail (A) and the slots on: before A lowers a request's
prefix bound, reserve() looks for a snapshot of the same prefix (salt, extra key, tokens[:E]) that ends past A's bound
and whose window covers all the draft history A would recompute (restore floor lo <= max(input - N, input - L), with lo
page-aligned like every ring floor: with the switch on, captures start at the page-aligned floor of E - L, so a slot
holds one page more than the window, charged to the KV budget with the rest). The request then keeps the stock bound,
capped at E, so the target reuses its cache up to E and B restores the window. The reservation binds the slot, its
version (a serial that is never reused, also across a flush), digest, end, salt, extra key and the snapshot generation
(bumped by every flush / weight update / inert switch). It is checked after the radix lookup (holds()) and again by
plan_floors before the restore (claim()). Nothing touches the slots during a radix match, so a mismatch after the
lookup means an unexpected state: the request then keeps its (valid) cache hit and drops the reservation rather than
repeating a match (a match stages hybrid-state slots and copies); before the restore a mismatch gives the request the
ordinary find(). Either way only the draft window can be shorter. A reserved slot is pinned against capture eviction
until it is restored or the next forward has drained (the scheduler matches and runs a batch in that order, so no
capture runs in between; a pin cannot outlive one forward). Deferral changes which requests the target re-prefills, so
it is armed only in a single-rank process (ranks would otherwise decide on their own snapshot state) and without
hierarchical / external KV storage (whose staged loads are planned outside the lookup).
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

RES_ATTR = "_inkling_dftail_ab"   # Req attribute: PENDING while a lookup plans, then a Reservation or None
PENDING = object()

STATE = {"slots": 0, "inert": "", "charged": None, "stores": weakref.WeakSet(),
         "stats": {"captures": 0, "restores": 0, "restored_positions": 0, "refreshes": 0, "misses": 0},
         # (A defers to B) only: armed by install; solo = the store's process is a single rank (set by build_store)
         "defer": False, "solo": False, "generation": 0,
         "ab_stats": {"reserved": 0, "reserved_restores": 0, "reserve_lost": 0, "pinned_skips": 0}}


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


def _page_up(x, page):
    return -(-int(x) // page) * page


def _window_floor(e, floor_end, length, page):
    """(A defers to B) The lowest page-aligned ring floor a snapshot can restore for an extend ending at floor_end:
    the page-aligned floor of floor_end - L when the snapshot holds it, else its own first row (page-aligned up)."""
    base = max(0, int(floor_end) - length)
    down = base // page * page
    return down if down >= e.lo else _page_up(max(e.lo, base), page)


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
        self.serial = 0    # entry versions: never reused, also across clear()
        self.epoch = 0     # forwards drained so far
        self.pins = {}     # slot -> [epoch, reservations not yet restored] (honoured while the epoch is current)

    def clear(self):
        self.entries = [None] * self.slots
        self.pins = {}

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
        if not STATE["defer"]:  # the released selection and versions
            slot = empty[0] if empty else min(range(self.slots), key=lambda s: self.entries[s].used)
            old = self.entries[slot]
            version = 0 if old is None else old.version + 1
            self.entries[slot] = Entry(digest, end, lo, tail, salt, extra, self._tick(), version)
            return slot
        free = [s for s in range(self.slots) if not self.pinned(s)]
        if not empty and not free:
            STATE["ab_stats"]["pinned_skips"] += 1
            return None  # every slot is reserved for a restore this forward: no capture
        slot = empty[0] if empty else min(free, key=lambda s: self.entries[s].used)
        self.serial += 1
        version = self.serial
        self.entries[slot] = Entry(digest, end, lo, tail, salt, extra, self._tick(), version)
        return slot

    def pinned(self, slot):
        pin = self.pins.get(slot)
        return pin is not None and pin[0] == self.epoch and pin[1] > 0

    def pin(self, slot):
        pin = self.pins.get(slot)
        self.pins[slot] = [self.epoch, pin[1] + 1 if pin is not None and pin[0] == self.epoch else 1]

    def unpin(self, slot, epoch):
        pin = self.pins.get(slot)
        if pin is not None and pin[0] == epoch == self.epoch and pin[1] > 0:
            pin[1] -= 1

    def cover(self, tokens, input_len, capped, salt=None, extra=None):
        """(slot, entry, lo) of the snapshot that lets the prefix tail defer: same prefix, end E with
        capped < E <= input_len, and a window that covers all the tail recomputes (lo <= max(capped, input_len - L),
        where lo is the restore floor after the whole prompt's extend). The longest E wins (most target reuse), then
        the lowest lo."""
        n, capped = int(input_len), int(capped)
        need = max(0, capped, n - self.length)
        best = None
        cache = {}
        for slot, e in enumerate(self.entries):
            if e is None or e.end <= capped or e.end > n or e.end > len(tokens) or e.salt != salt or e.extra != extra:
                continue
            lo = _window_floor(e, n, self.length, self.page)
            if lo > need or _tail(tokens, e.end) != e.tail:
                continue
            d = cache.get(e.end)
            if d is None:
                d = cache[e.end] = prefix_digest(tokens, e.end, salt, extra)
            if d != e.digest:
                continue
            if best is None or (e.end, -lo) > (best[1].end, -best[2]):
                best = (slot, e, lo)
        return best


def extra_rows(page):
    """Rows a slot holds beyond the window: one page with the switch on (page-aligned capture floor), else none."""
    return int(page) if STATE["defer"] and int(page) > 1 else 0


class Reservation:
    __slots__ = ("store", "slot", "version", "digest", "end", "salt", "extra", "generation", "input_len", "pin_epoch")

    def __init__(self, store, slot, entry, generation, input_len, pin_epoch):
        self.store, self.slot, self.version, self.digest = store, slot, entry.version, entry.digest
        self.end, self.salt, self.extra = entry.end, entry.salt, entry.extra
        self.generation, self.input_len, self.pin_epoch = generation, int(input_len), pin_epoch


def _live_store():
    if not (STATE["defer"] and STATE["solo"]) or STATE["inert"]:
        return None
    stores = [s for s in list(STATE["stores"]) if s.ring is not None]
    return stores[0] if len(stores) == 1 else None


def reserve(req, input_len, capped):
    """A Reservation of the snapshot that covers req's draft history above the prefix tail's bound, or None."""
    store = _live_store()
    if store is None or getattr(req, "positional_embed_overrides", None) is not None \
            or getattr(req, "return_logprob", False):
        return None
    hit = store.index.cover(req.full_untruncated_fill_ids, input_len, capped,
                            getattr(req, "cache_salt", None), getattr(req, "extra_key", None))
    if hit is None:
        return None
    slot, entry, _ = hit
    ix = store.index
    entry.used = ix._tick()
    ix.pin(slot)
    STATE["ab_stats"]["reserved"] += 1
    return Reservation(store, slot, entry, STATE["generation"], input_len, ix.epoch)


def drop(res):
    """Release a reservation's own pin, once (restored, superseded by a new lookup, or rejected). A pin of an earlier
    forward or of a cleared index has already lapsed and is never taken from another reservation."""
    if not isinstance(res, Reservation) or res.pin_epoch is None:
        return
    epoch, res.pin_epoch = res.pin_epoch, None
    if res.store.ring is not None and res.generation == STATE["generation"]:
        res.store.index.unpin(res.slot, epoch)


def _intact(res, store=None):
    st = res.store
    if (store is not None and st is not store) or st.ring is None or STATE["inert"]:
        return None
    if getattr(st.ring, "_inkling_snap", None) is not st or STATE["generation"] != res.generation:
        return None
    e = st.index.entries[res.slot]
    if e is None or e.version != res.version or e.digest != res.digest or e.end != res.end \
            or e.salt != res.salt or e.extra != res.extra:
        return None
    return e


def holds(res, req):
    """After the radix lookup: the reserved snapshot is unchanged and the reused prefix ends at or below its end."""
    if _intact(res) is None or getattr(req, "host_hit_length", 0):
        return False
    for name in ("prefix_len", "num_matched_prefix_tokens"):
        v = getattr(req, name, None)
        if v is not None and int(v) > res.end:
            return False
    return True


def claim(store, res, prefix_len, extend_end):
    """At the restore: (slot, entry, lo) for the actual prefix, or None when the reservation no longer holds."""
    drop(res)
    e = _intact(res, store)
    P = int(prefix_len)
    if e is None or P > res.end:
        return None
    floor_end = max(P, int(extend_end)) if extend_end is not None else P
    lo = _window_floor(e, floor_end, store.length, store.index.page)
    if P - lo <= 0 or floor_end - lo > store.ring.span:
        return None
    e.used = store.index._tick()
    return res.slot, e, lo


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
        self.extra = extra_rows(page)  # (A defers to B): captures start page-aligned
        rows = self.length + self.extra
        self.k_snap = [torch.empty((slots, rows) + tuple(b.shape[1:]), dtype=b.dtype, device=b.device)
                       for b in self.k_bufs]
        self.v_snap = [torch.empty((slots, rows) + tuple(b.shape[1:]), dtype=b.dtype, device=b.device)
                       for b in self.v_bufs]
        self.nbytes = sum(t.numel() * t.element_size() for t in self.k_snap + self.v_snap)
        self.floors = {}    # req pool slot -> (allocation generation, ring floor) as record_floors set it
        self.pending = []   # captures to copy when the forward returns
        self.device = ring.min_pos.device

    def clear(self):
        self.index.clear()
        self.pending = []
        if STATE["defer"]:
            STATE["generation"] += 1  # every reservation made before is void

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
    the generation of every fresh request. A fresh request's reservation (A defers to B) is consumed here: restored when
    it still holds, else the request gets the ordinary find(). Only LRU stamps and that request attribute change."""
    restores, fresh = [], {}
    ring = store.ring
    for i, req in enumerate(reqs):
        slot_idx = int(req.kv.req_pool_idx)
        generation = int(req_generation[slot_idx])
        if int(ring._floor_generation[slot_idx]) == generation:
            continue  # not fresh: a later chunk, or a running request in a mixed batch
        fresh[i] = (slot_idx, generation)
        P = int(prefix_lens[i])
        res = req.__dict__.pop(RES_ATTR, None) if hasattr(req, "__dict__") else None
        if P <= 0:
            drop(res)
            continue
        hit = None
        if isinstance(res, Reservation):
            try:
                hit = claim(store, res, P, getattr(req, "extend_end", None))
            except Exception:  # noqa: BLE001  (this request falls back to find())
                hit = None
            STATE["ab_stats"]["reserved_restores" if hit is not None else "reserve_lost"] += 1
        if hit is None:
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
        lo = end - store.length
        if store.extra:
            lo = max(0, lo) // store.index.page * store.index.page  # page-aligned: a restore can reach A's floor
        lo = max(lo, rec[1])
        if end - lo < MIN_RESTORE:
            continue
        tokens = req.full_untruncated_fill_ids
        salt, extra = getattr(req, "cache_salt", None), getattr(req, "extra_key", None)
        store.pending.append((slot_idx, generation, lo, end, prefix_digest(tokens, end, salt, extra),
                              _tail(tokens, end), salt, extra))
