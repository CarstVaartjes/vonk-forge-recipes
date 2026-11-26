"""Side-slot pressure: the packed store's overflow, fallback and staging capacity,
made visible to the scheduler so that exhausting it never stops the worker.

A sealed page needs page-head overflow slots for its escapes (2 x heads at most);
without them it stays raw in a fallback slot, and without one of those it stays
raw in a live staging slot (device._publish). Only when staging is exhausted too
are a page's rows lost (counted per lane, never asserted).

After each forward the guard copies the lanes' slot usage and loss counters to
pinned memory (one small matmul and two copies, no host sync). It reads them:

- before a forward's result commits (its tokens, its prefix-cache insert), and
- before a scheduling pass stashes a continuing chunked prefill into the prefix
  cache (it then waits for that chunk's forward),

which are the only places rows written by a forward become visible to anything
but that forward. A loss at either point aborts every request in flight, without
a prefix-cache insert and without committing the step: rows are lost only after
the overflow, fallback and staging pools are all exhausted, and attributing a
physical page to its owner is not reliable while pages move and are recycled.

Before each scheduling pass it also compares the worst-case sealable pages left
with one prefill chunk plus one page per running request:

1. short: evict unreferenced prefix-cache pages (their side slots return on
   release) and measure again;
2. still short: defer new prefill while requests are running, and let decode
   retract requests until the rest fits (`ScheduleBatch.check_decode_mem`), never
   below one request.

With the default pools (2,048 overflow heads, 64 fallback pages per visit) target
visits used no overflow slot in measured serving, so none of this engages and the
schedule is the stock one.
"""

import contextvars
import logging

import torch

LOG = logging.getLogger("inkling_kv12s")

_RECORDING = contextvars.ContextVar("inkling_kv12s_probe_depth", default=0)
GUARD = []


class SideGuard:
    RING = 8
    LOG_PERIOD = 256
    RELIEVE_ROUNDS = 4
    RECENT = 4

    def __init__(self, lanes, locks, events, *, prefill_tokens, page):
        from collections import deque

        from .storage import EVENT_LOST

        first = lanes[0]
        self.lanes = lanes
        self.page = page
        self.locks = locks        # (lanes, live + fallback + overflow) int32
        self.events = events      # (lanes, EVENT_WORDS) int32
        self.live = first.live
        self.fallback = first.fallback_pages
        self.overflow = first.overflow_locks.numel()
        self.page_heads = 2 * first.heads
        self.lost_word = EVENT_LOST
        self.prefill_pages = -(-max(1, int(prefill_tokens)) // page) + 2
        # Column sums of the lock rows by segment (live, fallback, overflow) in one matmul.
        segments = torch.zeros((locks.shape[1], 3), dtype=torch.float32, device=locks.device)
        segments[:self.live, 0] = 1
        segments[self.live:self.live + self.fallback, 1] = 1
        segments[self.live + self.fallback:, 2] = 1
        self.segments = segments
        cuda = locks.is_cuda
        self.host = torch.zeros((self.RING, len(lanes), 3), dtype=torch.float32, pin_memory=cuda)
        self.host_events = torch.zeros((self.RING,) + tuple(events.shape), dtype=torch.int32, pin_memory=cuda)
        self.ready = [torch.cuda.Event() if cuda else None for _ in range(self.RING)]
        self.position = 0
        self.last = None                          # probe of the last launched forward
        self.recent = deque(maxlen=self.RECENT)   # the last launched batches (overlap keeps two in flight)
        self.free_pages = None    # worst-case pages sealable without a live staging slot
        self.live_free = None
        self.lost_seen = 0
        self.high = (0, 0, 0)
        self.observed = 0
        self.pressure = False

    # -- measurement -------------------------------------------------------
    def _usage(self):
        # One launch (was a full-width int32 -> FP32 copy plus a [lanes, N] x [N, 3] cuBLAS matmul, ~38 us per step):
        # per lane the integer sums of the lock words over the live / fallback / overflow segments, as FP32. The
        # lock words are 0 / 1, so every sum is < 2**24 and equals the old FP32 matmul's value exactly.
        if self.locks.is_cuda and self.locks.dtype == torch.int32 and self.locks.is_contiguous():
            from .segsum import segment_sums

            return segment_sums(self.locks, self.live, self.fallback)
        return torch.mm(self.locks.to(torch.float32), self.segments)  # exact: counts < 2**24

    def record(self, batch=None):
        """On the forward's stream, after its kernels: queue the usage and loss copies."""
        index = self.position % self.RING
        self.position += 1
        self.host[index].copy_(self._usage(), non_blocking=True)
        self.host_events[index].copy_(self.events, non_blocking=True)
        if self.ready[index] is not None:
            self.ready[index].record()
        self.last = index
        if batch is not None and getattr(batch, "reqs", None):
            self.recent.append(list(batch.reqs))
        return index

    def _observe(self, rows):
        rows = [[int(v) for v in row] for row in rows]
        live_used = max(row[0] for row in rows)
        fallback_used = max(row[1] for row in rows)
        overflow_used = max(row[2] for row in rows)
        self.free_pages = min((self.fallback - row[1]) + (self.overflow - row[2]) // self.page_heads
                              for row in rows)
        self.live_free = self.live - live_used
        self.observed += 1
        high = tuple(max(a, b) for a, b in zip(self.high, (live_used, overflow_used, fallback_used)))
        if high != self.high and (self.observed % self.LOG_PERIOD == 0 or high[1:] != self.high[1:]):
            self.high = high
            LOG.info("inkling_kv12s: side-slot high water per visit: stage %d of %d, overflow %d of %d, "
                     "fallback %d of %d", high[0], self.live, high[1], self.overflow, high[2], self.fallback)

    def measure(self):
        """Exact usage now (synchronizes; only under pressure)."""
        if self.locks.is_cuda:
            torch.cuda.synchronize()
        self._observe(self._usage().tolist())

    # -- scheduler hooks ---------------------------------------------------
    def _read(self, index):
        """A recorded probe, after its forward: usage, and the losses so far."""
        if self.ready[index] is not None:
            self.ready[index].synchronize()
        self._observe(self.host[index].tolist())
        return int(self.host_events[index][:, self.lost_word].sum())

    def on_result(self, scheduler, batch, result):
        """Before a result commits: that forward has finished (its outputs are read next)."""
        index = getattr(result, "_inkling_kv12s_probe", None)
        if index is not None and self._read(index) > self.lost_seen:
            self._abort_in_flight(scheduler, batch)

    def before_schedule(self, scheduler, running_batch, last_batch):
        """Before a scheduling pass: a continuing chunked prefill is stashed into the
        prefix cache in this pass, before its forward's result, so wait for the last
        launched forward first. Then relieve pressure."""
        if getattr(scheduler, "chunked_req", None) is not None and self.last is not None:
            if self._read(self.last) > self.lost_seen:
                self._abort_in_flight(scheduler, last_batch, running_batch)
        self.relieve(scheduler, running_batch)

    def _abort_in_flight(self, scheduler, *batches):
        if self.events.is_cuda:
            torch.cuda.synchronize()
        lost = int(self.events[:, self.lost_word].sum().item())
        reqs = []
        # Requests of launched forwards that still hold their rows: in a batch, launched
        # (prefill), or chunked. A retracted one re-prefills from its tokens.
        holds = getattr(scheduler, "still_in_batch", None)
        chunked = getattr(scheduler, "chunked_req", None)

        def add(items, *, check=True):
            for req in items or ():
                if (req is not None and req not in reqs and not req.finished()
                        and (not check or holds is None or req is chunked or holds(req))):
                    reqs.append(req)

        collect = getattr(scheduler, "collect_inflight_reqs", None)
        add(collect() if collect is not None else ())  # last_batch may name requeued requests
        for batch in batches:
            add(getattr(batch, "reqs", None))
        add(getattr(getattr(scheduler, "running_batch", None), "reqs", None))
        for launched in self.recent:
            add(launched)
        add([chunked], check=False)
        LOG.error("inkling_kv12s: side KV slots exhausted; %d row loss(es) since the last check; aborting "
                  "all %d request(s) in flight: %s", lost - self.lost_seen, len(reqs), [req.rid for req in reqs])
        self.lost_seen = lost
        for req in reqs:
            self._abort(scheduler, req)

    @staticmethod
    def _abort(scheduler, req):
        """Abort without committing the in-flight step (its tokens may have read lost
        rows) and without a prefix-cache insert. A running or launched request leaves
        its request list now, so the step's result skips it; the chunked request is
        aborted at the top of this scheduling pass, before its stash."""
        from http import HTTPStatus

        from sglang.srt.managers.req_types import FINISH_ABORT

        reason = FINISH_ABORT("inkling_kv12s: packed KV side slots exhausted; request aborted",
                              HTTPStatus.SERVICE_UNAVAILABLE)
        req.skip_radix_cache_insert = True
        if req is getattr(scheduler, "chunked_req", None):
            req.request_discard(reason)
            scheduler._pending_chunked_abort_req = req
            return
        req.clear_requested_finish()
        scheduler.request_retirer.abort(req, reason=reason, finished_reason=reason.to_json())

    def _need(self, running):
        return self.prefill_pages + running

    def relieve(self, scheduler, running_batch):
        """Before scheduling: under pressure, evict prefix-cache pages and measure again."""
        if self.free_pages is None:
            return
        running = 0 if running_batch is None else running_batch.batch_size()
        need = self._need(running)
        if self.free_pages >= need:
            if self.pressure:
                LOG.info("inkling_kv12s: side-slot pressure relieved (%d sealable pages)", self.free_pages)
            self.pressure = False
            return
        from sglang.srt.mem_cache.base_prefix_cache import EvictParams

        tree = scheduler.tree_cache
        for _ in range(self.RELIEVE_ROUNDS):
            evictable = int(tree.full_evictable_size())
            if evictable <= 0:
                break
            short = need - self.free_pages
            tree.evict(EvictParams(num_tokens=min(evictable, max(64, 8 * short) * self.page)))
            flush = getattr(scheduler.token_to_kv_pool_allocator, "flush_deferred_full_frees", None)
            if flush is not None:
                flush()
            self.measure()
            if self.free_pages >= need:
                break
        if not self.pressure:
            LOG.warning("inkling_kv12s: side-slot pressure: %d sealable pages for %d running requests "
                        "and one prefill chunk (%d); evicting prefix-cache pages%s", self.free_pages, running,
                        need, "" if self.free_pages >= need else ", deferring prefill and retracting")
        self.pressure = self.free_pages < need

    def defer_prefill(self, running_batch):
        running = running_batch.batch_size() if running_batch is not None else 0
        decodes = running_batch is not None and not running_batch.is_empty() and not running_batch.is_prefill_only
        return self.free_pages is not None and decodes and self.free_pages < self._need(running)

    def decode_fits(self, requests, measure=False):
        if self.free_pages is None or requests <= 1:
            return True  # never retract the last request for side slots: the backstop covers it
        if measure:
            self.measure()  # inside the retraction loop: count what is free now
        # Each request may open one tail (a live slot) and seal one page; a seal
        # beyond the side pools stays raw in a live slot too.
        return self.live_free >= requests + max(0, requests - self.free_pages)


def guard():
    return GUARD[0] if GUARD else None


def record_after(result, batch=None):
    """Attach a usage probe to an outermost forward's result."""
    side = guard()
    if side is not None and result is not None:
        try:
            result._inkling_kv12s_probe = side.record(batch)
        except AttributeError:
            pass
    return result


def outermost(call, batch=None):
    if not _RECORDING.get():
        from .static import note_forward_stream

        note_forward_stream()  # a serving forward: its stream orders static-pool releases
    token = _RECORDING.set(_RECORDING.get() + 1)
    try:
        result = call()
    finally:
        _RECORDING.reset(token)
    return record_after(result, batch) if _RECORDING.get() == 0 else result
