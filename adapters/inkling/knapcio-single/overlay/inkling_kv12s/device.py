"""Stream-ordered CUDA writes and side-slot lifetime for packed page envelopes."""

import os

import torch
import triton
import triton.language as tl

from . import storage
from .storage import (CAP, CHUNK, DIM, EVENT_WORDS, EXP_LO, PACK_BATCH, PAGE, Layer, STAGE_MASK, STAGE_SHIFT,
                      raw_lane_bytes)

# Test only: send every sealed page-head through the overflow arena, to exercise
# side-slot exhaustion on real traffic (INKLING_KV12S_FORCE_OVERFLOW=1).
FORCE_OVERFLOW = os.environ.get("INKLING_KV12S_FORCE_OVERFLOW", "0").strip().lower() in ("1", "true", "yes", "on")

# DeviceLayer.events word offsets (storage.EVENT_*), for the kernels.
EVENT_FALLBACK = tl.constexpr(storage.EVENT_FALLBACK)
EVENT_KEPT = tl.constexpr(storage.EVENT_KEPT)
EVENT_LOST = tl.constexpr(storage.EVENT_LOST)


@triton.jit
def _claim(Locks, page, SLOTS: tl.constexpr):
    # Slot ownership is independent of physical page IDs. A whole-page move can
    # transfer the descriptor without rewriting the slot or its lock.
    slot = page.to(tl.int32) % SLOTS
    old = tl.atomic_cas(Locks + slot, 0, 1, sem="acquire")
    probes = 1
    while (old != 0) & (probes < SLOTS):
        slot = (slot + 1) % SLOTS
        old = tl.atomic_cas(Locks + slot, 0, 1, sem="acquire")
        probes += 1
    return tl.where(old == 0, slot, -1)


@triton.jit
def _direct(Loc, i, loc, n_rows, PAGE: tl.constexpr):
    row = loc % PAGE
    start = i - row
    end = start + PAGE - 1
    lo = tl.load(Loc + start, start >= 0, other=0)
    hi = tl.load(Loc + end, end < n_rows, other=0)
    return (lo == loc - row) & (hi == loc + PAGE - 1 - row)


@triton.jit
def _lose(Events):
    # No room left to keep a page's rows: count the loss. The host aborts every
    # request in flight before any result commits or is cached (pressure.py).
    tl.atomic_add(Events + EVENT_LOST, 1)


@triton.jit(debug=True, do_not_specialize=["n_rows"])
def _reserve(Loc, Flags, Locks, Events, n_rows, NPAGES: tl.constexpr,
             PAGE: tl.constexpr, STRIDE: tl.constexpr, STAGES: tl.constexpr,
             SHIFT: tl.constexpr, COUNT_MASK: tl.constexpr, TENTATIVE: tl.constexpr):
    i = tl.program_id(0)
    loc = tl.load(Loc + i).to(tl.int64)
    page, row = loc // PAGE, loc % PAGE
    tl.device_assert((loc <= 0) | ((page > 0) & (page < NPAGES)), "kv12s: invalid physical write row")
    if (page > 0) & (page < NPAGES):
        last = tl.load(Loc + i - 1, i > 0, other=0)
        if (row == 0) | (loc != last + 1):
            ptr = Flags + page * (STRIDE // 4)
            flags = tl.load(ptr)
            tl.device_assert(flags != 1, "kv12s: write to a committed page")
            tl.device_assert((flags < 2) | ((flags >> SHIFT) < STAGES), "kv12s: write to a committed fallback page")
            direct = False
            if not TENTATIVE:
                direct = _direct(Loc, i, loc, n_rows, PAGE)
            if not direct:
                if flags == 0:
                    slot = (row * 0 - 1).to(tl.int32)
                    if row == 0:
                        slot = _claim(Locks, page, STAGES)
                    if slot >= 0:
                        # This provisional descriptor is visible only to the
                        # scatter until the final publication kernel runs.
                        tl.store(ptr, (slot << SHIFT) | 2)
                    else:
                        # Staging exhausted, or an append to a page whose first
                        # rows were lost that way: the rows are not stored.
                        _lose(Events)
                else:
                    tl.device_assert((row < (flags & COUNT_MASK)) & ((flags >> SHIFT) < STAGES),
                                     "kv12s: staged append is missing its prefix")


@triton.jit(do_not_specialize=["n_rows"])
def _scatter(Loc, K, V, Flags, Stage, n_rows, NPAGES: tl.constexpr,
             PAGE: tl.constexpr, H: tl.constexpr, D: tl.constexpr, STRIDE: tl.constexpr,
             K0: tl.constexpr, K1: tl.constexpr, V0: tl.constexpr, V1: tl.constexpr,
             STAGES: tl.constexpr, SHIFT: tl.constexpr, TENTATIVE: tl.constexpr,
             BLOCK: tl.constexpr):
    i, kind = tl.program_id(0), tl.program_id(1)
    loc = tl.load(Loc + i).to(tl.int64)
    page = loc // PAGE
    if (page > 0) & (page < NPAGES):
        direct = False
        if not TENTATIVE:
            direct = _direct(Loc, i, loc, n_rows, PAGE)
        if not direct:
            flags = tl.load(Flags + page * (STRIDE // 4))
            slot = flags >> SHIFT
            if (flags >= 2) & (slot < STAGES):
                dim = tl.arange(0, BLOCK)
                if kind == 0:
                    words = tl.load(K + i.to(tl.int64) * K0 + dim // D * K1 + dim % D,
                                    dim < H * D, other=0)
                else:
                    words = tl.load(V + i.to(tl.int64) * V0 + dim // D * V1 + dim % D,
                                    dim < H * D, other=0)
                base = ((slot.to(tl.int64) * 2 + kind) * PAGE + loc % PAGE) * H * D
                tl.store(Stage + base + dim, words, dim < H * D)


@triton.jit
def _escape(bits, LO: tl.constexpr):
    exp = (bits >> 7) & 255
    inline = (exp >= LO) & (exp < LO + 15)
    return tl.where(inline, exp - LO, 15), (exp != 0) & ~inline


@triton.jit
def _pack_at(Loc, K, V, SM, EX, ESC, OFF, Refs, Flags, Stage, Overflow, Locks, Fail, i, head, kind,
             n_rows, NPAGES: tl.constexpr, PAGE: tl.constexpr, H: tl.constexpr,
             D: tl.constexpr, STRIDE: tl.constexpr, K0: tl.constexpr, K1: tl.constexpr,
             V0: tl.constexpr, V1: tl.constexpr, OVERFLOW: tl.constexpr, SHIFT: tl.constexpr,
             COUNT_MASK: tl.constexpr, CAP: tl.constexpr, CHUNK: tl.constexpr, LO: tl.constexpr,
             PROMOTE: tl.constexpr, LIVE: tl.constexpr, FORCE: tl.constexpr):
    # One (location, head, K/V) program of _pack; _pack_lanes runs the same body for every lane of a store.
    loc = tl.load(Loc + i).to(tl.int64)
    page = loc if PROMOTE else loc // PAGE
    if (page > 0) & (page < NPAGES) & (PROMOTE | (loc % PAGE == PAGE - 1)):
        base = page * STRIDE
        flags = tl.load(Flags + base // 4)
        direct = False
        if not PROMOTE:
            direct = _direct(Loc, i, loc, n_rows, PAGE)
        if (flags >= 2) | direct:
            if not PROMOTE or (((flags & COUNT_MASK) == PAGE + 1) & ((flags >> SHIFT) < LIVE)):
                pair = tl.arange(0, PAGE * D // 2)
                row = pair // (D // 2)
                dim = pair % (D // 2) * 2
                if direct:
                    src = i.to(tl.int64) - (PAGE - 1) + row
                    if kind == 0:
                        ptr = K + src * K0 + head * K1 + dim
                        a, b = tl.load(ptr), tl.load(ptr + 1)
                    else:
                        ptr = V + src * V0 + head * V1 + dim
                        a, b = tl.load(ptr), tl.load(ptr + 1)
                else:
                    slot = (flags >> SHIFT).to(tl.int64)
                    ptr = Stage + (((slot * 2 + kind) * PAGE + row) * H + head) * D + dim
                    a, b = tl.load(ptr), tl.load(ptr + 1)
                a, b = a.to(tl.uint16).to(tl.int32), b.to(tl.uint16).to(tl.int32)
                ea, xa = _escape(a, LO)
                eb, xb = _escape(b, LO)
                ranks = tl.cumsum(xa.to(tl.int32) + xb.to(tl.int32))
                count = tl.sum(xa.to(tl.int32) + xb.to(tl.int32), 0)
                lane = head * 2 + kind
                if FORCE:
                    spill = True  # test only (FORCE_OVERFLOW): every page-head takes the overflow path
                else:
                    spill = count > CAP
                if spill:
                    slot = _claim(Locks, page + lane, OVERFLOW)
                    if slot < 0:
                        # Arena exhausted: publication keeps this page raw in a
                        # fallback slot instead of sealing it (exact, no assert).
                        # Clear any stale ref left in a reused envelope first.
                        tl.store(Refs + base // 4 + lane, 0)
                        tl.atomic_or(Fail + page, 1)
                    else:
                        dst = Overflow + slot.to(tl.int64) * (PAGE * D) + pair * 2
                        tl.store(dst, a)
                        tl.store(dst + 1, b)
                        tl.store(Refs + base // 4 + lane, slot + 1)
                else:
                    elem = ((kind * PAGE + row) * H + head) * D + dim
                    sign_a = ((a >> 8) & 128) | (a & 127)
                    sign_b = ((b >> 8) & 128) | (b & 127)
                    tl.store(SM + base + elem, sign_a)
                    tl.store(SM + base + elem + 1, sign_b)
                    tl.store(EX + base + elem // 2, ea | (eb << 4))
                    stream = ESC + base // 4 + lane * CAP
                    entry = (row << 23) | (dim << 16)
                    tl.store(stream + ranks - xb.to(tl.int32) - 1, entry | a, xa)
                    tl.store(stream + ranks - 1, entry | (1 << 16) | b, xb)
                    offsets = OFF + base // 4 + lane * (PAGE // CHUNK + 1)
                    tl.store(offsets, 0)
                    tl.store(offsets + row // CHUNK + 1, ranks,
                             (row % CHUNK == CHUNK - 1) & (dim == D - 2))
                    tl.store(Refs + base // 4 + lane, 0)


@triton.jit(do_not_specialize=["n_rows"])
def _pack(Loc, K, V, SM, EX, ESC, OFF, Refs, Flags, Stage, Overflow, Locks, Fail,
          n_rows, NPAGES: tl.constexpr, PAGE: tl.constexpr, H: tl.constexpr,
          D: tl.constexpr, STRIDE: tl.constexpr, K0: tl.constexpr, K1: tl.constexpr,
          V0: tl.constexpr, V1: tl.constexpr, OVERFLOW: tl.constexpr, SHIFT: tl.constexpr,
          COUNT_MASK: tl.constexpr, CAP: tl.constexpr, CHUNK: tl.constexpr, LO: tl.constexpr,
          PROMOTE: tl.constexpr, LIVE: tl.constexpr, FORCE: tl.constexpr):
    _pack_at(Loc, K, V, SM, EX, ESC, OFF, Refs, Flags, Stage, Overflow, Locks, Fail,
             tl.program_id(0), tl.program_id(1), tl.program_id(2), n_rows, NPAGES, PAGE, H, D, STRIDE,
             K0, K1, V0, V1, OVERFLOW, SHIFT, COUNT_MASK, CAP, CHUNK, LO, PROMOTE, LIVE, FORCE)


@triton.jit
def _publish_at(Loc, K, V, Flags, Locks, Fail, Refs, OverflowLocks, Stage, Events, i, n_rows,
                NPAGES: tl.constexpr, PAGE: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
                STRIDE: tl.constexpr, K0: tl.constexpr, K1: tl.constexpr, V0: tl.constexpr, V1: tl.constexpr,
                SHIFT: tl.constexpr, COUNT_MASK: tl.constexpr, LIVE: tl.constexpr, FALLBACK: tl.constexpr,
                OVERFLOW: tl.constexpr, TENTATIVE: tl.constexpr, PROMOTE: tl.constexpr, BLOCK: tl.constexpr,
                LANES: tl.constexpr):
    # One location's program of _publish; _publish_lanes runs the same body for every lane of a store.
    loc = tl.load(Loc + i).to(tl.int64)
    page = loc if PROMOTE else loc // PAGE
    if (page > 0) & (page < NPAGES):
        ptr = Flags + page * (STRIDE // 4)
        flags = tl.load(ptr)
        if PROMOTE:
            end = (flags & COUNT_MASK) - 1
            # A committed fallback page (slot >= LIVE) is never promoted again.
            publish = (flags >= 2) & (end == PAGE) & ((flags >> SHIFT) < LIVE)
        else:
            end = loc % PAGE + 1
            following = tl.load(Loc + i + 1, i + 1 < n_rows, other=0)
            publish = (end == PAGE) | (following != loc + 1)
        if publish:
            slot = flags >> SHIFT
            # A direct seal reads the write inputs; any other publication needs
            # the page's staging slot. A page with neither lost its rows when
            # staging was exhausted (_reserve) and stays empty.
            direct = flags < 0  # False, as a value: a promoted page is always staged
            if not PROMOTE:
                direct = _direct(Loc, i, loc, n_rows, PAGE)
            held = (flags >= 2) | direct
            if not TENTATIVE and end == PAGE:
                if held:
                    if tl.load(Fail + page) != 0:
                        # Overflow arena exhausted while sealing: return the page-head
                        # slots this page did claim and keep the whole page raw, in the
                        # dedicated fallback range [LIVE, LIVE + FALLBACK) so live tails
                        # never lose staging capacity to fallback pages, or, with that
                        # range exhausted too, in a live staging slot (the host then
                        # relieves the pressure, pressure.py). Readers serve both exactly.
                        tl.store(Fail + page, 0)
                        lane = tl.arange(0, LANES)
                        lanes = lane < 2 * H
                        refs = Refs + page * (STRIDE // 4) + lane
                        ref = tl.load(refs, lanes, other=0)
                        tl.atomic_xchg(OverflowLocks + ref - 1, 0, sem="release",
                                       mask=lanes & (ref > 0) & (ref <= OVERFLOW))
                        tl.store(refs, tl.zeros_like(ref), lanes)
                        target = tl.zeros_like(slot) - 1
                        if FALLBACK > 0:
                            target = _claim(Locks + LIVE, page, FALLBACK)
                        fallback = target >= 0
                        if fallback:
                            target = target + LIVE
                        elif flags >= 2:
                            target = slot
                        else:
                            target = _claim(Locks, page, LIVE)
                        if target >= 0:
                            # A direct seal (also a full rewrite over a staged prefix)
                            # copies the write inputs; a staged page copies its slot
                            # unless it stays in that slot.
                            if direct | (target != slot):
                                words = tl.arange(0, BLOCK)
                                live = words < H * D
                                for row in range(PAGE):
                                    if direct:
                                        src = i.to(tl.int64) - (PAGE - 1) + row
                                        k = tl.load(K + src * K0 + words // D * K1 + words % D, live)
                                        v = tl.load(V + src * V0 + words // D * V1 + words % D, live)
                                    else:
                                        old = Stage + ((slot.to(tl.int64) * 2) * PAGE + row) * H * D + words
                                        k = tl.load(old, live)
                                        v = tl.load(old + PAGE * H * D, live)
                                    dst = Stage + ((target.to(tl.int64) * 2) * PAGE + row) * H * D + words
                                    tl.store(dst, k, live)
                                    tl.store(dst + PAGE * H * D, v, live)
                            tl.store(ptr, (target << SHIFT) | (PAGE + 1))
                            if (flags >= 2) & (target != slot):
                                # Every warp's copy out of the old slot must finish
                                # before another program can claim it (a kept direct
                                # page claims live slots in this same launch).
                                tl.debug_barrier()
                                tl.atomic_xchg(Locks + slot, 0, sem="release")
                            tl.atomic_add(Events + tl.where(fallback, EVENT_FALLBACK, EVENT_KEPT), 1)
                        else:
                            tl.store(ptr, 0)
                            _lose(Events)
                    else:
                        tl.store(ptr, 1)
                        if flags >= 2:
                            tl.atomic_xchg(Locks + slot, 0, sem="release")
                # else: rows lost at _reserve; the page stays empty.
            elif flags >= 2:
                tl.store(ptr, (slot << SHIFT) | (end + 1))


@triton.jit(do_not_specialize=["n_rows"])
def _publish(Loc, K, V, Flags, Locks, Fail, Refs, OverflowLocks, Stage, Events, n_rows,
             NPAGES: tl.constexpr, PAGE: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
             STRIDE: tl.constexpr, K0: tl.constexpr, K1: tl.constexpr, V0: tl.constexpr, V1: tl.constexpr,
             SHIFT: tl.constexpr, COUNT_MASK: tl.constexpr, LIVE: tl.constexpr, FALLBACK: tl.constexpr,
             OVERFLOW: tl.constexpr, TENTATIVE: tl.constexpr, PROMOTE: tl.constexpr, BLOCK: tl.constexpr,
             LANES: tl.constexpr):
    _publish_at(Loc, K, V, Flags, Locks, Fail, Refs, OverflowLocks, Stage, Events, tl.program_id(0), n_rows,
                NPAGES, PAGE, H, D, STRIDE, K0, K1, V0, V1, SHIFT, COUNT_MASK, LIVE, FALLBACK, OVERFLOW,
                TENTATIVE, PROMOTE, BLOCK, LANES)


@triton.jit
def _trim_at(Loc, Commit, Flags, Locks, i, WIDTH: tl.constexpr,
             NPAGES: tl.constexpr, PAGE: tl.constexpr, STRIDE: tl.constexpr, SHIFT: tl.constexpr,
             COUNT_MASK: tl.constexpr, LIVE: tl.constexpr):
    # One location's program of _trim; _trim_lanes runs the same body for every lane of a store.
    loc = tl.load(Loc + i).to(tl.int64)
    page = loc // PAGE
    col = i % WIDTH
    last_row = (loc % PAGE).to(tl.int32)
    if (page > 0) & (page < NPAGES) & ((last_row == PAGE - 1) | (col == WIDTH - 1)):
        ptr = Flags + page * (STRIDE // 4)
        flags = tl.load(ptr)
        # Fallback slots (>= LIVE) hold committed full pages: never trimmed.
        if (flags >= 2) & ((flags >> SHIFT) < LIVE):
            # The uniform verify reservation is contiguous within each page.
            # A ragged verify wrote only its leading rows, using the same IDs.
            first_col = tl.maximum(0, col - last_row)
            first_row = last_row - (col - first_col)
            accepted = tl.load(Commit + i // WIDTH)
            kept = first_row + tl.minimum(tl.maximum(accepted - first_col, 0), col - first_col + 1)
            tl.device_assert(kept < (flags & COUNT_MASK), "kv12s: unwritten row in accepted prefix")
            slot = flags >> SHIFT
            tl.store(ptr, tl.where(kept > 0, (slot << SHIFT) | (kept + 1), 0))
            if kept == 0:
                tl.atomic_xchg(Locks + slot, 0, sem="release")


@triton.jit(debug=True)
def _trim(Loc, Commit, Flags, Locks, WIDTH: tl.constexpr,
          NPAGES: tl.constexpr, PAGE: tl.constexpr, STRIDE: tl.constexpr, SHIFT: tl.constexpr,
          COUNT_MASK: tl.constexpr, LIVE: tl.constexpr):
    _trim_at(Loc, Commit, Flags, Locks, tl.program_id(0), WIDTH, NPAGES, PAGE, STRIDE, SHIFT, COUNT_MASK, LIVE)


# LaneGroup.deltas columns: where each lane's buffer starts, in elements from lane 0's buffer of the same kind.
L_FLAGS, L_STAGE_LOCKS, L_FAIL, L_REFS, L_OVERFLOW_LOCKS, L_STAGE, L_EVENTS, L_SM, L_EX, L_ESC, L_OFF, \
    L_OVERFLOW = (tl.constexpr(i) for i in range(12))
LANE_KINDS = 12


@triton.jit
def _at(Base, Deltas, lane, WHICH: tl.constexpr, KINDS: tl.constexpr):
    return Base + tl.load(Deltas + lane * KINDS + WHICH)


@triton.jit(debug=True)
def _trim_lanes(Loc, Commit, Flags, Locks, Deltas, WIDTH: tl.constexpr,
                NPAGES: tl.constexpr, PAGE: tl.constexpr, STRIDE: tl.constexpr, SHIFT: tl.constexpr,
                COUNT_MASK: tl.constexpr, LIVE: tl.constexpr, KINDS: tl.constexpr):
    lane = tl.program_id(1)
    _trim_at(Loc, Commit, _at(Flags, Deltas, lane, L_FLAGS, KINDS), _at(Locks, Deltas, lane, L_STAGE_LOCKS, KINDS),
             tl.program_id(0), WIDTH, NPAGES, PAGE, STRIDE, SHIFT, COUNT_MASK, LIVE)


@triton.jit(do_not_specialize=["n_rows"])
def _pack_lanes(Loc, SM, EX, ESC, OFF, Refs, Flags, Stage, Overflow, Locks, Fail, Deltas,
                n_rows, NPAGES: tl.constexpr, PAGE: tl.constexpr, H: tl.constexpr,
                D: tl.constexpr, STRIDE: tl.constexpr, K0: tl.constexpr, K1: tl.constexpr,
                OVERFLOW: tl.constexpr, SHIFT: tl.constexpr, COUNT_MASK: tl.constexpr, CAP: tl.constexpr,
                CHUNK: tl.constexpr, LO: tl.constexpr, LIVE: tl.constexpr, FORCE: tl.constexpr,
                KINDS: tl.constexpr):
    # The promotion (commit) of DeviceLayer.commit for every lane: grid (pages, heads, 2 * lanes).
    lane, kind = tl.program_id(2) // 2, tl.program_id(2) % 2
    stage = _at(Stage, Deltas, lane, L_STAGE, KINDS)
    _pack_at(Loc, stage, stage, _at(SM, Deltas, lane, L_SM, KINDS), _at(EX, Deltas, lane, L_EX, KINDS),
             _at(ESC, Deltas, lane, L_ESC, KINDS), _at(OFF, Deltas, lane, L_OFF, KINDS),
             _at(Refs, Deltas, lane, L_REFS, KINDS), _at(Flags, Deltas, lane, L_FLAGS, KINDS), stage,
             _at(Overflow, Deltas, lane, L_OVERFLOW, KINDS), _at(Locks, Deltas, lane, L_OVERFLOW_LOCKS, KINDS),
             _at(Fail, Deltas, lane, L_FAIL, KINDS), tl.program_id(0), tl.program_id(1), kind, n_rows,
             NPAGES, PAGE, H, D, STRIDE, K0, K1, K0, K1, OVERFLOW, SHIFT, COUNT_MASK, CAP, CHUNK, LO, True, LIVE,
             FORCE)


@triton.jit(do_not_specialize=["n_rows"])
def _publish_lanes(Loc, Flags, Locks, Fail, Refs, OverflowLocks, Stage, Events, Deltas, n_rows,
                   NPAGES: tl.constexpr, PAGE: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
                   STRIDE: tl.constexpr, K0: tl.constexpr, K1: tl.constexpr, SHIFT: tl.constexpr,
                   COUNT_MASK: tl.constexpr, LIVE: tl.constexpr, FALLBACK: tl.constexpr, OVERFLOW: tl.constexpr,
                   BLOCK: tl.constexpr, LANES: tl.constexpr, KINDS: tl.constexpr):
    # The promotion (commit) of DeviceLayer.commit for every lane: grid (pages, lanes).
    lane = tl.program_id(1)
    stage = _at(Stage, Deltas, lane, L_STAGE, KINDS)
    _publish_at(Loc, stage, stage, _at(Flags, Deltas, lane, L_FLAGS, KINDS),
                _at(Locks, Deltas, lane, L_STAGE_LOCKS, KINDS), _at(Fail, Deltas, lane, L_FAIL, KINDS),
                _at(Refs, Deltas, lane, L_REFS, KINDS), _at(OverflowLocks, Deltas, lane, L_OVERFLOW_LOCKS, KINDS),
                stage, _at(Events, Deltas, lane, L_EVENTS, KINDS), tl.program_id(0), n_rows,
                NPAGES, PAGE, H, D, STRIDE, K0, K1, K0, K1, SHIFT, COUNT_MASK, LIVE, FALLBACK, OVERFLOW,
                False, True, BLOCK, LANES)


@triton.jit
def _release(Ids, Flags, Refs, StageLocks, OverflowLocks, NPAGES: tl.constexpr, H: tl.constexpr,
             STRIDE: tl.constexpr, SHIFT: tl.constexpr, STAGES: tl.constexpr, OVERFLOW: tl.constexpr,
             RECYCLE: tl.constexpr, BLOCK: tl.constexpr):
    page = tl.load(Ids + tl.program_id(0)).to(tl.int64)
    if (page > 0) & (page < NPAGES):
        ptr = Flags + page * (STRIDE // 4)
        if RECYCLE:
            flags = tl.load(ptr)
            tl.store(ptr, 0)
            if flags >= 2:
                slot = flags >> SHIFT
                tl.atomic_xchg(StageLocks + slot, 0, sem="release", mask=slot < STAGES)
            elif flags == 1:
                lane = tl.arange(0, BLOCK)
                ref = tl.load(Refs + page * (STRIDE // 4) + lane, lane < H * 2, other=0)
                tl.atomic_xchg(OverflowLocks + ref - 1, 0, sem="release", mask=(ref > 0) & (ref <= OVERFLOW))
        else:
            tl.store(ptr, 0)


class DeviceLayer(Layer):
    """The same envelope as the reference writer, with device-owned side slots."""

    def __init__(self, raw, heads, page_stride, offset, staging_pages, overflow_heads, fallback_pages=0,
                 *, locks=None, events=None):
        # Slots [0, live) hold open tails; [live, live + fallback) hold full pages
        # that could not be sealed because the overflow arena was exhausted.
        from . import MAX_SIDE_SLOTS

        counts = (staging_pages, overflow_heads, fallback_pages)
        if (not all(isinstance(n, int) and not isinstance(n, bool) for n in counts)
                or not 1 <= staging_pages <= MAX_SIDE_SLOTS or not 1 <= overflow_heads <= MAX_SIDE_SLOTS
                or not 0 <= fallback_pages <= MAX_SIDE_SLOTS):
            raise ValueError(f"inkling_kv12s: invalid side-slot counts (staging, overflow, fallback) {counts}")
        super().__init__(raw, heads, page_stride, offset, staging_pages + fallback_pages, overflow_heads)
        self.live = staging_pages
        self.fallback_pages = fallback_pages
        # Stage then overflow locks, and the event words, may be rows of buffers
        # shared by all lanes (pressure.SideGuard reads them in one reduction).
        slots = staging_pages + fallback_pages + overflow_heads
        if locks is None:
            locks = torch.zeros(slots, dtype=torch.int32, device=self.device)
        if events is None:
            events = torch.zeros(EVENT_WORDS, dtype=torch.int32, device=self.device)
        if (locks.shape != (slots,) or locks.dtype != torch.int32 or not locks.is_contiguous()
                or events.shape != (EVENT_WORDS,) or events.dtype != torch.int32 or not events.is_contiguous()
                or locks.device != self.device or events.device != self.device):
            raise ValueError("inkling_kv12s: invalid side-slot lock or event buffers")
        self.stage_locks = locks[:staging_pages + fallback_pages]
        self.overflow_locks = locks[staging_pages + fallback_pages:]
        # Per-page "a page-head found no overflow slot" marks.
        self.fail = torch.zeros(self.pages, dtype=torch.int32, device=self.device)
        self.events = events
        self.fallbacks = events[storage.EVENT_FALLBACK:storage.EVENT_FALLBACK + 1]
        self.gather_workspace = None
        self.force_overflow = FORCE_OVERFLOW

    def write(self, locations, k, v, *, tentative=False):
        loc = locations.reshape(-1).contiguous()
        if not loc.numel():
            return
        k = k.reshape(-1, self.heads, DIM).to(torch.bfloat16).view(torch.int16)
        v = v.reshape_as(k).to(torch.bfloat16).view(torch.int16)
        if (k.shape[0] != loc.numel() or k.stride(-1) != 1 or v.stride(-1) != 1
                or k.device != self.device or v.device != self.device or loc.device != self.device
                or loc.dtype not in (torch.int32, torch.int64)):
            raise ValueError("inkling_kv12s: invalid device scatter geometry")
        grid = (loc.numel(),)
        _reserve[grid](loc, self.planes[-1], self.stage_locks, self.events, loc.numel(),
                       self.pages, PAGE, self.page_stride, self.live, STAGE_SHIFT, STAGE_MASK, tentative,
                       num_warps=1)
        _scatter[(loc.numel(), 2)](
            loc, k, v, self.planes[-1], self.stage, loc.numel(), self.pages,
            PAGE, self.heads, DIM, self.page_stride, *k.stride()[:2], *v.stride()[:2],
            self.live, STAGE_SHIFT, tentative, triton.next_power_of_2(self.heads * DIM),
        )
        if not tentative:
            self._launch_pack(loc, k, v, promote=False)
        self._launch_publish(loc, k, v, tentative=tentative, promote=False)

    def _launch_pack(self, loc, k, v, *, promote):
        _pack[(loc.numel(), self.heads, 2)](
            loc, k, v, *self.planes, self.stage, self.overflow, self.overflow_locks, self.fail,
            loc.numel(), self.pages, PAGE, self.heads, DIM, self.page_stride,
            *k.stride()[:2], *v.stride()[:2], self.overflow.shape[0],
            STAGE_SHIFT, STAGE_MASK, CAP, CHUNK, EXP_LO, promote, self.live, self.force_overflow, num_warps=8,
        )

    def _launch_publish(self, loc, k, v, *, tentative, promote):
        _publish[(loc.numel(),)](
            loc, k, v, self.planes[-1], self.stage_locks, self.fail, self.planes[4], self.overflow_locks,
            self.stage, self.events, loc.numel(), self.pages, PAGE, self.heads, DIM, self.page_stride,
            *k.stride()[:2], *v.stride()[:2], STAGE_SHIFT, STAGE_MASK, self.live, self.fallback_pages,
            self.overflow.shape[0], tentative, promote, triton.next_power_of_2(self.heads * DIM),
            triton.next_power_of_2(2 * self.heads), num_warps=4,
        )

    def commit(self, pages):
        pages = pages.reshape(-1).contiguous()
        if pages.numel():
            self._launch_pack(pages, self.stage, self.stage, promote=True)
            self._launch_publish(pages, self.stage, self.stage, tentative=False, promote=True)

    @staticmethod
    def settle_pages(locations):
        """(contiguous locations, the page-final candidate pages) for one verify step, shared by every layer.

        One entry per verify ROW instead of per location: a row is at most PAGE consecutive positions of one request
        and the paged allocation keeps loc % PAGE == position % PAGE (page_size == PAGE is checked at pool setup), so
        a row has at most one page-final location; its page (or 0 = nothing to commit) is the row maximum. The commit
        kernels then run B programs instead of B * W, most of which only exited. A row with two page-final locations
        would break that and fails loudly (device assert) instead of skipping a page."""
        loc = locations.contiguous()
        final = loc.remainder(PAGE) == PAGE - 1
        if loc.shape[1] > PAGE:
            return loc, torch.where(final, loc // PAGE, 0)
        torch._assert_async(final.sum(1).le(1).all(), "inkling_kv12s: two page-final locations in one verify row")
        return loc, torch.where(final, loc // PAGE, 0).amax(dim=1)

    def settle(self, locations, commit_lens, prepared=None):
        if (locations.ndim != 2 or commit_lens.shape != (locations.shape[0],)
                or locations.device != self.device or commit_lens.device != self.device
                or locations.dtype not in (torch.int32, torch.int64)
                or commit_lens.dtype not in (torch.int32, torch.int64)):
            raise ValueError("inkling_kv12s: invalid verify settlement geometry")
        if not locations.numel():
            return
        loc, pages = prepared if prepared is not None else self.settle_pages(locations)
        _trim[(loc.numel(),)](
            loc, commit_lens.contiguous(), self.planes[-1], self.stage_locks,
            loc.shape[1], self.pages, PAGE, self.page_stride, STAGE_SHIFT, STAGE_MASK, self.live, num_warps=1,
        )
        # One candidate per full page. A rejected candidate is either empty or
        # still staged with a shorter count. Already packed draft pages are skipped.
        self.commit(pages)

    def release(self, pages, *, recycle=True):
        pages = pages.reshape(-1).contiguous()
        if pages.numel():
            _release[(pages.numel(),)](
                pages, self.planes[-1], self.planes[4], self.stage_locks, self.overflow_locks,
                self.pages, self.heads, self.page_stride, STAGE_SHIFT,
                self.stage.shape[0], self.overflow.shape[0], recycle,
                triton.next_power_of_2(self.heads * 2), num_warps=1,
            )

    def clear(self):
        # Free space may currently be occupied by SWA/state. Stock zeros the
        # actual envelope when a FULL page is handed out.
        self.stage_locks.zero_()
        self.overflow_locks.zero_()

    def _snapshot_reference(self, *, words):
        """Build the host ownership mirror only at the synchronous snapshot API."""
        import heapq
        from .storage import EMPTY, PACKED, OpenPage

        ref = Layer.__new__(Layer)
        ref.__dict__.update(self.__dict__)
        flags = self.flags[words].tolist()
        ref.open = {}
        for page, flag in zip(words.tolist(), flags):
            if flag not in (EMPTY, PACKED):
                ref.open[page] = OpenPage(flag >> STAGE_SHIFT, (flag & STAGE_MASK) - 1)
        ref.spilled = {}
        ref._free_stage = (self.stage_locks[:self.live] == 0).nonzero().flatten().tolist()
        ref._free_overflow = (self.overflow_locks == 0).nonzero().flatten().tolist()
        heapq.heapify(ref._free_stage)
        heapq.heapify(ref._free_overflow)
        # The device writer keeps an unsealable page raw in staging instead of
        # failing, so the reference only validates geometry, not overflow room.
        ref._check_overflow = lambda batches: None
        return ref

    def read_rows(self, locations):
        pages = self._ids(list(dict.fromkeys(loc // PAGE for loc in locations)))
        return self._snapshot_reference(words=pages).read_rows(locations)

    def prepare_write(self, locations, k, v, *, tentative=False):
        pages = self._ids(list(dict.fromkeys(loc // PAGE for loc in locations if loc > 0)))
        reference = self._snapshot_reference(words=pages)
        reference.prepare_write(locations, k, v, tentative=tentative)
        groups = {}
        for row, loc in enumerate(locations):
            if loc > 0:
                groups.setdefault(loc // PAGE, []).append(row)
        # Close whole pages in the same batches checked by the reference.
        # Put the final tails last so the direct full pages don't need slots.
        batches = ([], [])
        for rows in groups.values():
            rows.sort(key=locations.__getitem__)
            full = not tentative and locations[rows[-1]] % PAGE == PAGE - 1
            batches[0 if full else 1].append(rows)

        def publish():
            for groups in batches:
                for start in range(0, len(groups), PACK_BATCH):
                    rows = [row for group in groups[start:start + PACK_BATCH] for row in group]
                    ids = torch.tensor(rows, device=k.device, dtype=torch.long)
                    loc = self._ids([locations[row] for row in rows])
                    self.write(loc, k.index_select(0, ids).to(self.device),
                               v.index_select(0, ids).to(self.device), tentative=tentative)
        return publish


class RawLane:
    """The MTP draft's global lane, kept as raw BF16 inside the packed page envelope.

    Draft K/V escapes the packed exponent window far more often than the target's
    (live serving filled thousands of page-head overflow slots in minutes), and the
    draft proposal reads the whole prefix with noncausal FA4. A raw page-major lane
    costs 4096 B/token like stock and gives FA4 a strided (pages, 128, heads, 128)
    cache at the original physical page IDs: no gather and the stock arithmetic.
    Moves copy whole envelopes, so the lane needs no side-slot lifetime."""

    is_raw = True

    def __init__(self, raw, heads, page_stride, offset, pages):
        self.heads = heads
        self.device = raw.device
        self.pages = pages
        self.page_stride = page_stride
        size = raw_lane_bytes(heads)
        block = raw[:pages * page_stride].view(pages, page_stride)[:, offset:offset + size]
        kv = block.view(torch.bfloat16).unflatten(-1, (2, PAGE, heads, DIM))
        self.k, self.v = kv[:, 0], kv[:, 1]

    def write(self, locations, k, v, *, tentative=False):
        loc = locations.reshape(-1).to(torch.int64)
        if not loc.numel():
            return
        k = k.reshape(-1, self.heads, DIM).to(torch.bfloat16)
        v = v.reshape(-1, self.heads, DIM).to(torch.bfloat16)
        if k.shape[0] != loc.numel() or v.shape[0] != loc.numel() or loc.device != self.device:
            raise ValueError("inkling_kv12s: invalid raw lane scatter geometry")
        # Masked rows carry location 0, the sink page, as in the stock view.
        page, row = loc // PAGE, loc % PAGE
        self.k[page, row] = k
        self.v[page, row] = v

    def read_rows(self, locations):
        loc = torch.as_tensor(locations, dtype=torch.int64, device=self.device)
        page, row = loc // PAGE, loc % PAGE
        return torch.stack((self.k[page, row], self.v[page, row]), dim=1)

    def prepare_write(self, locations, k, v, *, tentative=False):
        loc = torch.as_tensor(locations, dtype=torch.int64, device=self.device)

        def publish():
            self.write(loc, k.to(self.device), v.to(self.device))
        return publish

    # The packed lanes' lifetime hooks: nothing to settle, release or recycle.
    @staticmethod
    def settle_pages(locations):
        return None

    def settle(self, locations, commit_lens, prepared=None):
        pass

    def commit(self, pages):
        pass

    def release(self, pages, *, recycle=True):
        pass

    def clear(self):
        pass


class LaneGroup:
    """The packed lanes of one store, settled together after a verify step.

    DeviceLayer.settle runs _trim and then the promotion's _pack and _publish, once per lane. The lanes share no
    state: each has its own envelope bytes, staging, overflow and fallback slots, lock and event words and failure
    marks. So one launch of each kernel over (programs, lanes) does to every lane exactly what the per-lane loop
    does, in the same kernel order per lane: 3 launches per step instead of 3 per lane. Each lane's buffers are
    addressed from lane 0's by a fixed element offset (`deltas`), which needs one geometry for all lanes."""

    def __init__(self, lanes, deltas):
        self.lanes = lanes
        self.deltas = deltas

    @staticmethod
    def _buffers(lane):
        # In the order of the L_* columns.
        planes = lane.planes
        return (planes[-1], lane.stage_locks, lane.fail, planes[4], lane.overflow_locks, lane.stage, lane.events,
                planes[0], planes[1], planes[2], planes[3], lane.overflow)

    @staticmethod
    def _geometry(lane):
        return (lane.device, lane.heads, lane.pages, lane.page_stride, lane.live, lane.fallback_pages,
                lane.force_overflow, tuple((t.dtype, tuple(t.shape), t.stride()) for t in LaneGroup._buffers(lane)))

    @staticmethod
    def _disjoint(lanes):
        """No two lanes share writable state: the grouped launches run all lanes at once."""
        def span(t):
            return t.data_ptr(), t.data_ptr() + t.numel() * t.element_size()

        raws = {span(lane.raw) for lane in lanes}
        spans = list(raws)
        if len(raws) == 1:  # one envelope buffer (static.py): each lane owns its own slice of every page
            size = lanes[0].layout.size
            offsets = sorted(lane.offset for lane in lanes)
            if (any(b - a < size for a, b in zip(offsets, offsets[1:]))
                    or offsets[0] < 0 or offsets[-1] + size > lanes[0].page_stride):
                return False
        elif len(raws) != len(lanes):
            return False
        for lane in lanes:
            side = (lane.stage, lane.overflow, lane.fail, lane.stage_locks, lane.overflow_locks, lane.events)
            if not all(t.is_contiguous() for t in side):
                return False
            spans += [span(t) for t in side if t.numel()]
        spans.sort()
        return all(b[0] >= a[1] for a, b in zip(spans, spans[1:]))

    @classmethod
    def build(cls, lanes):
        """A group over `lanes`, or None when they cannot share the launches (the caller then settles per lane)."""
        lanes = tuple(lanes)
        if not lanes or any(type(lane) is not DeviceLayer for lane in lanes):
            return None
        first = lanes[0]
        if any(cls._geometry(lane) != cls._geometry(first) for lane in lanes[1:]) or not cls._disjoint(lanes):
            return None
        rows = []
        for lane in lanes:
            row = []
            for mine, ref in zip(cls._buffers(lane), cls._buffers(first)):
                delta = mine.data_ptr() - ref.data_ptr()
                if delta % mine.element_size():
                    return None
                row.append(delta // mine.element_size())
            rows.append(row)
        return cls(lanes, torch.tensor(rows, dtype=torch.int64, device=first.device))

    def settle(self, locations, commit_lens, prepared=None):
        """DeviceLayer.settle(locations, commit_lens, prepared) for every lane of the group."""
        first = self.lanes[0]
        if (locations.ndim != 2 or commit_lens.shape != (locations.shape[0],)
                or locations.device != first.device or commit_lens.device != first.device
                or locations.dtype not in (torch.int32, torch.int64)
                or commit_lens.dtype not in (torch.int32, torch.int64)):
            raise ValueError("inkling_kv12s: invalid verify settlement geometry")
        if not locations.numel():
            return
        loc, pages = prepared if prepared is not None else DeviceLayer.settle_pages(locations)
        lanes = len(self.lanes)
        _trim_lanes[(loc.numel(), lanes)](
            loc, commit_lens.contiguous(), first.planes[-1], first.stage_locks, self.deltas,
            loc.shape[1], first.pages, PAGE, first.page_stride, STAGE_SHIFT, STAGE_MASK, first.live, LANE_KINDS,
            num_warps=1,
        )
        pages = pages.reshape(-1).contiguous()
        if not pages.numel():
            return
        count = pages.numel()
        _pack_lanes[(count, first.heads, 2 * lanes)](
            pages, *first.planes, first.stage, first.overflow, first.overflow_locks, first.fail, self.deltas,
            count, first.pages, PAGE, first.heads, DIM, first.page_stride, *first.stage.stride()[:2],
            first.overflow.shape[0], STAGE_SHIFT, STAGE_MASK, CAP, CHUNK, EXP_LO, first.live, first.force_overflow,
            LANE_KINDS, num_warps=8,
        )
        _publish_lanes[(count, lanes)](
            pages, first.planes[-1], first.stage_locks, first.fail, first.planes[4], first.overflow_locks,
            first.stage, first.events, self.deltas, count, first.pages, PAGE, first.heads, DIM, first.page_stride,
            *first.stage.stride()[:2], STAGE_SHIFT, STAGE_MASK, first.live, first.fallback_pages,
            first.overflow.shape[0], triton.next_power_of_2(first.heads * DIM),
            triton.next_power_of_2(2 * first.heads), LANE_KINDS, num_warps=4,
        )
