"""kv12 device side: packed 12-bit side pool of the GLOBAL attention layers' KV, its packer, and the decode /
verify split-KV attention that reads packed pages (exact BF16 bits decoded in registers) plus raw tiles.

The attention kernels (`_kv12_partials_np`, `_kv12_partials`) are modified copies of SGLang's `_decode_partials`
(sglang/kernels/ops/attention/inkling2_decode_attention_sm90.py, Apache-2.0, Copyright the SGLang project authors):
the K / V tile loads are replaced by the packed-tile decode; grid, tile order, online softmax and partials are unchanged.
See NOTICE.

Packed planes live in NS side SLOTS (any physical page can own one), K and V separately (128 keys, head dim 128):
  SM  uint8 [NS, 128, KVH, 128]     sign << 7 | 7-bit mantissa
  EX  uint8 [NS, 128, KVH, 64]      4-bit exponent codes, element 2i low nibble: exp - 116 for 116..130, 15 = escape
                                      (decodes to a signed zero unless the escape stream has the value)
  ESC int32 [NS, KVH, 2, CAP]       escape entries of the slot-head (K at [.., 0, :], V at [.., 1, :]):
                                      (key << 23) | (d << 16) | BF16 bits, escaped values with exponent byte != 0
  OFF int32 [NS, KVH, 2, 5]         entry offsets of the four 32-key chunks + end (relative to the region)
Per physical page (P pages + a sink row P):
  FLAG int32 [P + 1, KVH]           (slot << 2) | state; state 0 = read raw, 1 = packed in `slot`, 2 = escape
                                      overflow (raw until rewritten), 3 = being packed
  SLOT_OF int32 [P + 1]             the page's slot or -1;  PAGE_OF int32 [NS + 1]: the slot's page or -1
  CLAIM int32 [P + 1]               dedupes a page queued by several requests in one find launch
  STAMP int32 [NS + 1]              last pack_step that packed or read the slot (LRU for slot stealing)
Free slots: a stack (FREE, TOP) filled at start; when it is empty, find steals a slot whose STAMP is older than AGE
pack_steps (its page's FLAG is cleared first; the pack kernel re-validates SLOT_OF / PAGE_OF ownership).
The raw BF16 pool is kept (speed lever): correctness only needs state 1 to imply that the slot's bytes equal the
page's current raw bytes. Pages are packed only when full and committed (every position < seq_len - q_max of a
request that reads them); every write path clears the page's FLAG (open pages here before packing; prefill / extend,
compaction moves, retraction / snapshot restores in patch.py). Invalidation never touches the slot bookkeeping, so it
is safe from any stream; slot allocation happens only in find launches on the serving stream.
"""

import torch
import triton
import triton.language as tl

PAGE = 128
G = 32  # keys per escape chunk
NC = PAGE // G
EXP_LO = 116
CAP = 64  # escape entries per (slot, head, tensor); more -> state 2 (raw)
FIND_BLK = 128  # pages per find-program
AGE = 16  # pack_steps a slot must be unused before it can be stolen


# ------------------------------------------------------------------------------------------------ pack side


@triton.jit
def _clear_range_kernel(PT, SeqLens, CuQ, FLAGS, SP, NPAGES, QLEN: tl.constexpr, KVH: tl.constexpr,
                        PAGE_: tl.constexpr):
    """FLAG := 0 for every page holding a position in [seq_len - q_len, seq_len) of request b. QLEN > 0: uniform
    (decode / verify open pages, with q_max), 0: per request from CuQ (prefill / extend post-hook)."""
    b = tl.program_id(0)
    seqlen = tl.load(SeqLens + b)
    if QLEN > 0:
        qlen = QLEN
    else:
        qlen = tl.load(CuQ + b + 1) - tl.load(CuQ + b)
    hk = tl.arange(0, KVH)
    if (seqlen > 0) & (qlen > 0):
        p0 = tl.maximum(seqlen - qlen, 0) // PAGE_
        p1 = (seqlen - 1) // PAGE_
        for p in range(p0, p1 + 1):
            pg = tl.load(PT + b.to(tl.int64) * SP + p)
            ok = (pg >= 0) & (pg < NPAGES)
            pg = tl.where(ok, pg, NPAGES)
            tl.store(FLAGS + pg * KVH + hk, tl.zeros((KVH,), tl.int32), ok)


@triton.jit
def _find_kernel(PT, SeqLens, FLAGS, SLOT_OF, PAGE_OF, CLAIM, STAMP, FREE, TOP, CLOCK, STEP, QUEUE, QCOUNT, STATS,
                 SP, NPAGES, NS, QCAP, QMAX, AGE_,
                 KVH: tl.constexpr, PAGE_: tl.constexpr, BLK: tl.constexpr):
    """Per (request, block of BLK page-table entries): every full committed page with a head in state 0 is claimed,
    gets a slot (its own, a free one, or a stolen LRU one) and is queued for _pack_kernel."""
    b = tl.program_id(0)
    blk = tl.program_id(1)
    seqlen = tl.load(SeqLens + b)
    committed = seqlen - QMAX
    full = tl.maximum(committed, 0) // PAGE_
    hk = tl.arange(0, KVH)
    i = blk * BLK + tl.arange(0, BLK)
    inb = i < full
    pg = tl.load(PT + b.to(tl.int64) * SP + i, inb, other=0)
    inb &= (pg >= 0) & (pg < NPAGES)
    f = tl.load(FLAGS + pg[:, None] * KVH + hk[None, :], inb[:, None], other=1)
    need = inb & (tl.min(f & 3, 1) == 0)
    if tl.sum(need.to(tl.int32)) > 0:
        now = tl.load(STEP)
        cl = tl.atomic_cas(CLAIM + tl.where(need, pg, NPAGES), tl.zeros((BLK,), tl.int32),
                           tl.full((BLK,), 1, tl.int32))
        got = need & (cl == 0)
        so = tl.load(SLOT_OF + pg, got, other=-1)
        # a mapping is only reused while the slot still names this page (a same-launch double steal can leave a
        # stale SLOT_OF; it heals here)
        own = tl.load(PAGE_OF + tl.where(so >= 0, so, NS), got & (so >= 0), other=-2)
        so = tl.where(own == pg, so, -1)
        lack = got & (so < 0)
        slot = so
        nl = tl.sum(lack.to(tl.int32))
        if nl > 0:
            # pop from the free stack (pops never overlap pushes: the stack is only filled at start)
            base = tl.atomic_add(TOP, -nl)
            rk = tl.cumsum(lack.to(tl.int32), 0) - 1
            idx = base - 1 - rk
            okp = lack & (idx >= 0)
            fs = tl.load(FREE + tl.where(okp, idx, 0), okp, other=-1)
            nbad = nl - tl.sum(okp.to(tl.int32))
            if nbad > 0:
                tl.atomic_add(TOP, nbad)
            slot = tl.where(okp, fs, slot)
            rest = lack & ~okp
            nr = tl.sum(rest.to(tl.int32))
            if nr > 0:
                # steal LRU slots (clock hand); ownership moves with a compare-and-swap on PAGE_OF
                c0 = tl.atomic_add(CLOCK, nr) & 0x3FFFFFFF
                rr = tl.cumsum(rest.to(tl.int32), 0) - 1
                cand = (c0 % NS + rr) % NS
                owner = tl.load(PAGE_OF + cand, rest, other=-1)
                st = tl.load(STAMP + cand, rest, other=0)
                old = rest & (owner >= 0) & (st < now - AGE_)
                won = tl.atomic_cas(PAGE_OF + tl.where(old, cand, NS), tl.where(old, owner, -2), pg)
                winner = old & (won == owner)
                ow = tl.where(winner, owner, NPAGES)
                tl.store(SLOT_OF + ow, tl.full((BLK,), -1, tl.int32), winner)
                tl.store(FLAGS + ow[:, None] * KVH + hk[None, :], tl.zeros((BLK, KVH), tl.int32), winner[:, None])
                slot = tl.where(winner, cand, slot)
                tl.atomic_add(STATS + 7, tl.sum(winner.to(tl.int64)))
        assigned = got & (slot >= 0)
        fresh = assigned & (so < 0)
        tl.store(SLOT_OF + pg, slot, fresh)
        tl.store(PAGE_OF + tl.where(fresh, slot, NS), pg, fresh)
        tl.store(STAMP + tl.where(assigned, slot, NS), tl.zeros((BLK,), tl.int32) + now, assigned)
        tl.store(CLAIM + pg, tl.zeros((BLK,), tl.int32), got & ~assigned)
        n = tl.sum(assigned.to(tl.int32))
        if n > 0:
            qb = tl.atomic_add(QCOUNT, n)
            dst = qb + tl.cumsum(assigned.to(tl.int32), 0) - 1
            tl.store(QUEUE + dst, pg, assigned & (dst < QCAP))
            tl.store(CLAIM + pg, tl.zeros((BLK,), tl.int32), assigned & (dst >= QCAP))


@triton.jit
def _pack_tensor(SRC, S0, S1, S2, SMP, EXP_, ESC, OFF, page, sl, hk, region, offb,
                 KVH: tl.constexpr, PAGE_: tl.constexpr, D: tl.constexpr, GK: tl.constexpr, NCH: tl.constexpr,
                 CAPN: tl.constexpr, EXPLO: tl.constexpr):
    """Pack one (page, head) of one tensor into slot `sl`; returns the escape count (> CAPN: overflow, entries clipped)."""
    rows = tl.arange(0, GK)
    d = tl.arange(0, D)
    dh = tl.arange(0, D // 2)
    cnt = tl.full((), 0, tl.int32)
    tl.store(OFF + offb, 0)
    for c in range(NCH):
        slot = c * GK + rows
        x = tl.load(SRC + page * S0 + slot[:, None] * S1 + hk * S2 + d[None, :])
        bits = x.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
        expo = (bits >> 7) & 255
        sm = ((bits >> 8) & 128) | (bits & 127)
        inwin = (expo >= EXPLO) & (expo <= EXPLO + 14)
        code = tl.where(inwin, expo - EXPLO, 15)
        lo, hi = tl.split(tl.reshape(code, (GK, D // 2, 2)))
        tl.store(SMP + sl * (PAGE_ * KVH * D) + slot[:, None] * (KVH * D) + hk * D + d[None, :], sm.to(tl.uint8))
        tl.store(EXP_ + sl * (PAGE_ * KVH * (D // 2)) + slot[:, None] * (KVH * (D // 2)) + hk * (D // 2)
                 + dh[None, :], (lo | (hi << 4)).to(tl.uint8))
        esc = (~inwin) & (expo != 0)
        flat = tl.reshape(esc.to(tl.int32), (GK * D,))
        n = tl.sum(flat)
        if n > 0:
            rank = tl.cumsum(flat, 0) - 1
            ent = (slot[:, None] << 23) | (d[None, :] << 16) | bits
            dst = cnt + rank
            tl.store(ESC + region + dst, tl.reshape(ent, (GK * D,)), (flat > 0) & (dst < CAPN))
        cnt += n
        tl.store(OFF + offb + c + 1, cnt)
    return cnt


@triton.jit
def _pack_kernel(K, V, SK0, SK1, SK2, SMK, EXK, SMV, EXV, ESC, OFF, FLAGS, SLOT_OF, PAGE_OF, CLAIM, QUEUE, QCOUNT,
                 QCAP, STATS,
                 KVH: tl.constexpr, PAGE_: tl.constexpr, D: tl.constexpr, GK: tl.constexpr, NCH: tl.constexpr,
                 CAPN: tl.constexpr, EXPLO: tl.constexpr):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    count = tl.minimum(tl.load(QCOUNT), QCAP)
    for item in range(pid, count * KVH, nprog):
        page = tl.load(QUEUE + item // KVH).to(tl.int64)
        hk = item % KVH
        s = tl.load(SLOT_OF + page).to(tl.int64)
        owned = s >= 0
        if owned:
            owned = tl.load(PAGE_OF + s) == page
        if owned:
            if tl.atomic_cas(FLAGS + page * KVH + hk, 0, 3) == 0:
                rk = ((s * KVH + hk) * 2 + 0) * CAPN
                rv = ((s * KVH + hk) * 2 + 1) * CAPN
                ok_ = ((s * KVH + hk) * 2 + 0) * (NCH + 1)
                ov = ((s * KVH + hk) * 2 + 1) * (NCH + 1)
                nk = _pack_tensor(K, SK0, SK1, SK2, SMK, EXK, ESC, OFF, page, s, hk, rk, ok_, KVH, PAGE_, D, GK, NCH,
                                  CAPN, EXPLO)
                nv = _pack_tensor(V, SK0, SK1, SK2, SMV, EXV, ESC, OFF, page, s, hk, rv, ov, KVH, PAGE_, D, GK, NCH,
                                  CAPN, EXPLO)
                state = tl.where((nk <= CAPN) & (nv <= CAPN), 1, 2)
                # publish only if no writer cleared the claim meanwhile (a clear issued on another stream after its
                # raw write must win: the bytes read above may be the old ones)
                tl.atomic_cas(FLAGS + page * KVH + hk, 3, ((s << 2) | state).to(tl.int32))
                tl.atomic_add(STATS + 4 + (state == 2).to(tl.int32), 1)
        if hk == 0:
            tl.store(CLAIM + page, 0)


# ------------------------------------------------------------------------------------------------ attention


@triton.jit
def _fetch(SM, EX, OFF, page, slot0, hk, cols, d,
           KVH: tl.constexpr, PAGE_: tl.constexpr, D: tl.constexpr, GK: tl.constexpr, NCH: tl.constexpr,
           T: tl.constexpr, BLOCK_N: tl.constexpr):
    slot = slot0 + cols
    sm = tl.load(SM + page * (PAGE_ * KVH * D) + slot[:, None] * (KVH * D) + hk * D + d[None, :])
    dh = tl.arange(0, D // 2)
    cb = tl.load(EX + page * (PAGE_ * KVH * (D // 2)) + slot[:, None] * (KVH * (D // 2)) + hk * (D // 2)
                 + dh[None, :])
    ob = ((page * KVH + hk) * 2 + T) * (NCH + 1) + slot0 // GK
    e0 = tl.load(OFF + ob)
    e1 = tl.load(OFF + ob + BLOCK_N // GK)
    return sm, cb, e0, e1


@triton.jit
def _finish(sm, cb, ESC, region, e0, e1, slot0, cols, d, EXPLO: tl.constexpr, BLOCK_N: tl.constexpr,
            D: tl.constexpr):
    s32 = sm.to(tl.int32)
    c32 = cb.to(tl.int32)
    code = tl.reshape(tl.join(c32 & 15, (c32 >> 4) & 15), (BLOCK_N, D))
    expo = tl.where(code == 15, 0, code + EXPLO)
    bits = ((s32 & 128) << 8) | (expo << 7) | (s32 & 127)
    for c in range(e0, e1):
        ent = tl.load(ESC + region + c)
        es = ((ent >> 23) & 127) - slot0
        ed = (ent >> 16) & 127
        bits = tl.where((cols[:, None] == es) & (d[None, :] == ed), ent & 0xFFFF, bits)
    return bits.to(tl.uint16).to(tl.bfloat16, bitcast=True)


@triton.jit
def _fetch_e(SM, EX, OFF, ESC, region, page, slot0, hk, cols, d,
             KVH: tl.constexpr, PAGE_: tl.constexpr, D: tl.constexpr, GK: tl.constexpr, NCH: tl.constexpr,
             T: tl.constexpr, BLOCK_N: tl.constexpr, NE: tl.constexpr, CAPN_: tl.constexpr):
    """_fetch plus the tile's first NE escape entries as one vector load (no per-escape load latency)."""
    sm, cb, e0, e1 = _fetch(SM, EX, OFF, page, slot0, hk, cols, d, KVH, PAGE_, D, GK, NCH, T, BLOCK_N)
    jj = tl.arange(0, NE)
    ents = tl.load(ESC + region + e0 + jj, ((e0 + jj) < e1) & ((e0 + jj) < CAPN_), other=0)
    return sm, cb, e0, e1, ents


@triton.jit
def _finish_e(sm, cb, ents, ESC, region, e0, e1, slot0, cols, d, EXPLO: tl.constexpr, BLOCK_N: tl.constexpr,
              D: tl.constexpr, NE: tl.constexpr):
    s32 = sm.to(tl.int32)
    c32 = cb.to(tl.int32)
    code = tl.reshape(tl.join(c32 & 15, (c32 >> 4) & 15), (BLOCK_N, D))
    expo = tl.where(code == 15, 0, code + EXPLO)
    bits = ((s32 & 128) << 8) | (expo << 7) | (s32 & 127)
    n = e1 - e0
    jj = tl.arange(0, NE)
    for c in range(0, tl.minimum(n, NE)):
        ent = tl.sum(tl.where(jj == c, ents, 0))  # entry c from registers
        es = ((ent >> 23) & 127) - slot0
        ed = (ent >> 16) & 127
        bits = tl.where((cols[:, None] == es) & (d[None, :] == ed), ent & 0xFFFF, bits)
    for c in range(e0 + NE, e1):  # rare: more than NE escapes in the tile
        ent = tl.load(ESC + region + c)
        es = ((ent >> 23) & 127) - slot0
        ed = (ent >> 16) & 127
        bits = tl.where((cols[:, None] == es) & (d[None, :] == ed), ent & 0xFFFF, bits)
    return bits.to(tl.uint16).to(tl.bfloat16, bitcast=True)


@triton.jit
def _kv12_partials_np(
    Q, K, V, SMK, EXK, SMV, EXV, ESC, OFF, FLAGS, STAMP, STEP, PageTable, SeqLens, Bias, OPartial, MPartial, LPartial,
    STATS, scale, batch, NPAGES, QMAX,
    SQ0: tl.constexpr, SQ1: tl.constexpr, SK0: tl.constexpr, SK1: tl.constexpr, SK2: tl.constexpr,
    SP: tl.constexpr, SB0: tl.constexpr, SB1: tl.constexpr, HEADS_K: tl.constexpr, QLEN: tl.constexpr,
    GROUP: tl.constexpr, ROWS: tl.constexpr, D: tl.constexpr, PAGE: tl.constexpr, EXTENT: tl.constexpr,
    BLOCK_N: tl.constexpr, GK: tl.constexpr, NCH: tl.constexpr, CAPN: tl.constexpr, EXPLO: tl.constexpr,
):
    # == stock _decode_partials (sglang inkling2_decode_attention_sm90, global layers: WINDOW = -1) with the K / V
    # tile loads replaced: a packed page-head (state 1, page fully committed) is decoded from its slot's bytes, any
    # other tile is loaded raw as stock. Same grid, tile order, online softmax, P rounding, partials. ==
    tl.extra.cuda.gdc_wait()
    tl.extra.cuda.gdc_launch_dependents()
    b = tl.program_id(0) // HEADS_K
    hk = tl.program_id(0) % HEADS_K
    split = tl.program_id(1)
    splits = tl.num_programs(1)
    seqlen = tl.load(SeqLens + b)
    first = 0
    tiles = tl.cdiv(seqlen - first // BLOCK_N * BLOCK_N, BLOCK_N)
    per_split = tl.cdiv(tiles, splits)
    tile_lo = first // BLOCK_N + split * per_split
    tile_hi = tl.minimum(tile_lo + per_split, first // BLOCK_N + tiles)
    committed = seqlen - QMAX

    rows = tl.arange(0, ROWS)
    head_mask = rows < QLEN * GROUP
    token = b * QLEN + rows // GROUP
    qh = hk * GROUP + rows % GROUP
    q_pos = seqlen - QLEN + rows // GROUP
    d = tl.arange(0, D)
    q = tl.load(Q + token[:, None] * SQ0 + qh[:, None] * SQ1 + d[None, :], head_mask[:, None], other=0)
    m = tl.full((ROWS,), float("-inf"), tl.float32)
    l = tl.zeros((ROWS,), tl.float32)
    acc = tl.zeros((ROWS, D), tl.float32)
    cols = tl.arange(0, BLOCK_N)
    last_tile = tl.maximum(first // BLOCK_N + tiles - 1, 0)
    page_n = tl.load(PageTable + b * SP + (tl.minimum(tile_lo, last_tile) * BLOCK_N) // PAGE, tile_lo < tile_hi,
                     other=0).to(tl.int64)
    now = tl.load(STEP)
    n_packed = tl.full((), 0, tl.int32)
    n_flag0 = tl.full((), 0, tl.int32)
    n_ovf = tl.full((), 0, tl.int32)
    max_page = tl.full((), 0, tl.int64)
    for tile in range(tile_lo, tile_hi):
        pos = tile * BLOCK_N + cols
        valid = (pos < seqlen) & (pos >= first)
        page = page_n  # page id fetched one tile ahead
        page_n = tl.load(PageTable + b * SP + (tl.minimum(tile + 1, last_tile) * BLOCK_N) // PAGE).to(tl.int64)
        slot0 = (tile * BLOCK_N) % PAGE
        pgf = tl.where((page >= 0) & (page < NPAGES), page, NPAGES)
        fl = tl.load(FLAGS + pgf * HEADS_K + hk)
        state = fl & 3
        sl = tl.where(state == 1, fl >> 2, 0).to(tl.int64)
        # the slot's packed bytes are requested right after FLAG (slot 0's, unused, for a raw tile)
        rgk = ((sl * HEADS_K + hk) * 2 + 0) * CAPN
        smk, cbk, k0, k1, ek = _fetch_e(SMK, EXK, OFF, ESC, rgk, sl, slot0, hk, cols, d, HEADS_K, PAGE, D, GK,
                                        NCH, 0, BLOCK_N, 16, CAPN)
        rgv = ((sl * HEADS_K + hk) * 2 + 1) * CAPN
        smv, cbv, v0, v1, ev = _fetch_e(SMV, EXV, OFF, ESC, rgv, sl, slot0, hk, cols, d, HEADS_K, PAGE, D, GK,
                                        NCH, 1, BLOCK_N, 16, CAPN)
        packed = (state == 1) & ((tile // (PAGE // BLOCK_N) + 1) * PAGE <= committed)
        n_packed += packed.to(tl.int32)
        n_flag0 += (state == 0).to(tl.int32)
        n_ovf += (state == 2).to(tl.int32)
        max_page = tl.maximum(max_page, page)
        kv_offset = page * SK0 + (slot0 + cols)[:, None] * SK1 + hk * SK2 + d[None, :]
        if packed:
            tl.store(STAMP + sl, now)  # LRU stamp of the slot
            k = _finish_e(smk, cbk, ek, ESC, rgk, k0, k1, slot0, cols, d, EXPLO, BLOCK_N, D, 16)
            k = tl.where(valid[:, None], k, 0.0)
        else:
            k = tl.load(K + kv_offset, valid[:, None], other=0)
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
        if packed:
            v = _finish_e(smv, cbv, ev, ESC, rgv, v0, v1, slot0, cols, d, EXPLO, BLOCK_N, D, 16)
            v = tl.where(valid[:, None], v, 0.0)
        else:
            v = tl.load(V + kv_offset, valid[:, None], other=0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m = m_new
    # coverage counters (int64): packed tiles, raw state-0 tiles, raw overflow tiles, all tiles, max page id
    tl.atomic_add(STATS + 0, n_packed.to(tl.int64))
    tl.atomic_add(STATS + 1, n_flag0.to(tl.int64))
    tl.atomic_add(STATS + 2, n_ovf.to(tl.int64))
    tl.atomic_add(STATS + 3, tl.maximum(tile_hi - tile_lo, 0).to(tl.int64))
    tl.atomic_max(STATS + 6, max_page)
    stat = (split * batch * QLEN + token) * (HEADS_K * GROUP) + qh
    tl.store(MPartial + stat, m, head_mask)
    tl.store(LPartial + stat, l, head_mask)
    tl.store(OPartial + stat[:, None] * D + d[None, :], acc, head_mask[:, None])


# ------------------------------------------------------------------------------------------------ host side


class LayerPack:
    """Side pool of one global cache layer (keyed by its key buffer's data pointer): n_slots packed page slots
    (0 = one per physical page) and the per-page FLAG / slot bookkeeping."""

    def __init__(self, k_cache, n_slots, qcap):
        pages, page, kvh, d = k_cache.shape
        assert page == PAGE and d == 128 and k_cache.dtype == torch.bfloat16
        dev = k_cache.device
        self.npages = pages
        self.ns = min(pages, n_slots) if n_slots > 0 else pages
        self.maxp = self.ns  # (log compatibility)
        self.kvh = kvh
        n = self.ns
        self.smk = torch.empty((n, PAGE, kvh, d), dtype=torch.uint8, device=dev)
        self.exk = torch.empty((n, PAGE, kvh, d // 2), dtype=torch.uint8, device=dev)
        self.smv = torch.empty_like(self.smk)
        self.exv = torch.empty_like(self.exk)
        self.esc = torch.empty((n, kvh, 2, CAP), dtype=torch.int32, device=dev)
        self.off = torch.zeros((n, kvh, 2, NC + 1), dtype=torch.int32, device=dev)
        self.flags = torch.zeros((pages + 1, kvh), dtype=torch.int32, device=dev)
        self.slot_of = torch.full((pages + 1,), -1, dtype=torch.int32, device=dev)
        self.claim = torch.zeros((pages + 1,), dtype=torch.int32, device=dev)
        self.page_of = torch.full((n + 1,), -1, dtype=torch.int32, device=dev)
        self.stamp = torch.zeros((n + 1,), dtype=torch.int32, device=dev)
        self.free = torch.arange(n - 1, -1, -1, dtype=torch.int32, device=dev)  # popped from the end: slot 0 first
        self.top = torch.full((1,), n, dtype=torch.int32, device=dev)
        self.clock = torch.zeros((1,), dtype=torch.int32, device=dev)
        self.step = torch.full((1,), AGE + 1, dtype=torch.int32, device=dev)
        self.key = (k_cache.data_ptr(), tuple(k_cache.shape), tuple(k_cache.stride()))
        self.qcap = qcap
        self.queue = torch.zeros((qcap,), dtype=torch.int32, device=dev)
        self.qcount = torch.zeros((1,), dtype=torch.int32, device=dev)
        # [packed tiles, raw state-0 tiles, raw overflow tiles, tiles, page-heads packed, overflowed, max page id,
        #  slots stolen]
        self.stats = torch.zeros((8,), dtype=torch.int64, device=dev)

    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in (
            self.smk, self.exk, self.smv, self.exv, self.esc, self.off, self.flags, self.slot_of, self.claim,
            self.page_of, self.stamp, self.free, self.queue))


def stock_config(batch, heads_k, qlen, span, sms):
    """The stock decode_partials launch heuristic (splits, block_n)."""
    max_splits = max(1, min(2 * sms // max(1, batch * heads_k), 64))
    block_n = 128 if qlen == 1 and span >= 128 * max_splits else 32
    splits = min(triton.cdiv(span, block_n), max_splits)
    return splits, block_n


def pack_step(lp, k_cache, v_cache, page_table, seqused_k, qmax, pack_programs=96):
    """Clear FLAG of the open pages, then claim / slot / queue and pack the full committed pages (graph-safe: fixed
    grids, device-side counts). qmax: an upper bound of every request's query length in this call."""
    bs = seqused_k.shape[0]
    assert v_cache.stride() == k_cache.stride() and k_cache.shape[1] == PAGE
    _clear_range_kernel[(bs,)](page_table, seqused_k, seqused_k, lp.flags, page_table.stride(0), lp.npages, qmax,
                               lp.kvh, PAGE, num_warps=1)
    lp.qcount.zero_()
    lp.step.add_(1)
    nblk = triton.cdiv(page_table.shape[1], FIND_BLK)
    _find_kernel[(bs, nblk)](
        page_table, seqused_k, lp.flags, lp.slot_of, lp.page_of, lp.claim, lp.stamp, lp.free, lp.top, lp.clock,
        lp.step, lp.queue, lp.qcount, lp.stats, page_table.stride(0), lp.npages, lp.ns, lp.qcap, qmax, AGE, lp.kvh,
        PAGE, FIND_BLK, num_warps=4)
    _pack_kernel[(pack_programs,)](
        k_cache, v_cache, k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), lp.smk, lp.exk, lp.smv, lp.exv,
        lp.esc, lp.off, lp.flags, lp.slot_of, lp.page_of, lp.claim, lp.queue, lp.qcount, lp.qcap, lp.stats, lp.kvh,
        PAGE, 128, G, NC, CAP, EXP_LO, num_warps=4)


def clear_written(lp, page_table, seqlens, cu_q=None, qlen=0):
    """FLAG := 0 for the pages of the written positions (cu_q: per-request lengths, else uniform qlen)."""
    bs = seqlens.shape[0]
    assert cu_q is not None or qlen > 0
    _clear_range_kernel[(bs,)](page_table, seqlens, cu_q if cu_q is not None else seqlens, lp.flags,
                               page_table.stride(0), lp.npages, qlen if cu_q is None else 0, lp.kvh, PAGE,
                               num_warps=1)


def clear_pages(lp, pages):
    """Eager FLAG := 0 for physical page ids (compaction moves, retraction restore); slots stay bound."""
    p = pages.to(device=lp.flags.device, dtype=torch.int64).clamp(0, lp.npages)
    lp.flags.index_fill_(0, p, 0)


def clear_all(lp):
    lp.flags.zero_()


def kv12_partials(lp, q, k_cache, v_cache, page_table, seqused_k, rel_bias, softmax_scale, *, config=None,
                  num_warps=4, qmax=None, prefetch=False, num_stages=1):
    """Partials (splits, tokens, 1, heads_q[, d]) like stock decode_partials (global layers: no window)."""
    assert not prefetch, "kv12: the lookahead variant was retired"
    tokens, heads_q, head_dim = q.shape
    batch = seqused_k.shape[0]
    qlen = tokens // max(batch, 1)
    assert qlen * batch == tokens
    _, page_size, heads_k, _ = k_cache.shape
    assert page_size == PAGE and v_cache.stride() == k_cache.stride()
    group = heads_q // heads_k
    if config is None:
        sms = torch.cuda.get_device_properties(q.device).multi_processor_count
        config = stock_config(batch, heads_k, qlen, page_table.shape[1] * page_size, sms)
    splits, block_n = config
    assert block_n in (32, 64), "kv12: 32/64-key tiles only (the caller passes stock's 128-key decode through)"
    if qmax is None:
        qmax = qlen
    o = torch.empty((splits, tokens, 1, heads_q, head_dim), dtype=torch.float32, device=q.device)
    m = torch.empty((splits, tokens, 1, heads_q), dtype=torch.float32, device=q.device)
    l = torch.empty_like(m)
    _kv12_partials_np[(batch * heads_k, splits)](
        q, k_cache, v_cache, lp.smk, lp.exk, lp.smv, lp.exv, lp.esc, lp.off, lp.flags, lp.stamp, lp.step, page_table,
        seqused_k, rel_bias, o, m, l, lp.stats, softmax_scale, batch, lp.npages, qmax,
        q.stride(0), q.stride(1), k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), page_table.stride(0),
        rel_bias.stride(0), rel_bias.stride(1), heads_k, qlen, group, max(16, triton.next_power_of_2(qlen * group)),
        head_dim, page_size, rel_bias.shape[-1], block_n, G, NC, CAP, EXP_LO,
        num_warps=num_warps, num_stages=num_stages, launch_pdl=True,
    )
    return o, m, l
