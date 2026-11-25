# inkling_kv12: lossless 12-bit KV for the global attention layers (decode / verify)

Switch: `INKLING_KV12=1`; unset = nothing is imported. Install it after `inkling_vattn` (its `rel_bias_partials`
wrapper must be the outer one). Options: `INKLING_KV12_MAX_PAGES` (packed side slots per global layer; default one per
page of the pool), `INKLING_KV12_QCAP` (pages packed per layer visit, default 8192), `INKLING_KV12_SPLITS` (attention
split count; default = the stock heuristic, which keeps the partials bit-identical to stock), `INKLING_KV12_STATS_S`
(coverage log period, default 60 s).

## What it does

Long-context decode is at the DRAM floor for the BF16 KV of the 8 global visits (layers 5 / 11 / 17 / 23; each core
pass has its own cache). Their K (after the L2 clip) and V (after the gain-free RMSNorm) have a narrow exponent
range: on a CPU oracle (K / V of real prompts, dumped from the model; not shipped), all 16 K/V tensors share the same top-15 exponents 116..130, and
2.8e-4 of the values fall outside (tiny values or exact zeros). This package keeps the BF16 pool and adds a packed side
pool per global layer:

| plane | per slot (one packed page of 128 keys) and KV head | content |
|---|---|---|
| SM | 128 x 128 B | sign << 7, 7-bit mantissa |
| EX | 128 x 64 B | 4-bit exponent code (exp - 116, 15 = escape) |
| ESC | 64 entries x 4 B, K and V | escaped values with a nonzero exponent byte: (key << 23) \| (d << 16) \| BF16 bits |
| OFF | 2 x 5 int32 | chunk offsets |

Per physical page: FLAG word per head `(slot << 2) | state` (0 raw, 1 packed, 2 escape overflow, 3 being packed),
SLOT_OF, CLAIM; per slot: PAGE_OF, STAMP (last pack step that packed or read it). Any page can own a slot (the
pool's band may hand out any page ids): slots come from a free stack, then from LRU stealing (clock hand, a slot
unused for 16 pack steps; ownership moves by compare-and-swap, the owner's FLAG is cleared first, the pack kernel
re-validates ownership and publishes with a compare-and-swap so a concurrent invalidation wins).

Packed bytes per key and head: 192 + ~1 instead of 256 (1.33x fewer). A page-head is packed when the page is full and
committed (all its positions < `seq_len - q_len` of the request reading it); the open page and unpacked or
overflowed page-heads are read raw.

Per global decode / verify visit (`rel_bias_partials` wrapper, compact path):
1. `_clear_range_kernel`: FLAG := 0 for the request's open pages; `_find_kernel` (grid batch x page blocks): claim,
   give a slot to and queue every full committed page with an unpacked head (atomic counter); `_pack_kernel` (96 programs) packs the queue. Steady state: nothing to do
   (two tiny launches); after a prefill: all its full pages once (~35 ms per visit-set at 128K).
2. `_kv12_partials_np` (default): the stock `_decode_partials` (global layers) with the K / V tile load replaced: a
   packed tile is decoded in registers, any other tile is loaded raw. The page id is fetched one tile ahead; the
   packed planes, chunk offsets and the tile's first 16 escape entries (one vector load) are requested together
   with FLAG, and the escapes then select their stored BF16 bits into the tile from registers. Same grid, tile order, online softmax, P rounding and partials; `combine_gate_sink` as stock.
   `INKLING_KV12_PREFETCH=1` selects `_kv12_partials` (packed bytes fetched one tile ahead; also bit-identical,
   slower on GB10 in this mixed raw / packed form).

Invalidation: post-hooks of `FlashAttentionBackend.forward_extend` and `forward_decode` clear FLAG of every page
that holds a written position after the stock store (prefill, chunked / mixed extend, verify, decode, whichever
attention kernel ran); `MHAView.load_cpu_copy` (retraction restore) clears the restored pages. Side pools are
allocated before the target runner captures its first graph (`capture_cuda_graphs` wrapper), so every captured
writer clears flags. HiCache and PD disaggregation are refused. Prefill attention keeps reading the raw pool.

## Measured (GB10, 2026-11-22; development check script, not shipped: it needs the oracle K/V dumps)

Full global verify visit (find / pack steady state + attention + `combine_gate_sink`), cold, CUDA-graph medians:

| shape | stock us | kv12 us | speedup |
|---|---:|---:|---:|
| 128K, 1 request x 8 rows, synthetic KV (oracle exponent histogram) | 2,296 | 1,809 | 1.269 |
| 128K, 1 x 8, real oracle KV (layer-5 bank, the most escapes) | 2,297 | 1,996 | 1.151 |
| 128K, 4 x 8, synthetic | 9,123 | 7,149 | 1.276 |
| 256K, 1 x 8, synthetic | 4,572 | 3,623 | 1.262 |

The real layer-5 bank is the worst tensor: its escapes cluster in a few heads (one head has an escape in 10 % of
its rows), each escape costs a whole-tile select, and that head's CTA is the slowest. `INKLING_KV12_SPLITS=48`
recovers ~3 points there (FP32 split boundaries differ from stock: rounding level).

One-time pack after a prefill: 4.4 ms per layer at 128K (1 request), 8.8 ms at 256K. The kernel-only prototype
(all pages packed, no raw branch) measured 1.25-1.32x at 32K-256K.

## Exactness

- Decoded tiles are bit-identical to the BF16 cache (GPU: all 65,536 BF16 patterns, inf / nan, escape-dense rows,
  real oracle K/V).
- Verify partials are bit-identical to stock `_decode_partials` at the stock launch config (int32 compare) for
  packed, mixed and raw tiles, eager and graph-replayed over a 60-step verify loop with page reuse; the gated output
  is byte-identical. Plain decode where stock picks 128-key tiles passes through to stock.
- Staleness: FLAG 1 implies the packed bytes equal the raw bytes, because a page is packed only when full and
  committed, committed KV is never rewritten while a request holds the page, and every forward that writes a page
  clears its FLAG (forward post-hooks; the find kernel also clears open pages before packing). The commit boundary
  uses q_max = tokens - batch + 1, so ragged verify rows are never treated as committed. A page freed and reused is
  rewritten by its new owner, which clears it.

## CUDA graphs

All per-call work is device-side on fixed grids (queue length read on device). A layer's side pool is allocated at its
first eager call (the graph backend runs two eager warm-ups before each capture); a first call inside a capture stays
stock and is logged.

## Memory

Per slot and global layer: 397 KB (0.77 x the raw page); for the 8 global layers 3.1 MB per slot (24 KB per token).
`INKLING_KV12_MAX_PAGES` sets the slot count (e.g. enough for the longest decoding contexts) instead of covering
the whole pool. The raw pool is unchanged, so lower `--mem-fraction-static`
or `--max-total-tokens` accordingly, or cap the coverage with `INKLING_KV12_MAX_PAGES`. A second stage could drop the
raw copy of packed pages (needs prefill to read packed pages exactly).

## Limits

- Global layers only (local layers read 1K keys); BF16 KV cache, page size 128, head dim 128, SM12x.
- Verify batches above 32 rows with `inkling_vattn`'s `bias` lever stay on vattn's kernel (raw).
- Writes outside `forward_extend` / the compact decode-verify path (disaggregation, host load-back) are not tracked:
  do not combine with HiCache / PD disaggregation.
