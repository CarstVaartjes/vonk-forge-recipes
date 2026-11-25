# inkling_scodes: scale codes for the NVFP4 routed experts (exact)

Switch: `INKLING_SCODES=1` (unset = nothing is imported). Exact: the GEMM's MMAs see bit-identical operands; the
output equals FlashInfer's stock kernel bit for bit (tested on real weights and routing, and self-checked live).
Idea and format precedent: Christopher Owen (B12X MXFP4-CSF), ported in spirit from our DeepSeek recipe's
`DSV41_MOE_SCALE_CODES`; no code reused.

## What it does

NVFP4 experts (layers 0-19) carry one UE4M3 scale byte per 16 weights: 11 % of their bytes (497,664 of 4,478,976
bytes per expert for w13 + w2). Each weight row is stored as a base byte plus a 5-bit code per scale (`encode.py`),
packed as 5 bit-planes of 128 bytes per 128 rows x 128 K (640 bytes instead of 1024). Per expert: 497,664 -> 315,648
scale bytes, i.e. **-4.06 % of the NVFP4 expert bytes** read per decode step (w13 -4.09 %, w2 -4.01 %; all 20
layers: 1,274.0 -> 808.1 MB of scales, 11,466 MB of NVFP4 expert bytes).
Measured on the checkpoint (all 40 tensors, `results/kbench/round2/scodes/logs/real_encode_check.log`): 11,794,049 of 11,796,480
rows span <= 31 codes (base = row minimum, exact). 2,431 rows (0.02 %) span 32-50 codes: they are escape rows (base
bit 7 set, codes 0..30 cover the best 31-value window, code 31 = escape); the kernel takes such an SFA word from the
resident stock scales (prefetched one k tile ahead). 3,313 escape words of 318.5 M. Round trip exact on every tensor.

The kernel (`csrc/`) is FlashInfer 0.7.4's SM120 CUTLASS NVFP4 grouped GEMM (`group_gemm_nvfp4_nt_groupwise`,
swap_ab=true, tile_m 128) with one collective changed (`csrc/scodes_sm120_mma_array.hpp`, a copy of CUTLASS's
`sm120_blockscaled_mma_array_tma.hpp`, all edits marked `scodes`):
- producer: one `cp.async.bulk` of the stage's code chunks on the stage mbarrier replaces the SFA TMA (tx bytes
  adjusted);
- consumers: after each `consumer_wait` the 256 MMA threads expand the codes into ONE shared SFA buffer laid out
  exactly as the stock TMA writes it (thread t -> SFA word t of each chunk: `base*0x01010101 + sum_b
  spread(nibble_b) << b`), a named barrier publishes it, and the stock smem->rmem copies and MMAs run unchanged.
  Chunks past K (w2 with tile_k 256) are zero words, like the TMA out-of-bounds fill.
Selected through a partial specialization on `TransformA == inkling_scodes::ScodesTransformA`; schedule, epilogue,
argument kernel and PDL launch are the stock ones. Built by FlashInfer's own JIT (`gen_jit_spec`, same nvcc flags),
cached under `~/.cache/flashinfer/0.7.4/121a/cached_ops/inkling_scodes_<hash>` (~3.5 min cold build, 5 tiles:
32x128, 32x256, 64x128, 64x256, 128x128; 128x256 would need more smem than stock and stays stock).

## How it hooks SGLang (post-import, no engine file edited)

- `Inkling2Experts.swizzle_nvfp4_scales` (called by `post_load_weights`, before the KV pool is sized): afterwards
  the swizzled w13/w2 scales of NVFP4 SM120 experts are encoded; the codes ride on the scale Parameter
  (`_inkling_scodes`). Raw scales stay resident (prefill > 3072 tokens and any fallback use them): **+0.81 GB**
  device memory for 20 layers.
- `Inkling2Experts._load_shard`: an in-place reload of a coded scale re-encodes into the same buffers (captured
  graphs keep valid addresses); scales that no longer fit exit 78.
- `inkling2_grouped_sm120.nvfp4_expert_matmul`: coded weights run the scodes kernel at the tile `_tiles` returns
  (so `inkling_kernels.moe_tiles` and `inkling_moe2` keep choosing tiles); a non-swap choice at <= 3072 tokens (the
  shipped moe_tiles rule for w13 at 9-64 tokens) runs coded swap 32x256 (bit-identical, tested); larger non-swap
  calls, 128x256 and uncoded weights run stock.
- Self-check: the first 3 eager calls per (n, k, tile) also run the stock function and compare bitwise (warm-up
  only, one host sync each); a mismatch exits 78. A configuration with no passed check runs stock during CUDA-graph
  capture. `INKLING_SCODES_CHECK=0` disables it.
- Patch targets are checked as source strings at `install()`; a change raises `RuntimeError` (loader exits 78).

### sitecustomize snippet (any position; it wraps nvfp4_expert_matmul, not _tiles)

```python
    if _on("INKLING_SCODES"):  # NVFP4 expert scale codes (exact)
        import inkling_scodes
        inkling_scodes.install()
```

and in `scripts/launch.sh`: `-e INKLING_SCODES="${SCODES:-}"`.

## Evidence (GB10, 2026-11-22; `results/kbench/round2/scodes/`)

- Format: all 40 real NVFP4 tensors (layers 0-19) encoded and decoded exactly (`logs/real_encode_check.log`).
- Bit identity (`tools/gpu_scodes.py`, `logs/gpu1.log`): real layers 0, 6, 17 (most escape rows), 19 plus "0x"
  (layer 0 with forced escape words in the first and last chunk of every 5th row), real routing (oracle top-8 ids)
  at 8 / 64 / 192 tokens x 2 steps plus a synthetic 160-rows-per-expert case, w13 and w2, all 5 tiles: 350/350
  outputs bit-identical to FlashInfer's stock kernel at the same tile, every output written; 10/10 identical to the
  stock NON-swap 64x256 call it replaces at 9-64 tokens; CUDA graph captured on one routing and replayed on two
  routings: identical to stock.
- Integration (`tools/gpu_integration.py`): hooks over moe_tiles + moe2 and over moe_tiles alone; eager and graph
  replay identical to stock at 8/64/192 tokens; corrupted codes trip the self-check (exit 78); reload re-encodes
  into the same buffers.
- Cold-L2 microbench (median of 30, 128 MiB read flush before each call), us per call, stock -> scodes at the
  shipped tile 32x256 (union 37 / 97 / 126 experts):

| layer | 8 tok w13 | 8 tok w2 | 64 tok w13 | 64 tok w2 | 192 tok w13 | 192 tok w2 |
|---|---|---|---|---|---|---|
| 0 | 494 -> 475 | 251 -> 242 | 1291 -> 1240 | 655 -> 632 | 1706 -> 1639 | 881 -> 843 |
| 6 | 495 -> 476 | 251 -> 242 | 1293 -> 1245 | 654 -> 632 | 1728 -> 1650 | 887 -> 840 |
| 17 | 496 -> 483 | 252 -> 246 | 1298 -> 1259 | 655 -> 641 | 1709 -> 1662 | 876 -> 856 |
| 19 | 495 -> 477 | 250 -> 246 | 1294 -> 1245 | 665 -> 637 | 1711 -> 1648 | 874 -> 844 |

  -3.5 to -4.5 % per call on low-escape layers (the byte cut is -4.1 %), -2.2 to -3 % on layer 17 (890 escape rows).
  Forced escapes in 20 % of rows ("0x") cost +7 %: escape words are a slow path, fine at the real 0.02 %.
- Registers: 118-168 per kernel, no local memory (stock gemm_sm120: 114-168).
