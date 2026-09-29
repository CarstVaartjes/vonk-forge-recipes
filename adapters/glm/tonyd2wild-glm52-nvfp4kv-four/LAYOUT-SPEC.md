# GLM-5.2 NVFP4 ds_mla KV record — LAYOUT SPEC (single source of truth)

Record name: `nvfp4_ds_mla` (GLM variant, 400 B/token). Replaces the fp8_ds_mla
656 B/token record for the main sparse-MLA KV cache. Latent = 576 dims
(kv_lora_rank 512 NoPE + 64 RoPE), same as fp8_ds_mla.

## Per-token record, 400 bytes, per-token contiguous (NO split footer)

| bytes       | content |
|-------------|---------|
| [0 : 256)   | 512 E2M1 (NVFP4) nibbles of the NoPE latent (dims 0:512), packed 2/byte: byte b = low nibble dim 2b, high nibble dim 2b+1 |
| [256 : 272) | 16 UE8M0 scale bytes, one per 32-dim NoPE block; block i covers dims [32i, 32i+32) |
| [272 : 400) | 64 bf16 RoPE values (dims 512:576), byte-identical to fp8_ds_mla's [528:656) region (already-roped values copied unchanged) |

Cache tensor shape: `(num_blocks, block_size, 400)` uint8 (vs fp8's `(num_blocks, block_size, 656)`).
Same 1-record-per-token style as fp8_ds_mla — deliberately NOT danielwoz's split
footer, so all existing per-token indexing logic transfers with a stride change.

## Quantization (store side)

Per 32-dim NoPE block:
- `amax = max(|x|)` over the block (fp32 math)
- exponent `E = ceil(log2(amax / 6.0))` (6.0 = E2M1 max magnitude); if amax == 0, E = -127
- stored scale byte `e = clamp(E + 127, 0, 255)` (UE8M0, biased)
- each value quantized as E2M1 of `x / 2^E` via hardware `cvt.rn.satfinite.e2m1x2.f32`
  (satfinite clamps |x|>6 to ±6 code — no explicit clamp needed)

RoPE half stored bf16 unchanged (no quantization) — identical bytes to fp8_ds_mla path.

## Dequantization (read side)

`val = e2m1_decode(nibble) * exp2(e_block - 127.0)`; RoPE read directly as bf16.
E2M1 code: sign<<3 | exp<<1 | mant, magnitudes {0, 0.5, 1, 1.5, 2, 3, 4, 6}.
This mirrors the existing fp8ds footer dequant in sparse_mla_kernels.py (which
is already `tl.exp2(encoded_scale - 127.0)` — there with per-64 blocks / 8
footer bytes on the 448-dim DSV4 variant of fp8_ds_mla); only the block width
and the value load change (1 byte fp8 -> a nibble + LUT/decode). The GLM 656B
fp8_ds_mla record's own 16 scale bytes at [512:528) are 4 fp32 per-128-tile
scales (sm12x_sparse_mla_attn.py) — same 16-byte scale region as ours,
different encoding.

## Memory math

- fp8_ds_mla: 656 B/token → 200,064-token pool in 10.95 GB (KNOWNGOOD)
- nvfp4_ds_mla: 400 B/token = 61.0% → same 10.95 GB pool: 317,312 tokens
  measured at the 200K config (317,279 at the live 316K config). The naive
  record-ratio estimate is ≈328K; the gap is per-token pool overhead the port
  does not shrink (the separate DSA indexer cache shares the same budget).
- (0xdfi's 368 B variant exists but uses a different scale/rope packing; we keep
  per-32 UE8M0 + bf16 rope for a mechanical, low-risk port. Optimize later.)

## Provenance / credits (keep with any derived code)

- Quant scheme + E2M1 packer inline-asm: danielwoz/vllm-dspark-nvfp4 (Apache-2.0),
  itself building on the DSpark fork lineage and Keys' NVFP4 image (v0.21).
- b12x CuTe e2m1 dequant helpers: b12x package (`cute/fp4.py`).
- fp8_ds_mla 656B layout + Triton dequant blocks: jasl/vllm deepseek_v4 path via
  CosmicRaisins/glm-5.2-gb10 kernel overlays.
- Recipe/target numbers: 0xdfi/GLM-5.2-1M-4x-DGX-Spark (FINDINGS).
