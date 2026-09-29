# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
NVFP4 (E2M1) store + dequant-gather Triton kernels for the GLM-5.2 sparse-MLA
main KV cache (``nvfp4_ds_mla`` record, 400 B/token) on GB10 / sm_121a.

Drop-in replacements for the two cache-side ops of the fp8_ds_mla path:
  * store  : replaces the KV insert (fp8 656 B/token -> nvfp4 400 B/token)
  * gather : replaces ``ops.cp_gather_and_upconvert_fp8_kv_cache`` for the
             prefill workspace path (see flashmla_sparse.py::
             _forward_fp8_kv_separate_prefill_decode), emitting bf16 [n, 576].

LOCKED RECORD LAYOUT (single source of truth: LAYOUT-SPEC.md).
Per-token record, 400 bytes, per-token contiguous (NO split footer):

  | bytes       | content                                                     |
  |-------------|-------------------------------------------------------------|
  | [0 : 256)   | 512 E2M1 (NVFP4) nibbles of the NoPE latent (dims 0:512),   |
  |             | packed 2/byte: byte b = low nibble dim 2b, high nibble      |
  |             | dim 2b+1                                                    |
  | [256 : 272) | 16 UE8M0 scale bytes, one per 32-dim NoPE block; block i    |
  |             | covers dims [32i, 32i+32)                                   |
  | [272 : 400) | 64 bf16 RoPE values (dims 512:576), byte-identical to       |
  |             | fp8_ds_mla's [528:656) region (already-roped, copied raw)   |

Cache tensor shape: ``(num_blocks, block_size, 400)`` uint8
(vs fp8_ds_mla's ``(num_blocks, block_size, 656)``).

QUANTIZATION (store side), per 32-dim NoPE block:
  amax = max(|x|) over the block (fp32 math)
  E    = ceil(log2(amax / 6.0))       (6.0 = E2M1 max magnitude)
  byte = clamp(E + 127, 0, 255)       (UE8M0, biased; amax == 0 -> byte 0)
  each value quantized as E2M1 of (x / 2^E) via the hardware
  ``cvt.rn.satfinite.e2m1x2.f32`` instruction (satfinite clamps |x| > 6 to the
  +-6 code, so no explicit clamp is needed).  RoPE half stays bf16 unchanged.

DEQUANTIZATION (read side):
  val = e2m1_decode(nibble) * exp2(scale_byte - 127.0); RoPE read raw as bf16.
  E2M1 code: sign<<3 | exp<<1 | mant, magnitudes {0, 0.5, 1, 1.5, 2, 3, 4, 6}.
  This mirrors the fp8ds footer dequant in sparse_mla_kernels.py (already
  ``tl.exp2(encoded_scale - 127.0)``, there per-64 blocks / 8 footer bytes on
  the 448-dim DSV4 fp8_ds_mla variant); only the block width and the value
  load change (one fp8 byte -> a nibble + E2M1 decode).

PROVENANCE / CREDITS (keep with any derived code):
  * Quant scheme, store-kernel structure and the ``_fp32x2_to_fp4x2`` inline-
    asm E2M1 packer: danielwoz/vllm-dspark-nvfp4 (Apache-2.0),
    patches/03-vllm-nvfp4/NEW_nvfp4_store.py — itself building on the DSpark
    fork lineage and Keys' NVFP4 image (v0.21).  Differences vs that kernel:
    448 -> 512 NoPE dims, per-64 -> per-32 quant blocks, split-footer ->
    per-token contiguous scales, and NO in-kernel RoPE (GLM-5.2's vLLM path
    ropes k_pe *before* the cache write, so the rope half is copied raw).
  * fp8_ds_mla 656 B layout + Triton dequant-gather structure (per-token
    record walk, ``exp2(scale - 127)``, int64 offset promotion): jasl/vllm
    deepseek_v4 path via CosmicRaisins/glm-5.2-gb10 kernel overlays
    (sm12x_sparse_mla_attn.py::_gather_dequant_fp8ds_kernel and
    sparse_mla_kernels.py fp8ds accumulate kernels).
  * b12x CuTe e2m1 dequant helpers consulted for the decode LUT: b12x package
    (cute/fp4.py).

This module is deliberately dependency-free (pure torch + triton, no vLLM
imports) so it can be vendored into a vLLM image.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

__all__ = [
    "store_nvfp4_glm_kv",
    "gather_dequant_nvfp4_glm",
    "dequant_nvfp4_rows",
    "selftest",
]

# ---------------------------------------------------------------------------
# nvfp4_ds_mla 400-byte layout constants (GLM-5.2 main sparse-MLA KV).
# Mirrors the _FP8DS_* constant block in sm12x_sparse_mla_attn.py.
# ---------------------------------------------------------------------------
_NVFP4_NOPE_DIM = 512                                # NoPE latent dims (kv_lora_rank)
_NVFP4_ROPE_DIM = 64                                 # bf16 RoPE dims
_NVFP4_QUANT_BLOCK = 32                              # dims per UE8M0 scale
_NVFP4_N_SCALES = _NVFP4_NOPE_DIM // _NVFP4_QUANT_BLOCK      # 16
_NVFP4_NOPE_PACKED_BYTES = _NVFP4_NOPE_DIM // 2              # 256
_NVFP4_SCALE_BYTE_OFF = _NVFP4_NOPE_PACKED_BYTES             # 256
_NVFP4_ROPE_BYTE_OFF = _NVFP4_SCALE_BYTE_OFF + _NVFP4_N_SCALES  # 272
_NVFP4_RECORD_BYTES = _NVFP4_ROPE_BYTE_OFF + 2 * _NVFP4_ROPE_DIM  # 400
_DQK = _NVFP4_NOPE_DIM + _NVFP4_ROPE_DIM             # 576 (bf16 workspace row)
_E2M1_MAX = 6.0                                      # largest E2M1 magnitude


# ---------------------------------------------------------------------------
# E2M1 packer — copied verbatim from danielwoz/vllm-dspark-nvfp4
# patches/03-vllm-nvfp4/NEW_nvfp4_store.py (which copied it from
# image-src/vllm_deepseek_v4/common/ops/fused_indexer_q.py:30) so this module
# is self-contained. Packs two fp32 lanes into one uint8: low nibble = x_lo,
# high nibble = x_hi (E2M1: sign<<3 | exp<<1 | mant; magnitudes
# {0,0.5,1,1.5,2,3,4,6}). ``satfinite`` clamps |x|>6 to the +-6 code, so no
# explicit clamp is needed after dividing by the per-block scale.
# ---------------------------------------------------------------------------
@triton.jit
def _fp32x2_to_fp4x2(x_lo, x_hi):
    # NOTE: $1 is high nibble, $2 is low nibble
    return tl.inline_asm_elementwise(
        """
        {
            .reg .b8 tmp;
            cvt.rn.satfinite.e2m1x2.f32 tmp, $1, $2;
            cvt.u32.u8 $0, tmp;
        }
        """,
        constraints="=r,f,f",
        args=[x_hi, x_lo],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    ).to(tl.uint8)


# ---------------------------------------------------------------------------
# E2M1 decode (branchless). Desk-checked against the LUT:
#
#   nibble = sign<<3 | code,  code = exp<<1 | mant
#   magnitude formula:  exp == 0 -> 0.5 * mant
#                       exp >= 1 -> (1 + 0.5 * mant) * 2^(exp - 1)
#
#   code  exp  mant  formula                    LUT value
#    0     0    0    0.5 * 0            = 0.0     0.0   ok
#    1     0    1    0.5 * 1            = 0.5     0.5   ok
#    2     1    0    (1 + 0.0) * 2^0    = 1.0     1.0   ok
#    3     1    1    (1 + 0.5) * 2^0    = 1.5     1.5   ok
#    4     2    0    (1 + 0.0) * 2^1    = 2.0     2.0   ok
#    5     2    1    (1 + 0.5) * 2^1    = 3.0     3.0   ok
#    6     3    0    (1 + 0.0) * 2^2    = 4.0     4.0   ok
#    7     3    1    (1 + 0.5) * 2^2    = 6.0     6.0   ok
#
# ``code >= 2`` is exactly ``exp >= 1``; the exp2(exp - 1) lane is finite
# (0.5) even on the discarded exp == 0 lanes, so tl.where is safe.
# ---------------------------------------------------------------------------
@triton.jit
def _e2m1_decode(nibble):
    """Decode a tensor of 4-bit E2M1 codes (values 0..15) to fp32."""
    n = nibble.to(tl.int32)
    code = n & 0x7
    sign = (n >> 3) & 0x1
    mant = (code & 0x1).to(tl.float32)
    exp = (code >> 1).to(tl.float32)
    mag = tl.where(code >= 2, (1.0 + 0.5 * mant) * tl.exp2(exp - 1.0), 0.5 * mant)
    return tl.where(sign != 0, -mag, mag)


# ---------------------------------------------------------------------------
# Shared record dequant: given a [ROWS] vector of record byte offsets into the
# cache, return (nope fp32 [ROWS, 512], rope raw uint16 [ROWS, 64]).
# Used by both the sequential gather kernel and the indexed rows kernel so the
# layout walk exists exactly once.
# ---------------------------------------------------------------------------
@triton.jit
def _dequant_nvfp4_record_rows(
    cache_ptr,            # uint8 base pointer of the cache
    cache_u16_ptr,        # same pointer viewed as uint16 (for the rope half)
    rec_byte,             # [ROWS] int64 byte offset of each 400 B record
    row_mask,             # [ROWS] bool
    NOPE_PACKED_BYTES: tl.constexpr,   # 256
    PAIRS_PER_SCALE_BLOCK: tl.constexpr,  # 16 (= QUANT_BLOCK // 2)
    SCALE_BYTE_OFF: tl.constexpr,      # 256
    ROPE_BYTE_OFF: tl.constexpr,       # 272 (even: uint16 view is aligned)
    ROPE_DIM: tl.constexpr,            # 64
):
    rec2d = rec_byte[:, None]
    m2d = row_mask[:, None]

    # ---- NoPE: 256 packed bytes; byte b = (low: dim 2b, high: dim 2b+1) ----
    byte_idx = tl.arange(0, NOPE_PACKED_BYTES)
    packed = tl.load(cache_ptr + rec2d + byte_idx[None, :], mask=m2d, other=0)
    packed_i32 = packed.to(tl.int32)

    # Scale byte for byte b: dims (2b, 2b+1) live in 32-dim block (2b)//32 =
    # b // 16, so bytes [16j, 16j+16) all share scale byte j (redundant loads,
    # same pattern as the fp8ds ``offsets // quant_block`` scale walk).
    scale_idx = byte_idx // PAIRS_PER_SCALE_BLOCK
    scale_byte = tl.load(
        cache_ptr + rec2d + SCALE_BYTE_OFF + scale_idx[None, :],
        mask=m2d,
        other=127,
    )
    block_scale = tl.exp2(scale_byte.to(tl.float32) - 127.0)

    lo = _e2m1_decode(packed_i32 & 0xF) * block_scale
    hi = _e2m1_decode((packed_i32 >> 4) & 0xF) * block_scale
    # interleave along the last dim: out[:, 2b] = lo[:, b], out[:, 2b+1] =
    # hi[:, b] — exactly the layout's (low nibble, high nibble) dim pairing.
    nope = tl.interleave(lo, hi)  # [ROWS, 512] fp32

    # ---- RoPE: 64 bf16 copied raw as uint16 (bit-exact, no arithmetic) ----
    # rec_byte and ROPE_BYTE_OFF are both even, so the // 2 is exact and the
    # uint16 view stays aligned.
    rope_idx = tl.arange(0, ROPE_DIM)
    rope_u16 = tl.load(
        cache_u16_ptr + (rec2d + ROPE_BYTE_OFF) // 2 + rope_idx[None, :],
        mask=m2d,
        other=0,
    )
    return nope, rope_u16


# ---------------------------------------------------------------------------
# A) NVFP4 store: quantize + insert one token record per program.
# ---------------------------------------------------------------------------
@triton.jit
def _store_nvfp4_glm_kv_kernel(
    # ── inputs (both fully preprocessed by the caller) ──
    kv_c_ptr,             # [num_tokens, 512] bf16 NoPE latent, already normed
    kv_c_stride0,
    k_pe_ptr,             # [num_tokens, 64] bf16 RoPE half, ALREADY roped
    k_pe_stride0,
    # ── metadata ──
    slot_mapping_ptr,     # [num_tokens] int64, -1 = skip
    # ── KV cache output (uint8 base pointer) ──
    kv_cache_ptr,
    kv_cache_block_size,  # pbs: tokens per paged cache block
    # ── constexprs ──
    NOPE_DIM: tl.constexpr,           # 512
    ROPE_DIM: tl.constexpr,           # 64
    QUANT_BLOCK: tl.constexpr,        # 32
    N_SCALES: tl.constexpr,           # 16 (= NOPE_DIM // QUANT_BLOCK)
    NOPE_PACKED_BYTES: tl.constexpr,  # 256 (= NOPE_DIM // 2)
    SCALE_BYTE_OFF: tl.constexpr,     # 256
    ROPE_BYTE_OFF: tl.constexpr,      # 272
    RECORD_BYTES: tl.constexpr,       # 400
    KV_BLOCK_STRIDE: tl.constexpr,    # kv_cache.stride(0), bytes per paged block
    E2M1_MAX: tl.constexpr,           # 6.0
):
    """One program per token. Writes the 400-byte nvfp4_ds_mla record.

    Unlike the danielwoz reference kernel there is NO in-kernel RoPE: GLM's
    vLLM path applies rotary embedding to k_pe before the cache write, so the
    rope half is a raw bf16 (uint16) copy — the cos_sin_cache machinery is
    dropped entirely.
    """
    token_idx = tl.program_id(0).to(tl.int64)

    slot_idx = tl.load(slot_mapping_ptr + token_idx)  # int64
    if slot_idx < 0:
        return

    kv_block_idx = slot_idx // kv_cache_block_size
    kv_pos_in_block = slot_idx % kv_cache_block_size

    # int64: block_idx * block_stride can exceed 2^31 with many KV-cache
    # blocks (same promotion as the fp8ds gather kernels).
    rec_byte = (
        kv_block_idx.to(tl.int64) * KV_BLOCK_STRIDE
        + kv_pos_in_block.to(tl.int64) * RECORD_BYTES
    )
    rec_ptr = kv_cache_ptr + rec_byte

    # ── Load NoPE latent [512] (bf16 -> fp32 is exact; no extra roundtrip
    # needed since the input is already bf16) ─────────────────────────────
    dim = tl.arange(0, NOPE_DIM)
    x = tl.load(kv_c_ptr + token_idx * kv_c_stride0 + dim).to(tl.float32)

    # ── Even/odd split (dim 2p = even, dim 2p+1 = odd) ────────────────────
    NUM_PAIRS: tl.constexpr = NOPE_DIM // 2                    # 256
    pair_2d = tl.reshape(x, (NUM_PAIRS, 2))
    even, odd = tl.split(pair_2d)  # each [256] fp32

    # Tile into (N_SCALES, HALF_BLOCK): scale block j = pairs [16j..16j+15]
    # = dims [32j..32j+31], exactly one UE8M0 block.
    HALF_BLOCK: tl.constexpr = QUANT_BLOCK // 2                # 16
    even_2d = tl.reshape(even, (N_SCALES, HALF_BLOCK))
    odd_2d = tl.reshape(odd, (N_SCALES, HALF_BLOCK))

    amax = tl.maximum(
        tl.max(tl.abs(even_2d), axis=1),
        tl.max(tl.abs(odd_2d), axis=1),
    )  # [N_SCALES]

    # UE8M0 block scale: E = ceil(log2(amax / 6)); byte = E + 127.
    # amax == 0: log2 -> -inf, the clamp pins E to -127 -> byte 0 (the
    # LAYOUT-SPEC zero-block encoding), and inv_scale = 2^127 is finite fp32,
    # so 0 * inv_scale = 0 quantizes to the exact-zero code. No epsilon guard
    # needed (deviation from the danielwoz 6*2^-126 floor, which encodes zero
    # blocks as byte 1; byte 0 is what our dequant/spec expects).
    log2_ratio = tl.ceil(tl.log2(amax * (1.0 / E2M1_MAX)))
    log2_ratio = tl.minimum(tl.maximum(log2_ratio, -127.0), 127.0)
    inv_scale = tl.exp2(-log2_ratio)                          # 1 / 2^E
    ue8m0 = (log2_ratio + 127.0).to(tl.uint8)                 # [N_SCALES]

    inv_scale_col = tl.reshape(inv_scale, (N_SCALES, 1))
    # low nibble = even (dim 2b), high nibble = odd (dim 2b+1). ``satfinite``
    # clamps |val / 2^E| > 6 to the +-6 code — no explicit clamp needed.
    packed = _fp32x2_to_fp4x2(even_2d * inv_scale_col, odd_2d * inv_scale_col)
    packed_flat = tl.reshape(packed, (NOPE_PACKED_BYTES,))    # [256] uint8

    # ── Record writes: [0:256) nibbles, [256:272) scales, [272:400) rope ──
    byte_idx = tl.arange(0, NOPE_PACKED_BYTES)
    tl.store(rec_ptr + byte_idx, packed_flat)

    scale_idx = tl.arange(0, N_SCALES)
    tl.store(rec_ptr + SCALE_BYTE_OFF + scale_idx, ue8m0)

    # RoPE: raw uint16 copy (bit-exact; matches fp8_ds_mla bytes [528:656)).
    # rec_byte, ROPE_BYTE_OFF and the strides are all even, so the uint16
    # views stay aligned and the // 2 is exact.
    rope_idx = tl.arange(0, ROPE_DIM)
    k_pe_u16 = k_pe_ptr.to(tl.pointer_type(tl.uint16))
    rope_bits = tl.load(k_pe_u16 + token_idx * k_pe_stride0 + rope_idx)
    cache_u16 = kv_cache_ptr.to(tl.pointer_type(tl.uint16))
    tl.store(cache_u16 + (rec_byte + ROPE_BYTE_OFF) // 2 + rope_idx, rope_bits)


def store_nvfp4_glm_kv(
    kv_c_normed: torch.Tensor,   # [num_tokens, 512] bf16 (NoPE latent, normed)
    k_pe: torch.Tensor,          # [num_tokens, 64] bf16 (RoPE half, ALREADY roped)
    kv_cache: torch.Tensor,      # uint8 (num_blocks, block_size, 400) or flat
    slot_mapping: torch.Tensor,  # [num_tokens] int64, -1 = skip
) -> None:
    """Quantize the GLM-5.2 main-KV latent to NVFP4 and insert into the cache.

    Both halves arrive fully preprocessed (kv_c_normed is post-RMSNorm, k_pe
    is post-RoPE), so the kernel only quantizes/packs/copies. ``kv_cache``
    may be the canonical ``(num_blocks, block_size, 400)`` uint8 view, a flat
    ``(total_slots, 400)`` view, or a 1-D contiguous byte view; records must
    be contiguous 400-byte rows in all cases.
    """
    assert kv_c_normed.dim() == 2 and kv_c_normed.shape[1] == _NVFP4_NOPE_DIM, (
        f"kv_c_normed must be [num_tokens, {_NVFP4_NOPE_DIM}], "
        f"got {tuple(kv_c_normed.shape)}"
    )
    assert kv_c_normed.stride(1) == 1, "kv_c_normed rows must be contiguous"
    assert k_pe.dim() == 2 and k_pe.shape[1] == _NVFP4_ROPE_DIM, (
        f"k_pe must be [num_tokens, {_NVFP4_ROPE_DIM}], got {tuple(k_pe.shape)}"
    )
    assert k_pe.dtype == torch.bfloat16, "k_pe must be bf16 (raw-copied)"
    assert k_pe.stride(1) == 1, "k_pe rows must be contiguous"
    assert kv_cache.dtype == torch.uint8, "kv_cache must be uint8"
    assert slot_mapping.dtype == torch.int64

    # Normalize the cache view to (blocks, block_size) + strides.
    if kv_cache.dim() == 1:
        assert kv_cache.is_contiguous()
        assert kv_cache.numel() % _NVFP4_RECORD_BYTES == 0
        kv_cache = kv_cache.view(-1, _NVFP4_RECORD_BYTES)
    if kv_cache.dim() == 2:
        # Flat [total_slots, 400]: treat every slot as its own "block".
        assert kv_cache.shape[1] == _NVFP4_RECORD_BYTES
        assert kv_cache.stride(1) == 1
        block_size = 1
        block_stride = kv_cache.stride(0)
    elif kv_cache.dim() == 3:
        assert kv_cache.shape[2] == _NVFP4_RECORD_BYTES, (
            f"expected last dim {_NVFP4_RECORD_BYTES}, got {kv_cache.shape[2]}"
        )
        assert kv_cache.stride(2) == 1
        assert kv_cache.stride(1) == _NVFP4_RECORD_BYTES, (
            "records must be contiguous within a block"
        )
        block_size = kv_cache.shape[1]
        block_stride = kv_cache.stride(0)
    else:
        raise ValueError(f"unsupported kv_cache rank {kv_cache.dim()}")
    assert block_stride % 2 == 0, "block stride must be even (uint16 rope view)"
    assert kv_cache.data_ptr() % 2 == 0

    # DP padding: slot_mapping may be shorter than kv_c_normed.
    num_tokens = slot_mapping.shape[0]
    assert kv_c_normed.shape[0] >= num_tokens
    assert k_pe.shape[0] >= num_tokens
    if num_tokens == 0:
        return

    grid = (num_tokens,)
    _store_nvfp4_glm_kv_kernel[grid](
        kv_c_normed,
        kv_c_normed.stride(0),
        k_pe,
        k_pe.stride(0),
        slot_mapping,
        kv_cache,
        block_size,
        NOPE_DIM=_NVFP4_NOPE_DIM,
        ROPE_DIM=_NVFP4_ROPE_DIM,
        QUANT_BLOCK=_NVFP4_QUANT_BLOCK,
        N_SCALES=_NVFP4_N_SCALES,
        NOPE_PACKED_BYTES=_NVFP4_NOPE_PACKED_BYTES,
        SCALE_BYTE_OFF=_NVFP4_SCALE_BYTE_OFF,
        ROPE_BYTE_OFF=_NVFP4_ROPE_BYTE_OFF,
        RECORD_BYTES=_NVFP4_RECORD_BYTES,
        KV_BLOCK_STRIDE=block_stride,
        E2M1_MAX=_E2M1_MAX,
        num_warps=4,
    )


# ---------------------------------------------------------------------------
# B) Sequential gather + dequant into the bf16 prefill workspace.
#    Replaces ops.cp_gather_and_upconvert_fp8_kv_cache for nvfp4_ds_mla.
# ---------------------------------------------------------------------------
@triton.jit
def _gather_dequant_nvfp4_glm_kernel(
    kv_cache_ptr,          # uint8 (num_blocks, block_size, 400)
    workspace_ptr,         # bf16 [workspace_len, 576]
    block_table_ptr,       # int32 [num_reqs, max_blocks]
    seq_lens_ptr,          # int32 [num_reqs]
    workspace_starts_ptr,  # int32 [num_reqs]
    bt_stride0,            # block_table.stride(0), elements
    ws_stride0,            # workspace.stride(0), elements
    NOPE_DIM: tl.constexpr,               # 512
    ROPE_DIM: tl.constexpr,               # 64
    NOPE_PACKED_BYTES: tl.constexpr,      # 256
    PAIRS_PER_SCALE_BLOCK: tl.constexpr,  # 16
    SCALE_BYTE_OFF: tl.constexpr,         # 256
    ROPE_BYTE_OFF: tl.constexpr,          # 272
    RECORD_BYTES: tl.constexpr,           # 400
    KV_BLOCK_STRIDE: tl.constexpr,        # kv_cache.stride(0), bytes
    BLOCK_SIZE: tl.constexpr,             # paged-cache tokens per block (64)
    TOKEN_TILE: tl.constexpr,             # tokens dequantized per program
):
    """Grid (token-tile, request) — tile on axis 0 because the tile count can
    exceed the 65535 CUDA grid-Y limit at long context (e.g. 328K tokens).
    Program p dequantizes tokens [p*TOKEN_TILE, (p+1)*TOKEN_TILE) of its
    request into workspace rows workspace_starts[r] + t; tails are masked.
    """
    tile = tl.program_id(0)
    req = tl.program_id(1)

    seq_len = tl.load(seq_lens_ptr + req)
    tile_start = tile * TOKEN_TILE
    if tile_start >= seq_len:
        return

    ws_start = tl.load(workspace_starts_ptr + req).to(tl.int64)

    t = tile_start + tl.arange(0, TOKEN_TILE)   # token positions in request
    t_mask = t < seq_len

    # Locate each token's record: block_table[r][t // BLOCK_SIZE], slot in
    # block t % BLOCK_SIZE.
    blk_col = t // BLOCK_SIZE
    pos_in_block = t % BLOCK_SIZE
    blk_id = tl.load(
        block_table_ptr + req * bt_stride0 + blk_col, mask=t_mask, other=0
    )
    # int64 promotion: blk_id * KV_BLOCK_STRIDE overflows int32 once the pool
    # crosses ~2^31 bytes (same fix as the fp8ds gather kernel).
    rec_byte = (
        blk_id.to(tl.int64) * KV_BLOCK_STRIDE
        + pos_in_block.to(tl.int64) * RECORD_BYTES
    )

    cache_u16 = kv_cache_ptr.to(tl.pointer_type(tl.uint16))
    nope, rope_bits = _dequant_nvfp4_record_rows(
        kv_cache_ptr,
        cache_u16,
        rec_byte,
        t_mask,
        NOPE_PACKED_BYTES=NOPE_PACKED_BYTES,
        PAIRS_PER_SCALE_BLOCK=PAIRS_PER_SCALE_BLOCK,
        SCALE_BYTE_OFF=SCALE_BYTE_OFF,
        ROPE_BYTE_OFF=ROPE_BYTE_OFF,
        ROPE_DIM=ROPE_DIM,
    )

    # Workspace row for token t of request r: workspace_starts[r] + t.
    ws_row = ws_start + t.to(tl.int64)          # [TOKEN_TILE] int64
    dim = tl.arange(0, NOPE_DIM)
    tl.store(
        workspace_ptr + ws_row[:, None] * ws_stride0 + dim[None, :],
        nope.to(workspace_ptr.dtype.element_ty),
        mask=t_mask[:, None],
    )
    # RoPE half: raw uint16 store into dims [512:576) (bit-exact copy).
    ws_u16 = workspace_ptr.to(tl.pointer_type(tl.uint16))
    rope_idx = tl.arange(0, ROPE_DIM)
    tl.store(
        ws_u16 + ws_row[:, None] * ws_stride0 + NOPE_DIM + rope_idx[None, :],
        rope_bits,
        mask=t_mask[:, None],
    )


def gather_dequant_nvfp4_glm(
    kv_cache: torch.Tensor,          # uint8 (num_blocks, block_size, 400)
    workspace: torch.Tensor,         # bf16 [workspace_len, 576] output
    block_table: torch.Tensor,       # int32 [num_reqs, max_blocks]
    seq_lens: torch.Tensor,          # int32 [num_reqs]
    workspace_starts: torch.Tensor,  # int32 [num_reqs]
    num_reqs: int,
    *,
    max_seq_len: int | None = None,
    token_tile: int = 8,
) -> None:
    """Gather + upconvert nvfp4_ds_mla cache rows into the bf16 workspace.

    For each request r and token position t < seq_lens[r], dequantizes the
    record at block_table[r][t // block_size] / slot t % block_size and writes
    the 576-dim bf16 row to workspace[workspace_starts[r] + t]. Drop-in for
    the fp8 path's ``ops.cp_gather_and_upconvert_fp8_kv_cache`` call.

    ``max_seq_len`` (optional) tightens the launch grid; when omitted the
    upper bound ``block_table.shape[1] * block_size`` is used so no host<->
    device sync is needed (out-of-range tiles exit immediately).
    """
    assert kv_cache.dtype == torch.uint8
    assert kv_cache.dim() == 3 and kv_cache.shape[2] == _NVFP4_RECORD_BYTES, (
        f"expected kv_cache (num_blocks, block_size, {_NVFP4_RECORD_BYTES}), "
        f"got {tuple(kv_cache.shape)}"
    )
    assert kv_cache.stride(2) == 1
    assert kv_cache.stride(1) == _NVFP4_RECORD_BYTES
    assert kv_cache.stride(0) % 2 == 0
    assert kv_cache.data_ptr() % 2 == 0
    assert workspace.dtype == torch.bfloat16
    assert workspace.dim() == 2 and workspace.shape[1] == _DQK, (
        f"expected workspace [n, {_DQK}], got {tuple(workspace.shape)}"
    )
    assert workspace.stride(1) == 1
    assert block_table.dtype == torch.int32 and block_table.dim() == 2
    assert seq_lens.dtype == torch.int32
    assert workspace_starts.dtype == torch.int32
    assert block_table.stride(1) == 1
    assert num_reqs <= block_table.shape[0]
    assert num_reqs <= seq_lens.shape[0]
    assert num_reqs <= workspace_starts.shape[0]
    assert num_reqs <= 65535, "num_reqs exceeds the CUDA grid-Y limit"

    if num_reqs == 0:
        return

    block_size = kv_cache.shape[1]
    seq_len_bound = (
        max_seq_len if max_seq_len is not None
        else block_table.shape[1] * block_size
    )
    if seq_len_bound <= 0:
        return

    grid = (triton.cdiv(seq_len_bound, token_tile), num_reqs)
    _gather_dequant_nvfp4_glm_kernel[grid](
        kv_cache,
        workspace,
        block_table,
        seq_lens,
        workspace_starts,
        block_table.stride(0),
        workspace.stride(0),
        NOPE_DIM=_NVFP4_NOPE_DIM,
        ROPE_DIM=_NVFP4_ROPE_DIM,
        NOPE_PACKED_BYTES=_NVFP4_NOPE_PACKED_BYTES,
        PAIRS_PER_SCALE_BLOCK=_NVFP4_QUANT_BLOCK // 2,
        SCALE_BYTE_OFF=_NVFP4_SCALE_BYTE_OFF,
        ROPE_BYTE_OFF=_NVFP4_ROPE_BYTE_OFF,
        RECORD_BYTES=_NVFP4_RECORD_BYTES,
        KV_BLOCK_STRIDE=kv_cache.stride(0),
        BLOCK_SIZE=block_size,
        TOKEN_TILE=token_tile,
        num_warps=4,
    )


# ---------------------------------------------------------------------------
# C) Indexed rows dequant (parity testing / fallback path).
# ---------------------------------------------------------------------------
@triton.jit
def _dequant_nvfp4_rows_kernel(
    cache_ptr,            # uint8 [total_slots, 400] (row-strided)
    indices_ptr,          # int64 [num_rows] slot ids
    out_ptr,              # bf16 [num_rows, 576]
    num_rows,
    cache_stride0,        # bytes per cache row
    out_stride0,          # elements per output row
    NOPE_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    NOPE_PACKED_BYTES: tl.constexpr,
    PAIRS_PER_SCALE_BLOCK: tl.constexpr,
    SCALE_BYTE_OFF: tl.constexpr,
    ROPE_BYTE_OFF: tl.constexpr,
    ROW_TILE: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * ROW_TILE + tl.arange(0, ROW_TILE)
    row_mask = rows < num_rows

    slot = tl.load(indices_ptr + rows, mask=row_mask, other=0)  # int64
    # slot is int64, so the product promotes to int64 (no 2^31 overflow).
    rec_byte = slot * cache_stride0

    cache_u16 = cache_ptr.to(tl.pointer_type(tl.uint16))
    nope, rope_bits = _dequant_nvfp4_record_rows(
        cache_ptr,
        cache_u16,
        rec_byte,
        row_mask,
        NOPE_PACKED_BYTES=NOPE_PACKED_BYTES,
        PAIRS_PER_SCALE_BLOCK=PAIRS_PER_SCALE_BLOCK,
        SCALE_BYTE_OFF=SCALE_BYTE_OFF,
        ROPE_BYTE_OFF=ROPE_BYTE_OFF,
        ROPE_DIM=ROPE_DIM,
    )

    out_row = rows.to(tl.int64)
    dim = tl.arange(0, NOPE_DIM)
    tl.store(
        out_ptr + out_row[:, None] * out_stride0 + dim[None, :],
        nope.to(out_ptr.dtype.element_ty),
        mask=row_mask[:, None],
    )
    out_u16 = out_ptr.to(tl.pointer_type(tl.uint16))
    rope_idx = tl.arange(0, ROPE_DIM)
    tl.store(
        out_u16 + out_row[:, None] * out_stride0 + NOPE_DIM + rope_idx[None, :],
        rope_bits,
        mask=row_mask[:, None],
    )


def dequant_nvfp4_rows(
    kv_cache: torch.Tensor,  # uint8 [total_slots, 400] flat (or 3D contiguous)
    indices: torch.Tensor,   # [n] int64 slot ids (must be in range)
) -> torch.Tensor:
    """Dequantize the nvfp4_ds_mla records selected by ``indices``.

    Returns bf16 [n, 576] (512 dequantized NoPE dims + 64 raw-copied RoPE
    dims). Used for parity testing and as a slow fallback; unlike the fp8ds
    gather this does NOT accept -1 sentinels — indices must be valid slots.
    """
    assert kv_cache.dtype == torch.uint8
    if kv_cache.dim() == 3:
        assert kv_cache.shape[2] == _NVFP4_RECORD_BYTES
        assert kv_cache.stride(2) == 1
        assert kv_cache.stride(1) == _NVFP4_RECORD_BYTES
        assert kv_cache.stride(0) == kv_cache.shape[1] * _NVFP4_RECORD_BYTES
        kv_cache = kv_cache.view(-1, _NVFP4_RECORD_BYTES)
    assert kv_cache.dim() == 2 and kv_cache.shape[1] == _NVFP4_RECORD_BYTES
    assert kv_cache.stride(1) == 1
    assert kv_cache.stride(0) % 2 == 0
    assert kv_cache.data_ptr() % 2 == 0
    assert indices.dim() == 1
    indices = indices.to(dtype=torch.int64, device=kv_cache.device)

    num_rows = indices.shape[0]
    out = torch.empty(
        (num_rows, _DQK), dtype=torch.bfloat16, device=kv_cache.device
    )
    if num_rows == 0:
        return out

    ROW_TILE = 8
    grid = (triton.cdiv(num_rows, ROW_TILE),)
    _dequant_nvfp4_rows_kernel[grid](
        kv_cache,
        indices,
        out,
        num_rows,
        kv_cache.stride(0),
        out.stride(0),
        NOPE_DIM=_NVFP4_NOPE_DIM,
        ROPE_DIM=_NVFP4_ROPE_DIM,
        NOPE_PACKED_BYTES=_NVFP4_NOPE_PACKED_BYTES,
        PAIRS_PER_SCALE_BLOCK=_NVFP4_QUANT_BLOCK // 2,
        SCALE_BYTE_OFF=_NVFP4_SCALE_BYTE_OFF,
        ROPE_BYTE_OFF=_NVFP4_ROPE_BYTE_OFF,
        ROW_TILE=ROW_TILE,
        num_warps=4,
    )
    return out


# ---------------------------------------------------------------------------
# D) GPU selftest: store -> dequant roundtrip + sentinel / parity checks.
# ---------------------------------------------------------------------------
def selftest() -> None:  # noqa: C901 (deliberately linear test script)
    """Roundtrip / invariance checks on GPU. Prints PASS/FAIL per check.

    Note on check (2): the strict E2M1 round-to-nearest bound is
    |err| <= 1.0 * 2^E_block (half the widest code gap, 4 -> 6), NOT
    0.5 * 2^E — 0.5 only bounds scaled values with |x/2^E| <= 4. The assert
    below uses the true 1.0 * 2^E bound (plus tiny fp slop).
    """
    if not torch.cuda.is_available():
        print("[selftest] SKIP: no CUDA device")
        return

    device = "cuda"
    torch.manual_seed(0)
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        status = "PASS" if ok else "FAIL"
        suffix = f" ({detail})" if detail else ""
        print(f"[selftest] {name}: {status}{suffix}")
        if not ok:
            failures.append(name)

    num_tokens, block_size, num_blocks = 128, 64, 3
    total_slots = num_blocks * block_size
    sentinel = 0xAB

    # Latents with per-32-block magnitude spread to exercise the scale range.
    kv_c32 = torch.randn(num_tokens, _NVFP4_NOPE_DIM, device=device)
    block_mag = 2.0 ** torch.randint(
        -8, 9, (_NVFP4_N_SCALES,), device=device
    ).to(torch.float32)
    kv_c32 *= block_mag.repeat_interleave(_NVFP4_QUANT_BLOCK)[None, :]
    kv_c32[0, 0:32] = 0.0          # single zero block
    kv_c32[5, :] = 0.0             # fully zero token
    kv_c = kv_c32.to(torch.bfloat16)
    k_pe = torch.randn(num_tokens, _NVFP4_ROPE_DIM, device=device).to(
        torch.bfloat16
    )

    kv_cache = torch.full(
        (num_blocks, block_size, _NVFP4_RECORD_BYTES),
        sentinel,
        dtype=torch.uint8,
        device=device,
    )
    # tokens 0..99 -> slots 0..99 (blocks 0/1), tokens 100..119 -> slots
    # 128..147 (block 2), tokens 120..127 -> -1 (skipped).
    slot_mapping = torch.full((num_tokens,), -1, dtype=torch.int64, device=device)
    slot_mapping[:100] = torch.arange(100, device=device)
    slot_mapping[100:120] = torch.arange(128, 148, device=device)

    store_nvfp4_glm_kv(kv_c, k_pe, kv_cache, slot_mapping)

    flat = kv_cache.view(total_slots, _NVFP4_RECORD_BYTES)
    mapped = slot_mapping[:120]
    out = dequant_nvfp4_rows(flat, mapped)  # [120, 576] bf16

    # (1) RoPE half is bit-identical.
    rope_bits_out = out[:, _NVFP4_NOPE_DIM:].contiguous().view(torch.int16)
    rope_bits_ref = k_pe[:120].contiguous().view(torch.int16)
    check("rope bit-identical", torch.equal(rope_bits_out, rope_bits_ref))

    # (2) NoPE error bounds. Use the *stored* scale bytes so a boundary-case
    # ulp difference between host and device log2 cannot skew the bound.
    x_ref = kv_c[:120].to(torch.float32)
    err = out[:, : _NVFP4_NOPE_DIM].to(torch.float32) - x_ref
    scale_bytes = flat[mapped, _NVFP4_SCALE_BYTE_OFF:_NVFP4_ROPE_BYTE_OFF]
    e_block = scale_bytes.to(torch.int32) - 127                      # [120, 16]
    bound = torch.ldexp(
        torch.ones_like(e_block, dtype=torch.float32), e_block
    ).repeat_interleave(_NVFP4_QUANT_BLOCK, dim=1)                   # [120, 512]
    max_ratio = (err.abs() / bound).max().item()
    check(
        "nope max-abs within 1.0 * 2^E_block",
        bool((err.abs() <= bound * (1.0 + 1e-5) + 1e-30).all()),
        f"max |err| / 2^E = {max_ratio:.3f}",
    )
    rel_rms = (err.norm() / x_ref.norm()).item()
    check("nope relative RMS < 0.15", rel_rms < 0.15, f"rel RMS = {rel_rms:.4f}")

    # (3) Zero blocks roundtrip to exact zeros (and encode scale byte 0).
    zeros_ok = (
        bool((out[0, 0:32] == 0).all())
        and bool((out[5, : _NVFP4_NOPE_DIM] == 0).all())
        and int(flat[mapped[0], _NVFP4_SCALE_BYTE_OFF].item()) == 0
        and bool(
            (flat[mapped[5], _NVFP4_SCALE_BYTE_OFF:_NVFP4_ROPE_BYTE_OFF] == 0).all()
        )
    )
    check("zero-block exact roundtrip (scale byte 0)", zeros_ok)

    # (4a) Slots never mapped keep their sentinel bytes.
    used = torch.zeros(total_slots, dtype=torch.bool, device=device)
    used[mapped] = True
    check("unmapped slots untouched", bool((flat[~used] == sentinel).all()))

    # (4b) An all -1 slot_mapping writes nothing at all.
    cache2 = torch.full(
        (1, block_size, _NVFP4_RECORD_BYTES),
        sentinel,
        dtype=torch.uint8,
        device=device,
    )
    all_skip = torch.full((8,), -1, dtype=torch.int64, device=device)
    store_nvfp4_glm_kv(kv_c[:8], k_pe[:8], cache2, all_skip)
    check("all -1 slot_mapping leaves cache untouched",
          bool((cache2 == sentinel).all()))

    # (5) Gather kernel parity vs the rows path (exercises partial tiles:
    # 100 and 20 are not multiples of the token tile or the block size).
    block_table = torch.tensor([[0, 1], [2, 0]], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([100, 20], dtype=torch.int32, device=device)
    workspace_starts = torch.tensor([0, 100], dtype=torch.int32, device=device)
    workspace = torch.full(
        (120, _DQK), 777.0, dtype=torch.bfloat16, device=device
    )
    gather_dequant_nvfp4_glm(
        kv_cache, workspace, block_table, seq_lens, workspace_starts, 2
    )
    check(
        "gather vs rows parity (bit-exact)",
        torch.equal(
            workspace.contiguous().view(torch.int16),
            out.contiguous().view(torch.int16),
        ),
    )

    if failures:
        print(f"[selftest] OVERALL: FAIL ({len(failures)} failed: {failures})")
        raise SystemExit(1)
    print("[selftest] OVERALL: PASS")


if __name__ == "__main__":
    selftest()
