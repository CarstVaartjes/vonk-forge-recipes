"""NVFP4 (E2M1 + UE8M0) KV-record support for the SM120 sparse-MLA decode.

Implements the read side of the GLM-5.2 ``nvfp4_ds_mla`` 400-byte KV record
(LAYOUT-SPEC.md — SINGLE SOURCE OF TRUTH; danielwoz/vllm-dspark-nvfp4 lineage):

    [0:256)   512 E2M1 nibbles of the NoPE latent, packed 2/byte
              (byte b = LOW nibble dim 2b, HIGH nibble dim 2b+1)
    [256:272) 16 UE8M0 scale bytes, one per 32-dim block
              (block i covers dims [32i, 32i+32); val = e2m1 * 2^(byte-127))
    [272:400) 64 bf16 RoPE values (byte-identical to fp8_ds_mla's [528:656))

The decode kernel's IO warp gathers the packed 272B/token payload+scale rows
DENSELY into the front of the (unchanged, 64x528B) kv_fp8 smem buffer, and
``s0k_nvfp4_expand_kv_smem`` below expands them IN PLACE to the standard GLM
fp8 compute rows (512 e4m3 bytes + 4 inline fp32 tile scales at [512:528)) so
S1..S7 in decode_math run BYTE-IDENTICAL to the fp8_ds_mla path. RoPE needs no
stage: its bytes are format-identical and land in the kv_rope buffer as usual.

Per-32-block scale realisation (tile-max shift): each 128-dim tile carries 4
UE8M0 block exponents e0..e3. We emit
    fp32 tile scale = 2^(max(e)-127)
    e4m3 payload    = e2m1(nibble) * 2^(e_i - max(e))
so payload*scale == e2m1 * 2^(e_i-127), the exact LAYOUT-SPEC dequant. The
shift is EXACT in e4m3 for blocks within 2^8 of the tile max; a block sitting
further below the tile max has its smallest magnitudes RN-rounded/flushed by
the e4m3 re-encode (error bounded by ~tile_amax/512 per element). This is the
closest correct alternative to native per-32 scaling given decode_math's
per-128 ARBITRARY_FP32 contract — flagged in the port summary.

Provenance / credits:
- E2M1/UE8M0 register helpers reused from b12x upstream ``b12x/cute/fp4.py``
  (FlashInfer team, Apache-2.0): ``e2m1x8_mul_residual_to_e4m3x8`` consumes
  nibble i -> byte i (low nibble first), exactly the LAYOUT-SPEC packing.
- Quant scheme + E2M1 packer: danielwoz/vllm-dspark-nvfp4 (Apache-2.0).
- Record layout: LAYOUT-SPEC.md (nvfp4_ds_mla, 400 B/token).
"""

from __future__ import annotations

from typing import Tuple

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Uint32
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm

from b12x.cute.fp4 import (
    e2m1x8_mul_residual_to_e4m3x8,
    ld_shared_i32,
    ld_shared_v4_u32,
    st_shared_v4_u32,
)

# nvfp4_ds_mla record geometry (LAYOUT-SPEC.md). Kept here as the module's
# authoritative constants; traits.py mirrors them in the GLM nvfp4 bundle.
NVFP4_RECORD_BYTES = 400  # 256 nibbles + 16 UE8M0 + 128 bf16 rope
NVFP4_PACK_BYTES = 272  # payload+scales staged per token (256 + 16)
NVFP4_SCALE_OFF = 256  # UE8M0 scale bytes offset within the record
NVFP4_ROPE_OFF = 272  # bf16 rope offset within the record (16B aligned)


@dsl_user_op
def st_shared_f32(smem_addr: Int32, value: Float32, *, loc=None, ip=None):
    """Store 32-bit float to shared memory. smem_addr is a u32 shared address."""
    llvm.inline_asm(
        None,
        [
            Int32(smem_addr).ir_value(loc=loc, ip=ip),
            Float32(value).ir_value(loc=loc, ip=ip),
        ],
        "st.shared.f32 [$0], $1;",
        "r,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def nvfp4_tile_scale_residuals(
    sc_u32: Uint32, *, loc=None, ip=None
) -> Tuple[Float32, Uint32, Uint32, Uint32, Uint32]:
    """Decode one 128-dim tile's 4 UE8M0 block-scale bytes (packed u32).

    Returns ``(tile_scale_f32, r0_h2, r1_h2, r2_h2, r3_h2)`` where
    ``tile_scale = 2^(max(e)-127)`` (0.0 when all four bytes are 0, matching
    the b12x ``cvt_e8m0_to_f32`` byte-0 convention) and ``r_i`` is the f16x2
    broadcast residual ``2^(e_i - max(e))`` (0.0 for byte-0 blocks, whose
    payload nibbles are all zero anyway). ``residual * tile_scale ==
    2^(e_i-127)`` exactly — the LAYOUT-SPEC dequant scale. ``ex2.approx.f32``
    is exact for the integer exponents used here (same reliance as the
    existing ``cvt_e8m0_to_f32``/``cvt_e8m0x4_to_f32x4`` upstream helpers).
    Byte order: LOW byte of ``sc_u32`` = the tile's first 32-dim block.
    """
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.i32(), T.i32(), T.i32(), T.i32()]),
        [Uint32(sc_u32).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .pred pz, p0, p1, p2, p3;
            .reg .u32 b0, b1, b2, b3, m01, m23, mm;
            .reg .s32 e, em;
            .reg .f32 f, s, g;
            .reg .b16 h;

            and.b32 b0, $5, 0x000000ff;
            bfe.u32 b1, $5, 8, 8;
            bfe.u32 b2, $5, 16, 8;
            shr.u32 b3, $5, 24;

            max.u32 m01, b0, b1;
            max.u32 m23, b2, b3;
            max.u32 mm, m01, m23;

            // tile scale = 2^(mm-127); byte 0 -> 0.0 (all-empty tile).
            setp.eq.u32 pz, mm, 0;
            cvt.s32.u32 em, mm;
            sub.s32 e, em, 127;
            cvt.rn.f32.s32 f, e;
            ex2.approx.f32 s, f;
            selp.f32 s, 0f00000000, s, pz;
            mov.f32 $0, s;

            // residual_i = 2^(b_i-mm) in [2^-255, 1]; f16 keeps it exactly
            // down to 2^-24 (subnormal), flushes below (negligible: those
            // values are < tile_amax * 2^-24). Byte-0 blocks -> residual 0.
            setp.eq.u32 p0, b0, 0;
            cvt.s32.u32 e, b0;
            sub.s32 e, e, em;
            cvt.rn.f32.s32 f, e;
            ex2.approx.f32 g, f;
            selp.f32 g, 0f00000000, g, p0;
            cvt.rn.f16.f32 h, g;
            mov.b32 $1, {h, h};

            setp.eq.u32 p1, b1, 0;
            cvt.s32.u32 e, b1;
            sub.s32 e, e, em;
            cvt.rn.f32.s32 f, e;
            ex2.approx.f32 g, f;
            selp.f32 g, 0f00000000, g, p1;
            cvt.rn.f16.f32 h, g;
            mov.b32 $2, {h, h};

            setp.eq.u32 p2, b2, 0;
            cvt.s32.u32 e, b2;
            sub.s32 e, e, em;
            cvt.rn.f32.s32 f, e;
            ex2.approx.f32 g, f;
            selp.f32 g, 0f00000000, g, p2;
            cvt.rn.f16.f32 h, g;
            mov.b32 $3, {h, h};

            setp.eq.u32 p3, b3, 0;
            cvt.s32.u32 e, b3;
            sub.s32 e, e, em;
            cvt.rn.f32.s32 f, e;
            ex2.approx.f32 g, f;
            selp.f32 g, 0f00000000, g, p3;
            cvt.rn.f16.f32 h, g;
            mov.b32 $4, {h, h};
        }
        """,
        "=f,=r,=r,=r,=r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), result, [2], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), result, [3], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), result, [4], loc=loc, ip=ip)),
    )


@cute.jit
def s0k_nvfp4_expand_kv_smem(
    kv_fp8_b: Int32,
    tid: Int32,
    *,
    bi: cutlass.Constexpr,
    pack_stride: cutlass.Constexpr,
    kv_smem_stride: cutlass.Constexpr,
    d_nope: cutlass.Constexpr,
    quant_tile: cutlass.Constexpr,
    num_scales: cutlass.Constexpr,
    num_threads: cutlass.Constexpr,
    barrier_id: cutlass.Constexpr,
):
    """Expand the packed nvfp4 staging rows into standard GLM fp8 rows IN PLACE.

    Input (written by the IO warp, DENSE at ``pack_stride``=272 from
    ``kv_fp8_b``): token r's payload at [272r : 272r+256) = packed E2M1
    nibbles, [272r+256 : 272r+272) = 16 UE8M0 block scales.

    Output (standard GLM fp8 compute rows at ``kv_smem_stride``=528): token
    r's e4m3 payload at [528r : 528r+512), 4 fp32 tile scales at
    [528r+512 : 528r+528). Total staging (64*272 = 17,408 B) fits inside the
    UNCHANGED 64*528 = 33,792 B kv_fp8 buffer -> zero extra smem.

    In-place safety: the output region overlaps the staging region, so phase 1
    pulls EVERY thread's source bytes into registers (17 u32/thread) and a
    CTA-math barrier separates it from the phase-2 writes. Runs on the 256
    math threads only (one thread per (token, 128-dim tile): 64*4 == 256);
    called between the chunk's mbarrier wait and S1.
    """
    # Mechanical GLM-only port (LAYOUT-SPEC.md): trace-time geometry checks.
    assert bi * num_scales == num_threads, "expansion needs 1 thread per tile"
    assert quant_tile == 128 and num_scales == 4 and d_nope == 512
    assert pack_stride == NVFP4_PACK_BYTES and kv_smem_stride == 528

    row = tid >> Int32(2)  # token 0..63 (num_scales == 4)
    ti = tid & Int32(3)  # 128-dim tile 0..3
    pack_row = kv_fp8_b + row * Int32(pack_stride)
    # 64 packed bytes per tile (128 dims / 2); 16B-aligned (272r + 64ti).
    pack_tile = pack_row + ti * Int32(quant_tile // 2)
    sc_addr = pack_row + Int32(d_nope // 2) + ti * Int32(4)

    # ── Phase 1: staging bytes -> registers (16 packed u32 + 1 scale u32). ──
    w = []
    for k in cutlass.range_constexpr(4):
        w0, w1, w2, w3 = ld_shared_v4_u32(pack_tile + Int32(16 * k))
        w += [w0, w1, w2, w3]
    sc_u32 = Uint32(ld_shared_i32(sc_addr))

    cute.arch.barrier(barrier_id=barrier_id, number_of_threads=num_threads)

    # ── Phase 2: decode + write the standard 528B GLM fp8 row. ──
    tile_scale, r0, r1, r2, r3 = nvfp4_tile_scale_residuals(sc_u32)
    res = [r0, r1, r2, r3]
    out_row = kv_fp8_b + row * Int32(kv_smem_stride)
    out_tile = out_row + ti * Int32(quant_tile)
    for blk in cutlass.range_constexpr(4):
        # Packed word w[4*blk+j] = dims [32blk+8j : 32blk+8j+8) of the tile;
        # e2m1x8_mul_residual_to_e4m3x8 keeps nibble i -> byte i (low nibble
        # = even dim), so stores stay dim-contiguous.
        l0, h0 = e2m1x8_mul_residual_to_e4m3x8(w[blk * 4 + 0], res[blk])
        l1, h1 = e2m1x8_mul_residual_to_e4m3x8(w[blk * 4 + 1], res[blk])
        l2, h2 = e2m1x8_mul_residual_to_e4m3x8(w[blk * 4 + 2], res[blk])
        l3, h3 = e2m1x8_mul_residual_to_e4m3x8(w[blk * 4 + 3], res[blk])
        st_shared_v4_u32(out_tile + Int32(32 * blk), l0, h0, l1, h1)
        st_shared_v4_u32(out_tile + Int32(32 * blk + 16), l2, h2, l3, h3)
    # Inline ARBITRARY_FP32 tile scale at [512 + 4*ti) of the row — the exact
    # slot the fp8 bulk copy would have populated from record bytes [512:528).
    st_shared_f32(out_row + Int32(d_nope) + ti * Int32(4), tile_scale)

    cute.arch.barrier(barrier_id=barrier_id, number_of_threads=num_threads)


def dequant_nvfp4_ds_mla_record_torch(records):
    """Pure-Torch oracle for the nvfp4_ds_mla record (validation only).

    ``records``: uint8 tensor [..., 400]. Returns ``(nope_f32 [..., 512],
    rope_bf16 [..., 64])`` using the LAYOUT-SPEC dequant
    ``val = e2m1(nibble) * 2^(scale_byte - 127)`` (byte 0 -> scale 0.0,
    matching the kernel/UE8M0 helper convention; the payload of such a block
    is all-zero nibbles, so the results agree with the spec either way).
    """
    import torch

    if records.dtype != torch.uint8 or records.shape[-1] != NVFP4_RECORD_BYTES:
        raise ValueError("expected uint8 [..., 400] nvfp4_ds_mla records")
    lead = records.shape[:-1]
    packed = records[..., :NVFP4_SCALE_OFF].to(torch.int64)
    nibbles = torch.stack(((packed & 0xF), (packed >> 4) & 0xF), dim=-1)
    nibbles = nibbles.reshape(*lead, 512)  # byte b -> dims (2b, 2b+1)
    mag_lut = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32,
        device=records.device,
    )
    mags = mag_lut[nibbles & 0x7]
    signs = 1.0 - 2.0 * ((nibbles >> 3) & 0x1).to(torch.float32)
    sc = records[..., NVFP4_SCALE_OFF:NVFP4_ROPE_OFF].to(torch.float32)
    scales = torch.where(sc == 0, torch.zeros_like(sc), torch.exp2(sc - 127.0))
    nope = (signs * mags).reshape(*lead, 16, 32) * scales[..., None]
    rope = (
        records[..., NVFP4_ROPE_OFF:NVFP4_RECORD_BYTES]
        .contiguous()
        .view(torch.bfloat16)
    )
    return nope.reshape(*lead, 512), rope
