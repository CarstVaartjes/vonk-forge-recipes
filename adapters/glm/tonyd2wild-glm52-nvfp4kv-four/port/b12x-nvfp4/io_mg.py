"""DSV4 MG prefill IO helper.

FlashInfer DSV4 prefill bulk-stages only the 448-byte NoPE FP8 payload into
shared memory. The 64-dim RoPE component is consumed from global/L2 by the math
warps, while the 8-byte UE8M0 footer is scalar-gathered into a contiguous smem
scale buffer.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint32

from b12x.cute.fp4 import (
    cp_async_bulk_g2s_mbar,
    get_ptr_as_int64,
    ld_global_nc_v2_u32,
    shared_ptr_to_u32,
    st_shared_u32,
)


_DSV4_IO_STRIDE = 576
_DSV4_NOPE_BYTES = 448
_DSV4_FOOTER_BYTES = 8
_IO_THREADS = 128

# GLM (ARBITRARY_FP32) KV gmem layout: per-token 656B contiguous record
# (512 e4m3 nope + 16 inline fp32 scales + 128 bf16 rope). NO grouped footer --
# the inline fp32 scales travel WITH the nope (bulk #1, 528B -> kv_fp8 row). The
# MG path reads rope from global/L2 (no smem staging), so only the 528B bulk is
# issued here.
_GLM_IO_STRIDE = 656
_GLM_NOPE_SCALE_BYTES = 528
# nvfp4 (LAYOUT-SPEC.md nvfp4_ds_mla, danielwoz lineage): GLM 400B record.
# 656 -> 400 record / 528 -> 272 payload because the nope latent packs as 256B
# E2M1 nibbles + 16B UE8M0 per-32-block scales. The 128B bf16 rope follows at
# gmem offset 272 (byte-identical format; still read from global/L2 by the
# math, via prefill_mg._glm_rope_base_off). Reached ONLY via the
# gmem_record_bytes / copy_bytes kwargs (defaults = fp8 values) so every
# existing caller is byte-identical.
_GLM_NVFP4_IO_STRIDE = 400
_GLM_NVFP4_PACK_BYTES = 272


@cute.jit
def io_issue_gather_dsv4_nope(
    kv_cache_u8: cute.Tensor,
    topk_indices: cute.Tensor,
    kv_fp8_dst_addr: Int32,
    kv_sc_dst_addr: Int32,
    token_idx_view: cute.Tensor,
    full_mbar_ptr,
    g_start: Int32,
    g_end: Int32,
    page_block_size: Int32,
    stride_kv_block: Int64,
    io_lane: Int32,
    *,
    bi: cutlass.Constexpr,
    kv_smem_stride: cutlass.Constexpr,
    io_threads: cutlass.Constexpr = _IO_THREADS,
):
    """Gather one BI=64 DSV4 prefill tile into MG smem.

    This is the DSV4-only equivalent of FlashInfer
    ``io_gather_scales`` + ``io_bulk_gather_tile`` with
    ``KV_SMEM_COPY_BYTES == D_NOPE``.
    """
    _ios = Int64(_DSV4_IO_STRIDE)
    _nope = Int32(_DSV4_NOPE_BYTES)
    _foot = Int32(_DSV4_FOOTER_BYTES)

    eo = Int32(0)
    for _ in cutlass.range_constexpr((bi + io_threads - 1) // io_threads):
        entry = eo + io_lane
        if entry < Int32(bi):
            cand_pos = g_start + entry
            idx_raw = Int32(-1)
            if cand_pos < g_end:
                idx_raw = Int32(topk_indices[cand_pos])
            token_idx_view[entry] = idx_raw

            f0 = Uint32(0)
            f1 = Uint32(0)
            if idx_raw >= Int32(0):
                block_idx = idx_raw // page_block_size
                local_idx = idx_raw - block_idx * page_block_size
                scale_base_off = (
                    Int64(block_idx) * stride_kv_block
                    + Int64(page_block_size) * _ios
                    + Int64(local_idx) * Int64(_foot)
                )
                f0, f1 = ld_global_nc_v2_u32(get_ptr_as_int64(kv_cache_u8, scale_base_off))
            s_byte = entry * _foot
            st_shared_u32(kv_sc_dst_addr + s_byte, f0)
            st_shared_u32(kv_sc_dst_addr + s_byte + Int32(4), f1)
        eo += Int32(io_threads)

    cute.arch.fence_acq_rel_cta()

    if io_lane == Int32(0):
        cute.arch.mbarrier_arrive_and_expect_tx(full_mbar_ptr, Int32(bi * _DSV4_NOPE_BYTES))

    full_mbar_u32 = shared_ptr_to_u32(full_mbar_ptr)
    eo = Int32(0)
    for _ in cutlass.range_constexpr((bi + io_threads - 1) // io_threads):
        entry = eo + io_lane
        if entry < Int32(bi):
            cand_pos = g_start + entry
            idx_raw = Int32(-1)
            if cand_pos < g_end:
                idx_raw = Int32(topk_indices[cand_pos])
            idx = idx_raw
            if idx < Int32(0):
                idx = Int32(0)
            block_idx = idx // page_block_size
            local_idx = idx - block_idx * page_block_size
            data_base_off = Int64(block_idx) * stride_kv_block + Int64(local_idx) * _ios
            cp_async_bulk_g2s_mbar(
                kv_fp8_dst_addr + entry * Int32(kv_smem_stride),
                get_ptr_as_int64(kv_cache_u8, data_base_off),
                _nope,
                full_mbar_u32,
            )
        eo += Int32(io_threads)


@cute.jit
def io_issue_gather_glm_mg(
    kv_cache_u8: cute.Tensor,
    topk_indices: cute.Tensor,
    kv_fp8_dst_addr: Int32,
    token_idx_view: cute.Tensor,
    full_mbar_ptr,
    g_start: Int32,
    g_end: Int32,
    page_block_size: Int32,
    stride_kv_block: Int64,
    io_lane: Int32,
    *,
    bi: cutlass.Constexpr,
    kv_smem_stride: cutlass.Constexpr,   # 528 fp8 (512 nope + 16 inline fp32)
                                         # nvfp4: 272 (dense packed staging rows)
    io_threads: cutlass.Constexpr = _IO_THREADS,
    # nvfp4: GLM record geometry, previously hardcoded 656/528. Defaults are the
    # fp8 values so every existing call site traces byte-identical; the nvfp4
    # caller passes 400/272. copy_bytes == kv_smem_stride in both modes (the
    # staged payload rows are dense); kept explicit rather than derived.
    gmem_record_bytes: cutlass.Constexpr = _GLM_IO_STRIDE,       # nvfp4: 656 -> 400
    copy_bytes: cutlass.Constexpr = _GLM_NOPE_SCALE_BYTES,       # nvfp4: 528 -> 272
):
    """Gather one BI=64 GLM prefill tile into MG smem.

    GLM analogue of ``io_issue_gather_dsv4_nope``: the per-token record's
    NoPE(+scales) payload (fp8: 528B e4m3+inline-fp32; nvfp4: 272B packed E2M1
    nibbles+UE8M0, expanded in-smem by the math before S1) bulk-copies into the
    kv_fp8 row. There is NO grouped UE8M0 footer (so NO scalar scale gather)
    and -- like the DSV4 MG path -- RoPE is read from global/L2 by the math
    (NOT staged to smem), so the 528-stride GLM KV fits the carveout for
    mg_n_hg==2. Single full mbarrier (the MG convention): the leader arrives +
    expect_tx over the BI payload bulks."""
    _ios = Int64(gmem_record_bytes)
    _nope = Int32(copy_bytes)

    # --- (1) per-entry validity index staging (no footer for GLM). ---
    eo = Int32(0)
    for _ in cutlass.range_constexpr((bi + io_threads - 1) // io_threads):
        entry = eo + io_lane
        if entry < Int32(bi):
            cand_pos = g_start + entry
            idx_raw = Int32(-1)
            if cand_pos < g_end:
                idx_raw = Int32(topk_indices[cand_pos])
            token_idx_view[entry] = idx_raw
        eo += Int32(io_threads)

    cute.arch.fence_acq_rel_cta()

    if io_lane == Int32(0):
        # nvfp4: expect_tx bi*528=33792 -> bi*copy_bytes (bi*272=17408 nvfp4)
        # because the mbarrier transaction count must equal the bulk-copied
        # bytes exactly; identical value (33792) on the fp8 path.
        cute.arch.mbarrier_arrive_and_expect_tx(full_mbar_ptr, Int32(bi * copy_bytes))

    full_mbar_u32 = shared_ptr_to_u32(full_mbar_ptr)
    eo = Int32(0)
    for _ in cutlass.range_constexpr((bi + io_threads - 1) // io_threads):
        entry = eo + io_lane
        if entry < Int32(bi):
            cand_pos = g_start + entry
            idx_raw = Int32(-1)
            if cand_pos < g_end:
                idx_raw = Int32(topk_indices[cand_pos])
            idx = idx_raw
            if idx < Int32(0):
                idx = Int32(0)
            block_idx = idx // page_block_size
            local_idx = idx - block_idx * page_block_size
            data_base_off = Int64(block_idx) * stride_kv_block + Int64(local_idx) * _ios
            # NoPE + inline fp32 scales (528B) -> kv_fp8 row.
            cp_async_bulk_g2s_mbar(
                kv_fp8_dst_addr + entry * Int32(kv_smem_stride),
                get_ptr_as_int64(kv_cache_u8, data_base_off),
                _nope,
                full_mbar_u32,
            )
        eo += Int32(io_threads)
