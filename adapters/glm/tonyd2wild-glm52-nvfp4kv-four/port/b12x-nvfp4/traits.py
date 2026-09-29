"""Per-model traits for the unified SM120 sparse-MLA CuTeDSL backend.

Pure Python (no `cute` import): this module is consumed both by the launcher
and by `smem.py`/`launch.py`, and its enums double as `cutlass.const_expr`
specialization keys (int-valued) and as `KernelCompileSpec` `KeyField` entries.

All per-model constants are transcribed VERBATIM from
`.sm120port/verified_traits.md` (DSV4 and GLM_NSA columns). DSV3.2 / POW2_FP32
are DROPPED per `.sm120port/scope_decisions.md`.

NVFP4 port: the GLM_NSA model additionally supports the `nvfp4_ds_mla` 400-byte
KV record (LAYOUT-SPEC.md, danielwoz/vllm-dspark-nvfp4 lineage) alongside the
KNOWNGOOD 656-byte fp8_ds_mla record. Selected via the new ``kv_layout``
argument; the fp8 trait bundle is BYTE-IDENTICAL to the pre-port bundle
(kv_layout/kv_pack_stride default to the fp8 values on every existing path).
"""

from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# const_expr specialization keys (int-valued so they can key cutlass.const_expr
# branches AND KernelCompileSpec KeyField entries). DSV3_2 / POW2_FP32 dropped.
# ---------------------------------------------------------------------------
class ModelType:
    DSV4 = 0
    GLM_NSA = 1


class ComputeMode:
    FP8 = 0
    BF16 = 1


class ScaleFormat:
    UE8M0_BYTE = 0  # DSV4: power-of-2 exponent bytes in an 8B footer.
    ARBITRARY_FP32 = 1  # GLM: arbitrary FP32 inline scales (reference.py).


class KVLayout:
    """GLM KV-cache record layout (const_expr specialization key).

    FP8_DS_MLA (KNOWNGOOD rollback path, 656 B/token):
        [0:512) e4m3 nope | [512:528) inline scales | [528:656) 64 bf16 rope
    NVFP4_DS_MLA (LAYOUT-SPEC.md, 400 B/token):
        [0:256) 512 E2M1 nibbles (byte b = low nibble dim 2b, high dim 2b+1)
        | [256:272) 16 UE8M0 per-32-block scales | [272:400) 64 bf16 rope
    """

    FP8_DS_MLA = 0
    NVFP4_DS_MLA = 1


@dataclass(frozen=True)
class UnifiedMLATraits:
    """Frozen, hashable trait bundle for one (model, compute, scale) tuple.

    Mirrors FlashInfer's ``KVCacheTraits<MT>`` + ``ComputeTraits<MT,CM>`` so the
    traced kernel can constant-fold every model-divergent point. Hashable so it
    is usable in ``functools.lru_cache`` / ``KernelCompileSpec`` keys.
    """

    model_type: int
    compute_mode: int
    scale_format: int
    d_nope: int
    d_rope: int
    d_v: int
    quant_tile: int
    num_scales: int
    n_v_chunks: int
    nt_per_warp_xv: int
    kv_gmem_stride: int
    kv_smem_stride: int
    q_nope_stride: int
    bi: int
    hpb: int
    block_threads: int
    math_threads: int
    bulk_tx_bytes: int
    v_has_rope: bool
    has_extra_cache: bool
    # nvfp4: new fields, DEFAULTED so every existing construction/consumer
    # (smem.py, launch.py) sees the identical fp8 bundle unchanged.
    # kv_layout: KVLayout.* record selector (const_expr specialization key).
    kv_layout: int = KVLayout.FP8_DS_MLA
    # kv_pack_stride: nonzero -> the IO warp gathers PACKED staging rows of
    # this many bytes/token (dense, into the front of the kv_fp8 buffer) and
    # the math warps expand them in place to kv_smem_stride rows before S1.
    # 0 -> no staging stage (fp8 path: IO gathers kv_smem_stride rows direct).
    kv_pack_stride: int = 0


def make_unified_traits(
    model_type: int,
    compute_mode: int,
    scale_format: int,
    kv_layout: int = KVLayout.FP8_DS_MLA,
) -> UnifiedMLATraits:
    """Build the trait bundle for one specialization tuple.

    Constants come straight from `.sm120port/verified_traits.md`. Raises
    ``ValueError`` for the dropped DSV3.2 / POW2_FP32 combinations and for any
    (model, scale_format) mismatch.
    """
    # BF16 compute_mode is deferred (both decode targets are FP8) but the enum
    # value is accepted so the const_expr branch can exist; FP8 is the only
    # validated path today.
    if compute_mode not in (ComputeMode.FP8, ComputeMode.BF16):
        raise ValueError(f"unsupported compute_mode {compute_mode!r}")
    if kv_layout not in (KVLayout.FP8_DS_MLA, KVLayout.NVFP4_DS_MLA):
        raise ValueError(f"unsupported kv_layout {kv_layout!r}")
    # nvfp4 is a GLM-only record (LAYOUT-SPEC.md); DSV4 keeps its compressed
    # 584B packed layout untouched.
    if kv_layout == KVLayout.NVFP4_DS_MLA and model_type != ModelType.GLM_NSA:
        raise ValueError(
            "KVLayout.NVFP4_DS_MLA is GLM_NSA-only (400B nvfp4_ds_mla record); "
            f"got model_type={model_type!r}"
        )
    # nvfp4 requires FP8 compute: the in-smem expansion stage rebuilds the GLM
    # FP8 row format (e4m3 + inline fp32 scales); the (deferred) BF16 compute
    # path expects a different smem row and has no nvfp4 expansion.
    if kv_layout == KVLayout.NVFP4_DS_MLA and compute_mode != ComputeMode.FP8:
        raise ValueError(
            "KVLayout.NVFP4_DS_MLA requires ComputeMode.FP8 (the nvfp4 "
            "expansion stage targets the GLM FP8 smem row format)"
        )

    if model_type == ModelType.DSV4:
        if scale_format != ScaleFormat.UE8M0_BYTE:
            raise ValueError(
                "DSV4 requires ScaleFormat.UE8M0_BYTE (footer); "
                f"got scale_format={scale_format!r}"
            )
        # DSV4 column of verified_traits.md (UE8M0_BYTE, V_HAS_ROPE=true).
        return UnifiedMLATraits(
            model_type=ModelType.DSV4,
            compute_mode=compute_mode,
            scale_format=ScaleFormat.UE8M0_BYTE,
            d_nope=448,
            d_rope=64,
            d_v=512,
            quant_tile=64,
            num_scales=7,  # 448/64
            n_v_chunks=7,
            nt_per_warp_xv=1,  # 64/8/8
            kv_gmem_stride=584,  # 448 + 128 + 8
            kv_smem_stride=464,  # 448 + 16
            q_nope_stride=464,
            bi=64,  # cands/chunk
            hpb=16,  # heads/CTA
            block_threads=288,  # 9 warps
            math_threads=256,  # 8 warps
            bulk_tx_bytes=36864,  # 64*(448+128); footer excluded (16-align caveat)
            v_has_rope=True,
            has_extra_cache=True,  # DSV4 dual-cache only
        )

    if model_type == ModelType.GLM_NSA:
        if scale_format != ScaleFormat.ARBITRARY_FP32:
            raise ValueError(
                "GLM_NSA requires ScaleFormat.ARBITRARY_FP32 (inline); "
                f"got scale_format={scale_format!r}"
            )
        if kv_layout == KVLayout.NVFP4_DS_MLA:
            # GLM_NSA + nvfp4_ds_mla (LAYOUT-SPEC.md 400B record). Every
            # COMPUTE-side constant (quant_tile/num_scales/n_v_chunks/
            # nt_per_warp_xv/kv_smem_stride/q_nope_stride) is UNCHANGED from
            # the fp8 column: the in-smem expansion stage (nvfp4.py) rebuilds
            # the standard 528B e4m3+fp32 rows before S1, so decode_math runs
            # byte-identical GLM fp8 math. Only the IO-side constants change.
            return UnifiedMLATraits(
                model_type=ModelType.GLM_NSA,
                compute_mode=compute_mode,
                scale_format=ScaleFormat.ARBITRARY_FP32,
                d_nope=512,
                d_rope=64,
                d_v=512,
                quant_tile=128,
                num_scales=4,  # 512/128 (post-expansion fp32 tile scales)
                n_v_chunks=4,
                nt_per_warp_xv=2,  # 128/8/8
                # nvfp4: 656 -> 400 because the per-token record shrinks to
                # 256B nibbles + 16B UE8M0 + 128B bf16 rope (LAYOUT-SPEC.md).
                kv_gmem_stride=400,
                # UNCHANGED (528 = 512 + 4*4): the POST-EXPANSION compute row.
                # The packed staging row is kv_pack_stride below.
                kv_smem_stride=528,
                q_nope_stride=528,
                bi=64,  # cands/chunk
                hpb=16,  # heads/CTA
                block_threads=288,
                math_threads=256,
                # nvfp4: 41984 -> 25600 because the IO warp now moves
                # 64*(272 payload+scales + 128 rope) = 64*400 bytes/chunk
                # (mbarrier complete_tx count must equal the bulk-copy bytes).
                bulk_tx_bytes=25600,
                v_has_rope=False,
                has_extra_cache=False,
                kv_layout=KVLayout.NVFP4_DS_MLA,
                # nvfp4: new; 272 = 256 nibble bytes + 16 UE8M0 scale bytes.
                # 272 = 16*17 -> every cp.async.bulk src/dst/size stays
                # 16B-aligned (record 400 = 16*25; rope gmem offset 272).
                kv_pack_stride=272,
            )
        # GLM_NSA column of verified_traits.md (ARBITRARY_FP32, V_HAS_ROPE=false).
        return UnifiedMLATraits(
            model_type=ModelType.GLM_NSA,
            compute_mode=compute_mode,
            scale_format=ScaleFormat.ARBITRARY_FP32,
            d_nope=512,
            d_rope=64,
            d_v=512,
            quant_tile=128,
            num_scales=4,  # 512/128
            n_v_chunks=4,
            nt_per_warp_xv=2,  # 128/8/8
            kv_gmem_stride=656,
            kv_smem_stride=528,  # 512 + 4*4
            q_nope_stride=528,
            bi=64,  # cands/chunk
            hpb=16,  # heads/CTA
            block_threads=288,
            math_threads=256,
            bulk_tx_bytes=41984,  # 64*(528+128)
            v_has_rope=False,
            has_extra_cache=False,
        )

    raise ValueError(
        f"unsupported model_type {model_type!r} (DSV3_2 is dropped; "
        "valid: ModelType.DSV4, ModelType.GLM_NSA)"
    )


def infer_model_type(q_head_dim: int, kv_dtype) -> tuple[int, int, int]:
    """Map (q_head_dim, kv_dtype) -> (model_type, compute_mode, scale_format).

    ``q_head_dim`` is ``d_nope + d_rope``:
      - DSV4:  448 + 64 = 512 -> (DSV4, FP8, UE8M0_BYTE)
      - GLM:   512 + 64 = 576 -> (GLM_NSA, FP8, ARBITRARY_FP32)

    Both decode targets are FP8 today; ``kv_dtype`` is accepted for the future
    BF16 const_expr branch but does not currently change the result.
    """
    if q_head_dim == 512:
        return (ModelType.DSV4, ComputeMode.FP8, ScaleFormat.UE8M0_BYTE)
    if q_head_dim == 576:
        return (ModelType.GLM_NSA, ComputeMode.FP8, ScaleFormat.ARBITRARY_FP32)
    raise ValueError(
        f"unsupported q_head_dim={q_head_dim!r}; expected 512 (DSV4) or 576 (GLM_NSA)"
    )


# nvfp4: GLM record-byte-stride -> KVLayout map (the cache tensor's last dim /
# per-token stride is the layout discriminator when no explicit kv_layout is
# threaded). String aliases accepted so vLLM-side glue can pass the cache_dtype
# style names straight through.
_GLM_RECORD_BYTES_TO_KV_LAYOUT = {
    656: KVLayout.FP8_DS_MLA,
    400: KVLayout.NVFP4_DS_MLA,
}
_KV_LAYOUT_NAMES = {
    "fp8_ds_mla": KVLayout.FP8_DS_MLA,
    "nvfp4_ds_mla": KVLayout.NVFP4_DS_MLA,
}


def infer_kv_layout(kv_layout=None, record_bytes: int | None = None) -> int:
    """Resolve a KVLayout from an explicit selector and/or the record stride.

    ``kv_layout`` may be None (infer / default), a ``KVLayout`` int, or one of
    the string names {"fp8_ds_mla", "nvfp4_ds_mla"}. When both an explicit
    selector and ``record_bytes`` are given they must AGREE (a mismatch is a
    config bug -> hard error, never a silent misread of the cache bytes).
    With neither, the KNOWNGOOD fp8_ds_mla layout is returned.
    """
    explicit: int | None
    if kv_layout is None:
        explicit = None
    elif isinstance(kv_layout, str):
        if kv_layout not in _KV_LAYOUT_NAMES:
            raise ValueError(
                f"unknown kv_layout {kv_layout!r}; "
                f"expected one of {sorted(_KV_LAYOUT_NAMES)}"
            )
        explicit = _KV_LAYOUT_NAMES[kv_layout]
    else:
        explicit = int(kv_layout)
        if explicit not in (KVLayout.FP8_DS_MLA, KVLayout.NVFP4_DS_MLA):
            raise ValueError(f"unknown kv_layout {kv_layout!r}")

    inferred = (
        _GLM_RECORD_BYTES_TO_KV_LAYOUT.get(int(record_bytes))
        if record_bytes is not None
        else None
    )
    if explicit is not None:
        if inferred is not None and inferred != explicit:
            raise ValueError(
                f"kv_layout {kv_layout!r} disagrees with the cache record "
                f"stride ({record_bytes} bytes/token)"
            )
        return explicit
    if inferred is not None:
        return inferred
    return KVLayout.FP8_DS_MLA
