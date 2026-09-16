#!/usr/bin/env python3
"""Fail closed if the digest-pinned Mia EXL3 image is incomplete."""

from pathlib import Path

required = (
    "/opt/glm53/chat_template.jinja",
    "/opt/glm53/dflash2_speculator.py",
    "/opt/glm53/qwen3_dflash2.py",
    "/opt/glm53/patch_dflash2.py",
    "/opt/glm53/patch_glm_eagle3.py",
    "/opt/glm53/patch_glm5_drafter_group.py",
    "/opt/glm53/patch_glm_video_placeholders.py",
    "/opt/glm53/patch_hybrid_prefix_hit.py",
    "/opt/glm53/patch_apc_per_group_retention.py",
    "/opt/glm53/patch_apc_no_store.py",
    "/opt/glm53/patch_kv_capacity_log.py",
    "/opt/glm53/patch_cache_reset.py",
    "/opt/glm53/patch_kpool_tail_slotmap.py",
    "/opt/glm53/patch_scheduler_decode_floor.py",
    "/opt/glm53/patch_suppress_stops_in_reasoning.py",
    "/opt/glm53/patch_xgrammar_termination.py",
    "/opt/glm53/patch_adaptive_k.py",
    "/opt/glm53/patch_dense_fp8.py",
    "/opt/glm53/exl3-fat-kernel/exl3_fat_gemm.cu",
    "/opt/glm53/exl3-fat-kernel/exl3_fat_gemm.cuh",
    "/opt/glm53/exl3-fat-kernel/exl3_fat_moe.cu",
    "/opt/glm53/exl3-fat-kernel/exl3_fat_moe.cuh",
    "/opt/glm53/test_scheduler_decode_floor_restart.py",
    "/opt/glm53/test_apc_per_group_retention.py",
    "/opt/glm53/test_apc_no_store.py",
    "/opt/glm53/test_kv_capacity_log.py",
    "/opt/glm53/test_cache_reset_endpoint.py",
    "/opt/glm53/test_kpool_tail_slotmap.py",
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/exl3.py",
)
missing = [path for path in required if not Path(path).is_file()]
if missing:
    raise SystemExit(f"incomplete Mia EXL3 runtime: {missing}")

# The E2 fat GEMM and the E3 grouped fat-expert kernels are compiled into a
# single exllamav3_ext translation unit (overlay/patch_exl3_fat_kernel.py).
# overlay/exl3.py also accepts the retired additive exl3_fat_moe_ext module, but
# this recipe deliberately ships upstream's one-extension layout, so assert
# every entry point on exllamav3_ext and nothing there to fall back to.
try:
    # torch first: the compiled extension needs libc10/libtorch already loaded,
    # so this block is deliberately not alphabetically sorted.
    import torch  # noqa: F401, I001  (load libc10/libtorch before the CUDA extension)
    import exllamav3_ext
except Exception as exc:  # pragma: no cover - exercised by image build
    raise SystemExit(f"incomplete EXL3 runtime: {exc}") from exc
required_symbols = (
    "exl3_moe",
    "exl3_fat_gemm",
    "exl3_fat_gemm_scatter",
    "exl3_fat_moe_gather",
    "exl3_fat_moe_gateup",
    "exl3_fat_moe_down",
    "exl3_fat_moe_tile_rows_gateup",
    "exl3_fat_moe_tile_rows_down",
)
missing_symbols = [
    name for name in required_symbols if not hasattr(exllamav3_ext, name)
]
if missing_symbols:
    raise SystemExit(f"incomplete EXL3 extension symbols: {missing_symbols}")

# --load-format instanttensor is declared by the Recipe; the loader is a
# separate wheel installed by the Dockerfile after the overlay tests.
try:
    import instanttensor  # noqa: F401
except Exception as exc:  # pragma: no cover - exercised by image build
    raise SystemExit(f"incomplete instanttensor loader: {exc}") from exc

print("Mia GLM 5.3 EXL3 DFlash2 runtime contract OK")
