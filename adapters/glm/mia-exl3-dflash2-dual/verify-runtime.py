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

# The overlay that rewrites an installed vLLM file is applied at image build
# time (the container root is read-only at run time), so the image has to prove
# it landed rather than trusting that some later start-time hook ran. Two
# independent facts: the kpool guard carries the overlay's own marker and is
# disabled, and the video alignment is installed as a .pth that every process
# imports.
kpool_path = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/"
    "sparse_attn_indexer_kpool.py"
)
if not kpool_path.is_file():
    raise SystemExit(f"incomplete GLM 5.3 overlay: {kpool_path} is absent")
kpool = kpool_path.read_text()
if "persistent_topk" not in kpool:
    raise SystemExit(
        "incomplete GLM 5.3 overlay: the GB10 persistent_topk guard is absent "
        "from sparse_attn_indexer_kpool.py, so the overlay did not run"
    )
if "if False and current_platform.is_cuda()" not in kpool:
    raise SystemExit(
        "incomplete GLM 5.3 overlay: the GB10 persistent_topk guard is still "
        "enabled in sparse_attn_indexer_kpool.py"
    )
video_pth = Path("/usr/local/lib/python3.12/dist-packages/glm53_video.pth")
if not video_pth.is_file():
    raise SystemExit(
        f"incomplete GLM 5.3 overlay: the video alignment is not installed: {video_pth}"
    )

print("Mia GLM 5.3 EXL3 DFlash2 runtime contract OK")
