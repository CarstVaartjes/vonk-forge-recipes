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
    "/opt/glm53/patch_kpool_tail_slotmap.py",
    "/opt/glm53/patch_scheduler_decode_floor.py",
    "/opt/glm53/patch_suppress_stops_in_reasoning.py",
    "/opt/glm53/patch_xgrammar_termination.py",
    "/opt/glm53/patch_adaptive_k.py",
    "/opt/glm53/patch_dense_fp8.py",
    "/opt/glm53/build_exl3_fat_moe_ext.py",
    "/opt/glm53/exl3-fat-kernel/exl3_fat_moe.cu",
    "/opt/glm53/exl3-fat-kernel/exl3_fat_moe.cuh",
    "/opt/glm53/test_kpool_tail_slotmap.py",
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/exl3.py",
)
missing = [path for path in required if not Path(path).is_file()]
if missing:
    raise SystemExit(f"incomplete Mia EXL3 runtime: {missing}")
try:
    import torch  # noqa: F401  (load libc10/libtorch before the CUDA extension)
    import exl3_fat_moe_ext
except Exception as exc:  # pragma: no cover - exercised by image build
    raise SystemExit(f"incomplete E3 grouped runtime: {exc}") from exc
required_symbols = (
    "exl3_fat_moe_gather",
    "exl3_fat_moe_gateup",
    "exl3_fat_moe_down",
    "exl3_fat_moe_tile_rows_gateup",
    "exl3_fat_moe_tile_rows_down",
)
missing_symbols = [name for name in required_symbols if not hasattr(exl3_fat_moe_ext, name)]
if missing_symbols:
    raise SystemExit(f"incomplete E3 grouped runtime symbols: {missing_symbols}")
print("Mia GLM 5.3 EXL3 DFlash2 runtime contract OK")
