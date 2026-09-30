#!/usr/bin/env python3
"""Fail closed if our packaging of the Reederey87 EXL3 image is incomplete.

Only files we ship are checked; upstream features and patch contents are not.
"""

import os
from pathlib import Path

required = (
    "/opt/glm53/chat_template.jinja",
    "/opt/glm53/dflash2_speculator.py",
    "/opt/glm53/exl3-fat-kernel/exl3_fat_gemm.cu",
    "/opt/glm53/exl3-fat-kernel/exl3_fat_gemm.cuh",
    "/opt/glm53/exl3-fat-kernel/exl3_fat_moe.cu",
    "/opt/glm53/exl3-fat-kernel/exl3_fat_moe.cuh",
    "/opt/glm53/patch_adaptive_k.py",
    "/opt/glm53/patch_align_floor.py",
    "/opt/glm53/patch_apc_no_store.py",
    "/opt/glm53/patch_apc_per_group_retention.py",
    "/opt/glm53/patch_cache_reset.py",
    "/opt/glm53/patch_dflash2.py",
    "/opt/glm53/patch_exl3_ext_aarch64.py",
    "/opt/glm53/patch_exl3_fat_kernel.py",
    "/opt/glm53/patch_exl3_ticket_scheduler.py",
    "/opt/glm53/patch_fine_grained_apc.py",
    "/opt/glm53/patch_glm5_drafter_group.py",
    "/opt/glm53/patch_glm_eagle3.py",
    "/opt/glm53/patch_glm_video_placeholders.py",
    "/opt/glm53/patch_hybrid_prefix_hit.py",
    "/opt/glm53/patch_indexer_workspace.py",
    "/opt/glm53/patch_kda_recurrent.py",
    "/opt/glm53/patch_kpool_tail_slotmap.py",
    "/opt/glm53/patch_kv_capacity_log.py",
    "/opt/glm53/patch_kv_merge_assert.py",
    "/opt/glm53/patch_mamba_null_gap_retirement.py",
    "/opt/glm53/patch_model_overrides.py",
    "/opt/glm53/patch_router_gemm_gb10.py",
    "/opt/glm53/patch_scheduler_decode_floor.py",
    "/opt/glm53/patch_suppress_stops_in_reasoning.py",
    "/opt/glm53/patch_w28_correctness.py",
    "/opt/glm53/patch_xgrammar_termination.py",
    "/opt/glm53/dflash2_model.py",
    "/opt/glm53/patch_prefix_cache_sparse_miss_metric.py",
)
missing = [path for path in required if not Path(path).is_file()]
if missing:
    raise SystemExit(f"incomplete Reederey87 EXL3 runtime: {missing}")
if not os.access("/opt/vonk/bin/vllm", os.X_OK):
    raise SystemExit("/opt/vonk/bin/vllm is not executable")
