#!/usr/bin/env python3
"""Fail closed if our packaging of the Mia EXL3 image is incomplete.

Only files we ship are checked; upstream features and patch contents are not.
"""

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
    "/opt/glm53/patch_kpool_tail_seed_stride.py",
    "/opt/glm53/patch_dflash2_exl3.py",
    "/opt/glm53/patch_loadclone.py",
    "/opt/glm53/patch_cold_load_uma.py",
    "/opt/glm53/patch_skip_cudagraph_profile.py",
    "/opt/glm53/patch_mamba_align_chunking.py",
    "/opt/glm53/patch_mamba_align_state_free.py",
    "/opt/glm53/patch_tool_choice_none.py",
    "/opt/glm53/patch_scheduler_decode_floor.py",
    "/opt/glm53/patch_suppress_stops_in_reasoning.py",
    "/opt/glm53/patch_xgrammar_termination.py",
    "/opt/glm53/patch_adaptive_k.py",
    "/opt/glm53/patch_dense_fp8.py",
    "/opt/glm53/patch_tp3_glm.py",
    "/opt/glm53/patch_exl3_ep_shard.py",
    "/opt/glm53/patch_exl3_expert_map.py",
    "/opt/glm53/drafter-config-tp3.json",
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
    "/opt/glm53/test_mamba_align_chunking.py",
    "/opt/glm53/test_mamba_align_state_free.py",
    "/opt/glm53/test_tool_choice_none.py",
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/exl3.py",
)
missing = [path for path in required if not Path(path).is_file()]
if missing:
    raise SystemExit(f"incomplete Mia EXL3 runtime: {missing}")

try:
    # torch first: the compiled extension needs libc10/libtorch already loaded.
    import torch  # noqa: F401, I001
    import exllamav3_ext  # noqa: F401
except Exception as exc:  # pragma: no cover - exercised by image build
    raise SystemExit(f"incomplete EXL3 runtime: {exc}") from exc

# --load-format instanttensor is declared by the Recipe; the loader is a
# separate wheel installed by the Dockerfile after the overlay tests.
try:
    import instanttensor  # noqa: F401
except Exception as exc:  # pragma: no cover - exercised by image build
    raise SystemExit(f"incomplete instanttensor loader: {exc}") from exc

# Video alignment is installed by our Dockerfile as a .pth every process imports.
video_pth = Path("/usr/local/lib/python3.12/dist-packages/glm53_video.pth")
if not video_pth.is_file():
    raise SystemExit(f"incomplete GLM 5.3 image: the video alignment is not installed: {video_pth}")

# The TP=3 shape fixes upstream applies at start are baked into this image.
site = Path("/usr/local/lib/python3.12/dist-packages/vllm")
markers = {
    site / "model_executor/layers/quantization/exl3.py": (
        "EXL3-EP-NO-INTRA-SHARD",
        "EXL3-PIN-EXPERT-MAP-RO",
    ),
    site / "model_executor/model_loader/weight_utils.py": ("[glm53-loadclone:v2]",),
}
for target, wanted in markers.items():
    text = target.read_text()
    absent = [marker for marker in wanted if marker not in text]
    if absent:
        raise SystemExit(f"incomplete TP=3 overlay: {target} lacks {absent}")
models = [
    site / "models/glm5next/nvidia/model.py",
    site / "model_executor/models/glm5next/nvidia/model.py",
]
patched = [path for path in models if path.is_file()]
if not patched or any("TP3-HEAD-PAD" not in path.read_text() for path in patched):
    raise SystemExit("incomplete TP=3 overlay: glm5next model.py lacks the head pad")

unreadable = [
    str(path) for path in site.rglob("*") if path.is_file() and not path.stat().st_mode & 0o004
]
if unreadable:
    raise SystemExit(f"installed vLLM files are not readable by the service account: {unreadable[:5]}")

print("Mia GLM 5.3 EXL3 DFlash2 TP3 runtime contract OK")
