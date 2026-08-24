"""Fail the image build unless the pinned Inkling DGX Spark runtime is complete."""

from __future__ import annotations

import importlib
import importlib.metadata
import platform

import torch
from sglang.srt.server_args import ServerArgs


REQUIRED_MODULES = (
    "sglang.srt.models.inkling",
    "sglang.srt.multimodal.inkling.processing_inkling",
    "sglang.srt.multimodal.processors.inkling",
    "sglang.srt.function_call.inkling_detector",
    "sglang.srt.parser.inkling_renderer",
)

REQUIRED_SERVER_FIELDS = {
    "attention_backend",
    "disable_prefill_cuda_graph",
    "dist_init_addr",
    "enable_multimodal",
    "fp4_gemm_runner_backend",
    "mamba_full_memory_ratio",
    "mamba_radix_cache_strategy",
    "mem_fraction_static",
    "model_path",
    "moe_runner_backend",
    "node_rank",
    "nnodes",
    "page_size",
    "quantization",
    "reasoning_parser",
    "swa_full_tokens_ratio",
    "tool_call_parser",
    "tp_size",
}


def main() -> None:
    if platform.machine() not in {"aarch64", "arm64"}:
        raise SystemExit("the Inkling Spark adapter must be built for linux/arm64")
    if not torch.version.cuda or not torch.version.cuda.startswith("13."):
        raise SystemExit("the Inkling Spark adapter requires the pinned CUDA 13 runtime")

    for module in REQUIRED_MODULES:
        importlib.import_module(module)

    fields = set(ServerArgs.__dataclass_fields__)
    missing = sorted(REQUIRED_SERVER_FIELDS - fields)
    if missing:
        raise SystemExit(f"the pinned SGLang image lacks Inkling launch fields: {missing}")

    nccl_version = importlib.metadata.version("nvidia-nccl-cu13")
    if nccl_version != "2.30.7":
        raise SystemExit(f"unexpected NCCL runtime: {nccl_version}")


if __name__ == "__main__":
    main()
