"""Fail the image build unless the packaged DeepSeek-V4.1-Flash SGLang kit is complete."""

from __future__ import annotations

import importlib
import os
import platform
import sys

from sglang.srt.server_args import ServerArgs

# Fields boot.py always or (with the recipe's EXTRA_SGLANG_ARGS) sets on the command line.
REQUIRED_SERVER_FIELDS = {
    "attention_backend",
    "chunked_prefill_size",
    "context_length",
    "cuda_graph_max_bs_decode",
    "dist_init_addr",
    "enable_cache_report",
    "enable_decoder_swa_bounded_replay",
    "ep_size",
    "fp8_gemm_backend",
    "max_running_requests",
    "max_total_tokens",
    "mem_fraction_static",
    "moe_runner_backend",
    "nnodes",
    "node_rank",
    "reasoning_parser",
    "sleep_on_idle",
    "speculative_algorithm",
    "speculative_dspark_block_size",
    "tool_call_parser",
    "tp_size",
    "watchdog_timeout",
}


def main() -> None:
    if platform.machine() not in {"aarch64", "arm64"}:
        raise SystemExit("the DeepSeek-V4.1-Flash Spark adapter must be built for linux/arm64")
    missing = sorted(REQUIRED_SERVER_FIELDS - set(ServerArgs.__dataclass_fields__))
    if missing:
        raise SystemExit(f"the pinned SGLang image lacks DeepSeek-V4.1 launch fields: {missing}")
    importlib.import_module("sglang.srt.models.deepseek_v4")
    importlib.import_module("sglang.srt.models.deepseek_v4_dspark")
    for path in (
        "/opt/dsv41/boot.py",
        "/opt/dsv41/adapter/sitecustomize.py",
        "/opt/dsv41/adapter/librow_store.so",
        "/opt/vonk/bin/sglang-serve",
    ):
        if not os.path.exists(path):
            raise SystemExit(f"missing {path}")


if __name__ == "__main__":
    sys.exit(main())
