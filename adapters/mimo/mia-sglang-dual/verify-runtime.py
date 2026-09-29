"""Fail the image build unless the pinned MiMo-V2.6 SGLang packaging is complete."""

from __future__ import annotations

import importlib
import json
import os
import platform
import shutil
import subprocess
import sys

import torch
from sglang.srt.server_args import ServerArgs

REQUIRED_SERVER_FIELDS = {
    "attention_backend",
    "chunked_prefill_size",
    "context_length",
    "cuda_graph_max_bs_decode",
    "disable_prefill_cuda_graph",
    "dist_init_addr",
    "enable_metrics",
    "enable_multi_layer_eagle",
    "enable_multimodal",
    "kv_cache_dtype",
    "max_running_requests",
    "max_total_tokens",
    "mem_fraction_static",
    "mm_attention_backend",
    "moe_a2a_backend",
    "moe_runner_backend",
    "nnodes",
    "node_rank",
    "page_size",
    "speculative_algorithm",
    "speculative_dflash_block_size",
    "speculative_draft_model_path",
    "speculative_eagle_topk",
    "speculative_num_draft_tokens",
    "speculative_num_steps",
    "swa_full_tokens_ratio",
    "tp_size",
    "weight_loader_drop_cache_after_load",
}


def main() -> None:
    if platform.machine() not in {"aarch64", "arm64"}:
        raise SystemExit("the MiMo-V2.6 Spark adapter must be built for linux/arm64")
    if not torch.version.cuda or not torch.version.cuda.startswith("13."):
        raise SystemExit("the MiMo-V2.6 Spark adapter requires the pinned CUDA 13 runtime")

    missing = sorted(REQUIRED_SERVER_FIELDS - set(ServerArgs.__dataclass_fields__))
    if missing:
        raise SystemExit(f"the pinned SGLang image lacks MiMo-V2.6 launch fields: {missing}")
    importlib.import_module("sglang.srt.models.mimo_v2")

    for path in (
        "/opt/mimo26/boot.py",
        "/opt/mimo26/patches/sitecustomize.py",
        "/opt/vonk/bin/sglang-serve",
    ):
        if not os.path.isfile(path):
            raise SystemExit(f"missing packaged file: {path}")
    if not os.access("/opt/vonk/bin/sglang-serve", os.X_OK):
        raise SystemExit("/opt/vonk/bin/sglang-serve is not executable")
    if shutil.which("ffprobe") is None:
        raise SystemExit("ffprobe is missing")

    # boot.py's own dry run: the default launch line must assemble without a GPU or a model.
    plan = subprocess.run(
        [sys.executable, "/opt/mimo26/boot.py", "plan"],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "NNODES": "1", "TP_SIZE": "2"},
    )
    arguments = json.loads(plan.stdout.strip().splitlines()[-1])["args"]
    for expected in ("--enable-multimodal", "--speculative-algorithm", "EAGLE"):
        if expected not in arguments:
            raise SystemExit(f"boot.py plan lost {expected}: {arguments}")


if __name__ == "__main__":
    main()
