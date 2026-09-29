#!/usr/bin/env python3
"""Check that every file the Mia patch sequence changes is present in the patched vLLM tree.

Structural only: the base image is pinned by digest in the Dockerfile and the
patch scripts assert their own anchors, so no tree digest is kept here.
"""

from __future__ import annotations

import sys
from pathlib import Path

TARGETS = (
    "config/vllm.py",
    "distributed/device_communicators/shm_broadcast.py",
    "entrypoints/openai/chat_completion/serving.py",
    "model_executor/layers/sparse_attn_indexer.py",
    "models/deepseek_v32/nvidia/attention.py",
    "models/deepseek_v4/attention.py",
    "models/deepseek_v4/common/ops/cache_utils.py",
    "models/deepseek_v4/compressor.py",
    "models/deepseek_v4/nvidia/flashmla.py",
    "models/deepseek_v4/nvidia/dspark.py",
    "models/deepseek_v4/nvidia/model.py",
    "models/deepseek_v4/sparse_mla.py",
    "tokenizers/deepseek_v4.py",
    "tokenizers/deepseek_v4_encoding.py",
    "v1/attention/backends/mla/flashmla_sparse.py",
    "v1/core/kv_cache_coordinator.py",
    "v1/core/sched/scheduler.py",
    "v1/engine/detokenizer.py",
    "v1/engine/input_processor.py",
    "v1/structured_output/__init__.py",
    "v1/worker/gpu/model_runner.py",
    "v1/worker/gpu/sample/sampler.py",
    "v1/worker/gpu/sample/thinking_budget_gpu.py",
)


def missing(root: Path) -> list[str]:
    return [name for name in TARGETS if not (root / name).is_file()]


if len(sys.argv) != 2:
    raise SystemExit("usage: verify-patched-tree.py ROOT")
absent = missing(Path(sys.argv[1]))
if absent:
    raise SystemExit("patched vLLM tree is missing: " + ", ".join(absent))
print(f"patched vLLM tree has all {len(TARGETS)} patch targets")
