#!/usr/bin/env python3
"""Idempotently cap only vLLM's synthetic kernel-warmup request batch."""

from __future__ import annotations

import argparse
from pathlib import Path


IMPORT_ANCHOR = "import numpy as np\nimport torch\n"
IMPORT_PATCH = "import numpy as np\nimport os\nimport torch\n"

BLOCK_ANCHOR = """    if max_blocks_per_req > 0:
        # Reserve block 0 (null block) and ensure we have enough blocks.
        # Encoder-only models allocate no KV blocks, so this cap doesn't apply.
        num_reqs = min(
            num_reqs,
            max(1, (model_runner.kv_cache_config.num_blocks - 1) // max_blocks_per_req),
        )

    req_ids = [f\"_warmup_{i}_\" for i in range(num_reqs)]
"""

BLOCK_PATCH = """    if max_blocks_per_req > 0:
        # Reserve block 0 (null block) and ensure we have enough blocks.
        # Encoder-only models allocate no KV blocks, so this cap doesn't apply.
        num_reqs = min(
            num_reqs,
            max(1, (model_runner.kv_cache_config.num_blocks - 1) // max_blocks_per_req),
        )

    # Inkling's vendored SM120/SM121 relative-attention FA4 kernel can issue
    # cudaErrorIllegalAddress when vLLM warms six uniform synthetic requests.
    # This opt-in cap affects startup warmup only; runtime max_num_seqs remains
    # unchanged so C1-C6 can be validated against the real scheduler.
    warmup_cap = int(os.environ.get(\"INKLING_WARMUP_MAX_NUM_SEQS\", \"0\"))
    if warmup_cap > 0:
        num_reqs = min(num_reqs, warmup_cap)

    req_ids = [f\"_warmup_{i}_\" for i in range(num_reqs)]
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("target", type=Path)
    args = parser.parse_args()

    text = args.target.read_text(encoding="utf-8")
    patched = "INKLING_WARMUP_MAX_NUM_SEQS" in text
    if args.check:
        if not patched:
            raise SystemExit("Inkling warmup cap is not installed")
        return
    if patched:
        return
    if IMPORT_ANCHOR not in text or BLOCK_ANCHOR not in text:
        raise SystemExit("Unsupported vLLM warmup.py: expected anchors not found")
    text = text.replace(IMPORT_ANCHOR, IMPORT_PATCH, 1)
    text = text.replace(BLOCK_ANCHOR, BLOCK_PATCH, 1)
    args.target.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
