#!/usr/bin/env python3
"""Backport Mia's Stage-C DSpark shared-expert tensor mapping fix.

The pinned DSpark overlay maps fused attention tensors, but omitted the two
shared-expert checkpoint shards that load into ``gate_up_proj``.  Refuse source
drift, make the exact upstream mapping change, and verify the result before the
runtime image can be published.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

DEFAULT_TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/dspark.py"
)
EXPECTED_SOURCE_SHA256 = "fdbfd9e57b052c7ef9584ad628fc66efe7df3735150e624097f2f330891c4f97"
EXPECTED_PATCHED_SHA256 = "e5fbedb76142b69041ec827dc37726c4b8fb14b8fdc9f6923be4dc220ce32afe"
OLD_MAPPING = (
    '_STACKED_PARAM_NAME_MAPPING = (\n'
    '    ("attn.fused_wqa_wkv", ".attn.wq_a", 0),\n'
    '    ("attn.fused_wqa_wkv", ".attn.wkv", 1),\n'
    ')\n'
)
NEW_MAPPING = (
    '_STACKED_PARAM_NAME_MAPPING = (\n'
    '    ("attn.fused_wqa_wkv", ".attn.wq_a", 0),\n'
    '    ("attn.fused_wqa_wkv", ".attn.wkv", 1),\n'
    '    ("shared_experts.gate_up_proj", ".shared_experts.w1", 0),\n'
    '    ("shared_experts.gate_up_proj", ".shared_experts.w3", 1),\n'
    ')\n'
)


def patch_text(source: str) -> tuple[str, str]:
    old_count = source.count(OLD_MAPPING)
    new_count = source.count(NEW_MAPPING)
    if old_count == 1 and new_count == 0:
        updated = source.replace(OLD_MAPPING, NEW_MAPPING, 1)
        compile(updated, "dspark.py", "exec")
        return updated, "applied"
    if old_count == 0 and new_count == 1:
        return source, "skipped"
    return source, f"drift:old={old_count},new={new_count}"


def main(argv: list[str]) -> int:
    target = Path(argv[1]) if len(argv) == 2 else DEFAULT_TARGET
    if len(argv) > 2:
        print(f"usage: {argv[0]} [DSPARK.py]", file=sys.stderr)
        return 2
    if not target.is_file():
        print(f"[dspark-shared-expert-loader] missing target: {target}", file=sys.stderr)
        return 1
    source = target.read_text(encoding="utf-8")
    updated, status = patch_text(source)
    if status not in {"applied", "skipped"}:
        print(
            f"[dspark-shared-expert-loader] source drift; refusing to patch ({status})",
            file=sys.stderr,
        )
        return 1
    source_digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    expected_source = (
        EXPECTED_SOURCE_SHA256 if status == "applied" else EXPECTED_PATCHED_SHA256
    )
    if source_digest != expected_source:
        print(
            "[dspark-shared-expert-loader] source digest drift; "
            f"expected {expected_source}, got {source_digest}",
            file=sys.stderr,
        )
        return 1
    if status == "applied":
        target.write_text(updated, encoding="utf-8")
    digest = hashlib.sha256(updated.encode("utf-8")).hexdigest()
    if digest != EXPECTED_PATCHED_SHA256:
        print(
            "[dspark-shared-expert-loader] patched digest mismatch; "
            f"expected {EXPECTED_PATCHED_SHA256}, got {digest}",
            file=sys.stderr,
        )
        return 1
    print(f"[dspark-shared-expert-loader] {status}: {target} sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
