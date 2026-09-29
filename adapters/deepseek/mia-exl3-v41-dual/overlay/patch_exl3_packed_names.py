#!/usr/bin/env python3
"""Teach vLLM packed-param maps that this checkpoint uses .mul1, not .mcg.

Stock EXL3 mappings often list (trellis, suh, svh, mcg). DeepSeek-V4.1-Flash
EXL3 2.9bpw stores the codebook marker as mul1 (0x83D6B12D). Without this,
expert/linear loaders skip every .mul1 tensor.
"""
from __future__ import annotations

from pathlib import Path
import sys


def _vllm_root() -> Path:
    import vllm

    return Path(vllm.__file__).resolve().parent


REPLACEMENTS = (
    ('("trellis", "suh", "svh", "mcg")', '("trellis", "suh", "svh", "mcg", "mul1")'),
    ("('trellis', 'suh', 'svh', 'mcg')", "('trellis', 'suh', 'svh', 'mcg', 'mul1')"),
    (
        '["trellis", "suh", "svh", "mcg"]',
        '["trellis", "suh", "svh", "mcg", "mul1"]',
    ),
    (
        "['trellis', 'suh', 'svh', 'mcg']",
        "['trellis', 'suh', 'svh', 'mcg', 'mul1']",
    ),
)


def patch_text(text: str) -> tuple[str, int]:
    hits = 0
    for old, new in REPLACEMENTS:
        if old in text and new not in text:
            n = text.count(old)
            text = text.replace(old, new)
            hits += n
    return text, hits


def main() -> int:
    root = _vllm_root()
    changed = 0
    scanned = 0
    for path in root.rglob("*.py"):
        rel = str(path.relative_to(root))
        if any(x in rel for x in ("/tests/", "/third_party/", "__pycache__")):
            continue
        text = path.read_text(errors="ignore")
        if "mcg" not in text or "trellis" not in text:
            continue
        scanned += 1
        new, hits = patch_text(text)
        if hits and new != text:
            path.write_text(new)
            changed += 1
            print(f"packed-suffix mul1: {rel} ({hits} edits)")
    print(f"packed-suffix scan: files_with_mcg_trellis={scanned} patched={changed}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"patch_exl3_packed_names skipped: {exc}", file=sys.stderr)
        raise SystemExit(0)
