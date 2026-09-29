#!/usr/bin/env python3
"""Put exl3 first in ModelConfig's quantization override list.

vLLM 0.30+ may already register EXL3; this is idempotent and tries several
paths because the deepseekv41-flash image is not the glm53-flash tree.
"""
from __future__ import annotations

from pathlib import Path
import sys

CANDIDATES = [
    Path("/usr/local/lib/python3.12/dist-packages/vllm/config/model.py"),
    Path("/usr/local/lib/python3.12/dist-packages/vllm/config.py"),
    Path("/usr/local/lib/python3.13/dist-packages/vllm/config/model.py"),
    Path("/usr/local/lib/python3.13/dist-packages/vllm/config.py"),
]


def _site_vllm() -> Path | None:
    try:
        import vllm

        return Path(vllm.__file__).resolve().parent
    except Exception:
        return None


def main() -> int:
    paths = list(CANDIDATES)
    site = _site_vllm()
    if site is not None:
        paths = [site / "config/model.py", site / "config.py", *paths]
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            unique.append(p)

    needle_old = '            overrides = [\n                "auto_gptq",\n'
    needle_new = '            overrides = [\n                "exl3",\n                "auto_gptq",\n'
    already = '            overrides = [\n                "exl3",\n'

    for p in unique:
        if not p.is_file():
            continue
        text = p.read_text()
        if already in text or '"exl3"' in text and "overrides = [" in text:
            print(f"exl3 already in ModelConfig overrides ({p})")
            return 0
        if text.count(needle_old) == 1:
            p.write_text(text.replace(needle_old, needle_new))
            print(f"exl3 added to ModelConfig overrides ({p})")
            return 0
        # Looser: insert before auto_gptq in an overrides list.
        if '"auto_gptq"' in text and "overrides" in text and '"exl3"' not in text:
            patched = text.replace(
                '"auto_gptq"',
                '"exl3",\n                "auto_gptq"',
                1,
            )
            if patched != text:
                p.write_text(patched)
                print(f"exl3 inserted before auto_gptq ({p})")
                return 0
        print(f"overrides list target missing or not unique in {p}", file=sys.stderr)

    print("no ModelConfig overrides file patched (exl3 may already be registered)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
