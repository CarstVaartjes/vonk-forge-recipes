#!/usr/bin/env python3
"""Install file-backed Engram into vLLM's engram.py (worker processes import this)."""
from __future__ import annotations

from pathlib import Path
import sys

MARK = "dsv41-engram-file-install"

FOOTER = '''
# [dsv41-engram-file-install]
def _dsv41_install_file_engram() -> None:
    import importlib.util
    import sys
    from pathlib import Path
    path = Path("/opt/dsv41/engram_file_backend.py")
    if not path.is_file():
        raise RuntimeError(f"missing {path} (file-backed Engram)")
    spec = importlib.util.spec_from_file_location("dsv41_engram_file_backend", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.install(sys.modules[__name__])
_dsv41_install_file_engram()
'''


def apply(text: str) -> tuple[str, str]:
    if MARK in text:
        return text, "skipped"
    if "class ParallelEngramEmbedding" not in text:
        return text, "missing:ParallelEngramEmbedding"
    if not text.endswith("\n"):
        text += "\n"
    return text + FOOTER, "applied"


def main() -> int:
    try:
        import vllm

        root = Path(vllm.__file__).resolve().parent
    except Exception as exc:
        print(f"WARN: vllm not importable; skip engram file install ({exc})", file=sys.stderr)
        return 0
    path = root / "models/deepseek_v4_1/common/engram.py"
    if not path.is_file():
        print(f"WARN: {path} missing", file=sys.stderr)
        return 1
    out, status = apply(path.read_text())
    if status == "applied":
        path.write_text(out)
        print(f"engram file install: applied ({path})")
        return 0
    if status == "skipped":
        print("engram file install: already applied")
        return 0
    print(f"WARN: engram file install: {status}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
