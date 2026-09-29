#!/usr/bin/env python3
"""CPU test: the memory-log footer applies once and compiles."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for _d in (HERE, ROOT / "overlay"):
    if (_d / "patch_memory_log.py").is_file():
        sys.path.insert(0, str(_d))
        break
from patch_memory_log import FOOTER, MARK, apply  # noqa: E402

SRC = "import gc\nimport os\n\nfrom vllm.logger import init_logger\n\nlogger = init_logger(__name__)\n\n\nclass Worker(object):\n    def load_model(self, *, load_dummy_weights=False):\n        return 1\n\n    def determine_available_memory(self):\n        return 2\n"


def test_apply_once() -> None:
    out, status = apply(SRC)
    assert status == "applied", status
    assert MARK in out and FOOTER.strip() in out
    compile(out, "gpu_worker.py", "exec")
    out2, status2 = apply(out)
    assert status2 == "skipped", status2
    assert out2 == out


def test_wraps_phase_methods() -> None:
    out, _ = apply(SRC)
    ns: dict = {}
    # Stub vllm.logger so the footer's module can execute standalone.
    import types

    vllm_pkg = types.ModuleType("vllm")
    vllm_logger = types.ModuleType("vllm.logger")
    vllm_logger.init_logger = lambda name: __import__("logging").getLogger(name)
    sys.modules.setdefault("vllm", vllm_pkg)
    sys.modules["vllm.logger"] = vllm_logger
    exec(compile(out, "gpu_worker.py", "exec"), ns)
    worker = ns["Worker"]()
    assert worker.load_model() == 1
    assert worker.determine_available_memory() == 2
    assert getattr(ns["Worker"].load_model, "_dsv41_mem_wrapped", False)


def test_missing_fails_closed() -> None:
    _, status = apply("class Foo:\n    pass\n")
    assert status.startswith("missing:"), status


if __name__ == "__main__":
    test_apply_once()
    test_wraps_phase_methods()
    test_missing_fails_closed()
    print("test_memory_log: ok")
