#!/usr/bin/env python3
"""The pinned-staging patch: applies once to the real vLLM loaders (when vLLM is
importable, i.e. inside the image build) and its helper is copy_-equivalent."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for _d in (HERE, ROOT / "overlay"):
    if (_d / "patch_h2d_stage.py").is_file():
        sys.path.insert(0, str(_d))
        break
from patch_h2d_stage import HELPER, JOBS, MARK, SHIM, apply  # noqa: E402


def _vllm_root() -> Path | None:
    try:
        import vllm

        return Path(vllm.__file__).resolve().parent
    except Exception:
        return None


def test_real_files_apply_once() -> None:
    root = _vllm_root()
    if root is None:
        print("test_h2d_stage: vllm not importable here; real-file check runs in the image build")
        return
    for rel, edits, footer in JOBS:
        text = (root / rel).read_text()
        out, status = apply(text, edits, footer)
        assert status in ("applied", "skipped"), (rel, status)
        if status == "applied":
            for old, new, count in edits:
                assert old not in out, (rel, old)
                assert out.count(new) == count, (rel, new, out.count(new))
        assert MARK in out
        compile(out, rel, "exec")
        out2, status2 = apply(out, edits, footer)
        assert status2 == "skipped" and out2 == out, (rel, status2)


def test_drift_is_loud() -> None:
    rel, edits, footer = JOBS[0]
    src = "def default_weight_loader(param, loaded_weight):\n    param.data.copy_(loaded_weight)\n"
    _, status = apply(src, edits, footer)
    assert status.startswith("missing:"), status


def test_helper_semantics_cpu() -> None:
    try:
        import torch
    except ImportError:
        print("test_h2d_stage: torch not importable here; helper check runs in the image build")
        return

    ns: dict = {"torch": torch}
    exec(HELPER, ns)
    h2d = ns["dsv41_h2d_copy"]
    src = torch.arange(12, dtype=torch.int16).view(3, 4)[:, :2]  # non-contiguous
    dst = torch.zeros(3, 2, dtype=torch.int16)
    h2d(dst, src)  # cpu->cpu falls through to copy_
    assert torch.equal(dst, src)
    assert not ns["_DSV41_PIN_POOL"]  # no pinned buffer on the CPU path
    assert "dsv41_h2d_copy" in SHIM
    if torch.cuda.is_available():
        d = torch.zeros(3, 2, dtype=torch.int16, device="cuda")
        h2d(d, src)
        assert torch.equal(d.cpu(), src)
        assert ns["_DSV41_PIN_POOL"][torch.int16].is_pinned()


if __name__ == "__main__":
    test_real_files_apply_once()
    test_drift_is_loud()
    test_helper_semantics_cpu()
    print("test_h2d_stage: ok")
