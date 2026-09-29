#!/usr/bin/env python3
"""CPU test: Engram secondary_weights removed; file-backed overlay anchors."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for _d in (HERE, ROOT / "overlay"):
    if (_d / "patch_engram_secondary.py").is_file():
        sys.path.insert(0, str(_d))
        break
from patch_engram_file import FOOTER, MARK as FILE_MARK, apply as apply_file  # noqa: E402
from patch_engram_secondary import (  # noqa: E402
    LLM_NEW,
    LLM_OLD,
    LOADER_NEW,
    LOADER_OLD,
    MARK,
    PREV_LLM_NEW,
    PREV_LOADER_NEW,
    VL_NEW,
    VL_OLD,
    apply_text,
    migrate_text,
)


def test_llm() -> None:
    src = "class DeepseekV41LLMForCausalLM:\n    def __init__(self):\n" + LLM_OLD + "        self.x = 1\n"
    out, status = apply_text(src, LLM_OLD, LLM_NEW)
    assert status == "applied", status
    assert MARK in out
    assert "self.secondary_weights" not in out
    out2, status2 = apply_text(out, LLM_OLD, LLM_NEW)
    assert status2 == "skipped", status2


def test_llm_migrates_old_secondary() -> None:
    src = "class X:\n" + PREV_LLM_NEW
    out, status = migrate_text(src, PREV_LLM_NEW, LLM_NEW)
    assert status == "migrated", status
    assert "self.secondary_weights" not in out
    assert MARK in out


def test_vl() -> None:
    src = "        self.language_model = DeepseekV41LLMForCausalLM()\n" + VL_OLD
    out, status = apply_text(src, VL_OLD, VL_NEW)
    assert status == "applied", status
    assert "self.secondary_weights" not in out


def test_loader() -> None:
    src = "        weights_iterator = safetensors_weights_iterator(hf_weights_files)\n" + LOADER_OLD
    out, status = apply_text(src, LOADER_OLD, LOADER_NEW)
    assert status == "applied", status
    assert "00047-of-00048" in out
    assert "return names_tensors" in out
    migrated, st = migrate_text(PREV_LOADER_NEW, PREV_LOADER_NEW, LOADER_NEW)
    assert st == "migrated"
    assert ".engram.embed." not in migrated or "if False" in migrated


def test_file_install_footer() -> None:
    src = "class ParallelEngramEmbedding(nn.Module):\n    pass\n"
    out, status = apply_file(src)
    assert status == "applied", status
    assert FILE_MARK in out
    assert "engram_file_backend.py" in out
    out2, status2 = apply_file(out)
    assert status2 == "skipped", status2
    assert FOOTER.strip() in out


def test_backend_does_not_del_closure_vars() -> None:
    for path in (
        ROOT / "overlay" / "engram_file_backend.py",
        HERE / "engram_file_backend.py",
    ):
        if path.is_file():
            text = path.read_text()
            break
    else:
        raise FileNotFoundError("engram_file_backend.py")
    import re

    assert "del orig_init" not in text
    assert "del cpu_offload, module" not in text
    # Any `del` that names a closure variable of _make_init/_make_lookup
    # (`module`, `orig_init`) turns it into a local and raises
    # UnboundLocalError on first use (the 16:48 boot died on first lookup).
    for m in re.finditer(r"^\s*del\s+([^\n#]+)", text, flags=re.M):
        names = {n.strip() for n in m.group(1).split(",")}
        assert not (names & {"module", "orig_init"}), m.group(0)
    assert 'device="cpu", pin_memory=True' in text or "device=\"cpu\", pin_memory=True" in text
    # lookup() must still drop the unused `background` flag, nothing else.
    assert "del background\n" in text


def test_missing_fails_closed() -> None:
    _, status = apply_text("class Foo:\n    pass\n", LLM_OLD, LLM_NEW)
    assert status.startswith("missing:"), status


if __name__ == "__main__":
    test_llm()
    test_llm_migrates_old_secondary()
    test_vl()
    test_loader()
    test_file_install_footer()
    test_backend_does_not_del_closure_vars()
    test_missing_fails_closed()
    print("test_engram_secondary: ok")
