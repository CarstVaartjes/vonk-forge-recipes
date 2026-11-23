#!/usr/bin/env python3
"""Regression checks for Mia's fail-closed issue 133 Triton patch."""

from __future__ import annotations

import importlib.util
import unittest

from repo_root import ROOT

PATCH = (
    ROOT
    / "adapters/deepseek/mia-vllm-vision/patches/hotfix-dsv4-issue133-triton-specialization.py"
)


def load_patch():
    spec = importlib.util.spec_from_file_location("mia_issue133_patch", PATCH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {PATCH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MiaIssue133PatchTest(unittest.TestCase):
    def test_applies_exactly_once(self) -> None:
        patch = load_patch()
        source = "from triton import language as tl\n" + patch.OLD + "    pass\n"
        updated, status = patch.patch_text(source)
        assert status == "applied"
        assert patch.MARK in updated
        assert "do_not_specialize_on_alignment" in updated
        assert "block_size: tl.constexpr" in updated

        second, second_status = patch.patch_text(updated)
        assert second_status == "skipped"
        assert second == updated

    def test_refuses_source_drift(self) -> None:
        patch = load_patch()
        source, status = patch.patch_text("def unrelated():\n    pass\n")
        assert status == "drift:old=0"
        assert patch.MARK not in source


if __name__ == "__main__":
    unittest.main()
