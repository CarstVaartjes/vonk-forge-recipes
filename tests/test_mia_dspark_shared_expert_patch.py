from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
SCRIPT = (
    ROOT
    / "adapters/deepseek/mia-vllm/patches/hotfix-dspark-shared-expert-loader.py"
)
LOADER = importlib.machinery.SourceFileLoader("dspark_shared_expert_patch", str(SCRIPT))
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
assert SPEC is not None
patch = importlib.util.module_from_spec(SPEC)
LOADER.exec_module(patch)


class SharedExpertPatchTests(unittest.TestCase):
    def test_adds_both_shared_expert_shards_and_is_idempotent(self) -> None:
        source = f"before\n{patch.OLD_MAPPING}\nafter\n"

        updated, status = patch.patch_text(source)
        repeated, repeated_status = patch.patch_text(updated)

        self.assertEqual(status, "applied")
        self.assertEqual(repeated_status, "skipped")
        self.assertEqual(repeated, updated)
        self.assertIn(
            '    ("shared_experts.gate_up_proj", ".shared_experts.w1", 0),\n',
            updated,
        )
        self.assertIn(
            '    ("shared_experts.gate_up_proj", ".shared_experts.w3", 1),\n',
            updated,
        )

    def test_fails_closed_on_ambiguous_or_changed_mapping(self) -> None:
        duplicate = patch.OLD_MAPPING + patch.OLD_MAPPING
        changed = patch.OLD_MAPPING.replace("fused_wqa_wkv", "different")

        self.assertEqual(patch.patch_text(duplicate)[1], "drift:old=2,new=0")
        self.assertEqual(patch.patch_text(changed)[1], "drift:old=0,new=0")


if __name__ == "__main__":
    unittest.main()
