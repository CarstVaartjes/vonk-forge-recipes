#!/usr/bin/env python3
"""Static checks for the pinned Anemll DSpark loader build gate."""

from __future__ import annotations

import runpy
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "adapters/deepseek/mia-vllm"


class MiaDSparkRuntimeContractTest(unittest.TestCase):
    def test_docker_build_runs_loader_verifier_before_patching(self) -> None:
        dockerfile = (ADAPTER / "Dockerfile").read_text(encoding="utf-8")
        verify_position = dockerfile.index("verify-dspark-runtime.py")
        patch_position = dockerfile.index("apply-build-patches.py", verify_position)
        self.assertLess(verify_position, patch_position)

    def test_verifier_accepts_the_exact_required_mapping(self) -> None:
        verifier = ADAPTER / "verify-dspark-runtime.py"
        source = "\n".join(
            (
                '("gate_up_proj", "w1", 0),',
                '("gate_up_proj", "w3", 1),',
                'is_layer_param = name.startswith("model.layers.")',
                "name = name.replace(weight_name, param_name)",
            )
        ).encode()
        fake_target = MagicMock()
        fake_target.is_file.return_value = True
        fake_target.read_bytes.return_value = source
        with (
            patch("pathlib.Path", return_value=fake_target),
            patch("hashlib.sha256") as digest,
        ):
            digest.return_value.hexdigest.return_value = (
                "efe33c32d37ed7f26d869d94626f1415906d31218ec0ee44d79bb2b815b8cf39"
            )
            runpy.run_path(str(verifier))


if __name__ == "__main__":
    unittest.main()
