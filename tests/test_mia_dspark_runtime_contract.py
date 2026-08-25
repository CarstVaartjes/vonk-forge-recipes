#!/usr/bin/env python3
"""Static checks for the pinned Anemll DSpark loader build gate."""

from __future__ import annotations

import json
import runpy
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "adapters/deepseek/mia-vllm"
RECIPE = ROOT / "recipes/deepseek-v4-flash-0731-mia-dual.json"
PATCH_BUNDLE = ROOT / "patch-bundles/mia-deepseek-v4-flash-0731.json"
MIA_REVISION = "70a7cc4b49664e83b51e9b73c0ed41db18ac3190"
MIA_ARCHIVE_SHA256 = "7d17f73ca4f444f8518d8535a237e7f3c7b3c3a8d6f0f5a36bbabdfcfe2b5b02"


class MiaDSparkRuntimeContractTest(unittest.TestCase):
    def test_recipe_pins_latest_reviewed_mia_source(self) -> None:
        recipe = json.loads(RECIPE.read_text(encoding="utf-8"))
        patch_bundle = json.loads(PATCH_BUNDLE.read_text(encoding="utf-8"))
        dockerfile = (ADAPTER / "Dockerfile").read_text(encoding="utf-8")

        self.assertEqual(patch_bundle["source"]["revision"], MIA_REVISION)
        self.assertEqual(
            patch_bundle["source"]["archive_sha256"], MIA_ARCHIVE_SHA256
        )
        self.assertTrue(recipe["provenance"]["source_reference"].endswith(MIA_REVISION))
        self.assertIn(MIA_REVISION, dockerfile)
        self.assertIn(MIA_ARCHIVE_SHA256, dockerfile)

    def test_recipe_persists_compile_caches_and_nccl_diagnostics(self) -> None:
        recipe = json.loads(RECIPE.read_text(encoding="utf-8"))
        environment = {
            item["name"]: item["value"]
            for item in recipe["runtime"]["environment"]
        }

        self.assertEqual(
            environment["B12X_CUTE_COMPILE_CACHE_DIR"],
            "/outputs/cache/b12x-cute-compile",
        )
        self.assertEqual(environment["TORCH_FR_BUFFER_SIZE"], "2000")
        self.assertEqual(environment["TORCH_NCCL_DUMP_ON_TIMEOUT"], "1")
        self.assertEqual(environment["TORCH_NCCL_ENABLE_MONITORING"], "1")
        self.assertTrue(environment["TORCH_FR_DUMP_TEMP_FILE"].startswith("/outputs/"))
        self.assertTrue(
            environment["TORCH_NCCL_DEBUG_INFO_PIPE_FILE"].startswith("/outputs/")
        )

        wrapper = (ADAPTER / "vllm-wrapper.py").read_text(encoding="utf-8")
        self.assertIn("/outputs/cache/b12x-cute-compile", wrapper)
        self.assertIn("/outputs/cache/nccl-fr", wrapper)

    def test_sampling_shape_canary_remains_qualification_evidence(self) -> None:
        recipe = json.loads(RECIPE.read_text(encoding="utf-8"))
        benchmarks = {
            benchmark["name"]: benchmark["configuration"]
            for benchmark in recipe["validation"]["benchmarks"]
        }

        self.assertEqual(benchmarks["sampler-shape-canary"]["top_k"], 40)
        self.assertEqual(benchmarks["sampler-shape-canary"]["top_p"], "0.9")
        self.assertIn("candidate", recipe["metadata"]["tags"])

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
