#!/usr/bin/env python3
"""Static checks for the pinned Anemll DSpark loader build gate."""

from __future__ import annotations

import hashlib
import json
import runpy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "adapters/deepseek/mia-vllm"
RECIPE = ROOT / "recipes/deepseek-v4-flash-0731-mia-dual.json"
PATCH_BUNDLE = ROOT / "patch-bundles/mia-deepseek-v4-flash-0731.json"
MIA_REVISION = "03d4776fa8977ed9f7fd077c4776ba164c7127b6"
MIA_ARCHIVE_SHA256 = "578e4a84b900794d23c91cd64d263f9196a86fb612b46c6b8c4232031e2040c2"


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

    def test_xgrammar_termination_fix_is_source_exact_and_verified(self) -> None:
        patch_path = ADAPTER / "patches/hotfix-vllm-issue136-xgrammar-termination.py"
        patch_bytes = patch_path.read_bytes()
        patch_bundle = json.loads(PATCH_BUNDLE.read_text(encoding="utf-8"))
        patch_contract = next(
            item
            for item in patch_bundle["patches"]
            if item["path"].endswith("issue136-xgrammar-termination.py")
        )
        apply_script = (ADAPTER / "apply-build-patches.py").read_text(encoding="utf-8")
        dockerfile = (ADAPTER / "Dockerfile").read_text(encoding="utf-8")

        self.assertEqual(
            hashlib.sha256(patch_bytes).hexdigest(), patch_contract["sha256"]
        )
        self.assertIn(
            'run("hotfix-vllm-issue136-xgrammar-termination.py")', apply_script
        )
        self.assertIn(
            "hotfix-vllm-issue136-xgrammar-termination.py --status", dockerfile
        )

    def test_xgrammar_stock_to_patched_apply_executes_successfully(self) -> None:
        patch_path = (
            ADAPTER / "patches/hotfix-vllm-issue136-xgrammar-termination.py"
        )
        module = runpy.run_path(str(patch_path), run_name="issue136_apply_test")
        old_region = module["OLD_REGION"]
        new_region = module["NEW_REGION"]
        stock = b"class SyntheticGrammar:\n" + old_region + b"\n"
        patched = stock.replace(old_region, new_region)
        versions = {
            "vllm": module["EXPECTED_VLLM_VERSION"],
            "xgrammar": module["EXPECTED_XGRAMMAR_VERSION"],
        }
        exact_fixture_constants = {
            "STOCK_SIZE": len(stock),
            "STOCK_SHA256": hashlib.sha256(stock).hexdigest(),
            "PATCHED_SIZE": len(patched),
            "PATCHED_SHA256": hashlib.sha256(patched).hexdigest(),
        }

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "backend_xgrammar.py"
            target.write_bytes(stock)
            target.chmod(0o640)
            with patch.dict(module["apply"].__globals__, exact_fixture_constants):
                result = module["apply"](target, versions.__getitem__)

            self.assertEqual(result.outcome, "applied")
            self.assertEqual(result.pre_sha256, exact_fixture_constants["STOCK_SHA256"])
            self.assertEqual(
                result.post_sha256, exact_fixture_constants["PATCHED_SHA256"]
            )
            self.assertEqual(result.vllm_version, versions["vllm"])
            self.assertEqual(result.xgrammar_version, versions["xgrammar"])
            self.assertEqual(target.read_bytes(), patched)
            self.assertEqual(target.stat().st_mode & 0o777, 0o640)

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
