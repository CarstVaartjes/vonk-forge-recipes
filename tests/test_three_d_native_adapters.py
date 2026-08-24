from __future__ import annotations

import ast
import hashlib
import importlib.machinery
import importlib.util
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CATALOG_TOOL = ROOT / "tools/build-catalog-index"
LOADER = importlib.machinery.SourceFileLoader("three_d_catalog_tool", str(CATALOG_TOOL))
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
assert SPEC is not None
CATALOG = importlib.util.module_from_spec(SPEC)
sys.modules[LOADER.name] = CATALOG
LOADER.exec_module(CATALOG)

CASES = {
    "skintokens-pytorch-single": (
        "adapters/three-d/skintokens",
        "runtime-distributions/skintokens-tokenrig-arm64.json",
    ),
    "triposg-pytorch-single": (
        "adapters/three-d/triposg",
        "runtime-distributions/triposg-native-arm64.json",
    ),
    "hunyuan3d-omni-pytorch-single": (
        "adapters/three-d/hunyuan3d-omni",
        "runtime-distributions/hunyuan3d-omni-native-arm64.json",
    ),
}


class NativeThreeDAdapterTests(unittest.TestCase):
    def test_recipes_bind_exact_native_context_and_runtime(self) -> None:
        for slug, (context_name, runtime_name) in CASES.items():
            with self.subTest(slug=slug):
                recipe = json.loads((ROOT / f"recipes/{slug}.json").read_text())
                runtime = json.loads((ROOT / runtime_name).read_text())
                context = ROOT / context_name
                archive, _metadata, digest = CATALOG.source_bundle(context)
                self.assertEqual(recipe["build"]["context"]["path"], context_name)
                self.assertEqual(recipe["build"]["context"]["sha256"], digest)
                self.assertEqual(recipe["build"]["context"]["expected_bytes"], len(archive))
                self.assertEqual(
                    recipe["runtime"]["distribution"]["content_sha256"],
                    hashlib.sha256(CATALOG.canonical(runtime)).hexdigest(),
                )
                tags = set(recipe["metadata"]["tags"])
                self.assertIn("candidate", tags)
                self.assertFalse(tags.intersection({"metadata-only", "non-executable", "integration-required"}))

    def test_runtime_is_offline_and_source_authorities_are_immutable(self) -> None:
        for _slug, (_context_name, runtime_name) in CASES.items():
            with self.subTest(runtime=runtime_name):
                runtime = json.loads((ROOT / runtime_name).read_text())
                self.assertTrue(runtime["build"]["offline_after_installation"])
                self.assertEqual(runtime["security"]["network_mode"], "none")
                self.assertRegex(runtime["source"]["revision"], r"^[a-f0-9]{40}$")
                self.assertRegex(runtime["source"]["archive_sha256"], r"^[a-f0-9]{64}$")
                self.assertIn("@sha256:", runtime["image"])

    def test_entrypoints_are_syntax_valid_and_have_no_runtime_downloads(self) -> None:
        forbidden = ("snapshot_download", "hf_hub_download", "requests.get", "requests.post", "urlopen(", "curl ")
        for _slug, (context_name, _runtime_name) in CASES.items():
            with self.subTest(context=context_name):
                source = (ROOT / context_name / "run.py").read_text()
                ast.parse(source)
                for marker in forbidden:
                    self.assertNotIn(marker, source)
                self.assertIn("output.glb", source)
                self.assertIn("b\"glTF\"", source)


if __name__ == "__main__":
    unittest.main()
