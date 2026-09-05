from __future__ import annotations

import json
import runpy
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANARY_SLUG = "ltx-2-5-22b-distilled-fp8-cast-diffusers-single"
BF16_SLUG = "ltx-2-5-22b-distilled-bf16-diffusers-single"
CANARY_ADAPTER = ROOT / "adapters/video/ltx25-diffusers-fp8-canary"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def adapter_module():
    path = CANARY_ADAPTER / "run.py"
    module = types.ModuleType("ltx25_fp8_canary_adapter"); module.__file__ = str(path)
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)
    return module


class Ltx25Fp8CanaryTests(unittest.TestCase):
    def test_canary_reuses_exact_model_selection_and_has_bounded_resources(self) -> None:
        canary = load(ROOT / "recipes" / f"{CANARY_SLUG}.json")
        bf16 = load(ROOT / "recipes" / f"{BF16_SLUG}.json")
        self.assertEqual(canary["models"], bf16["models"])
        self.assertEqual(canary["topology"]["roles"][0]["resources"]["memory"]["startup_peak_bytes"], 110_000_000_000)
        self.assertLessEqual(max(canary["topology"]["roles"][0]["resources"]["memory"]["startup_peak_bytes"], 120_000_000_000), 126_946_283_520)
        self.assertIn("canary", canary["metadata"]["tags"])

    def test_canary_profile_cannot_fall_back_to_bf16(self) -> None:
        adapter = adapter_module()
        self.assertEqual(adapter.PROFILES, {"fp8-cast-sequential-offload"})
        self.assertEqual(adapter._profile(None), "fp8-cast-sequential-offload")
        with self.assertRaises(ValueError): adapter._profile("bf16-model-offload")
        source = (CANARY_ADAPTER / "run.py").read_text(encoding="utf-8")
        self.assertIn("storage_dtype=torch.float8_e4m3fn", source)

    def test_source_bundle_path_and_release_bind_current_recipe(self) -> None:
        recipe = load(ROOT / "recipes" / f"{CANARY_SLUG}.json")
        archive, _, bundle_digest = runpy.run_path(str(ROOT / "tools/build-catalog-index"))["source_bundle"](CANARY_ADAPTER)
        self.assertEqual(recipe["execution"]["build"]["context"]["path"], "adapters/video/ltx25-diffusers-fp8-canary")
        self.assertTrue(bundle_digest and len(archive) > 0)
        release = load(ROOT / "recipe-releases" / f"{CANARY_SLUG}.json")
        self.assertEqual(release["history"][0]["recipe_content_sha256"], __import__("hashlib").sha256(json.dumps(recipe, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest())


if __name__ == "__main__":
    unittest.main()
