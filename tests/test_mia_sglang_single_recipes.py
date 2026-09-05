from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


class MiaSglangSingleRecipeTests(unittest.TestCase):
    def _capture(self, wrapper: Path, arguments: list[str]) -> list[str]:
        with tempfile.TemporaryDirectory() as directory:
            launcher = Path(directory) / "sglang" / "launch_server.py"; launcher.parent.mkdir()
            launcher.write_text("import json, os, sys\nopen(os.environ['CAPTURE'], 'w').write(json.dumps(sys.argv[1:]))")
            capture = Path(directory) / "arguments.json"
            result = subprocess.run([sys.executable, str(wrapper), *arguments], check=False, env={**os.environ, "PYTHONPATH": directory, "CAPTURE": str(capture)})
            self.assertEqual(result.returncode, 0)
            return json.loads(capture.read_text())

    def test_ling_profile_binds_two_model_slots_and_wrapper_mounts(self) -> None:
        recipe = load("recipes/ling-3-0-flash-dspark-sglang-single.json")
        self.assertEqual(len(recipe["models"]), 2)
        mounts = {item["mount"]["target"] for selection in recipe["models"] for item in selection["files"]}
        self.assertEqual(mounts, {"/models/target", "/models/draft"})
        captured = self._capture(ROOT / "adapters/ling/flash-dspark-single/sglang-serve", ["--model-path", "/models", "--served-model-name", "ling-3-0-flash", "--tensor-parallel-size", "1", "--context-length", "262144", "--mem-fraction-static", "0.80", "--host", "0.0.0.0", "--port", "30000"])
        self.assertEqual(captured[captured.index("--model-path") + 1], "/models/target")
        self.assertEqual(captured[captured.index("--speculative-draft-model-path") + 1], "/models/draft")
        self.assertIn("DSPARK", captured)

    def test_ling_profile_has_the_declared_admission_envelope(self) -> None:
        recipe = load("recipes/ling-3-0-flash-dspark-sglang-single.json")
        self.assertEqual(recipe["settings"]["context_tokens"]["value"], 262144)
        memory = recipe["topology"]["roles"][0]["resources"]["memory"]
        self.assertEqual(max(memory["startup_peak_bytes"], memory["steady_state_bytes"] + memory["runtime_growth_bytes"]) + memory["system_reserve_bytes"], 126_000_000_000)
        self.assertIn("262144 context", recipe["metadata"]["description"])

    def test_qwen27_profile_binds_external_dspark_draft(self) -> None:
        recipe = load("recipes/qwen3-8-27b-nvfp4-dspark-sglang-single.json")
        self.assertEqual(recipe["topology"]["node_count"], 1)
        self.assertEqual(len(recipe["models"]), 2)
        captured = self._capture(ROOT / "adapters/qwen/qwen38-27b-dspark-single/sglang-serve", ["--model-path", "/models", "--served-model-name", "qwen3-8-27b", "--tensor-parallel-size", "1", "--context-length", "262144", "--mem-fraction-static", "0.80", "--host", "0.0.0.0", "--port", "30000"])
        self.assertIn("DSPARK", captured)
        self.assertIn("--enable-torch-compile", captured)
        self.assertNotIn("--continuous-decode-steps", captured)

    def test_sglang_recipes_bind_immutable_runtime_arguments(self) -> None:
        for path in ("recipes/ling-3-0-flash-dspark-sglang-single.json", "recipes/qwen3-8-27b-nvfp4-dspark-sglang-single.json"):
            recipe = load(path)
            self.assertEqual(recipe["runtime"]["engine"], "sglang")
            self.assertTrue(recipe["runtime"]["entrypoint"])
            self.assertIn(recipe["execution"]["mode"], {"image", "build"})


if __name__ == "__main__":
    unittest.main()
