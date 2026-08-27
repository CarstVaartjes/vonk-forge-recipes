from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(path: str) -> dict[str, object]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


class MiaSglangSingleRecipeTests(unittest.TestCase):
    def _capture(self, wrapper: Path, arguments: list[str]) -> list[str]:
        with tempfile.TemporaryDirectory() as directory:
            launcher = Path(directory) / "sglang" / "launch_server.py"
            launcher.parent.mkdir()
            launcher.write_text(
                "import json, os, sys\n"
                "open(os.environ['CAPTURE'], 'w').write(json.dumps(sys.argv[1:]))\n",
                encoding="utf-8",
            )
            capture = Path(directory) / "arguments.json"
            result = subprocess.run(
                [sys.executable, str(wrapper), *arguments],
                check=False,
                env={**os.environ, "PYTHONPATH": directory, "CAPTURE": str(capture)},
            )
            self.assertEqual(result.returncode, 0)
            return json.loads(capture.read_text(encoding="utf-8"))

    def test_ling_profile_binds_external_dspark_draft(self) -> None:
        recipe = load("recipes/ling-3-0-flash-dspark-sglang-single.json")
        self.assertEqual(
            [artifact["id"] for artifact in recipe["artifacts"]],
            ["target", "draft"],
        )
        captured = self._capture(
            ROOT / "adapters/ling/flash-dspark-single/sglang-serve",
            [
                "--model-path", "/models", "--served-model-name", "ling-3-0-flash",
                "--tensor-parallel-size", "1", "--context-length", "262144",
                "--mem-fraction-static", "0.75", "--host", "0.0.0.0", "--port", "30000",
            ],
        )
        self.assertEqual(captured[captured.index("--model-path") + 1], "/models/target")
        self.assertEqual(
            captured[captured.index("--speculative-draft-model-path") + 1],
            "/models/draft",
        )
        self.assertIn("DSPARK", captured)
        self.assertIn("--enable-linear-replayssm-spec", captured)

    def test_qwen27_profile_binds_external_dspark_draft(self) -> None:
        recipe = load("recipes/qwen3-8-27b-nvfp4-dspark-sglang-single.json")
        self.assertEqual(recipe["topology"]["node_count"], 1)
        captured = self._capture(
            ROOT / "adapters/qwen/qwen38-27b-dspark-single/sglang-serve",
            [
                "--model-path", "/models", "--served-model-name", "qwen3-8-27b",
                "--tensor-parallel-size", "1", "--context-length", "262144",
                "--mem-fraction-static", "0.90", "--host", "0.0.0.0", "--port", "30000",
            ],
        )
        self.assertEqual(captured[captured.index("--model-path") + 1], "/models/target")
        self.assertEqual(
            captured[captured.index("--speculative-draft-model-path") + 1],
            "/models/draft",
        )
        self.assertIn("EAGLE", captured)
        self.assertIn("--disable-prefill-cuda-graph", captured)

    def test_runtime_images_and_sources_are_immutable(self) -> None:
        for path, revision in (
            ("runtime-distributions/sglang-ling-3-0-flash-0e5e40d8-arm64.json", "0e5e40d8f1460976cd7190ae479c210f0642c120"),
            ("runtime-distributions/sglang-qwen38-27b-c4271c3f-arm64.json", "c4271c3fe1262fc2adbd162c33b25de5255251c5"),
        ):
            runtime = load(path)
            self.assertEqual(runtime["source"]["revision"], revision)
            self.assertIn("@sha256:", runtime["image"])
            self.assertTrue(runtime["build"]["offline_after_installation"])


if __name__ == "__main__":
    unittest.main()
