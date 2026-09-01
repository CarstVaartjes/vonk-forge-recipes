from __future__ import annotations

import hashlib
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
            recipe["dependencies"],
            [
                {
                    "kind": "model-version",
                    "publisher": "inclusionai",
                    "slug": "ling-3-0-flash-dspark-8e5d9988",
                    "content_sha256": "8785c5e09b6f5ccda7b780a94926650593563f59cf640acf7ab7f83f3b799343",
                }
            ],
        )
        self.assertEqual(
            [artifact["id"] for artifact in recipe["artifacts"]],
            ["target", "draft"],
        )
        captured = self._capture(
            ROOT / "adapters/ling/flash-dspark-single/sglang-serve",
            [
                "--model-path", "/models", "--served-model-name", "ling-3-0-flash",
                "--tensor-parallel-size", "1", "--context-length", "262144",
                "--mem-fraction-static", "0.80", "--host", "0.0.0.0", "--port", "30000",
            ],
        )
        self.assertEqual(captured[captured.index("--model-path") + 1], "/models/target")
        self.assertEqual(
            captured[captured.index("--speculative-draft-model-path") + 1],
            "/models/draft",
        )
        self.assertIn("DSPARK", captured)
        self.assertIn("--enable-linear-replayssm-spec", captured)

    def test_ling_full_context_profile_has_a_126_gb_admission_envelope(self) -> None:
        recipe = load("recipes/ling-3-0-flash-dspark-sglang-single.json")
        arguments = {
            argument["name"]: argument["value"]
            for argument in recipe["runtime"]["arguments"]
        }
        self.assertEqual(arguments["context-length"], 262144)
        self.assertEqual(arguments["mem-fraction-static"], "0.80")

        memory = recipe["topology"]["roles"][0]["resources"]["memory"]
        self.assertEqual(memory["startup_peak_bytes"], 119_000_000_000)
        self.assertEqual(memory["steady_state_bytes"], 114_000_000_000)
        self.assertEqual(memory["runtime_growth_bytes"], 5_000_000_000)
        self.assertEqual(memory["system_reserve_bytes"], 7_000_000_000)
        workload_peak = max(
            memory["startup_peak_bytes"],
            memory["steady_state_bytes"] + memory["runtime_growth_bytes"],
        )
        self.assertEqual(workload_peak + memory["system_reserve_bytes"], 126_000_000_000)

        description = recipe["metadata"]["description"]
        for evidence in (
            "262144 context",
            "328898-token KV pool",
            "12.5 GB of KV memory",
            "22.0 GB free GPU-side memory",
            "20 GB host MemAvailable",
            "119 GB startup",
            "114 GB steady state",
            "5 GB runtime growth",
            "7 GB system reserve",
            "126 GB admission envelope",
        ):
            self.assertIn(evidence, description)
        self.assertEqual(
            recipe["provenance"]["source_reference"],
            "https://github.com/MiaAI-Lab/Ling-3.0-Flash-SGLang-DSpark-DGX-Spark/blob/ca840cb8d032353e24648aeee06312b0938348f6/README.md",
        )

        release = load("recipe-releases/ling-3-0-flash-dspark-sglang-single.json")
        latest = release["history"][0]
        self.assertEqual(latest["version"], "1.0.3")
        self.assertEqual(latest["upgrade_effect"], "metadata-only")
        recipe_digest = hashlib.sha256(
            json.dumps(
                recipe,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode()
        ).hexdigest()
        self.assertEqual(latest["recipe_content_sha256"], recipe_digest)
        details = next(
            change["details"]
            for entry in release["history"]
            for change in entry.get("changes", [])
            if "details" in change and "328898-token" in change["details"]
        )
        self.assertIn("262144", details)
        self.assertIn("328898-token", details)
        self.assertEqual(
            next(
                change["references"]
                for entry in release["history"]
                for change in entry.get("changes", [])
                if change.get("references")
                and "328898-token" in change.get("details", "")
            ),
            [recipe["provenance"]["source_reference"]],
        )

    def test_ling_target_ledger_uses_the_exact_int4_revision(self) -> None:
        target_set = load("model-targets/language.json")
        target = next(
            item
            for item in target_set["targets"]
            if item["model"] == "Ling 3.0 Flash"
        )
        self.assertEqual(
            target["source"],
            "https://huggingface.co/inclusionAI/Ling-3.0-flash-int4/tree/7a27e9eb8179b2c2eb71eb214f0dab14ec6a63f2",
        )
        self.assertEqual(target["version"], "INT4 7a27e9eb + DSpark 8e5d9988")

    def test_qwen27_profile_binds_external_dspark_draft(self) -> None:
        recipe = load("recipes/qwen3-8-27b-nvfp4-dspark-sglang-single.json")
        self.assertEqual(recipe["topology"]["node_count"], 1)
        self.assertEqual(
            recipe["dependencies"],
            [
                {
                    "kind": "model-version",
                    "publisher": "radixark",
                    "slug": "qwen3-8-27b-dspark-b3c99101",
                    "content_sha256": "03d09c5dd8d95d901da360644010ad546521625eef37a26df26cc3624a1f8937",
                }
            ],
        )
        draft = next(
            artifact for artifact in recipe["artifacts"] if artifact["id"] == "draft"
        )
        self.assertEqual(
            draft["revision"],
            "b3c9910194dc7ae53fea3d95a1959b654f495416",
        )
        self.assertEqual(draft["download_bytes"], 3714763499)
        captured = self._capture(
            ROOT / "adapters/qwen/qwen38-27b-dspark-single/sglang-serve",
            [
                "--model-path", "/models", "--served-model-name", "qwen3-8-27b",
                "--tensor-parallel-size", "1", "--context-length", "262144",
                "--mem-fraction-static", "0.80", "--host", "0.0.0.0", "--port", "30000",
            ],
        )
        self.assertEqual(captured[captured.index("--model-path") + 1], "/models/target")
        self.assertEqual(
            captured[captured.index("--speculative-draft-model-path") + 1],
            "/models/draft",
        )
        self.assertEqual(
            captured[captured.index("--speculative-algorithm") + 1],
            "DSPARK",
        )
        self.assertEqual(
            captured[captured.index("--speculative-dspark-block-size") + 1],
            "7",
        )
        self.assertIn("--num-continuous-decode-steps", captured)
        self.assertNotIn("--continuous-decode-steps", captured)
        self.assertNotIn("--speculative-num-steps", captured)
        self.assertNotIn("--speculative-eagle-topk", captured)
        self.assertIn("--enable-torch-compile", captured)
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
