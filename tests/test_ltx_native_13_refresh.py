from __future__ import annotations

import hashlib
import json
import runpy
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "runtime-distributions/ltx2-pipelines-1-3-arm64.json"
SOURCE_REVISION = "a95ab856bf29407b6b066ede0abe1846050db56c"
SOURCE_ARCHIVE_SHA256 = (
    "4698fc5f635196edc08e891f209402d6b80e0b64d6c55589266e2448966500e8"
)
RECIPES = {
    "ltx-2-19b-dev-fp4-pytorch-single": "adapters/video/ltx2-pytorch",
    "ltx-2-19b-dev-bf16-diffusers-single": (
        "adapters/video/ltx23-sync-native-disk"
    ),
    "ltx-2-19b-distilled-diffusers-single": "adapters/video/ltx2-sync-native",
    "ltx-2-19b-distilled-fp8-diffusers-single": "adapters/video/ltx2-sync-native",
    "ltx-2-3-22b-distilled-1-1-diffusers-single": (
        "adapters/video/ltx23-sync-native-disk"
    ),
}
MEMORY_ENVELOPES = {
    "ltx-2-19b-dev-fp4-pytorch-single": (89, 75, 8, 8),
    "ltx-2-19b-dev-bf16-diffusers-single": (89, 75, 8, 8),
    "ltx-2-19b-distilled-diffusers-single": (116, 102, 8, 8),
    "ltx-2-19b-distilled-fp8-diffusers-single": (96, 82, 8, 8),
    "ltx-2-3-22b-distilled-1-1-diffusers-single": (93, 77, 8, 8),
}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    payload = json.dumps(
        load(path),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class LtxNative13RefreshTests(unittest.TestCase):
    def test_runtime_pins_exact_13_source_and_required_arm64_dependencies(self) -> None:
        runtime = load(RUNTIME)
        self.assertEqual(runtime["identity"]["slug"], "ltx2-pipelines-1-3-arm64")
        self.assertEqual(runtime["source"]["revision"], SOURCE_REVISION)
        self.assertEqual(
            runtime["source"]["archive_sha256"], SOURCE_ARCHIVE_SHA256
        )
        dependencies = {
            item["name"]: item["version"] for item in runtime["dependencies"]
        }
        self.assertEqual(dependencies["LTX Core and Pipelines"], "1.3.0+a95ab856")
        self.assertEqual(dependencies["TorchVision"], "0.28.0+cu132")
        self.assertEqual(dependencies["cuDNN"], "9.24.0.43")
        self.assertEqual(dependencies["Transformers"], "5.14.1")

        tags = set(runtime["metadata"]["tags"])
        self.assertTrue(
            {
                "pipeline-output",
                "dfr",
                "keyframe-aware-decode",
                "oom-safe-tiling",
            }
            <= tags
        )
        description = runtime["metadata"]["description"]
        for capability in (
            "PipelineOutput",
            "DFR",
            "keyframe-aware",
            "OOM",
        ):
            self.assertIn(capability, description)

    def test_all_native_recipes_bind_runtime_source_bundle_and_release(self) -> None:
        runtime_digest = digest(RUNTIME)
        source_bundle = runpy.run_path(str(ROOT / "tools/build-catalog-index"))[
            "source_bundle"
        ]
        bundle_cache: dict[str, tuple[int, str]] = {}

        for slug, adapter in RECIPES.items():
            recipe_path = ROOT / "recipes" / f"{slug}.json"
            recipe = load(recipe_path)
            distribution = recipe["runtime"]["distribution"]
            self.assertEqual(distribution["slug"], "ltx2-pipelines-1-3-arm64")
            self.assertEqual(distribution["content_sha256"], runtime_digest)
            self.assertIn("LTX 1.3.0", recipe["metadata"]["description"])

            if adapter not in bundle_cache:
                archive, _, bundle_digest = source_bundle(ROOT / adapter)
                bundle_cache[adapter] = (len(archive), bundle_digest)
            expected_bytes, bundle_digest = bundle_cache[adapter]
            context = recipe["build"]["context"]
            self.assertEqual(context["path"], adapter)
            self.assertEqual(context["sha256"], bundle_digest)
            self.assertEqual(context["expected_bytes"], expected_bytes)

            release = load(ROOT / "recipe-releases" / f"{slug}.json")
            self.assertEqual(
                release["history"][0]["recipe_content_sha256"], digest(recipe_path)
            )
            self.assertEqual(release["history"][0]["upgrade_effect"], "rebuild")

    def test_builds_are_archive_verified_and_fail_closed_on_pipeline_shape(self) -> None:
        for adapter in set(RECIPES.values()):
            root = ROOT / adapter
            dockerfile = (root / "Dockerfile").read_text(encoding="utf-8")
            runner_name = "pipelines/run.py" if adapter.endswith("ltx2-pytorch") else "run.py"
            runner = (root / runner_name).read_text(encoding="utf-8")
            requirements = (root / "requirements.lock").read_text(encoding="utf-8")

            self.assertIn(SOURCE_REVISION, dockerfile)
            self.assertIn(SOURCE_ARCHIVE_SHA256, dockerfile)
            self.assertIn("sha256sum --check --strict", dockerfile)
            self.assertIn("torchvision==0.28.0", dockerfile)
            self.assertIn("nvidia-cudnn-cu13==9.24.0.43", requirements)
            self.assertIn('LTX_PIPELINES_VERSION = "1.3.0"', runner)
            self.assertIn('"keyframes"', runner)
            self.assertIn('"video_latent"', runner)
            self.assertIn("_verify_ltx_runtime_contract()", runner)

    def test_optional_13_features_do_not_overstate_interfaces_or_memory(self) -> None:
        for slug, adapter in RECIPES.items():
            recipe = load(ROOT / "recipes" / f"{slug}.json")
            memory = recipe["topology"]["roles"][0]["resources"]["memory"]
            actual = tuple(
                memory[key] // 1_000_000_000
                for key in (
                    "startup_peak_bytes",
                    "steady_state_bytes",
                    "runtime_growth_bytes",
                    "system_reserve_bytes",
                )
            )
            self.assertEqual(actual, MEMORY_ENVELOPES[slug])

            runner_name = "pipelines/run.py" if adapter.endswith("ltx2-pytorch") else "run.py"
            runner = (ROOT / adapter / runner_name).read_text(encoding="utf-8")
            self.assertNotIn("ltx_pipelines.dfr_pipeline", runner)
            self.assertNotIn("--num-generated-keyframes", runner)
            self.assertNotIn("--compile", runner)

        ltx23 = load(
            ROOT / "recipes/ltx-2-3-22b-distilled-1-1-diffusers-single.json"
        )
        description = ltx23["metadata"]["description"]
        self.assertIn("OOM fix is not used to lower admission bounds", description)
        self.assertEqual(
            ltx23["topology"]["roles"][0]["resources"]["memory"][
                "startup_peak_bytes"
            ],
            93_000_000_000,
        )


if __name__ == "__main__":
    unittest.main()
