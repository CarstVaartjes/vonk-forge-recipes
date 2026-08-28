from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SLUG = "flux-2-klein-4b-nvfp4-comfyui-single"
MODEL_SLUG = "flux-2-klein-4b-nvfp4-1db2b2f7"
MODEL_REVISION = "1db2b2f776c24b76f1122e5f69ab1949fc620068"


def load(relative: str) -> dict[str, object]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def canonical_digest(document: dict[str, object]) -> str:
    payload = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


class Flux2NVFP4RecipeTests(unittest.TestCase):
    def test_official_nvfp4_artifact_and_companions_are_exact(self) -> None:
        model = load(f"model-versions/{MODEL_SLUG}.json")
        self.assertEqual(model["source"]["revision"], MODEL_REVISION)
        self.assertEqual(model["lineage"]["relation"], "official")
        self.assertEqual(model["format"]["quantization"], "nvfp4")

        artifacts = {item["id"]: item for item in model["artifacts"]}
        self.assertEqual(
            (
                artifacts["diffusion"]["sha256"],
                artifacts["diffusion"]["download_bytes"],
            ),
            (
                "d8c5007b6a3bbbdfd38538bbcef5101a55dfde81894f58d2e3c8701cdef3542b",
                2_460_413_488,
            ),
        )
        self.assertEqual(model["sizes"]["installed_bytes"], 2_460_413_488)

    def test_recipe_reuses_hash_locked_core_workflow_conservatively(self) -> None:
        recipe = load(f"recipes/{SLUG}.json")
        self.assertTrue(
            {"candidate", "executable", "nvfp4"} <= set(recipe["metadata"]["tags"])
        )
        self.assertEqual(
            recipe["runtime"]["distribution"]["slug"], "comfyui-0-33-4-cuda13-arm64"
        )
        arguments = {
            item["name"]: item["value"] for item in recipe["runtime"]["arguments"]
        }
        workflow = ROOT / "adapters/media/comfyui-core/workflows/flux-2-klein-4b.json"
        self.assertEqual(
            arguments["workflow-sha256"],
            hashlib.sha256(workflow.read_bytes()).hexdigest(),
        )
        resources = recipe["topology"]["roles"][0]["resources"]
        artifacts = {item["id"]: item for item in recipe["artifacts"]}
        self.assertEqual(
            artifacts["text-encoder"]["revision"],
            "sha256:6c671498573ac2f7a5501502ccce8d2b08ea6ca2f661c458e708f36b36edfc5a",
        )
        self.assertEqual(
            artifacts["vae"]["revision"],
            "sha256:868fe7b343cc8f3a19dbcfcafbc3d5f888802be3f89bd81b65b3621a066ce8f3",
        )
        self.assertEqual(resources["disk"]["artifact_bytes"], 10_841_606_828)
        self.assertGreaterEqual(
            resources["memory"]["startup_peak_bytes"], 36_000_000_000
        )
        self.assertGreaterEqual(
            resources["memory"]["system_reserve_bytes"], 8_000_000_000
        )

    def test_release_and_target_ledger_bind_current_recipe(self) -> None:
        recipe = load(f"recipes/{SLUG}.json")
        release = load(f"recipe-releases/{SLUG}.json")
        self.assertEqual(release["version"], "1.0.0")
        self.assertEqual(
            release["history"][0]["recipe_content_sha256"], canonical_digest(recipe)
        )

        targets = load("model-targets/image.json")["targets"]
        target = next(item for item in targets if SLUG in item.get("recipe_slugs", []))
        self.assertEqual(target["catalog_model_version"], MODEL_SLUG)
        self.assertEqual(target["status"], "candidate")

    def test_qwen_comfy_binding_uses_upstream_monolith_without_runtime_merge(
        self,
    ) -> None:
        model = load("model-versions/qwen-image-edit-2511-comfyui-b6a07947.json")
        self.assertEqual(
            model["source"]["revision"],
            "b6a0794717d3f5600f85c5edcdcd0c0eb93d7446",
        )
        self.assertEqual(model["lineage"]["relation"], "derived")
        self.assertEqual(
            (model["artifacts"][0]["sha256"], model["sizes"]["installed_bytes"]),
            (
                "ae42d927b5fac4f278b9a894554c727e619727a63622976f2d95625be4bce08c",
                40_861_031_560,
            ),
        )

        recipe = load("recipes/qwen-image-edit-2511-comfyui-single.json")
        tags = set(recipe["metadata"]["tags"])
        self.assertTrue({"candidate", "executable", "comfyui"} <= tags)
        artifacts = {item["id"]: item for item in recipe["artifacts"]}
        self.assertEqual(set(artifacts), {"target", "text-encoder", "vae"})
        self.assertEqual(
            artifacts["target"]["revision"],
            "sha256:ae42d927b5fac4f278b9a894554c727e619727a63622976f2d95625be4bce08c",
        )
        self.assertEqual(
            recipe["topology"]["roles"][0]["resources"]["disk"]["artifact_bytes"],
            50_499_508_486,
        )
        workflow = load(
            "adapters/media/comfyui-core/workflows/qwen-image-edit-2511-bf16.json"
        )
        self.assertNotIn("snapshot_models", workflow)
        self.assertEqual(
            {item["artifact_id"] for item in workflow["models"]},
            {"target", "text-encoder", "vae"},
        )
        self.assertEqual(
            (workflow["inputs"]["minimum"], workflow["inputs"]["maximum"]),
            (1, 2),
        )

        targets = load("model-targets/image.json")["targets"]
        target = next(item for item in targets if item["model"] == "Qwen Image Edit")
        self.assertEqual(target["harnesses"], ["diffusers", "comfyui"])
        self.assertEqual(
            target["recipe_slugs"],
            [
                "qwen-image-edit-2511-diffusers-single",
                "qwen-image-edit-2511-comfyui-single",
            ],
        )


if __name__ == "__main__":
    unittest.main()
