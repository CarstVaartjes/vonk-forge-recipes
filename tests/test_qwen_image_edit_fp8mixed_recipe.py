from __future__ import annotations

import hashlib
import json
import runpy
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SLUG = "qwen-image-edit-2511-fp8mixed-comfyui-single"
MODEL_SLUG = "qwen-image-edit-2511-comfyui-fp8mixed-4c7c4ea2"
MODEL_REVISION = "4c7c4ea236326cbae56d403d22a03c6cd86ad9a0"
TARGET_SHA256 = "c9fdc158e46d3b61ef75f21ae866ca2fe808bf4a53643120d1c1e87c19280a4e"
TARGET_BYTES = 20_533_762_817
ADAPTER = ROOT / "adapters/media/comfyui-qwen-image-edit-2511-fp8mixed"
SHARED_ADAPTER = ROOT / "adapters/media/comfyui-core"
SHARED_RECIPES = (
    "flux-2-klein-4b-comfyui-single",
    "qwen-image-edit-2511-comfyui-single",
    "wan-2-2-i2v-14b-comfyui-single",
    "wan-2-2-t2v-14b-comfyui-single",
    "wan-2-2-ti2v-5b-comfyui-single",
)


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


class QwenImageEditFP8MixedRecipeTests(unittest.TestCase):
    def test_exact_upstream_fp8mixed_artifact_is_truthfully_bound(self) -> None:
        model = load(f"model-versions/{MODEL_SLUG}.json")
        self.assertEqual(model["source"]["revision"], MODEL_REVISION)
        self.assertEqual(model["lineage"]["relation"], "quantized")
        self.assertEqual(model["format"]["quantization"], "fp8mixed")
        self.assertEqual(model["format"]["precision"], "mixed")
        artifact = model["artifacts"][0]
        self.assertEqual(
            (artifact["path"], artifact["sha256"], artifact["download_bytes"]),
            (
                "split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors",
                TARGET_SHA256,
                TARGET_BYTES,
            ),
        )
        self.assertEqual(model["sizes"]["installed_bytes"], TARGET_BYTES)
        self.assertNotIn("int8", json.dumps(model).lower())

    def test_recipe_uses_isolated_exact_source_bundle_and_stable_runtime(self) -> None:
        recipe = load(f"recipes/{SLUG}.json")
        self.assertTrue(
            {"candidate", "executable", "fp8", "fp8mixed", "mixed-precision"}
            <= set(recipe["metadata"]["tags"])
        )
        self.assertEqual(
            recipe["model"]["content_sha256"],
            canonical_digest(load(f"model-versions/{MODEL_SLUG}.json")),
        )
        source_bundle = runpy.run_path(str(ROOT / "tools/build-catalog-index"))[
            "source_bundle"
        ]
        archive, _, digest = source_bundle(ADAPTER)
        context = recipe["build"]["context"]
        self.assertEqual(context["path"], ADAPTER.relative_to(ROOT).as_posix())
        self.assertEqual(context["sha256"], digest)
        self.assertEqual(context["expected_bytes"], len(archive))
        self.assertEqual(
            recipe["runtime"]["distribution"]["slug"],
            "comfyui-0-34-0-cuda13-arm64",
        )
        dockerfile = (ADAPTER / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("12d5279438bfefc058a269eae805ceab6047777f", dockerfile)
        self.assertIn("6d8ab87ec1250e60101f0caf9e11658834c29d9cd76c9174e2b84ec9436f4886", dockerfile)
        self.assertEqual(
            (ADAPTER / "comfyui_job.py").read_bytes(),
            (SHARED_ADAPTER / "comfyui_job.py").read_bytes(),
        )

    def test_shared_comfyui_source_bundle_and_recipes_use_the_current_digest(
        self,
    ) -> None:
        source_bundle = runpy.run_path(str(ROOT / "tools/build-catalog-index"))[
            "source_bundle"
        ]
        _, _, shared_digest = source_bundle(SHARED_ADAPTER)
        self.assertEqual(
            shared_digest,
            "ab8f73d84fa6b14bf00beb4b3bd6188033759f1607549d9a7ca09d2c07c30b38",
        )
        for slug in SHARED_RECIPES:
            with self.subTest(slug=slug):
                context = load(f"recipes/{slug}.json")["build"]["context"]
                self.assertEqual(context["path"], "adapters/media/comfyui-core")
                self.assertEqual(context["sha256"], shared_digest)

    def test_workflow_and_resources_match_the_fp8mixed_contract(self) -> None:
        recipe = load(f"recipes/{SLUG}.json")
        arguments = {
            item["name"]: item["value"] for item in recipe["runtime"]["arguments"]
        }
        workflow = ADAPTER / "workflows/qwen-image-edit-2511-fp8mixed.json"
        self.assertEqual(
            arguments["workflow-sha256"],
            hashlib.sha256(workflow.read_bytes()).hexdigest(),
        )
        document = json.loads(workflow.read_text(encoding="utf-8"))
        target_model = next(
            item for item in document["models"] if item["artifact_id"] == "target"
        )
        self.assertEqual(
            target_model["filename"],
            "qwen_image_edit_2511_fp8mixed.safetensors",
        )
        self.assertEqual(
            document["prompt"]["1"]["inputs"]["unet_name"],
            target_model["filename"],
        )
        artifacts = {item["id"]: item for item in recipe["artifacts"]}
        self.assertEqual(set(artifacts), {"target", "text-encoder", "vae"})
        self.assertEqual(artifacts["target"]["revision"], f"sha256:{TARGET_SHA256}")
        self.assertEqual(
            recipe["topology"]["roles"][0]["resources"]["disk"][
                "artifact_bytes"
            ],
            30_172_239_743,
        )
        self.assertEqual(
            (document["inputs"]["minimum"], document["inputs"]["maximum"]),
            (1, 2),
        )

    def test_release_and_target_matrix_bind_current_candidate(self) -> None:
        recipe = load(f"recipes/{SLUG}.json")
        release = load(f"recipe-releases/{SLUG}.json")
        self.assertEqual(release["version"], "1.0.3")
        self.assertEqual(
            release["history"][0]["recipe_content_sha256"],
            canonical_digest(recipe),
        )
        targets = load("model-targets/image.json")["targets"]
        target = next(
            item
            for item in targets
            if item.get("catalog_model_version") == MODEL_SLUG
        )
        self.assertEqual(target["status"], "candidate")
        self.assertEqual(target["recipe_slugs"], [SLUG])
        self.assertEqual(target["harnesses"], ["comfyui"])


if __name__ == "__main__":
    unittest.main()
