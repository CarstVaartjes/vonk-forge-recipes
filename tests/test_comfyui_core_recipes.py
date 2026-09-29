from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMFY_RECIPES = (
    "flux-2-klein-4b-comfyui-single",
    "qwen-image-2512-comfyui-single",
    "qwen-image-2512-fp8-lightning-comfyui-single",
    "qwen-image-edit-2511-comfyui-single",
    "wan-2-2-i2v-14b-comfyui-single",
    "wan-2-2-t2v-14b-comfyui-single",
    "wan-2-2-ti2v-5b-comfyui-single",
)
CORE_NODES = {
    "CFGGuider",
    "CFGNorm",
    "CLIPLoader",
    "CLIPTextEncode",
    "ConditioningZeroOut",
    "CreateVideo",
    "DiffusersLoader",
    "EmptyFlux2LatentImage",
    "EmptyHunyuanLatentVideo",
    "EmptySD3LatentImage",
    "Flux2Scheduler",
    "FluxKontextImageScale",
    "FluxKontextMultiReferenceLatentMethod",
    "KSampler",
    "KSamplerAdvanced",
    "KSamplerSelect",
    "LoadImage",
    "LoraLoaderModelOnly",
    "ModelSamplingAuraFlow",
    "ModelSamplingSD3",
    "RandomNoise",
    "SamplerCustomAdvanced",
    "SaveImage",
    "SaveVideo",
    "TextEncodeQwenImageEditPlus",
    "UNETLoader",
    "VAEDecode",
    "VAEEncode",
    "VAELoader",
    "Wan22ImageToVideoLatent",
    "WanImageToVideo",
}
COMFY_REVISION = "8ff6dc384ba5c410266b40e137799e049459d4f2"
COMFY_ARCHIVE_SHA256 = (
    "fa58882988cbb5902dbff71626ca7a82c2548fe34d588c482c8859306734a39f"
)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class ComfyUICoreRecipeTests(unittest.TestCase):
    def test_all_comfy_recipes_use_hash_locked_core_workflows(self) -> None:
        for slug in COMFY_RECIPES:
            recipe = load(ROOT / "recipes" / f"{slug}.json")
            self.assertIn(
                recipe["interfaces"][0]["adapter"], {"image-job", "video-job"}, slug
            )
            self.assertEqual(recipe["topology"]["node_count"], 1, slug)
            arguments = {
                item["name"]: item["value"] for item in recipe["runtime"]["arguments"]
            }
            workflow_hash = arguments["workflow-sha256"]
            self.assertEqual(len(workflow_hash), 64)
            context = ROOT / recipe["execution"]["build"]["context"]["path"]
            workflow = context / "workflows" / Path(arguments["workflow"]).name
            self.assertTrue(workflow.is_file())
            self.assertEqual(
                hashlib.sha256(workflow.read_bytes()).hexdigest(), workflow_hash
            )
            document = load(workflow)
            self.assertTrue(document["text_inputs"]["prompt"]["required"])
            self.assertEqual(document["text_inputs"]["prompt"]["slot"], "prompt")
            self.assertEqual(
                document["text_inputs"]["prompt"]["maximum_bytes"], 16 * 1024
            )
            recipe_slots = {
                slot["id"]: slot for slot in recipe["interfaces"][0]["input"]["slots"]
            }
            workflow_inputs = document["inputs"]
            image_slot = recipe_slots.get("image")
            if image_slot is None:
                self.assertEqual(
                    (workflow_inputs["minimum"], workflow_inputs["maximum"]), (0, 0)
                )
                self.assertNotIn("image", recipe_slots)
            else:
                self.assertEqual(workflow_inputs["slot"], "image")
                self.assertEqual(
                    (workflow_inputs["minimum"], workflow_inputs["maximum"]),
                    (image_slot["min_files"], image_slot["max_files"]),
                )
            self.assertLessEqual(
                {node["class_type"] for node in document["prompt"].values()}, CORE_NODES
            )
            output = recipe["interfaces"][0]["output"]["slots"][0]
            self.assertEqual(output["min_files"], 1)
            self.assertEqual(output["max_files"], 1)
            self.assertTrue(output["media_types"] in (["image/png"], ["video/mp4"]))
            if "result" in document:
                self.assertEqual(document["result"]["mime"], output["media_types"][0])
                self.assertEqual(document["result"]["count"], output["min_files"])
            else:
                self.assertEqual(output["media_types"], ["image/png"])

    def test_recipe_execution_and_model_closure_are_explicit(self) -> None:
        for slug in COMFY_RECIPES:
            recipe = load(ROOT / "recipes" / f"{slug}.json")
            self.assertTrue(recipe["models"])
            build = recipe["execution"]["build"]
            dockerfile = (ROOT / build["dockerfile"]).read_text(encoding="utf-8")
            self.assertIn(COMFY_REVISION, dockerfile)
            self.assertIn(COMFY_ARCHIVE_SHA256, dockerfile)
            self.assertNotIn("ComfyUI-Manager", dockerfile)
            self.assertNotIn("custom_nodes", dockerfile)

    def test_mounted_models_are_exactly_what_each_workflow_loads(self) -> None:
        # comfyui_job.link_models resolves every workflow model to
        # /models/<artifact_id> and fails when that mount is absent.
        recipes = [
            recipe
            for recipe in map(load, sorted((ROOT / "recipes").glob("*.json")))
            if recipe["runtime"]["engine"] == "comfyui"
        ]
        self.assertGreaterEqual(len(recipes), len(COMFY_RECIPES))
        for recipe in recipes:
            with self.subTest(recipe=recipe["identity"]["slug"]):
                arguments = {
                    item["name"]: item["value"]
                    for item in recipe["runtime"]["arguments"]
                }
                context = ROOT / recipe["execution"]["build"]["context"]["path"]
                workflow = load(
                    context / "workflows" / Path(arguments["workflow"]).name
                )
                mounted = {}
                for selection in recipe["models"]:
                    model = load(ROOT / "models" / f"{selection['model']['slug']}.json")
                    paths = {item["id"]: item["path"] for item in model["files"]}
                    for item in selection["files"]:
                        mounted[item["mount"]["target"]] = paths[item["file_id"]]
                expected = {
                    f"/models/{item['artifact_id']}": item
                    for item in workflow["models"]
                }
                self.assertEqual(set(mounted), set(expected))
                for target, item in expected.items():
                    if item["category"] != "diffusion_models":
                        self.assertEqual(
                            Path(mounted[target]).name, item["filename"], target
                        )

    def test_qwen_2512_quality_defaults_remain_bounded(self) -> None:
        recipe = load(ROOT / "recipes/qwen-image-2512-comfyui-single.json")
        arguments = {
            item["name"]: item["value"] for item in recipe["runtime"]["arguments"]
        }
        self.assertIn("workflow-sha256", arguments)
        self.assertEqual(recipe["interfaces"][0]["adapter"], "image-job")

    def test_core_source_is_pinned_without_custom_node_supply_chain(self) -> None:
        recipe = load(ROOT / "recipes/qwen-image-2512-comfyui-single.json")
        build = recipe["execution"].get("build")
        if build:
            dockerfile = ROOT / build["dockerfile"]
            dockerfile_text = dockerfile.read_text(encoding="utf-8")
            self.assertIn(COMFY_REVISION, dockerfile_text)
            self.assertIn(COMFY_ARCHIVE_SHA256, dockerfile_text)
            self.assertNotIn("custom_nodes", dockerfile_text)


if __name__ == "__main__":
    unittest.main()
