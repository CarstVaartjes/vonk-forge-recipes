from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMFY_RECIPES = ("flux-2-klein-4b-comfyui-single", "qwen-image-2512-comfyui-single", "qwen-image-2512-fp8-lightning-comfyui-single", "qwen-image-edit-2511-comfyui-single", "wan-2-2-i2v-14b-comfyui-single", "wan-2-2-t2v-14b-comfyui-single", "wan-2-2-ti2v-5b-comfyui-single")
CORE_NODES = {
    "CFGGuider", "CFGNorm", "CLIPLoader", "CLIPTextEncode", "ConditioningZeroOut",
    "CreateVideo", "DiffusersLoader", "EmptyFlux2LatentImage", "EmptyHunyuanLatentVideo",
    "EmptySD3LatentImage", "Flux2Scheduler", "FluxKontextImageScale",
    "FluxKontextMultiReferenceLatentMethod", "KSampler", "KSamplerAdvanced", "KSamplerSelect",
    "LoadImage", "LoraLoaderModelOnly", "ModelSamplingAuraFlow", "ModelSamplingSD3", "RandomNoise",
    "SamplerCustomAdvanced", "SaveImage", "SaveVideo", "TextEncodeQwenImageEditPlus",
    "UNETLoader", "VAEDecode", "VAEEncode", "VAELoader", "Wan22ImageToVideoLatent",
    "WanImageToVideo",
}
COMFY_REVISION = "12d5279438bfefc058a269eae805ceab6047777f"
COMFY_ARCHIVE_SHA256 = "6d8ab87ec1250e60101f0caf9e11658834c29d9cd76c9174e2b84ec9436f4886"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class ComfyUICoreRecipeTests(unittest.TestCase):
    def test_all_comfy_recipes_use_hash_locked_core_workflows(self) -> None:
        for slug in COMFY_RECIPES:
            recipe = load(ROOT / "recipes" / f"{slug}.json")
            self.assertIn(recipe["interfaces"][0]["adapter"], {"image-job", "video-job"}, slug)
            self.assertEqual(recipe["topology"]["node_count"], 1, slug)
            arguments = {item["name"]: item["value"] for item in recipe["runtime"]["arguments"]}
            workflow_hash = arguments["workflow-sha256"]
            self.assertEqual(len(workflow_hash), 64)
            context = ROOT / recipe["execution"]["build"]["context"]["path"]
            workflow = context / "workflows" / Path(arguments["workflow"]).name
            self.assertTrue(workflow.is_file())
            self.assertEqual(hashlib.sha256(workflow.read_bytes()).hexdigest(), workflow_hash)
            document = load(workflow)
            self.assertTrue(document["text_inputs"]["prompt"]["required"])
            self.assertEqual(document["text_inputs"]["prompt"]["slot"], "prompt")
            self.assertEqual(document["text_inputs"]["prompt"]["maximum_bytes"], 16 * 1024)
            recipe_slots = {slot["id"]: slot for slot in recipe["interfaces"][0]["input"]["slots"]}
            workflow_inputs = document["inputs"]
            image_slot = recipe_slots.get("image")
            if image_slot is None:
                self.assertEqual((workflow_inputs["minimum"], workflow_inputs["maximum"]), (0, 0))
                self.assertNotIn("image", recipe_slots)
            else:
                self.assertEqual(workflow_inputs["slot"], "image")
                self.assertEqual((workflow_inputs["minimum"], workflow_inputs["maximum"]), (image_slot["min_files"], image_slot["max_files"]))
            self.assertLessEqual({node["class_type"] for node in document["prompt"].values()}, CORE_NODES)
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
            self.assertIn(recipe["execution"]["mode"], {"build", "image"})
            self.assertTrue(recipe["models"])
            self.assertTrue(all(item["mount"]["read_only"] for selection in recipe["models"] for item in selection["files"]))
            build = recipe["execution"].get("build")
            self.assertIsNotNone(build)
            dockerfile = (ROOT / build["dockerfile"]).read_text(encoding="utf-8")
            self.assertIn(COMFY_REVISION, dockerfile)
            self.assertIn(COMFY_ARCHIVE_SHA256, dockerfile)
            self.assertNotIn("ComfyUI-Manager", dockerfile)
            self.assertNotIn("custom_nodes", dockerfile)

    def test_qwen_2512_quality_defaults_remain_bounded(self) -> None:
        recipe = load(ROOT / "recipes/qwen-image-2512-comfyui-single.json")
        arguments = {item["name"]: item["value"] for item in recipe["runtime"]["arguments"]}
        self.assertIn("workflow-sha256", arguments)
        self.assertEqual(recipe["interfaces"][0]["adapter"], "image-job")
        self.assertEqual(recipe["validation"]["benchmarks"][0]["configuration"]["steps"], 50)
        self.assertEqual(recipe["validation"]["benchmarks"][0]["configuration"]["width"], 1328)
        self.assertEqual(recipe["validation"]["benchmarks"][0]["configuration"]["height"], 1328)

    def test_core_source_is_pinned_without_custom_node_supply_chain(self) -> None:
        recipe = load(ROOT / "recipes/qwen-image-2512-comfyui-single.json")
        build = recipe["execution"].get("build")
        if build:
            dockerfile = ROOT / build["dockerfile"]
            dockerfile_text = dockerfile.read_text(encoding="utf-8")
            self.assertIn(COMFY_REVISION, dockerfile_text)
            self.assertIn(COMFY_ARCHIVE_SHA256, dockerfile_text)
            self.assertNotIn("custom_nodes", dockerfile_text)


if __name__ == "__main__": unittest.main()
