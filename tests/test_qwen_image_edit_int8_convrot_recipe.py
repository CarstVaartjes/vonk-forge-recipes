from __future__ import annotations

import hashlib
import json
import runpy
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SLUG = "qwen-image-edit-2511-int8-convrot-comfyui-single"
MODEL_SLUG = "qwen-image-edit-2511-comfyui-int8-convrot-e9e85de7"
MODEL_REVISION = "e9e85de74a8f48c1e3e2656617626348675a2f21"
TARGET_SHA256 = "11b5af5ac601821d73930c84846c9a158e67177356daf927ce1c8d10f3963829"
TARGET_BYTES = 20_499_083_824
WORKFLOW_REVISION = "d3b4a9e89573162b005961865164c18c8ae2206b"
ADAPTER = ROOT / "adapters/media/comfyui-qwen-image-edit-2511-int8-convrot"
SHARED_ADAPTER = ROOT / "adapters/media/comfyui-core"


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


class QwenImageEditINT8ConvRotRecipeTests(unittest.TestCase):
    def test_exact_official_int8_convrot_artifact_is_bound(self) -> None:
        model = load(f"model-versions/{MODEL_SLUG}.json")
        self.assertEqual(model["source"]["revision"], MODEL_REVISION)
        self.assertEqual(model["lineage"]["relation"], "quantized")
        self.assertEqual(model["format"]["quantization"], "int8_tensorwise_convrot")
        self.assertEqual(model["format"]["precision"], "mixed")
        artifact = model["artifacts"][0]
        self.assertEqual(
            (artifact["path"], artifact["sha256"], artifact["download_bytes"]),
            (
                (
                    "split_files/diffusion_models/"
                    "qwen_image_edit_2511_int8_convrot.safetensors"
                ),
                TARGET_SHA256,
                TARGET_BYTES,
            ),
        )
        self.assertEqual(model["sizes"]["installed_bytes"], TARGET_BYTES)

    def test_isolated_source_bundle_and_runtime_are_compatible(self) -> None:
        recipe = load(f"recipes/{SLUG}.json")
        self.assertTrue(
            {"candidate", "executable", "int8", "convrot", "mixed-precision"}
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
        runtime = load("runtime-distributions/comfyui-0-34-0-cuda13-arm64.json")
        comfyui = next(item for item in runtime["dependencies"] if item["name"] == "ComfyUI")
        self.assertGreaterEqual(
            tuple(map(int, comfyui["version"].split("."))),
            (0, 27, 0),
        )
        dockerfile = (ADAPTER / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("12d5279438bfefc058a269eae805ceab6047777f", dockerfile)
        self.assertIn("6d8ab87ec1250e60101f0caf9e11658834c29d9cd76c9174e2b84ec9436f4886", dockerfile)
        self.assertEqual(
            (ADAPTER / "comfyui_job.py").read_bytes(),
            (SHARED_ADAPTER / "comfyui_job.py").read_bytes(),
        )

    def test_official_non_turbo_workflow_and_resource_contract(self) -> None:
        recipe = load(f"recipes/{SLUG}.json")
        arguments = {
            item["name"]: item["value"] for item in recipe["runtime"]["arguments"]
        }
        workflow = ADAPTER / "workflows/qwen-image-edit-2511-int8-convrot.json"
        self.assertEqual(
            arguments["workflow-sha256"],
            hashlib.sha256(workflow.read_bytes()).hexdigest(),
        )
        document = json.loads(workflow.read_text(encoding="utf-8"))
        self.assertIn(WORKFLOW_REVISION, document["upstream_template"])
        target = next(item for item in document["models"] if item["artifact_id"] == "target")
        self.assertEqual(
            target["filename"],
            "qwen_image_edit_2511_int8_convrot.safetensors",
        )
        prompt = document["prompt"]
        self.assertEqual(prompt["1"]["inputs"]["unet_name"], target["filename"])
        self.assertEqual(prompt["4"]["inputs"]["shift"], 3.1)
        sampler = prompt["14"]["inputs"]
        self.assertEqual(
            (sampler["steps"], sampler["cfg"], sampler["sampler_name"], sampler["scheduler"]),
            (40, 3.0, "euler", "simple"),
        )
        self.assertNotIn("LoraLoaderModelOnly", {node["class_type"] for node in prompt.values()})
        artifacts = {item["id"]: item for item in recipe["artifacts"]}
        self.assertEqual(set(artifacts), {"target", "text-encoder", "vae"})
        self.assertEqual(artifacts["target"]["revision"], f"sha256:{TARGET_SHA256}")
        resources = recipe["topology"]["roles"][0]["resources"]
        self.assertEqual(resources["disk"]["artifact_bytes"], 30_137_560_750)
        self.assertEqual(sum(resources["disk"].values()), 95_137_560_750)
        memory = resources["memory"]
        memory_envelope = max(
            memory["startup_peak_bytes"],
            memory["steady_state_bytes"] + memory["runtime_growth_bytes"],
        ) + memory["system_reserve_bytes"]
        self.assertEqual(memory_envelope, 108_000_000_000)

    def test_inputs_outputs_release_and_target_matrix_are_exact(self) -> None:
        recipe = load(f"recipes/{SLUG}.json")
        interface = recipe["interfaces"][0]
        inputs = {slot["id"]: slot for slot in interface["input"]["slots"]}
        self.assertEqual((inputs["prompt"]["min_files"], inputs["prompt"]["max_files"]), (1, 1))
        self.assertEqual((inputs["image"]["min_files"], inputs["image"]["max_files"]), (1, 2))
        output = interface["output"]["slots"][0]
        self.assertEqual((output["min_files"], output["max_files"]), (1, 1))
        self.assertEqual(output["media_types"], ["image/png"])
        self.assertEqual(
            recipe["validation"]["validators"],
            [{"interface": "image-job", "checks": ["artifact.mime.image-png"]}],
        )
        release = load(f"recipe-releases/{SLUG}.json")
        self.assertEqual(release["version"], "1.0.3")
        self.assertEqual(
            release["history"][0]["recipe_content_sha256"],
            canonical_digest(recipe),
        )
        targets = load("model-targets/image.json")["targets"]
        target = next(item for item in targets if item.get("catalog_model_version") == MODEL_SLUG)
        self.assertEqual(target["status"], "candidate")
        self.assertEqual(target["recipe_slugs"], [SLUG])
        self.assertEqual(target["harnesses"], ["comfyui"])


if __name__ == "__main__":
    unittest.main()
