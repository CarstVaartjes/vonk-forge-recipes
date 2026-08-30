from __future__ import annotations

import hashlib
import json
import runpy
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SLUG = "qwen-image-2512-fp8-lightning-comfyui-single"
MODEL_SLUG = "qwen-image-2512-lightning-fp32-a52649c9"
BASE_MODEL_SLUG = "qwen-image-2512-comfyui-fp8-7beb7b64"
COMFY_REVISION = "7a131a3afadc8200120f67f9236311a2c48b7445"
QWEN_REVISION = "7beb7b647f04469fbe64ba8adc2bb0d7e5e9f73f"
LIGHTNING_REVISION = "a52649c9d0f6e1a248bff13f0df33bb8a2abdb52"
ADAPTER = ROOT / "adapters/media/comfyui-qwen-image-2512-fp8-lightning"
SHARED_RUNNER = ROOT / "adapters/media/comfyui-core/comfyui_job.py"


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


class QwenImage2512FP8LightningRecipeTests(unittest.TestCase):
    def test_model_version_closes_the_exact_upstream_pair(self) -> None:
        model = load(f"model-versions/{MODEL_SLUG}.json")
        self.assertEqual(model["source"]["revision"], LIGHTNING_REVISION)
        self.assertEqual(model["format"]["quantization"], "none")
        artifacts = {item["id"]: item for item in model["artifacts"]}
        self.assertEqual(set(artifacts), {"qwen-image-2512-lightning-4steps-v1-0-fp32"})
        self.assertEqual(
            (
                artifacts["qwen-image-2512-lightning-4steps-v1-0-fp32"]["revision"],
                artifacts["qwen-image-2512-lightning-4steps-v1-0-fp32"]["sha256"],
                artifacts["qwen-image-2512-lightning-4steps-v1-0-fp32"]["download_bytes"],
            ),
            (
                LIGHTNING_REVISION,
                "ad12117461cb41e2ea637fec8df6392ce8e8550c47fbe2b829ed3deb98262066",
                1_698_951_104,
            ),
        )
        base = load(f"model-versions/{BASE_MODEL_SLUG}.json")
        self.assertEqual(base["source"]["revision"], QWEN_REVISION)
        self.assertEqual(base["lineage"]["relation"], "quantized")
        self.assertEqual(base["format"]["quantization"], "fp8")
        artifacts = {item["id"]: item for item in base["artifacts"]}
        self.assertEqual(
            (
                artifacts["qwen-image-2512-fp8-transformer"]["revision"],
                artifacts["qwen-image-2512-fp8-transformer"]["sha256"],
                artifacts["qwen-image-2512-fp8-transformer"]["download_bytes"],
            ),
            (
                QWEN_REVISION,
                "5dc80554d5d83390046a2f4a94ece06afb7700bf7b0aaf8bde9769793875876b",
                20_430_679_144,
            ),
        )
        self.assertEqual(base["sizes"]["installed_bytes"], 20_430_679_144)
        self.assertEqual(model["sizes"]["installed_bytes"], 1_698_951_104)

    def test_recipe_has_an_isolated_pinned_offline_source_bundle(self) -> None:
        recipe = load(f"recipes/{SLUG}.json")
        self.assertEqual(
            recipe["model"]["content_sha256"],
            canonical_digest(load(f"model-versions/{MODEL_SLUG}.json")),
        )
        self.assertEqual(len(recipe["dependencies"]), 1)
        self.assertEqual(recipe["dependencies"][0]["slug"], BASE_MODEL_SLUG)
        self.assertEqual(
            recipe["dependencies"][0]["content_sha256"],
            canonical_digest(load(f"model-versions/{BASE_MODEL_SLUG}.json")),
        )
        source_bundle = runpy.run_path(str(ROOT / "tools/build-catalog-index"))[
            "source_bundle"
        ]
        archive, _, digest = source_bundle(ADAPTER)
        context = recipe["build"]["context"]
        self.assertEqual(context["path"], ADAPTER.relative_to(ROOT).as_posix())
        self.assertEqual(context["sha256"], digest)
        self.assertEqual(context["expected_bytes"], len(archive))
        dockerfile = (ADAPTER / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn(COMFY_REVISION, dockerfile)
        self.assertNotIn("custom_nodes", dockerfile)
        self.assertEqual(
            (ADAPTER / "comfyui_job.py").read_bytes(), SHARED_RUNNER.read_bytes()
        )
        self.assertFalse(recipe["runtime"]["security"]["host_network"])

    def test_workflow_is_the_exact_four_step_blueprint_profile(self) -> None:
        recipe = load(f"recipes/{SLUG}.json")
        arguments = {
            item["name"]: item["value"] for item in recipe["runtime"]["arguments"]
        }
        workflow_path = ADAPTER / "workflows/qwen-image-2512-fp8-lightning.json"
        workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
        self.assertEqual(
            arguments["workflow-sha256"],
            hashlib.sha256(workflow_path.read_bytes()).hexdigest(),
        )
        self.assertIn(COMFY_REVISION, workflow["upstream_template"])
        self.assertEqual(
            {item["artifact_id"] for item in workflow["models"]},
            {"target", "lightning-lora", "text-encoder", "vae"},
        )
        self.assertEqual(workflow["prompt"]["2"]["class_type"], "LoraLoaderModelOnly")
        self.assertEqual(workflow["prompt"]["2"]["inputs"]["strength_model"], 1.0)
        self.assertEqual(workflow["prompt"]["3"]["inputs"]["shift"], 3.1)
        sampler = workflow["prompt"]["9"]["inputs"]
        self.assertEqual(
            (sampler["steps"], sampler["cfg"], sampler["sampler_name"], sampler["scheduler"]),
            (4, 1.0, "euler", "simple"),
        )

    def test_single_spark_fit_is_conservative_and_directly_installable(self) -> None:
        recipe = load(f"recipes/{SLUG}.json")
        role = recipe["topology"]["roles"][0]
        disk = role["resources"]["disk"]
        memory = role["resources"]["memory"]
        self.assertEqual(disk["artifact_bytes"], 31_768_107_174)
        self.assertEqual(sum(disk.values()), 96_768_107_174)
        required = max(
            memory["startup_peak_bytes"],
            memory["steady_state_bytes"] + memory["runtime_growth_bytes"],
        )
        self.assertEqual(required + memory["system_reserve_bytes"], 108_000_000_000)
        self.assertLess(required + memory["system_reserve_bytes"], 128_000_000_000)
        self.assertTrue({"candidate", "executable", "fp8", "lightning"} <= set(recipe["metadata"]["tags"]))

    def test_release_and_target_bind_the_unaccepted_candidate(self) -> None:
        recipe = load(f"recipes/{SLUG}.json")
        release = load(f"recipe-releases/{SLUG}.json")
        self.assertEqual(release["version"], "1.0.0")
        self.assertEqual(
            release["history"][0]["recipe_content_sha256"],
            canonical_digest(recipe),
        )
        targets = load("model-targets/image.json")["targets"]
        target = next(
            item for item in targets if item.get("catalog_model_version") == MODEL_SLUG
        )
        self.assertEqual(target["status"], "candidate")
        self.assertEqual(target["harnesses"], ["comfyui"])
        self.assertEqual(target["recipe_slugs"], [SLUG])


if __name__ == "__main__":
    unittest.main()
