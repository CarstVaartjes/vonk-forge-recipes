from __future__ import annotations

import hashlib
import importlib.util
import io
import itertools
import json
import tempfile
import traceback
import unittest
from email.message import Message
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

ROOT = Path(__file__).resolve().parents[1]
COMFY_RECIPES = (
    "flux-2-klein-4b-comfyui-single",
    "flux-2-klein-4b-nvfp4-comfyui-single",
    "qwen-image-2512-comfyui-single",
    "qwen-image-2512-fp8-lightning-comfyui-single",
    "qwen-image-edit-2511-comfyui-single",
    "qwen-image-edit-2511-fp8mixed-comfyui-single",
    "qwen-image-edit-2511-int8-convrot-comfyui-single",
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
COMFY_REVISION = "83071e1aec311d31e773d64d6872181b3bad0fe2"
COMFY_ARCHIVE_SHA256 = (
    "fbcb570f26aaf67911d15e164907a0992fb9d5780ea17546f1e0476e9943977c"
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

    def test_http_failure_evidence_keeps_node_errors_for_every_adapter(self) -> None:
        for path in sorted((ROOT / "adapters/media").glob("comfyui-*/comfyui_job.py")):
            spec = importlib.util.spec_from_file_location("comfyui_http_test", path)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            bodies = (
                b'{"error": "prompt_outputs_failed_validation", "node_errors": {"14": {"errors": [{"message": "Required input is missing", "details": "format.codec"}]}}}',
                b"invalid response: \xff",
                b"x" * (64 * 1024 + 1),
                b"",
            )
            for body in bodies:
                with self.subTest(adapter=path.parent.name, body_length=len(body)):
                    error = HTTPError(
                        module.BASE_URL + "/prompt",
                        400,
                        "Bad Request",
                        Message(),
                        io.BytesIO(body),
                    )
                    with (
                        patch.object(module, "urlopen", side_effect=error) as request,
                        self.assertRaises(HTTPError) as caught,
                    ):
                        module.request_json(
                            "/prompt", body={"prompt": {}, "client_id": "test"}
                        )
                    self.assertIs(caught.exception, error)
                    evidence = "".join(traceback.format_exception(error))
                    self.assertIn("ComfyUI /prompt HTTP 400 response body:", evidence)
                    self.assertIn(
                        body[: 64 * 1024].decode("utf-8", errors="replace"), evidence
                    )
                    self.assertEqual(
                        "[response body truncated]" in evidence, len(body) > 64 * 1024
                    )
                    self.assertTrue(error.fp.closed)
                    request.assert_called_once()

    def test_model_links_point_to_files_inside_artifact_mounts(self) -> None:
        for path in sorted((ROOT / "adapters/media").glob("comfyui-*/comfyui_job.py")):
            spec = importlib.util.spec_from_file_location("comfyui_mount_test", path)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            for directory_mount, source_name in (
                (False, "wan_2.1_vae.safetensors"),
                (True, "wan_2.1_vae.safetensors"),
                (True, "upstream-alias.safetensors"),
            ):
                with (
                    self.subTest(
                        adapter=path.parent.name, directory_mount=directory_mount
                    ),
                    tempfile.TemporaryDirectory() as temporary,
                ):
                    root = Path(temporary)
                    mounts = root / "mounts"
                    mounts.mkdir()
                    artifact = mounts / "vae"
                    filename = "wan_2.1_vae.safetensors"
                    if directory_mount:
                        source = artifact / "split_files" / "vae" / source_name
                        source.parent.mkdir(parents=True)
                    else:
                        source = artifact
                    source.write_bytes(b"fixture checkpoint")
                    with patch.object(module, "MODEL_MOUNT_ROOT", mounts):
                        module.link_models(
                            {
                                "models": [
                                    {
                                        "artifact_id": "vae",
                                        "category": "vae",
                                        "filename": filename,
                                    }
                                ]
                            },
                            root / "models",
                        )
                    linked = root / "models/vae" / filename
                    self.assertTrue(linked.is_file())
                    self.assertEqual(linked.resolve(), source.resolve())

    def test_model_mount_rejects_missing_or_ambiguous_checkpoint(self) -> None:
        for path in sorted((ROOT / "adapters/media").glob("comfyui-*/comfyui_job.py")):
            spec = importlib.util.spec_from_file_location(
                "comfyui_invalid_mount_test", path
            )
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            for count in (0, 2):
                with (
                    self.subTest(adapter=path.parent.name, matching_files=count),
                    tempfile.TemporaryDirectory() as temporary,
                ):
                    root = Path(temporary)
                    artifact = root / "mounts/vae"
                    artifact.mkdir(parents=True)
                    for index in range(count):
                        checkpoint = artifact / str(index) / "wan_2.1_vae.safetensors"
                        checkpoint.parent.mkdir()
                        checkpoint.write_bytes(b"fixture checkpoint")
                    with (
                        patch.object(module, "MODEL_MOUNT_ROOT", artifact.parent),
                        self.assertRaisesRegex(ValueError, "exactly one.*wan_2.1_vae"),
                    ):
                        module.link_models(
                            {
                                "models": [
                                    {
                                        "artifact_id": "vae",
                                        "category": "vae",
                                        "filename": "wan_2.1_vae.safetensors",
                                    }
                                ]
                            },
                            root / "models",
                        )
                    self.assertFalse(
                        (root / "models/vae/wan_2.1_vae.safetensors").exists()
                    )

    def test_successful_http_response_is_unchanged(self) -> None:
        path = ROOT / "adapters/media/comfyui-core/comfyui_job.py"
        spec = importlib.util.spec_from_file_location("comfyui_success_test", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with patch.object(
            module, "urlopen", return_value=io.BytesIO(b'{"prompt_id": "accepted"}')
        ):
            self.assertEqual(
                module.request_json("/prompt", body={"prompt": {}}),
                {"prompt_id": "accepted"},
            )

    def test_comfyui_writes_its_startup_directories_inside_the_job(self) -> None:
        # Hardware 2026-10-16: ComfyUI 0.38 creates its user directory (and
        # its database there) and its temp directory at startup; under the
        # read-only image tree that was "Read-only file system:
        # '/opt/comfyui/user'" and the job exited before serving.
        adapters = sorted((ROOT / "adapters/media").glob("comfyui-*/comfyui_job.py"))
        self.assertTrue(adapters)
        for path in adapters:
            with self.subTest(adapter=path.parent.name):
                spec = importlib.util.spec_from_file_location(
                    f"comfyui_job_{path.parent.name.replace('-', '_')}", path
                )
                assert spec is not None and spec.loader is not None
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                with tempfile.TemporaryDirectory() as directory:
                    temporary = Path(directory)
                    command = module.comfyui_command(
                        temporary, temporary / "output", temporary / "extra.yaml"
                    )
                    flags = dict(itertools.pairwise(command))
                    for flag in (
                        "--user-directory",
                        "--temp-directory",
                        "--output-directory",
                    ):
                        self.assertTrue(
                            Path(flags[flag]).is_relative_to(temporary), flag
                        )
                    # ComfyUI requires the user directory to exist already.
                    self.assertTrue(Path(flags["--user-directory"]).is_dir())


if __name__ == "__main__":
    unittest.main()
