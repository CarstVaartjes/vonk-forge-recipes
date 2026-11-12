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


def _adapter(path: Path):
    spec = importlib.util.spec_from_file_location(
        f"comfyui_controls_{path.parent.name.replace('-', '_')}", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _recipe_job(slug: str) -> tuple[dict, dict, dict]:
    """The recipe, its workflow and the argument values a default job passes."""

    recipe = load(ROOT / "recipes" / f"{slug}.json")
    knobs = recipe["settings"]["knobs"]
    arguments = {
        item["name"]: knobs[item["setting"]]["value"]
        if item["setting"] is not None
        else item["value"]
        for item in recipe["runtime"]["arguments"]
    }
    context = ROOT / recipe["execution"]["build"]["context"]["path"]
    workflow = load(context / "workflows" / Path(arguments["workflow"]).name)
    return recipe, workflow, arguments


def _prompt(adapter, workflow: dict, arguments: dict) -> dict:
    """Build the submitted prompt exactly as the adapter's main() does."""

    inputs = [f"input-{index}.png" for index in range(workflow["inputs"]["minimum"])]
    with patch.dict("os.environ", {"VONK_PROMPT": "A lantern in fog."}):
        replacements = adapter.workflow_replacements(
            workflow, inputs, arguments["seed"]
        )
    replacements.update(
        adapter.control_replacements(
            workflow,
            {name: arguments.get(name) for name in adapter.WORKFLOW_CONTROLS},
        )
    )
    unresolved = adapter.workflow_placeholders(workflow["prompt"]) - replacements.keys()
    if unresolved:
        raise ValueError(f"workflow placeholders are unresolved: {sorted(unresolved)}")
    return adapter.substitute(workflow["prompt"], replacements)


CONTROLLED_ADAPTERS = (
    ROOT / "adapters/media/comfyui-core/comfyui_job.py",
    ROOT / "adapters/media/comfyui-qwen-image-edit-2511-fp8mixed/comfyui_job.py",
)


class ComfyUIWorkflowControlTests(unittest.TestCase):
    def test_recipes_on_controlled_adapters_build_complete_prompts(self) -> None:
        # Every recipe on an adapter with workflow controls, knobs or not,
        # resolves every placeholder from its own arguments and knob defaults;
        # a workflow without controls accepts none.
        for slug in COMFY_RECIPES:
            recipe, workflow, arguments = _recipe_job(slug)
            script = (
                ROOT
                / recipe["execution"]["build"]["context"]["path"]
                / "comfyui_job.py"
            )
            if script not in CONTROLLED_ADAPTERS:
                continue
            with self.subTest(recipe=slug):
                adapter = _adapter(script)
                prompt = _prompt(adapter, workflow, arguments)
                self.assertFalse(adapter.workflow_placeholders(prompt))
                bound = {
                    item["name"]
                    for item in recipe["runtime"]["arguments"]
                    if item["setting"] is not None
                }
                self.assertEqual(bound, set(recipe["settings"]["knobs"]))
                self.assertEqual(bound - {"seed"}, set(workflow.get("controls", {})))
                if not workflow.get("controls"):
                    with self.assertRaisesRegex(ValueError, "does not accept"):
                        adapter.control_replacements(workflow, {"width": 1024})

    def test_flux_nvfp4_defaults_reproduce_the_fixed_workflow(self) -> None:
        _, workflow, arguments = _recipe_job("flux-2-klein-4b-nvfp4-comfyui-single")
        adapter = _adapter(CONTROLLED_ADAPTERS[0])
        fixed = load(
            ROOT / "adapters/media/comfyui-core/workflows/flux-2-klein-4b.json"
        )
        self.assertEqual(
            _prompt(adapter, workflow, arguments),
            _prompt(adapter, fixed, {"seed": arguments["seed"]}),
        )
        wide = _prompt(
            adapter,
            workflow,
            {**arguments, "width": 1280, "height": 720, "num-inference-steps": 6},
        )
        self.assertEqual(
            wide["6"]["inputs"], {"width": 1280, "height": 720, "batch_size": 1}
        )
        self.assertEqual(
            wide["10"]["inputs"], {"steps": 6, "width": 1280, "height": 720}
        )

    def test_qwen_edit_fp8mixed_defaults_keep_the_forty_step_schedule(self) -> None:
        _, workflow, arguments = _recipe_job(
            "qwen-image-edit-2511-fp8mixed-comfyui-single"
        )
        adapter = _adapter(CONTROLLED_ADAPTERS[1])
        sampler = _prompt(adapter, workflow, arguments)["14"]["inputs"]
        self.assertEqual((sampler["steps"], sampler["seed"]), (40, 0))
        sampler = _prompt(
            adapter, workflow, {**arguments, "seed": 7, "num-inference-steps": 8}
        )["14"]["inputs"]
        self.assertEqual((sampler["steps"], sampler["seed"]), (8, 7))

    def test_controls_refuse_missing_or_out_of_bounds_values(self) -> None:
        _, workflow, arguments = _recipe_job("flux-2-klein-4b-nvfp4-comfyui-single")
        for path in CONTROLLED_ADAPTERS:
            adapter = _adapter(path)
            for change, message in (
                ({"width": 1000}, "multiples of 16"),
                ({"width": 240}, "256..2048"),
                ({"height": 2064}, "256..2048"),
                ({"num-inference-steps": 0}, "1..16"),
                ({"num-inference-steps": 17}, "1..16"),
                ({"height": None}, "requires --height"),
                ({"seed": -1}, "--seed must be"),
                ({"seed": adapter.MAX_SEED + 1}, "--seed must be"),
            ):
                with (
                    self.subTest(adapter=path.parent.name, change=change),
                    self.assertRaisesRegex(ValueError, message),
                ):
                    _prompt(adapter, workflow, {**arguments, **change})
            undeclared = {**workflow, "controls": {}}
            with self.assertRaisesRegex(ValueError, "unresolved.*__VONK_WIDTH__"):
                _prompt(adapter, undeclared, {"seed": 0})
            for bounds in (
                {"minimum": 0, "maximum": 4},
                {"minimum": 8, "maximum": 4},
                {"minimum": 16, "maximum": 2000, "multiple_of": 16},
                {"minimum": 1, "maximum": 4, "step": 1},
            ):
                with (
                    self.subTest(adapter=path.parent.name, bounds=bounds),
                    self.assertRaisesRegex(ValueError, "num-inference-steps"),
                ):
                    adapter.control_replacements(
                        {"controls": {"num-inference-steps": bounds}},
                        {"num-inference-steps": 4},
                    )

    def test_controls_arrive_through_the_command_line(self) -> None:
        adapter = _adapter(CONTROLLED_ADAPTERS[0])
        argv = [
            "comfyui-job",
            "--workflow=flux.json",
            "--workflow-sha256=" + "0" * 64,
            "--output-mime=image/png",
            "--seed=3",
            "--width=1280",
            "--height=720",
            "--num-inference-steps=4",
        ]
        with patch("sys.argv", argv):
            args = adapter.parse_args()
        self.assertEqual(
            (args.seed, args.width, args.height, args.num_inference_steps),
            (3, 1280, 720, 4),
        )
        with patch("sys.argv", argv[:5]):
            args = adapter.parse_args()
        self.assertEqual(
            (args.width, args.height, args.num_inference_steps), (None,) * 3
        )


if __name__ == "__main__":
    unittest.main()
