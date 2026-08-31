import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "adapters" / "media" / "comfyui-core"
RECIPE_SLUGS = (
    "flux-2-klein-4b-comfyui-single",
    "qwen-image-2512-comfyui-single",
    "qwen-image-2512-fp8-lightning-comfyui-single",
    "qwen-image-edit-2511-comfyui-single",
    "wan-2-2-i2v-14b-comfyui-single",
    "wan-2-2-t2v-14b-comfyui-single",
    "wan-2-2-ti2v-5b-comfyui-single",
)
ALL_COMFY_RECIPE_SLUGS = (
    "flux-2-klein-4b-nvfp4-comfyui-single",
    "qwen-image-edit-2511-fp8mixed-comfyui-single",
    "qwen-image-edit-2511-int8-convrot-comfyui-single",
    *RECIPE_SLUGS,
)
COMFY_REVISION = "12d5279438bfefc058a269eae805ceab6047777f"
COMFY_ARCHIVE_SHA256 = (
    "6d8ab87ec1250e60101f0caf9e11658834c29d9cd76c9174e2b84ec9436f4886"
)
RUNTIME_SLUG = "comfyui-0-34-0-cuda13-arm64"
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


class ComfyUICoreRecipeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        spec = importlib.util.spec_from_file_location(
            "comfyui_job", ADAPTER / "comfyui_job.py"
        )
        assert spec and spec.loader
        cls.module = importlib.util.module_from_spec(spec)
        previous = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            spec.loader.exec_module(cls.module)
        finally:
            sys.dont_write_bytecode = previous

    def test_all_recipes_bind_hash_locked_core_workflows(self) -> None:
        for slug in RECIPE_SLUGS:
            with self.subTest(slug=slug):
                recipe = json.loads((ROOT / "recipes" / f"{slug}.json").read_text())
                arguments = {
                    item["name"]: item["value"]
                    for item in recipe["runtime"]["arguments"]
                }
                context = ROOT / recipe["build"]["context"]["path"]
                workflow = context / Path(arguments["workflow"]).name
                if not workflow.exists():
                    workflow = context / "workflows" / Path(arguments["workflow"]).name
                self.assertEqual(
                    hashlib.sha256(workflow.read_bytes()).hexdigest(),
                    arguments["workflow-sha256"],
                )
                document = json.loads(workflow.read_text())
                prompt_input = document["text_inputs"]["prompt"]
                self.assertTrue(prompt_input["required"])
                self.assertEqual(prompt_input["slot"], "prompt")
                self.assertEqual(prompt_input["maximum_bytes"], 16384)
                node_types = {
                    node["class_type"] for node in document["prompt"].values()
                }
                self.assertLessEqual(node_types, CORE_NODES)
                self.assertTrue(node_types & {"SaveImage", "SaveVideo"})
                artifact_ids = {artifact["id"] for artifact in recipe["artifacts"]}
                workflow_ids = (
                    {item["artifact_id"] for item in document.get("models", [])}
                    | {
                        item["artifact_id"]
                        for item in document.get("diffusers_snapshots", [])
                    }
                    | {
                        item["artifact_id"]
                        for item in document.get("snapshot_models", [])
                    }
                )
                self.assertEqual(artifact_ids, workflow_ids)

    def test_runtime_is_pinned_and_has_no_custom_node_supply_chain(self) -> None:
        dockerfile = (ADAPTER / "Dockerfile").read_text()
        self.assertIn(COMFY_REVISION, dockerfile)
        self.assertIn(COMFY_ARCHIVE_SHA256, dockerfile)
        self.assertNotIn("ComfyUI-Manager", dockerfile)
        self.assertNotIn("custom_nodes", dockerfile)
        for slug in RECIPE_SLUGS:
            recipe = json.loads((ROOT / "recipes" / f"{slug}.json").read_text())
            recipe_dockerfile = (
                ROOT / recipe["build"]["context"]["path"] / "Dockerfile"
            ).read_text()
            self.assertIn(COMFY_REVISION, recipe_dockerfile)
            self.assertNotIn("ComfyUI-Manager", recipe_dockerfile)
            self.assertNotIn("custom_nodes", recipe_dockerfile)
            self.assertIn("candidate", recipe["metadata"]["tags"])
            self.assertEqual(recipe["execution"]["harness"]["slug"], "comfyui")
            self.assertFalse(recipe["runtime"]["security"]["host_network"])

    def test_all_comfy_recipes_bind_the_exact_034_runtime_closure(self) -> None:
        runtime = json.loads(
            (ROOT / "runtime-distributions" / f"{RUNTIME_SLUG}.json").read_text()
        )
        canonical = json.dumps(
            runtime,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
        runtime_digest = hashlib.sha256(canonical).hexdigest()
        self.assertEqual(runtime["source"]["revision"], COMFY_REVISION)
        self.assertEqual(runtime["source"]["archive_sha256"], COMFY_ARCHIVE_SHA256)
        dependencies = {item["name"]: item for item in runtime["dependencies"]}
        self.assertEqual(dependencies["ComfyUI"]["version"], "0.34.0")
        self.assertEqual(
            dependencies["comfyui-workflow-templates"]["version"], "0.11.48"
        )
        for slug in ALL_COMFY_RECIPE_SLUGS:
            with self.subTest(slug=slug):
                recipe = json.loads(
                    (ROOT / "recipes" / f"{slug}.json").read_text()
                )
                distribution = recipe["runtime"]["distribution"]
                self.assertEqual(distribution["slug"], RUNTIME_SLUG)
                self.assertEqual(distribution["content_sha256"], runtime_digest)

    def test_qwen_2512_uses_the_published_quality_defaults(self) -> None:
        recipe = json.loads(
            (ROOT / "recipes/qwen-image-2512-comfyui-single.json").read_text()
        )
        workflow = json.loads(
            (
                ROOT
                / recipe["build"]["context"]["path"]
                / "workflows/qwen-image-2512-bf16.json"
            ).read_text()
        )
        self.assertEqual(
            workflow["prompt"]["6"]["inputs"]["text"],
            "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，"
            "过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。",
        )
        arguments = {
            item["name"]: item["value"] for item in recipe["runtime"]["arguments"]
        }
        self.assertEqual(
            arguments["workflow-sha256"],
            hashlib.sha256(
                (
                    ROOT
                    / recipe["build"]["context"]["path"]
                    / "workflows/qwen-image-2512-bf16.json"
                ).read_bytes()
            ).hexdigest(),
        )
        benchmark = recipe["validation"]["benchmarks"][0]["configuration"]
        self.assertEqual(
            (benchmark["width"], benchmark["height"], benchmark["steps"]),
            (1328, 1328, 50),
        )

    def test_lightning_targets_claim_only_implemented_harnesses(self) -> None:
        targets = json.loads((ROOT / "model-targets/image.json").read_text())["targets"]
        lightning = [
            item for item in targets if item["model"] == "Qwen Image 2512 Lightning"
        ]
        self.assertEqual(len(lightning), 2)
        target = next(item for item in lightning if item["harnesses"] == ["diffusers"])
        self.assertEqual(
            target["recipe_slugs"],
            ["qwen-image-2512-lightning-diffusers-single"],
        )
        target = next(item for item in lightning if item["harnesses"] == ["comfyui"])
        self.assertEqual(
            target["recipe_slugs"],
            ["qwen-image-2512-fp8-lightning-comfyui-single"],
        )

    def test_wan_workflows_declare_exact_prompt_and_video_contracts(self) -> None:
        expected = {
            "wan-2-2-i2v-14b-comfyui-single": (640, 640, 16, 81),
            "wan-2-2-t2v-14b-comfyui-single": (640, 640, 16, 81),
            "wan-2-2-ti2v-5b-comfyui-single": (1280, 704, 24, 121),
        }
        for slug, media in expected.items():
            with self.subTest(slug=slug):
                recipe = json.loads((ROOT / "recipes" / f"{slug}.json").read_text())
                arguments = {
                    item["name"]: item["value"]
                    for item in recipe["runtime"]["arguments"]
                }
                workflow = json.loads(
                    (
                        ADAPTER / "workflows" / Path(arguments["workflow"]).name
                    ).read_text()
                )
                self.assertTrue(workflow["text_inputs"]["prompt"]["required"])
                self.assertEqual(workflow["text_inputs"]["prompt"]["slot"], "prompt")
                if slug != "wan-2-2-t2v-14b-comfyui-single":
                    self.assertEqual(workflow["inputs"]["slot"], "image")
                self.assertEqual(workflow["result"]["mime"], "video/mp4")
                self.assertEqual(workflow["result"]["count"], 1)
                video = workflow["result"]["video"]
                self.assertEqual(video["codec"], "h264")
                self.assertEqual(
                    (video["width"], video["height"], video["fps"], video["frames"]),
                    media,
                )
                benchmark = recipe["validation"]["benchmarks"][0]["configuration"]
                self.assertEqual(
                    (
                        benchmark["width"],
                        benchmark["height"],
                        benchmark["fps"],
                        benchmark["frames"],
                    ),
                    media,
                )
                interface = recipe["interfaces"][0]
                slots = {slot["id"]: slot for slot in interface["input"]["slots"]}
                self.assertEqual(slots["prompt"]["media_types"], ["text/plain"])
                self.assertEqual(
                    (slots["prompt"]["min_files"], slots["prompt"]["max_files"]), (1, 1)
                )
                if slug == "wan-2-2-t2v-14b-comfyui-single":
                    self.assertNotIn("image", slots)
                else:
                    self.assertEqual(
                        (slots["image"]["min_files"], slots["image"]["max_files"]),
                        (1, 1),
                    )
                output = interface["output"]
                self.assertEqual(output["max_total_bytes"], 536870912)
                self.assertEqual(output["slots"][0]["media_types"], ["video/mp4"])
                self.assertEqual(
                    (output["slots"][0]["min_files"], output["slots"][0]["max_files"]),
                    (1, 1),
                )

    def test_empty_required_prompt_is_rejected_before_model_linking(self) -> None:
        document = {
            "text_inputs": {
                "prompt": {"required": True, "maximum_characters": 4096},
            },
            "prompt": {},
        }
        arguments = SimpleNamespace(
            workflow="wan.json",
            workflow_sha256="0" * 64,
            output_mime="video/mp4",
            seed=0,
            output_dir="/outputs",
        )
        with (
            mock.patch.object(self.module, "parse_args", return_value=arguments),
            mock.patch.object(self.module, "load_workflow", return_value=document),
            mock.patch.object(self.module, "load_input_manifest", return_value=None),
            mock.patch.object(self.module, "input_names", return_value=[]),
            mock.patch.object(self.module, "link_models") as link_models,
            mock.patch.dict(os.environ, {"VONK_PROMPT": " \t "}),
            self.assertRaisesRegex(ValueError, "non-empty prompt"),
        ):
            self.module.main()
        link_models.assert_not_called()

    def test_manifest_prompt_is_verified_and_excluded_from_image_inputs(self) -> None:
        document = {
            "inputs": {"minimum": 1, "maximum": 1, "slot": "image"},
            "text_inputs": {
                "prompt": {
                    "required": True,
                    "slot": "prompt",
                    "maximum_bytes": 16384,
                    "maximum_characters": 4096,
                }
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompt = root / "prompt.txt"
            image = root / "reference.png"
            prompt.write_text("A red panda dancing", encoding="utf-8")
            image.write_bytes(b"png")
            write_input_manifest(root, {"prompt": prompt, "image": image})
            previous = self.module.INPUT_ROOT
            self.module.INPUT_ROOT = root
            try:
                manifest = self.module.load_input_manifest()
                names = self.module.input_names(document, manifest)
                replacements = self.module.workflow_replacements(
                    document, names, 7, manifest
                )
            finally:
                self.module.INPUT_ROOT = previous
        self.assertEqual(names, ["reference.png"])
        self.assertEqual(replacements["__VONK_PROMPT__"], "A red panda dancing")
        self.assertEqual(replacements["__VONK_INPUT_1__"], "reference.png")
        self.assertEqual(replacements["__VONK_SEED__"], 7)

    def test_qwen_edit_single_image_repeats_second_reference_deterministically(
        self,
    ) -> None:
        document = json.loads(
            (ADAPTER / "workflows/qwen-image-edit-2511-bf16.json").read_text()
        )
        with mock.patch.dict(os.environ, {"VONK_PROMPT": "Edit the reference"}):
            replacements = self.module.workflow_replacements(
                document,
                ["only.png"],
                0,
                None,
            )
        self.assertEqual(replacements["__VONK_INPUT_1__"], "only.png")
        self.assertEqual(replacements["__VONK_INPUT_2__"], "only.png")

    def test_output_selection_rejects_zero_multiple_and_unexpected_media(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "comfy"
            destination = root / "result"
            output.mkdir()
            with self.assertRaises(FileNotFoundError):
                self.module.copy_output(output, destination, "video/mp4")

            (output / "one.mp4").write_bytes(b"one")
            (output / "two.mp4").write_bytes(b"two")
            with self.assertRaisesRegex(RuntimeError, "2 video/mp4 outputs"):
                self.module.copy_output(output, destination, "video/mp4")

            (output / "two.mp4").unlink()
            (output / "preview.png").write_bytes(b"preview")
            with self.assertRaisesRegex(RuntimeError, "unexpected media outputs"):
                self.module.copy_output(output, destination, "video/mp4")

    def test_mp4_contract_accepts_only_exact_declared_media(self) -> None:
        contract = {
            "mime": "video/mp4",
            "count": 1,
            "video": {
                "codec": "h264",
                "width": 640,
                "height": 640,
                "fps": 16,
                "frames": 81,
            },
        }
        exact_probe = {
            "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": "5.0625"},
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 640,
                    "height": 640,
                    "avg_frame_rate": "16/1",
                    "nb_read_frames": "81",
                    "duration": "5.0625",
                }
            ],
        }
        completed = subprocess_result(exact_probe)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "comfy"
            destination = root / "result"
            source.mkdir()
            (source / "wan.mp4").write_bytes(b"valid-video-placeholder")
            with mock.patch.object(
                self.module.subprocess, "run", return_value=completed
            ) as run:
                result = self.module.copy_output(
                    source, destination, "video/mp4", contract
                )
            self.assertEqual(result.read_bytes(), b"valid-video-placeholder")
            self.assertIn("-count_frames", run.call_args.args[0])

        invalid = {
            "codec": ("codec_name", "vp9", "codec must be h264"),
            "width": ("width", 704, "width must be 640"),
            "height": ("height", 704, "height must be 640"),
            "fps": ("avg_frame_rate", "15/1", "frame rate must be 16 fps"),
            "frames": ("nb_read_frames", "80", "exactly 81 frames"),
            "duration": ("duration", "6.0", "duration must match"),
        }
        for name, (field, value, message) in invalid.items():
            with self.subTest(field=name):
                probe = json.loads(json.dumps(exact_probe))
                probe["streams"][0][field] = value
                with (
                    tempfile.NamedTemporaryFile(suffix=".mp4") as media,
                    mock.patch.object(
                        self.module.subprocess,
                        "run",
                        return_value=subprocess_result(probe),
                    ),
                    self.assertRaisesRegex(RuntimeError, message),
                ):
                    self.module.validate_mp4(Path(media.name), contract)


def subprocess_result(document: dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(stdout=json.dumps(document))


def write_input_manifest(root: Path, files: dict[str, Path]) -> None:
    entries = []
    total = 0
    for slot, path in files.items():
        payload = path.read_bytes()
        total += len(payload)
        entries.append(
            {
                "slot": slot,
                "name": path.name,
                "media_type": "text/plain" if path.suffix == ".txt" else "image/png",
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    (root / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "total_bytes": total, "files": entries}),
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
