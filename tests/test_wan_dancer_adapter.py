from __future__ import annotations

import ast
import base64
import contextlib
import hashlib
import io
import json
import math
import re
import subprocess
import sys
import tarfile
import tempfile
import types
import unittest
import wave
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "contracts" / "src"))
from generated_catalog import GENERATED
from vonk_forge_contracts import (
    document_sha256,
)

ADAPTER = ROOT / "adapters/video/wan-dancer-native"
RECIPE = ROOT / "recipes/wan-dancer-14b-pytorch-single.json"
MODEL = ROOT / "models/wan-dancer-14b.json"
ARCHIVE_SHA256 = "92c529d7727c75c6515ea990d27883a45bf566587cc9f5d325a0a488b9fa1649"
PACKAGE = GENERATED / "packages/wan-dancer-14b-pytorch-single.tar.gz"
BASE_IMAGE = {
    "repository": "nvcr.io/nvidia/cuda",
    "digest": "2eaf346843c93ae617a718818f4a804e2c46c0d97a85392212ec5b310ee57962",
}
BUILD_NETWORK_HOSTS = [
    "download.pytorch.org",
    "download-r2.pytorch.org",
    "pypi.org",
    "pypi.nvidia.com",
    "files.pythonhosted.org",
    "archive.ubuntu.com",
    "security.ubuntu.com",
    "ports.ubuntu.com",
]
PLATFORM_OWNED_CACHE_ENVIRONMENT = {
    "FLASHINFER_WORKSPACE_BASE",
    "TILELANG_CACHE_DIR",
    "B12X_CUTE_COMPILE_CACHE_DIR",
    "TORCH_FR_DUMP_TEMP_FILE",
    "TORCH_NCCL_DEBUG_INFO_PIPE_FILE",
    "XDG_CACHE_HOME",
    "HF_HOME",
    "VLLM_CACHE_ROOT",
    "TRITON_CACHE_DIR",
}


def canonical_digest(path: Path) -> str:
    document = json.loads(path.read_text(encoding="utf-8"))
    return document_sha256(document)


def package_manifest() -> dict[str, object]:
    with tarfile.open(fileobj=io.BytesIO(PACKAGE.read_bytes()), mode="r:gz") as archive:
        return json.load(archive.extractfile("manifest.json"))


def load_runner():
    path = ADAPTER / "run.py"
    module = types.ModuleType("wan_dancer_runner")
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)  # noqa: S102
    return module


class WanDancerAuthorityTests(unittest.TestCase):
    def test_recipe_resolves_complete_immutable_authorities(self) -> None:
        recipe = json.loads(RECIPE.read_text(encoding="utf-8"))
        self.assertEqual(
            recipe["models"][0]["model"]["content_sha256"], canonical_digest(MODEL)
        )
        build = recipe["execution"]["build"]
        self.assertEqual(build["base_image"], BASE_IMAGE)
        self.assertEqual(build["context"], {"path": "adapters/video/wan-dancer-native"})
        self.assertEqual(
            build["dockerfile"], "adapters/video/wan-dancer-native/Dockerfile"
        )
        self.assertEqual(
            build["patches"],
            [{"path": "adapters/video/wan-dancer-native/patch-upstream.py"}],
        )
        self.assertEqual(build["network"], {"hosts": BUILD_NETWORK_HOSTS})
        self.assertEqual(
            json.loads(MODEL.read_text())["source"]["revision"],
            "85ce88dd8d025459dcf0fe93982d6da8b9002957",
        )
        self.assertEqual(
            recipe["provenance"]["source_reference"],
            "https://github.com/Wan-Video/Wan-Dancer/tree/"
            "e6c87a94ec733230dac15b924c015f6e6501e618",
        )
        self.assertEqual(recipe["topology"]["node_count"], 1)
        memory = recipe["topology"]["roles"][0]["resources"]["memory"]
        self.assertEqual(memory["peak_bytes"], 106132436089)
        self.assertEqual(memory["reserve_bytes"], 8000000000)
        runtime = recipe["runtime"]
        self.assertEqual(
            set(runtime),
            {"engine", "entrypoint", "arguments", "environment", "lifecycle"},
        )
        self.assertEqual(runtime["engine"], "pytorch-pipeline")
        environment_names = {item["name"] for item in runtime["environment"]}
        self.assertFalse(PLATFORM_OWNED_CACHE_ENVIRONMENT & environment_names)
        manifest = package_manifest()
        self.assertEqual(manifest["schema_version"], 2)
        self.assertEqual(manifest["package_type"], "recipe")
        self.assertEqual(
            manifest["build_inputs"],
            [
                {
                    "kind": "oci-image",
                    "platform": "linux/arm64",
                    "reference": "nvcr.io/nvidia/cuda:13.4.1-runtime-ubuntu24.04@sha256:"
                    "2eaf346843c93ae617a718818f4a804e2c46c0d97a85392212ec5b310ee57962",
                }
            ],
        )
        self.assertEqual(
            manifest["recipe_content_sha256"],
            document_sha256(recipe),
        )
        package_paths = {item["path"] for item in manifest["files"]}
        self.assertTrue(
            {
                build["context"]["path"] + "/Dockerfile",
                build["dockerfile"],
                build["patches"][0]["path"],
                "models/wan-dancer-14b.json",
                "recipe.json",
            }
            <= package_paths
        )
        with tarfile.open(
            fileobj=io.BytesIO(PACKAGE.read_bytes()), mode="r:gz"
        ) as archive:
            for path in (
                build["dockerfile"],
                build["patches"][0]["path"],
                "models/wan-dancer-14b.json",
            ):
                packaged = archive.extractfile(path)
                self.assertIsNotNone(packaged)
                payload = packaged.read()
                if path == "models/wan-dancer-14b.json":
                    self.assertEqual(
                        document_sha256(json.loads(payload)), canonical_digest(MODEL)
                    )
                else:
                    self.assertEqual(payload, (ROOT / path).read_bytes())
        interface = recipe["interfaces"][0]
        slots = {slot["id"]: slot for slot in interface["input"]["slots"]}
        self.assertEqual(set(slots), {"prompt", "reference-image", "music", "controls"})
        self.assertEqual(
            (slots["prompt"]["min_files"], slots["prompt"]["max_files"]), (1, 1)
        )
        self.assertEqual(
            (slots["controls"]["min_files"], slots["controls"]["max_files"]), (0, 1)
        )
        output = interface["output"]
        self.assertEqual(output["max_total_bytes"], 536870912)
        self.assertEqual(output["slots"][0]["media_types"], ["video/mp4"])
        self.assertIn("H.264/AAC", output["slots"][0]["description"])

    def test_vendored_source_is_bounded_and_fail_closed_patch_applies(self) -> None:
        parts = sorted((ADAPTER / "vendor").glob("*.part-*"))
        self.assertEqual(len(parts), 4)
        self.assertTrue(all(part.stat().st_size < 12 * 1024 * 1024 for part in parts))
        archive = b"".join(part.read_bytes() for part in parts)
        self.assertEqual(hashlib.sha256(archive).hexdigest(), ARCHIVE_SHA256)

        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            archive_path = root / "source.tar.gz"
            archive_path.write_bytes(archive)
            with tarfile.open(archive_path, "r:gz") as source:
                source.extractall(root / "unpacked", filter="data")
            source_root = next((root / "unpacked").iterdir())
            subprocess.run(
                ["python3", str(ADAPTER / "patch-upstream.py"), str(source_root)],
                check=True,
            )
            global_stage = (source_root / "gen_video/gen_video_global.py").read_text()
            local_stage = (source_root / "gen_video/gen_video_local.py").read_text()
            self.assertIn(
                'ModelConfig(path="/models/global_model.safetensors"', global_stage
            )
            self.assertIn(
                'ModelConfig(path="/models/local_model.safetensors"', local_stage
            )
            self.assertNotIn('offload_device="cpu"', global_stage + local_stage)
            self.assertNotIn("enable_vram_management()", global_stage + local_stage)
            self.assertNotIn("use_usp=True", global_stage + local_stage)
            self.assertNotIn("dist.get_rank()", global_stage + local_stage)
            self.assertIn("cfg_scale=args.cfg_scale", local_stage)
            for stage in (global_stage, local_stage):
                self.assertIn('"--cfg_scale",\n        type=float,', stage)

    def test_runtime_has_no_model_download_path(self) -> None:
        runner = (ADAPTER / "run.py").read_text(encoding="utf-8")
        dockerfile = (ADAPTER / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn(
            'org.opencontainers.image.revision="e6c87a94ec733230dac15b924c015f6e6501e618"',
            dockerfile,
        )
        self.assertNotIn("snapshot_download", runner)
        self.assertNotIn("from_pretrained", runner)
        self.assertIn("HF_HUB_OFFLINE=1", dockerfile)
        self.assertIn("TRANSFORMERS_OFFLINE=1", dockerfile)
        self.assertIn("USER 10001:10001", dockerfile)


class WanDancerInputTests(unittest.TestCase):
    def test_generation_canvas_binds_smaller_creator_dimensions(self) -> None:
        runner = load_runner()
        self.assertEqual(runner._generation_dimensions({}), (512, 512))
        self.assertEqual(
            runner._generation_dimensions({"height": 720, "width": 352}), (720, 352)
        )
        self.assertEqual(
            runner._generation_dimensions({"height": 512, "width": 512}), (512, 512)
        )
        for height, width in ((720, 1280), (1280, 720)):
            with self.assertRaisesRegex(ValueError, "at most 262144 pixels"):
                runner._generation_dimensions({"height": height, "width": width})

    def test_generation_canvas_rejects_1280_square(self) -> None:
        runner = load_runner()

        with self.assertRaisesRegex(ValueError, "at most 262144 pixels"):
            runner._generation_dimensions({"height": 1280, "width": 1280})

    def test_expected_output_is_derived_from_reference_aspect_and_music(self) -> None:
        runner = load_runner()

        class Source:
            size = (64, 64)

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def verify(self):
                return None

        pil = SimpleNamespace(
            Image=SimpleNamespace(open=lambda _path: Source()),
            UnidentifiedImageError=ValueError,
        )
        with mock.patch.dict(sys.modules, {"PIL": pil}):
            output = runner._expected_output(
                Path("person.png"),
                target_height=1280,
                target_width=720,
                music_duration=1.0,
            )
        self.assertEqual(output, (720, 720, 24, 0.8))

    def test_output_requires_derived_h264_aac_media_contract(self) -> None:
        runner = load_runner()
        probe = {
            "format": {
                "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
                "duration": "0.800000",
            },
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 720,
                    "height": 720,
                    "avg_frame_rate": "30/1",
                    "nb_read_frames": "24",
                    "duration": "0.800000",
                },
                {
                    "codec_type": "audio",
                    "codec_name": "aac",
                    "sample_rate": "44100",
                    "duration": "0.800000",
                },
            ],
        }
        completed = SimpleNamespace(stdout=json.dumps(probe))
        with mock.patch.object(runner.subprocess, "run", return_value=completed) as run:
            runner._verify_output(
                Path("dance.mp4"),
                60,
                expected_width=720,
                expected_height=720,
                expected_frames=24,
                expected_duration=0.8,
            )
        self.assertIn("-count_frames", run.call_args.args[0])

        invalid_cases = (
            (("format", "format_name"), "matroska", "MP4 container"),
            (("streams", 0, "codec_name"), "vp9", "H.264 video"),
            (("streams", 1, "codec_name"), "opus", "codec must be AAC"),
            (("streams", 1, "sample_rate"), "48000", "must be 44100 Hz"),
            (("streams", 0, "width"), 704, "dimensions"),
            (("streams", 0, "avg_frame_rate"), "29/1", "must be 30 fps"),
            (("streams", 0, "nb_read_frames"), "23", "exactly 24 frames"),
            (("streams", 0, "duration"), "0.6", "video duration"),
            (("streams", 1, "duration"), "1.1", "audio duration"),
            (("format", "duration"), "1.1", "MP4 duration"),
        )
        for path, value, message in invalid_cases:
            with self.subTest(path=path):
                invalid = json.loads(json.dumps(probe))
                target = invalid
                for part in path[:-1]:
                    target = target[part]
                target[path[-1]] = value
                with (
                    mock.patch.object(
                        runner.subprocess,
                        "run",
                        return_value=SimpleNamespace(stdout=json.dumps(invalid)),
                    ),
                    self.assertRaisesRegex(RuntimeError, message),
                ):
                    runner._verify_output(
                        Path("dance.mp4"),
                        60,
                        expected_width=720,
                        expected_height=720,
                        expected_frames=24,
                        expected_duration=0.8,
                    )

    def test_request_requires_image_music_prompt_and_rejects_traversal(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as value:
            inputs = Path(value)
            runner.INPUT_ROOT = inputs
            (inputs / "person.jpg").write_bytes(b"jpeg")
            (inputs / "music.wav").write_bytes(b"wav")
            (inputs / "request.json").write_text(
                json.dumps(
                    {
                        "reference_image": "person.jpg",
                        "music": "music.wav",
                        "prompt": "A precise K-pop dance",
                        "style": "k-pop",
                    }
                ),
                encoding="utf-8",
            )
            request = runner._request()
            self.assertEqual(request["style"], "k-pop")
            self.assertEqual(
                runner._input_file(
                    request["reference_image"], "reference_image", runner.IMAGE_SUFFIXES
                ),
                inputs / "person.jpg",
            )
            with self.assertRaises(ValueError):
                runner._input_file(
                    "../person.jpg", "reference_image", runner.IMAGE_SUFFIXES
                )

    def test_manifest_slots_supply_prompt_image_music_and_optional_controls(
        self,
    ) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as value:
            inputs = Path(value)
            runner.INPUT_ROOT = inputs
            files = {
                "prompt": inputs / "prompt.txt",
                "reference-image": inputs / "person.png",
                "music": inputs / "music.wav",
                "controls": inputs / "controls.json",
            }
            files["prompt"].write_text("A precise street dance", encoding="utf-8")
            files["reference-image"].write_bytes(b"png")
            files["music"].write_bytes(b"wav")
            files["controls"].write_text(
                json.dumps({"style": "street", "num_inference_steps_global": 2}),
                encoding="utf-8",
            )
            entries = []
            total = 0
            media_types = {
                "prompt": "text/plain",
                "reference-image": "image/png",
                "music": "audio/wav",
                "controls": "application/json",
            }
            for slot, path in files.items():
                payload = path.read_bytes()
                total += len(payload)
                entries.append(
                    {
                        "slot": slot,
                        "name": path.name,
                        "media_type": media_types[slot],
                        "size_bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                )
            (inputs / "manifest.json").write_text(
                json.dumps(
                    {"schema_version": 1, "total_bytes": total, "files": entries}
                ),
                encoding="utf-8",
            )
            manifest = runner._manifest()
            request = runner._request(manifest)
            self.assertEqual(request["prompt"], "A precise street dance")
            self.assertEqual(request["reference_image"], "person.png")
            self.assertEqual(request["music"], "music.wav")
            self.assertEqual(request["style"], "street")
            self.assertEqual(request["num_inference_steps_global"], 2)

    def test_manifest_rejects_unknown_slots_and_mismatched_media_types(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as value:
            inputs = Path(value)
            runner.INPUT_ROOT = inputs
            prompt = inputs / "prompt.txt"
            prompt.write_text("Dance", encoding="utf-8")

            def write_manifest(slot: str, media_type: str) -> None:
                payload = prompt.read_bytes()
                (inputs / "manifest.json").write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "total_bytes": len(payload),
                            "files": [
                                {
                                    "slot": slot,
                                    "name": prompt.name,
                                    "media_type": media_type,
                                    "size_bytes": len(payload),
                                    "sha256": hashlib.sha256(payload).hexdigest(),
                                }
                            ],
                        }
                    ),
                    encoding="utf-8",
                )

            write_manifest("extra", "text/plain")
            with self.assertRaisesRegex(ValueError, "unsupported slot"):
                runner._manifest()
            write_manifest("prompt", "application/json")
            with self.assertRaisesRegex(ValueError, "prompt slot contract"):
                runner._manifest()


class _Device:
    def __init__(self, kind: str) -> None:
        self.type = kind


class _Parameter:
    def __init__(self, *, is_meta: bool) -> None:
        self.is_meta = is_meta


class _Construction:
    """Records the device the pinned init_weights_on_device builds under."""

    devices: ClassVar[list[str]] = []

    @classmethod
    @contextlib.contextmanager
    def init_weights_on_device(cls, device: _Device) -> Iterator[None]:
        cls.devices.append(device.type)
        yield


class _Model:
    """The torch.nn.Module surface load_model_from_single_file uses (no torch in CI).

    Parameters are real tensors unless the constructor runs under meta, and
    load_state_dict(assign=True) replaces the ones the checkpoint names.
    """

    PARAMETERS = ("blocks.0.ffn.weight", "music_encoder.0.self_attn.in_proj_weight")

    def __init__(self, **_config: object) -> None:
        on_meta = _Construction.devices[-1] == "meta"
        self.parameters = {
            name: _Parameter(is_meta=on_meta) for name in self.PARAMETERS
        }
        self.moved = False

    @staticmethod
    def state_dict_converter() -> SimpleNamespace:
        return SimpleNamespace(from_civitai=lambda state_dict, **_: (state_dict, {}))

    def eval(self) -> _Model:
        return self

    def load_state_dict(
        self, state_dict: dict[str, object], *, assign: bool, strict: bool
    ) -> None:
        assert assign and not strict  # the pinned call
        for name in state_dict:
            if name in self.parameters:
                self.parameters[name] = _Parameter(is_meta=False)

    def named_parameters(self) -> Iterator[tuple[str, _Parameter]]:
        return iter(self.parameters.items())

    def to(self, *, dtype: object, device: object) -> _Model:
        if any(parameter.is_meta for parameter in self.parameters.values()):
            raise NotImplementedError("Cannot copy out of meta tensor; no data!")
        self.moved = True
        return self


class WanDancerLoaderTests(unittest.TestCase):
    """The patched pinned loader, executed from the patched source itself."""

    source: Path
    _directory: tempfile.TemporaryDirectory[str]

    @classmethod
    def setUpClass(cls) -> None:
        cls._directory = tempfile.TemporaryDirectory()
        root = Path(cls._directory.name)
        archive = b"".join(
            part.read_bytes() for part in sorted((ADAPTER / "vendor").glob("*.part-*"))
        )
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as source:
            source.extractall(root, filter="data")
        cls.source = next(root.iterdir())
        subprocess.run(
            [sys.executable, str(ADAPTER / "patch-upstream.py"), str(cls.source)],
            check=True,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._directory.cleanup()

    def _loader(self) -> Callable[..., tuple[list[str], list[_Model]]]:
        path = self.source / "diffsynth/models/model_manager.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "load_model_from_single_file"
        )
        namespace: dict[str, Any] = {
            "torch": SimpleNamespace(device=_Device, float32="float32"),
            "init_weights_on_device": _Construction.init_weights_on_device,
            "logger": SimpleNamespace(info=lambda *_: None),
        }
        module = ast.Module(body=[function], type_ignores=[])
        exec(compile(module, str(path), "exec"), namespace)  # noqa: S102
        return namespace["load_model_from_single_file"]

    def _load(self, state_dict: dict[str, object]) -> _Model:
        _Construction.devices.clear()
        _, models = self._loader()(
            state_dict, ["wan_video_dit"], [_Model], "civitai", "bf16", "cuda"
        )
        return models[0]

    def test_models_are_built_on_meta_and_moved_once_loaded(self) -> None:
        model = self._load({name: object() for name in _Model.PARAMETERS})
        self.assertEqual(_Construction.devices, ["meta"])
        self.assertTrue(model.moved)

    def test_parameter_missing_from_the_checkpoint_fails_with_its_name(self) -> None:
        missing = "music_encoder.0.self_attn.in_proj_weight"
        complete = {name: object() for name in _Model.PARAMETERS}
        with self.assertRaisesRegex(RuntimeError, re.escape(missing)):
            self._load({k: v for k, v in complete.items() if k != missing})

    def test_dit_constructor_leaves_device_placement_to_the_loader(self) -> None:
        tree = ast.parse(
            (self.source / "diffsynth/models/wan_video_dit.py").read_text(
                encoding="utf-8"
            )
        )
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "MusicEncoderLayer"
        ]
        self.assertTrue(calls)
        for call in calls:
            self.assertNotIn("device", {keyword.arg for keyword in call.keywords})


class WanDancerSmokeMusicTests(unittest.TestCase):
    """The smoke music must satisfy upstream's frame math, not only the adapter.

    gen_video_global.py:353 computes ``30 / int(rows / 149 + 0.5)``, where rows
    is the 30 fps onset-envelope length (``1 + samples // 512`` at 15,360 Hz), so
    music shorter than about 2.47 s divides by zero after the models load.
    gen_video_local.py keeps one 149-frame segment only up to 149 / 30 seconds.
    """

    def test_smoke_music_fits_upstream_frame_math_and_the_output_assertion(
        self,
    ) -> None:
        runner = load_runner()
        definition = json.loads(
            (
                ROOT / "qualification/recipes/wan-dancer-14b-pytorch-single.json"
            ).read_text(encoding="utf-8")
        )["recipes"]
        fixtures = json.loads(
            (ROOT / "qualification/shared.json").read_text(encoding="utf-8")
        )["fixtures"]
        recipe = json.loads(RECIPE.read_text(encoding="utf-8"))
        validation = recipe["validation"]["serving"]["checks"][0]["request"]
        sources = [
            base64.b64decode(
                (ROOT / "qualification" / fixtures[item["fixture"]]["path"]).read_text()
            )
            for inputs in [definition["inputs"]]
            + [case["inputs"] for case in definition["cases"]]
            for item in inputs
            if item["slot"] == "music"
        ]
        sources.append((ROOT / validation["input_slots"]["music"]).read_bytes())
        (metadata,) = [
            item
            for item in definition["assertions"]
            if item["kind"] == "video-metadata"
        ]
        for content in sources:
            with wave.open(io.BytesIO(content), "rb") as source:
                frames, rate = source.getnframes(), source.getframerate()
            duration = frames / rate
            rows = 1 + math.ceil(frames * 30 * 512 / rate) // 512
            self.assertGreaterEqual(int(rows / 149.0 + 0.5), 1, (duration, rows))
            self.assertLess(duration, 149 / 30)
            self.assertLessEqual(1.0, duration)
            output_frames = int(
                (duration - runner.OUTPUT_TRIM_SECONDS) * runner.OUTPUT_FPS
            )
            self.assertEqual(metadata["frame_count"], output_frames)


if __name__ == "__main__":
    unittest.main()
