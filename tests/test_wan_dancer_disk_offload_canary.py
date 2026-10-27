from __future__ import annotations

import ast
import hashlib
import io
import json
import subprocess
import sys
import tarfile
import tempfile
import types
import unittest
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "contracts" / "src"))
from generated_catalog import GENERATED
from vonk_forge_contracts import (
    RecipeDefinition,
    document_sha256,
)

from qualification.memory_need import require_admissible_declaration

ADAPTER = ROOT / "adapters/video/wan-dancer-diffsynth-disk"
RECIPE = ROOT / "recipes/wan-dancer-14b-disk-offload-pytorch-single.json"
MODEL = ROOT / "models/wan-dancer-14b.json"
ARCHIVE = (
    ADAPTER / "vendor/diffsynth-studio-84f93fc4907b6c193be5501bab0b5c37f383033c.tar.gz"
)
ARCHIVE_SHA256 = "5f0dfef5351341613e2c8ba96806bddb576e5e1f44aaa104c5d5c388cf44bc1b"
DIFFSYNTH_REVISION = "84f93fc4907b6c193be5501bab0b5c37f383033c"
PACKAGE = GENERATED / "packages/wan-dancer-14b-disk-offload-pytorch-single.tar.gz"
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
    return document_sha256(json.loads(path.read_text(encoding="utf-8")))


def package_manifest() -> dict[str, object]:
    with tarfile.open(fileobj=io.BytesIO(PACKAGE.read_bytes()), mode="r:gz") as archive:
        return json.load(archive.extractfile("manifest.json"))


def load_module(filename: str, name: str):
    path = ADAPTER / filename
    module = types.ModuleType(name)
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)  # noqa: S102
    return module


class WanDancerDiskOffloadAuthorityTests(unittest.TestCase):
    def test_canary_resolves_pinned_model_runtime_and_bounded_memory(self) -> None:
        recipe = json.loads(RECIPE.read_text(encoding="utf-8"))
        self.assertEqual(
            recipe["models"][0]["model"]["content_sha256"], canonical_digest(MODEL)
        )
        build = recipe["execution"]["build"]
        self.assertEqual(build["base_image"], BASE_IMAGE)
        self.assertEqual(
            build["context"], {"path": "adapters/video/wan-dancer-diffsynth-disk"}
        )
        self.assertEqual(
            build["dockerfile"], "adapters/video/wan-dancer-diffsynth-disk/Dockerfile"
        )
        self.assertEqual(
            build["patches"],
            [{"path": "adapters/video/wan-dancer-diffsynth-disk/patch-runtime.py"}],
        )
        self.assertEqual(build["network"], {"hosts": BUILD_NETWORK_HOSTS})
        dockerfile = (ADAPTER / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn(
            'org.opencontainers.image.revision="84f93fc4907b6c193be5501bab0b5c37f383033c"',
            dockerfile,
        )
        self.assertIn(
            "diffsynth-studio-84f93fc4907b6c193be5501bab0b5c37f383033c.tar.gz",
            dockerfile,
        )
        self.assertEqual(
            json.loads(MODEL.read_text())["source"]["revision"],
            "85ce88dd8d025459dcf0fe93982d6da8b9002957",
        )
        self.assertEqual(
            recipe["provenance"]["source_reference"],
            "https://github.com/modelscope/DiffSynth-Studio/tree/"
            "84f93fc4907b6c193be5501bab0b5c37f383033c",
        )
        self.assertTrue(
            {"experimental", "disk-offload", "physical-oom-gated", "not-default"}
            <= set(recipe["metadata"]["tags"])
        )

        memory = recipe["topology"]["roles"][0]["resources"]["memory"]
        envelope = memory["peak_bytes"] + memory["reserve_bytes"]
        self.assertEqual(envelope, 126_000_000_000)
        require_admissible_declaration(RecipeDefinition.model_validate(recipe))

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

    def test_vendored_diffsynth_source_is_exact_and_offline_patch_applies(self) -> None:
        self.assertEqual(
            hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(), ARCHIVE_SHA256
        )
        dockerfile = (ADAPTER / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn(
            f'org.opencontainers.image.revision="{DIFFSYNTH_REVISION}"',
            dockerfile,
        )
        self.assertIn(
            f"diffsynth-studio-{DIFFSYNTH_REVISION}.tar.gz",
            dockerfile,
        )

        with tempfile.TemporaryDirectory() as value:
            source_root = Path(value) / "source"
            source_root.mkdir()
            with tarfile.open(ARCHIVE, "r:gz") as source:
                source.extractall(source_root, filter="data")
            unpacked = next(source_root.iterdir())
            subprocess.run(
                ["python3", str(ADAPTER / "patch-runtime.py"), str(unpacked)],
                check=True,
            )
            config = (unpacked / "diffsynth/core/loader/config.py").read_text()
            data_init = (unpacked / "diffsynth/core/data/__init__.py").read_text()
            self.assertNotIn("from modelscope import", config)
            self.assertNotIn("from huggingface_hub import", config)
            self.assertIn("network model loading is disabled", config)
            self.assertNotIn("UnifiedDataset", data_init)

            # Disk offload materializes only wrapped modules; the music encoder's
            # attention must be wrapped or its in_proj params stay on meta.
            music_encoder = (unpacked / "diffsynth/models/wantodance.py").read_text()
            self.assertIn("self.self_attn = nn.MultiheadAttention(", music_encoder)
            maps_path = unpacked / "diffsynth/configs/vram_management_module_maps.py"
            maps: dict[str, dict[str, dict[str, str]]] = {}
            exec(compile(maps_path.read_bytes(), str(maps_path), "exec"), maps)  # noqa: S102
            wan_model_map = maps["VRAM_MANAGEMENT_MODULE_MAPS"][
                "diffsynth.models.wan_video_dit.WanModel"
            ]
            self.assertEqual(
                wan_model_map["torch.nn.MultiheadAttention"],
                "diffsynth.core.vram.layers.AutoWrappedModule",
            )

    def test_pinned_global_stage_emits_one_keyframe_per_latent(self) -> None:
        generator = load_module("generate.py", "wan_dancer_disk_generator")
        with tarfile.open(ARCHIVE, "r:gz") as source:
            members = {
                Path(member.name).relative_to(Path(member.name).parts[0]).as_posix(): (
                    member
                )
                for member in source.getmembers()
                if member.isfile()
            }

            def pinned_method(
                path: str, class_name: str, method: str, namespace: dict[str, Any]
            ) -> Callable[..., Any]:
                payload = source.extractfile(members[path])
                assert payload is not None
                tree = ast.parse(payload.read())
                owner = next(
                    node
                    for node in tree.body
                    if isinstance(node, ast.ClassDef) and node.name == class_name
                )
                function = next(
                    node
                    for node in owner.body
                    if isinstance(node, ast.FunctionDef) and node.name == method
                )
                module = ast.Module(body=[function], type_ignores=[])
                exec(compile(module, path, "exec"), namespace)  # noqa: S102
                return namespace[method]

            initialize_noise = pinned_method(
                "diffsynth/pipelines/wan_video.py",
                "WanVideoUnit_NoiseInitializer",
                "process",
                {"WanVideoPipeline": object},
            )
            decode_framewise = pinned_method(
                "diffsynth/models/wan_video_vae.py",
                "WanVideoVAE",
                "decode_framewise",
                {
                    "torch": SimpleNamespace(
                        concat=lambda parts, dim: [
                            frame for part in parts for frame in part
                        ]
                    )
                },
            )

        pipe = SimpleNamespace(
            vae=SimpleNamespace(model=SimpleNamespace(z_dim=16), upsampling_factor=8),
            generate_noise=lambda shape, seed, rand_device: shape,
        )
        noise = initialize_noise(
            None, pipe, 1280, 720, generator.FRAME_COUNT, 0, "cpu", None, None
        )["noise"]
        self.assertEqual(noise[2], generator.GLOBAL_KEYFRAMES)

        class Latents:
            shape = noise

            def __getitem__(self, key):
                return key

        decoded = []
        vae = SimpleNamespace(
            single_decode=lambda latent, _device: decoded.append(latent[2]) or [latent]
        )
        frames = decode_framewise(vae, Latents(), "cuda")
        self.assertEqual(len(frames), generator.GLOBAL_KEYFRAMES)
        self.assertEqual(
            decoded, [slice(k, k + 1) for k in range(generator.GLOBAL_KEYFRAMES)]
        )

    def test_runtime_uses_disk_mapping_process_isolation_and_no_downloads(self) -> None:
        generator = (ADAPTER / "generate.py").read_text(encoding="utf-8")
        runner = (ADAPTER / "run.py").read_text(encoding="utf-8")
        dockerfile = (ADAPTER / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn('"offload_dtype": "disk"', generator)
        self.assertIn('"offload_device": "disk"', generator)
        self.assertIn("VRAM_LIMIT_GIB = 64.0", generator)
        self.assertIn('"--stage",\n            "global"', runner)
        self.assertIn('"--stage",\n            "local"', runner)
        self.assertIn("DIFFSYNTH_SKIP_DOWNLOAD=True", dockerfile)
        self.assertIn("HF_HUB_OFFLINE=1", dockerfile)
        self.assertIn("USER 10001:10001", dockerfile)
        self.assertNotIn("git clone", dockerfile)


class WanDancerDiskOffloadInputTests(unittest.TestCase):
    def test_generation_canvas_accepts_720p_and_rejects_a_larger_canvas(self) -> None:
        runner = load_module("run.py", "wan_dancer_disk_runner")
        self.assertEqual(
            runner._generation_dimensions({"height": 720, "width": 1280}),
            (720, 1280),
        )
        self.assertEqual(
            runner._generation_dimensions({"height": 1280, "width": 720}),
            (1280, 720),
        )
        with self.assertRaisesRegex(ValueError, "at most 921600 pixels"):
            runner._generation_dimensions({"height": 1280, "width": 1280})

    def _local_keyframes(self, source_frames: int, output_frames: int):
        generator = load_module("generate.py", "wan_dancer_disk_generator")

        class Image:
            @staticmethod
            def new(_mode, _size, _color):
                return "black"

        class VideoData:
            def __init__(self, **_kwargs):
                self.frames = [f"keyframe-{index}" for index in range(source_frames)]

            def __len__(self):
                return len(self.frames)

            def __getitem__(self, index):
                return self.frames[index]

        modules = {
            "PIL": SimpleNamespace(Image=Image),
            "diffsynth": types.ModuleType("diffsynth"),
            "diffsynth.utils": types.ModuleType("diffsynth.utils"),
            "diffsynth.utils.data": SimpleNamespace(VideoData=VideoData),
        }
        with mock.patch.dict(sys.modules, modules):
            return generator._local_keyframes(
                Path("global.mp4"), output_frames, 1280, 720
            )

    def test_local_keyframes_place_every_global_keyframe_on_its_latent_frame(
        self,
    ) -> None:
        generator = load_module("generate.py", "wan_dancer_disk_generator")
        runner = load_module("run.py", "wan_dancer_disk_runner")
        from PIL import Image

        with tempfile.TemporaryDirectory() as value:
            image = Path(value) / "reference.png"
            Image.new("RGB", (720, 1280)).save(image)
            # The longest music run.py accepts gives the most output frames.
            *_, output_frames, _ = runner._expected_output(
                image, target_height=1280, target_width=720, music_duration=5.0
            )
        self.assertEqual(output_frames, 144)

        frames, mask = self._local_keyframes(generator.GLOBAL_KEYFRAMES, output_frames)
        positions = [index for index, flag in enumerate(mask) if flag]
        self.assertEqual(len(mask), generator.FRAME_COUNT)
        self.assertEqual(positions, [4 * k for k in range(38)])
        self.assertEqual(positions[-1], generator.FRAME_COUNT - 1)
        self.assertEqual(
            [frames[index] for index in positions],
            [f"keyframe-{k}" for k in range(38)],
        )
        self.assertEqual(
            {frames[index] for index in range(len(frames)) if not mask[index]},
            {"black"},
        )

    def test_local_stage_rejects_a_global_video_with_one_frame_per_local_frame(
        self,
    ) -> None:
        generator = load_module("generate.py", "wan_dancer_disk_generator")
        with self.assertRaisesRegex(SystemExit, "expected 38"):
            self._local_keyframes(generator.FRAME_COUNT, 24)


class _FakeParameter:
    def __init__(self, *, is_meta: bool) -> None:
        self.is_meta = is_meta


class _FakeModule:
    """The torch.nn.Module traversal surface the disk net uses (no torch in CI)."""

    def __init__(self, **members: object) -> None:
        self._parameters: dict[str, _FakeParameter] = {}
        self._modules: dict[str, _FakeModule] = {}
        for name, value in members.items():
            setattr(self, name, value)

    def __setattr__(self, name: str, value: object) -> None:
        if isinstance(value, _FakeParameter):
            self._parameters[name] = value
        elif isinstance(value, _FakeModule):
            self._modules[name] = value
        else:
            object.__setattr__(self, name, value)

    def __getattr__(self, name: str):
        for members in ("_modules", "_parameters"):
            if name in self.__dict__.get(members, {}):
                return self.__dict__[members][name]
        raise AttributeError(name)

    def named_children(self):
        return iter(self._modules.items())

    def named_modules(self, prefix: str = ""):
        yield prefix, self
        for name, child in self._modules.items():
            yield from child.named_modules(f"{prefix}.{name}" if prefix else name)

    def named_parameters(self, prefix: str = "", recurse: bool = True):
        modules = self.named_modules(prefix) if recurse else [(prefix, self)]
        for path, module in modules:
            for name, parameter in module._parameters.items():
                yield (f"{path}.{name}" if path else name), parameter


class _AutoTorchModule(_FakeModule):
    """diffsynth.core.vram.layers.AutoTorchModule (layers.py:8): sets ``name``."""


class _AutoWrappedModule(_AutoTorchModule):
    """layers.py:93-129: wraps ``module``; with offload_dtype "disk" it sets
    ``name``, ``disk_map``, ``disk_offload = True`` and ``required_params`` from
    ``module.named_parameters()``; otherwise only ``disk_offload = False``."""

    def __init__(self, module: _FakeModule, *, disk: bool = True) -> None:
        super().__init__(module=module, disk_offload=disk)
        if disk:
            self.required_params = [name for name, _ in module.named_parameters()]


class _AutoWrappedLinear(_AutoTorchModule):
    """layers.py:295-341: owns ``weight``/``bias`` itself, sets ``disk_offload``
    and has no ``required_params``; load_from_disk reads ``name.weight/bias``."""

    def __init__(self) -> None:
        super().__init__(
            weight=_FakeParameter(is_meta=True),
            bias=_FakeParameter(is_meta=True),
            disk_offload=True,
        )


class WanDancerDiskLoadableNetTests(unittest.TestCase):
    def _require(self, pipe: _FakeModule) -> None:
        generator = load_module("generate.py", "wan_dancer_disk_net")
        layers = types.ModuleType("diffsynth.core.vram.layers")
        layers.AutoTorchModule = _AutoTorchModule  # type: ignore[attr-defined]
        modules = {
            "diffsynth": types.ModuleType("diffsynth"),
            "diffsynth.core": types.ModuleType("diffsynth.core"),
            "diffsynth.core.vram": types.ModuleType("diffsynth.core.vram"),
            "diffsynth.core.vram.layers": layers,
        }
        with mock.patch.dict(sys.modules, modules):
            generator._require_disk_loadable(pipe)

    @staticmethod
    def _pipe(self_attn: _FakeModule) -> _FakeModule:
        layer = _FakeModule(self_attn=self_attn, linear1=_AutoWrappedLinear())
        dit = _FakeModule(
            # Root parameter materialized from the disk map (layers.py:563).
            music_bias=_FakeParameter(is_meta=False),
            music_encoder=_FakeModule(**{"0": layer}),
        )
        visual = _FakeModule(proj=_FakeParameter(is_meta=True))
        image_encoder = _FakeModule(
            model=_FakeModule(
                visual=_AutoWrappedModule(visual),
                log_scale=_FakeParameter(is_meta=True),
            )
        )
        return _FakeModule(dit=dit, image_encoder=image_encoder)

    @staticmethod
    def _attention() -> _FakeModule:
        return _FakeModule(
            in_proj_weight=_FakeParameter(is_meta=True),
            in_proj_bias=_FakeParameter(is_meta=True),
            out_proj=_AutoWrappedLinear(),
        )

    def test_unwrapped_nested_parameter_fails_with_its_names(self) -> None:
        with self.assertRaises(SystemExit) as raised:
            self._require(self._pipe(self._attention()))
        message = str(raised.exception)
        self.assertIn("dit.music_encoder.0.self_attn.in_proj_weight", message)
        self.assertIn("dit.music_encoder.0.self_attn.in_proj_bias", message)
        self.assertNotIn("out_proj", message)
        self.assertNotIn("log_scale", message)

    def test_wrapped_parameters_pass(self) -> None:
        self._require(self._pipe(_AutoWrappedModule(self._attention())))

    def test_wrapper_without_disk_offload_does_not_cover(self) -> None:
        with self.assertRaises(SystemExit) as raised:
            self._require(self._pipe(_AutoWrappedModule(self._attention(), disk=False)))
        self.assertIn("self_attn.module.in_proj_weight", str(raised.exception))

    def test_only_reviewed_unread_parameters_may_stay_meta(self) -> None:
        pipe = self._pipe(_AutoWrappedModule(self._attention()))
        pipe.image_encoder.model.logit_bias = _FakeParameter(is_meta=True)
        with self.assertRaises(SystemExit) as raised:
            self._require(pipe)
        self.assertIn("image_encoder.model.logit_bias", str(raised.exception))
        self.assertNotIn("log_scale", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
