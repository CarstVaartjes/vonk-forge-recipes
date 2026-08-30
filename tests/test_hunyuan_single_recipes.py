from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import json
import sys
import tempfile
import types
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
CATALOG_TOOL = ROOT / "tools/build-catalog-index"
LOADER = importlib.machinery.SourceFileLoader("hunyuan_catalog_tool", str(CATALOG_TOOL))
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
assert SPEC is not None
CATALOG = importlib.util.module_from_spec(SPEC)
sys.modules[LOADER.name] = CATALOG
LOADER.exec_module(CATALOG)

RECIPE_SLUGS = (
    "hunyuan-video-15-distilled-diffusers-single",
    "hunyuan-video-15-i2v-step-distilled-diffusers-single",
    "hunyuan-video-15-t2v-diffusers-single",
    "hunyuan-video-foley-xl-pytorch-single",
    "hunyuan-video-foley-xxl-pytorch-single",
    "hunyuan3d-omni-pytorch-single",
    "hunyuanocr-1-5-vllm-dflash-single",
)
TERRITORY_NOTICE = (
    "The Tencent Hunyuan Community License Agreement does not apply in the "
    "European Union, United Kingdom, or South Korea."
)


def canonical_digest(path: Path) -> str:
    document = json.loads(path.read_text(encoding="utf-8"))
    return hashlib.sha256(CATALOG.canonical(document)).hexdigest()


def load_module(path: Path, name: str, stubs: dict[str, types.ModuleType] | None = None):
    previous: dict[str, types.ModuleType | None] = {}
    for module_name, module in (stubs or {}).items():
        previous[module_name] = sys.modules.get(module_name)
        sys.modules[module_name] = module
    try:
        module = types.ModuleType(name)
        module.__file__ = str(path)
        exec(  # noqa: S102 - isolated adapter contract loading
            compile(path.read_text(encoding="utf-8"), str(path), "exec"),
            module.__dict__,
        )
        return module
    finally:
        for module_name, original in previous.items():
            if original is None:
                del sys.modules[module_name]
            else:
                sys.modules[module_name] = original


def video_module():
    torch = types.ModuleType("torch")
    diffusers = types.ModuleType("diffusers")
    diffusers.HunyuanVideo15ImageToVideoPipeline = object
    diffusers.HunyuanVideo15Pipeline = object
    diffusers_utils = types.ModuleType("diffusers.utils")
    diffusers_utils.export_to_video = object
    pil = types.ModuleType("PIL")
    pil.Image = object
    return load_module(
        ROOT / "adapters/video/hunyuan-video-15-native/run.py",
        "hunyuan_video_adapter",
        {
            "torch": torch,
            "diffusers": diffusers,
            "diffusers.utils": diffusers_utils,
            "PIL": pil,
        },
    )


class HunyuanSingleRecipeAuthorityTests(unittest.TestCase):
    def test_all_seven_recipes_close_exact_catalog_and_release_references(self) -> None:
        for slug in RECIPE_SLUGS:
            with self.subTest(slug=slug):
                recipe_path = ROOT / f"recipes/{slug}.json"
                recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
                model = recipe["model"]
                model_path = ROOT / f"model-versions/{model['slug']}.json"
                runtime = recipe["runtime"]["distribution"]
                runtime_path = ROOT / f"runtime-distributions/{runtime['slug']}.json"
                release = json.loads(
                    (ROOT / f"recipe-releases/{slug}.json").read_text(encoding="utf-8")
                )
                context = ROOT / recipe["build"]["context"]["path"]
                archive, _files, digest = CATALOG.source_bundle(context)

                self.assertEqual(model["content_sha256"], canonical_digest(model_path))
                self.assertEqual(runtime["content_sha256"], canonical_digest(runtime_path))
                self.assertEqual(
                    release["history"][0]["recipe_content_sha256"],
                    canonical_digest(recipe_path),
                )
                self.assertEqual(recipe["build"]["context"]["sha256"], digest)
                self.assertEqual(
                    recipe["build"]["context"]["expected_bytes"], len(archive)
                )
                self.assertTrue(set(recipe["metadata"]["tags"]) >= {"executable", "candidate"})

    def test_current_upstream_authorities_remain_exact(self) -> None:
        expected_runtime_heads = {
            "hunyuan-video-foley-native-df7b005-cuda13-arm64": (
                "df7b005b5023df2a9b73e1d66dd51d452799884e"
            ),
            "hunyuan3d-omni-native-arm64": "4d47c0cc2bd0c4281963a7314ab330a5af36bfa8",
            "hunyuanocr-1-5-vllm-dflash-arm64": (
                "c55965d3da1e6f41987abec8068f2e70851318bc"
            ),
        }
        for slug, revision in expected_runtime_heads.items():
            runtime = json.loads(
                (ROOT / f"runtime-distributions/{slug}.json").read_text(encoding="utf-8")
            )
            self.assertEqual(runtime["source"]["revision"], revision)

        expected_model_heads = {
            "hunyuan-video-15-distilled": "1abb14f06518f37448dcf3a6917dd086dd7045c7",
            "hunyuan-video-15-i2v-step-distilled": (
                "854c04a4c8a53d990b418c7478f0802c0fc8c726"
            ),
            "hunyuan-video-15-t2v": "f4dbc4a1efa4ac8ea56680cdf79d9f455105e814",
            "hunyuan-video-foley-xl": "3abd4e833b95b8db0fc9c687afc52483a48e9a97",
            "hunyuan-video-foley-xxl": "3abd4e833b95b8db0fc9c687afc52483a48e9a97",
            "hunyuan3d-omni": "70e803bfb4e127d534049d8ab8c8cb511780d485",
            "hunyuanocr-1-5-47644ecc": (
                "47644ecc4fc854efa4f505155158831f36773ee4"
            ),
        }
        for slug, revision in expected_model_heads.items():
            model = json.loads(
                (ROOT / f"model-versions/{slug}.json").read_text(encoding="utf-8")
            )
            self.assertEqual(model["source"]["revision"], revision)

    def test_every_hunyuan_model_fails_closed_in_excluded_territories(self) -> None:
        for slug in RECIPE_SLUGS:
            recipe = json.loads((ROOT / f"recipes/{slug}.json").read_text())
            model = json.loads(
                (ROOT / f"model-versions/{recipe['model']['slug']}.json").read_text()
            )
            restrictions = model["license"]["territorial_restrictions"]
            self.assertEqual(restrictions["denied_jurisdictions"], ["EU", "GB", "KR"])
            if slug != "hunyuan3d-omni-pytorch-single":
                self.assertEqual(restrictions["notice"], TERRITORY_NOTICE)

    def test_offloaded_hunyuan_video_envelope_fits_healthy_128gb_spark(self) -> None:
        for slug in RECIPE_SLUGS[:3]:
            recipe = json.loads((ROOT / f"recipes/{slug}.json").read_text())
            memory = recipe["topology"]["roles"][0]["resources"]["memory"]
            required = max(
                memory["startup_peak_bytes"],
                memory["steady_state_bytes"] + memory["runtime_growth_bytes"],
            )
            self.assertEqual(required, 112_000_000_000)
            self.assertEqual(required + memory["system_reserve_bytes"], 120_000_000_000)


class HunyuanMediaAdapterTests(unittest.TestCase):
    def test_video_probe_requires_exact_mp4_contract(self) -> None:
        module = video_module()
        valid = {
            "streams": [
                {
                    "codec_type": "video",
                    "width": 848,
                    "height": 480,
                    "nb_read_frames": "121",
                }
            ],
            "format": {
                "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
                "duration": "5.042",
            },
        }
        completed = types.SimpleNamespace(stdout=json.dumps(valid))
        with patch.object(module.subprocess, "run", return_value=completed):
            module._validate_video(Path("output.mp4"), 480)
        valid["streams"][0]["nb_read_frames"] = "120"
        completed.stdout = json.dumps(valid)
        with (
            patch.object(module.subprocess, "run", return_value=completed),
            self.assertRaisesRegex(SystemExit, "invalid 121-frame MP4"),
        ):
            module._validate_video(Path("output.mp4"), 480)

    def test_video_and_foley_reject_wrong_signed_input_extensions(self) -> None:
        video = video_module()
        foley = load_module(
            ROOT / "adapters/audio/hunyuan-video-foley-native/pipelines/run.py",
            "hunyuan_foley_adapter",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "payload.bin").write_bytes(b"not media")
            (root / "manifest.json").write_text(
                json.dumps({"files": [{"slot": "image", "name": "payload.bin"}]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(SystemExit, "one supported file"):
                video._one_image(root)
            (root / "manifest.json").write_text(
                json.dumps({"files": [{"slot": "video", "name": "payload.bin"}]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(SystemExit, "one supported video"):
                foley._one_video(root)

    def test_foley_validates_wav_and_enforces_timeout_before_atomic_publish(self) -> None:
        adapter_path = ROOT / "adapters/audio/hunyuan-video-foley-native/pipelines/run.py"
        foley = load_module(adapter_path, "hunyuan_foley_wav_adapter")
        with tempfile.TemporaryDirectory() as directory:
            valid = Path(directory) / "valid.wav"
            with wave.open(str(valid), "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(48_000)
                audio.writeframes(b"\x00\x00" * 480)
            foley._validate_wav(valid)
            invalid = Path(directory) / "invalid.wav"
            invalid.write_bytes(b"not a wav")
            with self.assertRaisesRegex(SystemExit, "invalid WAV"):
                foley._validate_wav(invalid)

        source = adapter_path.read_text(encoding="utf-8")
        self.assertIn("timeout=args.timeout_seconds", source)
        self.assertLess(source.index("_validate_wav(outputs[0])"), source.index("os.replace("))
        self.assertIn('staging = args.output_dir / ".staging"', source)


if __name__ == "__main__":
    unittest.main()
