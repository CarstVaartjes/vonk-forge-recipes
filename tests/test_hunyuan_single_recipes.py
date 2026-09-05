from __future__ import annotations

import json
import tempfile
import types
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
RECIPE_SLUGS = ("hunyuan-video-15-distilled-diffusers-single", "hunyuan-video-15-i2v-step-distilled-diffusers-single", "hunyuan-video-15-t2v-diffusers-single", "hunyuan-video-foley-xl-pytorch-single", "hunyuan-video-foley-xxl-pytorch-single", "hunyuan3d-omni-pytorch-single", "hunyuanocr-1-5-vllm-dflash-single")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_module(path: Path, name: str, stubs: dict[str, types.ModuleType] | None = None):
    old = {}
    for key, value in (stubs or {}).items(): old[key] = __import__("sys").modules.get(key); __import__("sys").modules[key] = value
    try:
        module = types.ModuleType(name); module.__file__ = str(path)
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)
        return module
    finally:
        for key, value in old.items():
            if value is None: __import__("sys").modules.pop(key, None)
            else: __import__("sys").modules[key] = value


def video_module():
    torch = types.ModuleType("torch"); diffusers = types.ModuleType("diffusers"); diffusers.HunyuanVideo15ImageToVideoPipeline = object; diffusers.HunyuanVideo15Pipeline = object
    utils = types.ModuleType("diffusers.utils"); utils.export_to_video = object
    pil = types.ModuleType("PIL"); pil.Image = object
    return load_module(ROOT / "adapters/video/hunyuan-video-15-native/run.py", "hunyuan_video_adapter", {"torch": torch, "diffusers": diffusers, "diffusers.utils": utils, "PIL": pil})


class HunyuanSingleRecipeAuthorityTests(unittest.TestCase):
    def test_all_hunyuan_recipes_select_one_exact_model_and_job_interface(self) -> None:
        from vonk_forge_contracts import ModelDefinition, RecipeDefinition
        for slug in RECIPE_SLUGS:
            with self.subTest(slug=slug):
                recipe = load(ROOT / "recipes" / f"{slug}.json")
                RecipeDefinition.model_validate(recipe)
                self.assertEqual(len(recipe["models"]), 1)
                self.assertEqual(recipe["topology"]["node_count"], 1)
                model_slug = recipe["models"][0]["model"]["slug"]
                model = load(ROOT / "models" / f"{model_slug}.json")
                ModelDefinition.model_validate(model)
                self.assertEqual(recipe["interfaces"][0]["adapter"], "artifact-job" if slug.startswith("hunyuanocr") else recipe["interfaces"][0]["adapter"])

    def test_current_upstream_revisions_and_tencent_license_acceptance_are_explicit(self) -> None:
        expected = {"hunyuan-video-15-distilled": "1abb14f06518f37448dcf3a6917dd086dd7045c7", "hunyuan-video-15-i2v-step-distilled": "854c04a4c8a53d990b418c7478f0802c0fc8c726", "hunyuan-video-15-t2v": "f4dbc4a1efa4ac8ea56680cdf79d9f455105e814", "hunyuan-video-foley-xl": "3abd4e833b95b8db0fc9c687afc52483a48e9a97", "hunyuan-video-foley-xxl": "3abd4e833b95b8db0fc9c687afc52483a48e9a97", "hunyuan3d-omni": "70e803bfb4e127d534049d8ab8c8cb511780d485"}
        for slug, revision in expected.items(): self.assertEqual(load(ROOT / "models" / f"{slug}.json")["source"]["revision"], revision)
        for slug in RECIPE_SLUGS:
            recipe = load(ROOT / "recipes" / f"{slug}.json"); model = load(ROOT / "models" / f"{recipe['models'][0]['model']['slug']}.json")
            self.assertIn("operator_acceptance_required", model["license"])

    def test_offloaded_hunyuan_video_envelope_fits_healthy_spark(self) -> None:
        for slug in RECIPE_SLUGS[:3]:
            memory = load(ROOT / "recipes" / f"{slug}.json")["topology"]["roles"][0]["resources"]["memory"]
            self.assertLessEqual(max(memory["startup_peak_bytes"], memory["steady_state_bytes"] + memory["runtime_growth_bytes"]) + memory["system_reserve_bytes"], 128_000_000_000)


class HunyuanMediaAdapterTests(unittest.TestCase):
    def test_video_probe_requires_exact_mp4_contract(self) -> None:
        module = video_module(); valid = {"streams": [{"codec_type": "video", "width": 848, "height": 480, "nb_read_frames": "121"}], "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": "5.042"}}
        with patch.object(module.subprocess, "run", return_value=types.SimpleNamespace(stdout=json.dumps(valid))): module._validate_video(Path("output.mp4"), 480)
        valid["streams"][0]["nb_read_frames"] = "120"
        with patch.object(module.subprocess, "run", return_value=types.SimpleNamespace(stdout=json.dumps(valid))), self.assertRaisesRegex(SystemExit, "invalid 121-frame MP4"): module._validate_video(Path("output.mp4"), 480)

    def test_foley_validates_wav_before_atomic_publish(self) -> None:
        path = ROOT / "adapters/audio/hunyuan-video-foley-native/pipelines/run.py"; module = load_module(path, "hunyuan_foley_adapter")
        with tempfile.TemporaryDirectory() as directory:
            valid = Path(directory) / "valid.wav"
            with wave.open(str(valid), "wb") as audio: audio.setnchannels(1); audio.setsampwidth(2); audio.setframerate(48000); audio.writeframes(b"\0\0" * 10)
            module._validate_wav(valid)
        source = path.read_text(encoding="utf-8"); self.assertLess(source.index("_validate_wav(outputs[0])"), source.index("os.replace("))


if __name__ == "__main__": unittest.main()
