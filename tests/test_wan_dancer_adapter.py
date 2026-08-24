from __future__ import annotations

import hashlib
import json
import subprocess
import tarfile
import tempfile
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "adapters/video/wan-dancer-native"
RECIPE = ROOT / "recipes/wan-dancer-14b-pytorch-single.json"
RUNTIME = ROOT / "runtime-distributions/wan-dancer-native-e6c87a9-cuda13-arm64.json"
MODEL = ROOT / "model-versions/wan-dancer-14b.json"
ARCHIVE_SHA256 = "92c529d7727c75c6515ea990d27883a45bf566587cc9f5d325a0a488b9fa1649"


def canonical_digest(path: Path) -> str:
    document = json.loads(path.read_text(encoding="utf-8"))
    payload = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def load_runner():
    path = ADAPTER / "run.py"
    module = types.ModuleType("wan_dancer_runner")
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    return module


class WanDancerAuthorityTests(unittest.TestCase):
    def test_recipe_resolves_complete_immutable_authorities(self) -> None:
        recipe = json.loads(RECIPE.read_text(encoding="utf-8"))
        self.assertEqual(recipe["model"]["content_sha256"], canonical_digest(MODEL))
        self.assertEqual(
            recipe["runtime"]["distribution"]["content_sha256"],
            canonical_digest(RUNTIME),
        )
        self.assertEqual(recipe["artifacts"][0]["revision"], "85ce88dd8d025459dcf0fe93982d6da8b9002957")
        self.assertEqual(recipe["topology"]["node_count"], 1)
        self.assertIn(
            {"source": "inputs", "target": "/inputs", "read_only": True},
            recipe["runtime"]["security"]["mounts"],
        )

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
            self.assertIn('ModelConfig(path="/models/global_model.safetensors"', global_stage)
            self.assertIn('ModelConfig(path="/models/local_model.safetensors"', local_stage)
            self.assertNotIn("use_usp=True", global_stage + local_stage)
            self.assertNotIn("dist.get_rank()", global_stage + local_stage)
            self.assertIn("cfg_scale=args.cfg_scale", local_stage)

    def test_runtime_has_no_model_download_path(self) -> None:
        runner = (ADAPTER / "run.py").read_text(encoding="utf-8")
        dockerfile = (ADAPTER / "Dockerfile").read_text(encoding="utf-8")
        self.assertNotIn("snapshot_download", runner)
        self.assertNotIn("from_pretrained", runner)
        self.assertIn("HF_HUB_OFFLINE=1", dockerfile)
        self.assertIn("TRANSFORMERS_OFFLINE=1", dockerfile)


class WanDancerInputTests(unittest.TestCase):
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
                runner._input_file(request["reference_image"], "reference_image", runner.IMAGE_SUFFIXES),
                inputs / "person.jpg",
            )
            with self.assertRaises(ValueError):
                runner._input_file("../person.jpg", "reference_image", runner.IMAGE_SUFFIXES)


if __name__ == "__main__":
    unittest.main()
