from __future__ import annotations

import hashlib
import json
import runpy
import tempfile
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ADAPTER_ROOT = ROOT / "adapters/video/minimax-h3-modular-diffusers"
ADAPTER_PATH = ADAPTER_ROOT / "minimax_h3.py"
RECIPE_PATH = ROOT / "recipes/minimax-h3-diffusers-single.json"
RUNTIME_PATH = ROOT / "runtime-distributions/minimax-h3-modular-diffusers-arm64.json"
MODEL_PATH = ROOT / "model-versions/minimax-h3.json"
DIFFUSERS_REVISION = "efabd60d61c2b7aabf9f182bee6b5b6058980304"
MODEL_REVISION = "42ed227ee7df40d41602854ae760620d6eb651fe"


def _canonical_digest(path: Path) -> str:
    document = json.loads(path.read_text(encoding="utf-8"))
    content = json.dumps(
        document,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def _adapter_module():
    module = types.ModuleType("minimax_h3_adapter")
    module.__file__ = str(ADAPTER_PATH)
    exec(
        compile(ADAPTER_PATH.read_text(encoding="utf-8"), str(ADAPTER_PATH), "exec"),
        module.__dict__,
    )
    return module


class MiniMaxH3AuthorityTests(unittest.TestCase):
    def test_recipe_resolves_exact_local_authorities(self) -> None:
        recipe = json.loads(RECIPE_PATH.read_text(encoding="utf-8"))
        runtime = json.loads(RUNTIME_PATH.read_text(encoding="utf-8"))
        model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))

        self.assertEqual(
            recipe["model"]["content_sha256"], _canonical_digest(MODEL_PATH)
        )
        self.assertEqual(
            recipe["runtime"]["distribution"]["content_sha256"],
            _canonical_digest(RUNTIME_PATH),
        )
        self.assertEqual(model["source"]["revision"], MODEL_REVISION)
        self.assertEqual(runtime["source"]["revision"], DIFFUSERS_REVISION)
        self.assertIsNone(recipe["execution"]["patch_bundle"])
        self.assertEqual(recipe["artifacts"][0]["revision"], MODEL_REVISION)
        self.assertEqual(
            recipe["runtime"]["environment"],
            [
                {"name": "HF_HUB_OFFLINE", "value": "1"},
                {"name": "TRANSFORMERS_OFFLINE", "value": "1"},
            ],
        )

        tags = set(recipe["metadata"]["tags"])
        self.assertIn("candidate", tags)
        self.assertTrue({"video", "audio", "multimodal", "modular-diffusers"} <= tags)
        self.assertTrue(
            {"metadata-only", "non-executable", "integration-required"}.isdisjoint(tags)
        )

    def test_signed_source_bundle_matches_recipe(self) -> None:
        recipe = json.loads(RECIPE_PATH.read_text(encoding="utf-8"))
        source_bundle = runpy.run_path(str(ROOT / "tools/build-catalog-index"))[
            "source_bundle"
        ]
        archive, _, digest = source_bundle(ADAPTER_ROOT)
        context = recipe["build"]["context"]
        self.assertEqual(context["sha256"], digest)
        self.assertEqual(context["expected_bytes"], len(archive))

    def test_container_and_adapter_forbid_hidden_runtime_downloads(self) -> None:
        dockerfile = (ADAPTER_ROOT / "Dockerfile").read_text(encoding="utf-8")
        source = ADAPTER_PATH.read_text(encoding="utf-8")
        self.assertIn(f"diffusers.git@{DIFFUSERS_REVISION}", dockerfile)
        self.assertIn(
            f'org.opencontainers.image.revision="{MODEL_REVISION}"', dockerfile
        )
        self.assertGreaterEqual(source.count("local_files_only=True"), 4)
        self.assertIn("pretrained_model_name_or_path=str(MODEL_ROOT)", source)
        self.assertIn(
            'transformer_name = "transformer_ref" if workflow == "ref2va" else "transformer"',
            source,
        )
        self.assertIn("**{transformer_name: transformer}", source)
        self.assertNotIn("huggingface.co", source)
        self.assertNotIn("MiniMaxAI/MiniMax-H3", source)

    def test_export_is_atomic_and_joint_audio_video_is_verified(self) -> None:
        source = ADAPTER_PATH.read_text(encoding="utf-8")
        for required in (
            '"videos"',
            '"audio"',
            '"sampling_rate"',
            "encode_video(",
            "_verify_joint_av(temporary)",
            "audio.sample_rate != 32000",
            'audio.layout.name != "stereo"',
            "os.replace(temporary, destination)",
        ):
            self.assertIn(required, source)


class MiniMaxH3InputContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = _adapter_module()
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.module.INPUT_ROOT = Path(self.temporary.name)

    def test_request_rejects_unknown_fields_and_traversal(self) -> None:
        request = self.module.INPUT_ROOT / "request.json"
        request.write_text('{"unknown": true}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unsupported fields"):
            self.module._load_request()
        with self.assertRaisesRegex(ValueError, "inside /inputs"):
            self.module._safe_input_path("../escape.png", "first_image")

    def test_request_accepts_bounded_generation_controls(self) -> None:
        request = self.module.INPUT_ROOT / "request.json"
        request.write_text(
            json.dumps(
                {"prompt": "snowy fox", "num_frames": 124, "width": 960, "height": 544}
            ),
            encoding="utf-8",
        )
        value = self.module._load_request()
        self.assertEqual(self.module._prompt(value["prompt"]), "snowy fox")
        self.assertEqual(self.module._num_frames(value["num_frames"]), 124)
        self.assertEqual(
            self.module._positive_multiple(value["width"], "width", 1344), 960
        )
        self.assertEqual(
            self.module._positive_multiple(value["height"], "height", 768), 544
        )
        with self.assertRaisesRegex(ValueError, "120..345"):
            self.module._num_frames(346)


if __name__ == "__main__":
    unittest.main()
