from __future__ import annotations

import hashlib
import json
import runpy
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECIPE_PATH = ROOT / "recipes/minimax-h3-fl2va-diffusers-single.json"
FULL_RECIPE_PATH = ROOT / "recipes/minimax-h3-diffusers-single.json"
MODEL_PATH = ROOT / "model-versions/minimax-h3-fl2va-42ed227e.json"
RUNTIME_PATH = ROOT / "runtime-distributions/minimax-h3-fl2va-modular-diffusers-arm64.json"
ADAPTER_PATH = (
    ROOT
    / "adapters/video/minimax-h3-fl2va-modular-diffusers/minimax_h3.py"
)


def canonical_digest(path: Path) -> str:
    document = json.loads(path.read_text(encoding="utf-8"))
    content = json.dumps(
        document,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(content).hexdigest()


class MiniMaxH3Fl2vaRecipeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.recipe = json.loads(RECIPE_PATH.read_text(encoding="utf-8"))
        self.full_recipe = json.loads(FULL_RECIPE_PATH.read_text(encoding="utf-8"))
        self.model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))

    def test_variant_is_separate_candidate_and_keeps_full_snapshot(self) -> None:
        self.assertEqual(
            self.recipe["identity"]["slug"],
            "minimax-h3-fl2va-diffusers-single",
        )
        tags = set(self.recipe["metadata"]["tags"])
        self.assertTrue({"executable", "candidate", "fl2va", "artifact-slim"} <= tags)
        self.assertNotIn("accepted", tags)
        self.assertEqual(
            self.full_recipe["artifacts"][0]["installed_bytes"], 498_474_749_480
        )
        self.assertNotIn("include_paths", self.full_recipe["artifacts"][0])

    def test_model_and_runtime_authorities_are_exact(self) -> None:
        self.assertEqual(
            self.recipe["model"]["content_sha256"], canonical_digest(MODEL_PATH)
        )
        self.assertEqual(
            self.recipe["runtime"]["distribution"]["content_sha256"],
            canonical_digest(RUNTIME_PATH),
        )
        self.assertEqual(
            self.model["source"]["revision"],
            "42ed227ee7df40d41602854ae760620d6eb651fe",
        )

    def test_filtered_snapshot_matches_exact_immutable_manifest(self) -> None:
        artifact = self.recipe["artifacts"][0]
        entries = self.model["artifacts"]
        paths = [entry["path"] for entry in entries]
        self.assertEqual(artifact["include_paths"], paths)
        self.assertEqual(len(paths), 62)
        self.assertEqual(len(paths), len(set(paths)))
        self.assertEqual(sum(entry["installed_bytes"] for entry in entries), 144_051_160_615)
        self.assertEqual(artifact["installed_bytes"], 144_051_160_615)
        self.assertTrue(all(len(entry["sha256"]) == 64 for entry in entries))
        required = {
            "model_index.json",
            "modular_model_index.json",
            "text_encoder/model.safetensors.index.json",
            "transformer/diffusion_pytorch_model.safetensors.index.json",
            "vae/diffusion_pytorch_model.safetensors.index.json",
            "audio_vae/diffusion_pytorch_model.safetensors",
        }
        self.assertTrue(required <= set(paths))
        forbidden = ("transformer_ref/", "FL2VA/", "Ref2VA/", "assets/", "docs/", "scripts/")
        self.assertFalse(any(path.startswith(forbidden) for path in paths))

    def test_interface_is_truthful_fl2va_only(self) -> None:
        interface = self.recipe["interfaces"][0]
        slots = {slot["id"]: slot for slot in interface["input"]["slots"]}
        self.assertEqual(set(slots), {"prompt", "request", "keyframes"})
        self.assertEqual(slots["keyframes"]["max_files"], 2)
        self.assertNotIn("video/mp4", interface["input"]["media_types"])
        self.assertNotIn("audio/wav", interface["input"]["media_types"])
        self.assertIn("Ref2VA references are not supported", slots["request"]["description"])

    def test_adapter_rejects_ref2va_request_before_loading(self) -> None:
        module = runpy.run_path(str(ADAPTER_PATH))
        with tempfile.TemporaryDirectory() as directory:
            input_root = Path(directory)
            module["_load_request"].__globals__["INPUT_ROOT"] = input_root
            (input_root / "request.json").write_text(
                '{"references":[{"type":"image","path":"subject.png"}]}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "FL2VA-only variant"):
                module["_load_request"]()

    def test_resource_contract_uses_exact_closure(self) -> None:
        build = self.recipe["build"]["resources"]
        disk = self.recipe["topology"]["roles"][0]["resources"]["disk"]
        memory = self.recipe["topology"]["roles"][0]["resources"]["memory"]
        self.assertEqual(build["download_bytes"], 150_051_160_615)
        self.assertEqual(build["temporary_bytes"], 288_102_321_230)
        self.assertEqual(disk["artifact_bytes"], 144_051_160_615)
        self.assertEqual(disk["staging_bytes"], 288_102_321_230)
        self.assertEqual(
            sum(disk.values()),
            454_153_481_845,
        )
        self.assertEqual(
            memory["startup_peak_bytes"] + memory["system_reserve_bytes"],
            120_000_000_000,
        )


if __name__ == "__main__":
    unittest.main()
