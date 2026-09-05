from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs/upstream-media-image-coverage-2026-09-05.json"

MEDIA_IMAGE_IDS = (
    "vonk-forge/flux-2-klein-4b-comfyui-single",
    "vonk-forge/flux-2-klein-4b-nvfp4-comfyui-single",
    "vonk-forge/hunyuan-video-15-distilled-diffusers-single",
    "vonk-forge/hunyuan-video-15-i2v-step-distilled-diffusers-single",
    "vonk-forge/hunyuan-video-15-t2v-diffusers-single",
    "vonk-forge/ltx-2-5-22b-distilled-bf16-diffusers-single",
    "vonk-forge/ltx-2-5-22b-distilled-fp8-cast-diffusers-single",
    "vonk-forge/minimax-h3-diffusers-single",
    "vonk-forge/minimax-h3-fl2va-diffusers-single",
    "vonk-forge/nvidia-qwen-image-flash-diffusers-single",
    "vonk-forge/qwen-image-2512-comfyui-single",
    "vonk-forge/qwen-image-2512-diffusers-single",
    "vonk-forge/qwen-image-2512-fp8-lightning-comfyui-single",
    "vonk-forge/qwen-image-2512-lightning-diffusers-single",
    "vonk-forge/qwen-image-edit-2511-comfyui-single",
    "vonk-forge/qwen-image-edit-2511-diffusers-single",
    "vonk-forge/qwen-image-edit-2511-fp8mixed-comfyui-single",
    "vonk-forge/qwen-image-edit-2511-int8-convrot-comfyui-single",
    "vonk-forge/qwen-image-edit-2511-lightning-diffusers-single",
    "vonk-forge/qwen-image-layered-diffusers-single",
    "vonk-forge/wan-2-2-i2v-14b-comfyui-single",
    "vonk-forge/wan-2-2-t2v-14b-comfyui-single",
    "vonk-forge/wan-2-2-ti2v-5b-comfyui-single",
)

COMFY_REVISION = "250b2e9551a7bc7a8ebb5beb07e0fecd2983e04a"
COMFY_ARCHIVE_SHA256 = "40d10b36b57ba819cc524d1cb19e7b04b00356ba5254941d27e20b705f22e5ee"
DIFFUSERS_REVISION = "c5469b7ceb606edd7ba6570dcd17d38590a18db6"
HUNYUAN_REVISION = "a3608b512ed7248499a44c61d954965ed9bdae4d"
MINIMAX_REVISION = "efabd60d61c2b7aabf9f182bee6b5b6058980304"
SOURCE_REFRESH_COMMIT = "f4268482b625ca9f8bd4d644171ab2638f20f2a0"
QWEN_EDIT_ARTIFACTS = {
    "vonk-forge/qwen-image-edit-2511-comfyui-single": (
        "b6a0794717d3f5600f85c5edcdcd0c0eb93d7446",
        "split_files/diffusion_models/qwen_image_edit_2511_bf16.safetensors",
        "ae42d927b5fac4f278b9a894554c727e619727a63622976f2d95625be4bce08c",
        40861031560,
    ),
    "vonk-forge/qwen-image-edit-2511-fp8mixed-comfyui-single": (
        "4c7c4ea236326cbae56d403d22a03c6cd86ad9a0",
        "split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors",
        "c9fdc158e46d3b61ef75f21ae866ca2fe808bf4a53643120d1c1e87c19280a4e",
        20533762817,
    ),
    "vonk-forge/qwen-image-edit-2511-int8-convrot-comfyui-single": (
        "e9e85de74a8f48c1e3e2656617626348675a2f21",
        "split_files/diffusion_models/qwen_image_edit_2511_int8_convrot.safetensors",
        "11b5af5ac601821d73930c84846c9a158e67177356daf927ce1c8d10f3963829",
        20499083824,
    ),
}


def load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


class MediaImageCoverageTests(unittest.TestCase):
    def test_report_covers_exact_media_image_domain(self) -> None:
        report = load(REPORT)
        rows = report["rows"]
        assert isinstance(rows, list)
        self.assertEqual(report["scope"], "media_image")
        self.assertEqual(report["coverage_count"], len(MEDIA_IMAGE_IDS))
        self.assertEqual([row["recipe_id"] for row in rows], list(MEDIA_IMAGE_IDS))
        self.assertEqual(report["review_inputs"]["source_refresh_commit"], SOURCE_REFRESH_COMMIT)

    def test_every_row_binds_current_source_and_recipe_closure(self) -> None:
        report = load(REPORT)
        for row in report["rows"]:
            with self.subTest(recipe=row["recipe_id"]):
                recipe_path = ROOT / row["source_path"]
                recipe = load(recipe_path)
                self.assertEqual(recipe["identity"]["publisher"] + "/" + recipe["identity"]["slug"], row["recipe_id"])
                self.assertEqual(row["implementation_commit"], SOURCE_REFRESH_COMMIT)
                self.assertTrue(row["checked_at"].startswith("2026-09-05T"))
                self.assertTrue(row["source_evidence"])
                self.assertTrue(row["reason"])

                runtime = row["runtime"]
                dockerfile = (ROOT / runtime["dockerfile"]).read_text(encoding="utf-8")
                if runtime["family"] == "ComfyUI":
                    self.assertEqual(runtime["pinned_revision"], COMFY_REVISION)
                    self.assertIn(COMFY_REVISION, dockerfile)
                    self.assertIn(COMFY_ARCHIVE_SHA256, dockerfile)
                    self.assertEqual(row["decision"], "applied_executable_update")
                elif runtime["family"] == "Diffusers/HunyuanVideo native adapter":
                    self.assertEqual(runtime["pinned_revision"], HUNYUAN_REVISION)
                    self.assertIn(HUNYUAN_REVISION, dockerfile)
                    self.assertIn("diffusers==0.39.0", (ROOT / runtime["context"] / "requirements.lock").read_text())
                    self.assertEqual(row["decision"], "retained_intentional_variant")
                    self.assertIsNone(row["named_blocker"])
                    review = row["compatibility_review"]
                    self.assertEqual(review["verdict"], "retained_compatible_pinned_wheel")
                    self.assertIn("neither timesteps nor sigmas", review["adapter_call_contract"])
                    self.assertTrue(review["unchanged_relevant_files"])
                    changed = {item["path"]: item for item in review["changed_relevant_files"]}
                    scheduler = changed["src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py"]
                    self.assertNotEqual(scheduler["sha256_pinned"], scheduler["sha256_current"])
                    self.assertIn("explicit timesteps", scheduler["change"])
                    self.assertTrue(review["wheel_matches_pinned_source_for_scheduler"])
                    self.assertTrue(review["current_tree_requires_new_wheel"])
                    self.assertEqual(
                        review["wheel_lock"]["scheduler_file_sha256"],
                        scheduler["sha256_pinned"],
                    )
                    self.assertEqual(
                        review["wheel_lock"]["wheel_sha256"],
                        "912aca51b5787365110806e984d5555735bf8a461073bb8459029d0bca7870ef",
                    )
                elif runtime["family"] == "Diffusers/MiniMax H3 modular fork":
                    self.assertEqual(runtime["pinned_revision"], MINIMAX_REVISION)
                    self.assertIn(MINIMAX_REVISION, dockerfile)
                    self.assertEqual(row["decision"], "retained_intentional_variant")
                else:
                    self.assertEqual(runtime["pinned_revision"], DIFFUSERS_REVISION)
                    self.assertIn(DIFFUSERS_REVISION, dockerfile)
                    self.assertEqual(row["decision"], "applied_executable_update")

                recipe_model_slugs = {
                    selection["model"]["slug"] for selection in recipe["models"]
                }
                self.assertEqual(
                    len(row["models"]),
                    len(recipe_model_slugs),
                )
                for model_ref in row["models"]:
                    self.assertIn(model_ref["status"], {"already_current", "retained_exact_variant"})
                    model_path = next(
                        path
                        for path in (ROOT / "models").glob("*.json")
                        if load(path)["identity"]["slug"]
                        == next(
                            selection["model"]["slug"]
                            for selection in recipe["models"]
                            if selection["id"] == model_ref["selection_id"]
                        )
                    )
                    model = load(model_path)
                    self.assertEqual(model["source"]["revision"], model_ref["revision"])
                    self.assertEqual(model["source"]["repository"].removeprefix("https://huggingface.co/"), model_ref["repository"])
                    self.assertIn(model_ref["revision"], model_ref["source_url"])

    def test_exact_qwen_edit_variants_are_recorded_as_retained_artifacts(self) -> None:
        report = load(REPORT)
        rows = {
            row["recipe_id"]: row
            for row in report["rows"]
            if "qwen-image-edit-2511" in row["recipe_id"]
            and "diffusers" not in row["recipe_id"]
        }
        self.assertEqual(len(rows), 3)
        for row in rows.values():
            pinned_revision, selected_path, selected_sha, selected_size = QWEN_EDIT_ARTIFACTS[
                row["recipe_id"]
            ]
            model = row["models"][0]
            self.assertEqual(model["repository"], "Comfy-Org/Qwen-Image-Edit_ComfyUI")
            self.assertEqual(model["status"], "retained_exact_variant")
            self.assertEqual(model["observed_current_head"], "984166f60a9b1fcede5e9b9287b7a7aebc050010")
            comparison = row["artifact_comparison"]
            self.assertEqual(comparison["pinned_commit"], pinned_revision)
            self.assertEqual(comparison["selected_path"], selected_path)
            selected = comparison["selected_file"]
            self.assertEqual(selected["status"], "unchanged_exact_sha256_and_size")
            self.assertEqual(selected["sha256_pinned"], selected["sha256_current"])
            self.assertEqual(selected["size_pinned"], selected["size_current"])
            self.assertEqual(selected["sha256_pinned"], selected_sha)
            self.assertEqual(selected["size_pinned"], selected_size)
            self.assertEqual(comparison["changed_runtime_or_config_paths"], [])
            self.assertEqual(comparison["changed_metadata_paths"], ["README.md"])
            self.assertIn("exact LFS SHA256", row["reason"])


if __name__ == "__main__":
    unittest.main()
