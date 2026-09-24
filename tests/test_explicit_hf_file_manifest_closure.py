from __future__ import annotations

import unittest
from pathlib import Path

from vonk_forge_contracts import ModelDefinition, RecipeDefinition, content_sha256

ROOT = Path(__file__).resolve().parents[1]
MODEL_RECIPE_PAIRS = {
    "hunyuan-video-15-t2v": "hunyuan-video-15-t2v-diffusers-single",
    "hunyuan-video-15-distilled": "hunyuan-video-15-distilled-diffusers-single",
    "hunyuan-video-15-i2v-step-distilled": "hunyuan-video-15-i2v-step-distilled-diffusers-single",
    "skintokens": "skintokens-pytorch-single",
    "minimax-h3": "minimax-h3-diffusers-single",
    "triposg": "triposg-pytorch-single",
}
FULL_MODEL_SELECTIONS = {
    "hunyuan-video-15-t2v",
    "hunyuan-video-15-distilled",
    "hunyuan-video-15-i2v-step-distilled",
    "minimax-h3",
    "triposg",
}


class ExplicitHuggingFaceManifestTests(unittest.TestCase):
    def test_model_provenance_binds_an_exact_pinned_readme_digest(self) -> None:
        for slug in (*MODEL_RECIPE_PAIRS, "bria-rmbg-1-4"):
            with self.subTest(model=slug):
                path = ROOT / "models" / f"{slug}.json"
                model = ModelDefinition.model_validate_json(
                    path.read_text(encoding="utf-8")
                )
                self.assertEqual(model.identity.slug, path.stem)
                self.assertNotIn("snapshot", {item.path for item in model.files})
                readme = next(item for item in model.files if item.path == "README.md")
                evidence_url = (
                    f"{model.source.repository}/blob/{model.source.revision}/README.md"
                )
                self.assertEqual(model.provenance.source_url, evidence_url)
                self.assertEqual(model.provenance.evidence_digest, readme.sha256)
                self.assertEqual(model.capabilities.provenance.source_url, evidence_url)
                self.assertEqual(
                    model.capabilities.provenance.evidence_digest, readme.sha256
                )
                self.assertTrue(
                    all(
                        fact.evidence_digest == readme.sha256
                        for fact in model.capabilities.facts
                    )
                )

    def test_recipe_file_selectors_resolve_exact_models_and_offline_closure(
        self,
    ) -> None:
        for model_slug, recipe_slug in MODEL_RECIPE_PAIRS.items():
            with self.subTest(recipe=recipe_slug):
                model_path = ROOT / "models" / f"{model_slug}.json"
                model = ModelDefinition.model_validate_json(
                    model_path.read_text(encoding="utf-8")
                )
                recipe = RecipeDefinition.model_validate_json(
                    (ROOT / "recipes" / f"{recipe_slug}.json").read_text(
                        encoding="utf-8"
                    )
                )
                selection = next(item for item in recipe.models if item.id == "primary")
                self.assertEqual(selection.model.content_sha256, content_sha256(model))
                by_id = {item.id: item for item in model.files}
                selected_ids = {item.file_id for item in selection.files}
                self.assertEqual(len(selected_ids), len(selection.files))
                self.assertTrue(selected_ids <= set(by_id))
                self.assertNotIn("snapshot", selected_ids)
                self.assertTrue(
                    all(
                        set(selector.roles)
                        == {role.name for role in recipe.topology.roles}
                        for selector in selection.files
                    )
                )
                if model_slug in FULL_MODEL_SELECTIONS:
                    self.assertEqual(selected_ids, set(by_id))

    def test_skintokens_selects_both_required_checkpoints(self) -> None:
        recipe = RecipeDefinition.model_validate_json(
            (ROOT / "recipes/skintokens-pytorch-single.json").read_text()
        )
        model = ModelDefinition.model_validate_json(
            (ROOT / "models/skintokens.json").read_text()
        )
        files = {item.id: item.path for item in model.files}
        selected = {files[item.file_id] for item in recipe.models[0].files}
        self.assertEqual(
            selected,
            {
                "experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt",
                "experiments/skin_vae_2_10_32768/last.ckpt",
            },
        )

    def test_triposg_binds_bria_config_and_weights_with_license_gate(self) -> None:
        recipe = RecipeDefinition.model_validate_json(
            (ROOT / "recipes/triposg-pytorch-single.json").read_text(encoding="utf-8")
        )
        triposg = ModelDefinition.model_validate_json(
            (ROOT / "models/triposg.json").read_text(encoding="utf-8")
        )
        bria = ModelDefinition.model_validate_json(
            (ROOT / "models/bria-rmbg-1-4.json").read_text(encoding="utf-8")
        )
        selections = {item.id: item for item in recipe.models}
        self.assertEqual(set(selections), {"primary", "background-removal"})
        bria_selection = selections["background-removal"]
        self.assertEqual(bria_selection.model.content_sha256, content_sha256(bria))
        self.assertEqual(
            bria_selection.model.content_sha256,
            next(
                dependency.content_sha256
                for dependency in triposg.dependencies
                if dependency.slug == bria.identity.slug
            ),
        )
        by_id = {item.id: item for item in bria.files}
        selected_paths = {by_id[item.file_id].path for item in bria_selection.files}
        self.assertEqual(selected_paths, {"config.json", "model.safetensors"})
        self.assertTrue(
            all(
                item.mount.target == "/models/rmbg" and item.roles == ["entrypoint"]
                for item in bria_selection.files
            )
        )
        self.assertEqual(bria.access.visibility, "public")
        self.assertFalse(bria.access.gated)
        self.assertTrue(bria.license.operator_acceptance_required)
        self.assertEqual(bria.license.spdx, "other")
        self.assertEqual(
            bria.license.url,
            "https://bria.ai/bria-huggingface-model-license-agreement/",
        )


if __name__ == "__main__":
    unittest.main()
