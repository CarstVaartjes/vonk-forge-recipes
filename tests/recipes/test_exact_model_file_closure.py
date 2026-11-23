from __future__ import annotations

import json
import unittest
from pathlib import Path

from catalog_documents import model_digest
from documents import model_display
from repo_root import ROOT

from vonk_forge_contracts.model import ModelDefinition, ModelSource
from vonk_forge_contracts.recipe import RecipeDefinition

PRIMARY_MODELS = {
    "mova-360p-diffusers-single": "mova-360p",
    "mova-720p-diffusers-single": "mova-720p",
    "hunyuan3d-omni-pytorch-single": "hunyuan3d-omni",
    "hunyuan-video-foley-xl-pytorch-single": "hunyuan-video-foley-xl",
    "hunyuan-video-foley-xxl-pytorch-single": "hunyuan-video-foley-xxl",
}
OMITTED_PRIMARY_PATHS = {
    "hunyuan3d-omni-pytorch-single": {"model/pytorch_model_ema.bin"},
    "hunyuan-video-foley-xl-pytorch-single": {"hunyuanvideo_foley.pth"},
    "hunyuan-video-foley-xxl-pytorch-single": {"hunyuanvideo_foley_xl.pth"},
}
REQUIRED_PRIMARY_CHECKPOINTS = {
    "hunyuan3d-omni-pytorch-single": "model/pytorch_model.bin",
    "hunyuan-video-foley-xl-pytorch-single": "hunyuanvideo_foley_xl.pth",
    "hunyuan-video-foley-xxl-pytorch-single": "hunyuanvideo_foley.pth",
}


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


class ExactModelFileClosureTests(unittest.TestCase):
    def test_primary_selection_matches_variant_specific_manifest(self) -> None:
        for recipe_slug, model_slug in PRIMARY_MODELS.items():
            with self.subTest(recipe=recipe_slug):
                recipe = RecipeDefinition.model_validate_json(
                    (ROOT / "recipes" / f"{recipe_slug}.json").read_text()
                )
                primary = next(
                    selection
                    for selection in recipe.models
                    if selection.id == "primary"
                )
                model_doc = load_json(ROOT / "models" / f"{model_slug}.json")
                model = ModelDefinition.model_validate(model_doc)
                files_by_id = {item.id: item for item in model.files}
                manifest_paths = {item.path for item in model.files}
                omitted_paths = OMITTED_PRIMARY_PATHS.get(recipe_slug, set())
                assert not omitted_paths & manifest_paths
                selected_ids = {item.file_id for item in primary.files}
                assert selected_ids <= set(files_by_id)
                selected_paths = {files_by_id[file_id].path for file_id in selected_ids}

                assert primary.model.slug == model_slug
                assert primary.model.content_sha256 == model_digest(model_slug)
                assert len(manifest_paths) > 1
                assert "snapshot" not in manifest_paths
                assert selected_paths == manifest_paths
                assert len(primary.files) == len(selected_ids), (
                    "no manifest file may be selected twice"
                )
                assert all(item.roles == ["entrypoint"] for item in primary.files)
                required_checkpoint = REQUIRED_PRIMARY_CHECKPOINTS.get(recipe_slug)
                if required_checkpoint is not None:
                    assert required_checkpoint in manifest_paths
                    assert required_checkpoint in selected_paths

    def test_mova_license_links_target_the_pinned_model_cards(self) -> None:
        for model_slug in ("mova-360p", "mova-720p"):
            with self.subTest(model=model_slug):
                model = ModelDefinition.model_validate(
                    load_json(ROOT / "models" / f"{model_slug}.json")
                )
                model_license = model_display(model_slug).license
                assert model_license.spdx == "Apache-2.0"
                assert isinstance(model.source, ModelSource)
                assert (
                    model_license.url
                    == f"{model.source.repository}/blob/{model.source.revision}/README.md"
                )

    def test_pytorch_checkpoint_models_do_not_claim_safetensors(self) -> None:
        for model_slug in (
            "hunyuan3d-omni",
            "hunyuan-video-foley-xl",
            "hunyuan-video-foley-xxl",
        ):
            with self.subTest(model=model_slug):
                model = ModelDefinition.model_validate(
                    load_json(ROOT / "models" / f"{model_slug}.json")
                )
                weight_paths = {
                    item.path for item in model.files if "weights" in item.roles
                }
                assert weight_paths
                assert all(
                    Path(path).suffix in {".bin", ".pth"} for path in weight_paths
                )

    def test_required_offline_dependencies_are_bound_to_adapter_mounts(self) -> None:
        expected_auxiliary = {
            "hunyuan3d-omni-pytorch-single": ("dinov2-large",),
            "hunyuan-video-foley-xl-pytorch-single": ("siglip2", "clap"),
            "hunyuan-video-foley-xxl-pytorch-single": ("siglip2", "clap"),
        }
        auxiliary_docs = {
            "dinov2-large": "dinov2-large-47b73eef",
            "siglip2": "siglip2-base-patch16-512-a89f5c50",
            "clap": "larger-clap-general-ada0c23a",
        }
        expected_mounts = {
            "dinov2-large": "/models/dinov2-large",
            "siglip2": "/models/siglip2",
            "clap": "/models/clap",
        }

        for recipe_slug, aux_ids in expected_auxiliary.items():
            with self.subTest(recipe=recipe_slug):
                recipe = RecipeDefinition.model_validate_json(
                    (ROOT / "recipes" / f"{recipe_slug}.json").read_text()
                )
                selections = {item.id: item for item in recipe.models}
                primary_slug = PRIMARY_MODELS[recipe_slug]
                primary_doc = ModelDefinition.model_validate(
                    load_json(ROOT / "models" / f"{primary_slug}.json")
                )
                dependency_refs = {
                    (item.publisher, item.slug, item.content_sha256)
                    for item in primary_doc.dependencies
                }
                selected_refs = set()

                for auxiliary_id in aux_ids:
                    doc_slug = auxiliary_docs[auxiliary_id]
                    aux_model = ModelDefinition.model_validate(
                        load_json(ROOT / "models" / f"{doc_slug}.json")
                    )
                    selection = selections[auxiliary_id]
                    selected_ref = (
                        selection.model.publisher,
                        selection.model.slug,
                        selection.model.content_sha256,
                    )
                    selected_refs.add(selected_ref)
                    assert selection.model.slug == doc_slug
                    assert selection.model.content_sha256 == model_digest(doc_slug)
                    assert {item.file_id for item in selection.files} == {
                        item.id for item in aux_model.files
                    }
                    assert all(
                        item.mount.target == expected_mounts[auxiliary_id]
                        and item.roles == ["entrypoint"]
                        for item in selection.files
                    )
                assert selected_refs == dependency_refs

        hunyuan_source = (ROOT / "adapters/three-d/hunyuan3d-omni/run.py").read_text(
            encoding="utf-8"
        )
        assert 'DINO = Path("/models/dinov2-large")' in hunyuan_source

        foley_source = (
            ROOT / "adapters/audio/hunyuan-video-foley-native/pipelines/run.py"
        ).read_text(encoding="utf-8")
        assert 'Path("/models/siglip2/config.json")' in foley_source
        assert 'Path("/models/clap/config.json")' in foley_source


if __name__ == "__main__":
    unittest.main()
