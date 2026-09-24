# ruff: noqa: S102 -- isolated synthetic upstream modules are executed in tests.
from __future__ import annotations

import ast
import json
import tempfile
import types
import unittest
from pathlib import Path

from vonk_forge_contracts.canonical import content_sha256
from vonk_forge_contracts.model import ModelDefinition
from vonk_forge_contracts.recipe import RecipeDefinition
from vonk_forge_contracts.resolver import (
    validate_model_references,
    validate_recipe_models,
)

ROOT = Path(__file__).resolve().parents[1]
ADAPTER_ROOT = ROOT / "adapters/three-d/step1x-3d"
PREPARE_PATH = ADAPTER_ROOT / "prepare_upstream.py"
RUN_PATH = ADAPTER_ROOT / "run.py"
DOCKERFILE_PATH = ADAPTER_ROOT / "Dockerfile"


def _model_definitions() -> dict[tuple[str, str], ModelDefinition]:
    documents = [
        ModelDefinition.model_validate(json.loads(path.read_text(encoding="utf-8")))
        for path in sorted((ROOT / "models").glob("*.json"))
    ]
    validate_model_references(documents)
    return {
        (model.identity.publisher, model.identity.slug): model for model in documents
    }


def _prepare_module() -> types.ModuleType:
    module = types.ModuleType("step1x_prepare")
    module.__file__ = str(PREPARE_PATH)
    exec(
        compile(PREPARE_PATH.read_text(encoding="utf-8"), str(PREPARE_PATH), "exec"),
        module.__dict__,
    )
    return module


class Step1XUpstreamPatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = _prepare_module()
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_pipeline_import_does_not_require_pymeshlab(self) -> None:
        pipeline_path = (
            self.root / "step1x3d_geometry/models/pipelines/pipeline_utils.py"
        )
        pipeline_path.parent.mkdir(parents=True)
        pipeline_path.write_text(
            """import pymeshlab

def load_mesh(path):
    return pymeshlab.MeshSet()

def trimesh2pymeshlab(mesh: trimesh.Trimesh):
    return pymeshlab.MeshSet()

def pymeshlab2trimesh(mesh: pymeshlab.MeshSet):
    return mesh

def import_mesh(mesh):
    return pymeshlab.MeshSet()

def remove_degenerate_face(mesh):
    return pymeshlab.MeshSet()
""",
            encoding="utf-8",
        )

        self.module.patch_pipeline_utils(self.root)

        source = pipeline_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        self.assertTrue(source.startswith("from __future__ import annotations\n"))
        top_level_imports = {
            alias.name
            for node in tree.body
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        self.assertNotIn("pymeshlab", top_level_imports)
        functions = {
            node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
        }
        for name in (
            "load_mesh",
            "trimesh2pymeshlab",
            "pymeshlab2trimesh",
            "import_mesh",
            "remove_degenerate_face",
        ):
            self.assertIsInstance(functions[name].body[0], ast.Import)
            self.assertEqual(functions[name].body[0].names[0].name, "pymeshlab")

        namespace: dict[str, object] = {}
        exec(compile(source, str(pipeline_path), "exec"), namespace)
        self.assertIn("pymeshlab2trimesh", namespace)

    def test_sharp_and_normal_select_different_geometry_embeddings(self) -> None:
        encoder_path = (
            self.root / "step1x3d_geometry/models/conditional_encoders/label_encoder.py"
        )
        encoder_path.parent.mkdir(parents=True)
        encoder_path.write_text(
            """GEOMETRY_QUALITY_MAPPING = {"normal": 0, "smooth": 1, "sharp": 2}

class LabelEncoder:
    embedding_table_geometry_quality = (
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
    )

    def encode_label(self, label):
        return self.embedding_table_geometry_quality[
            GEOMETRY_QUALITY_MAPPING[label["geometry_type"][0]]
        ]
""",
            encoding="utf-8",
        )

        self.module.patch_label_encoder(self.root)

        namespace: dict[str, object] = {}
        source = encoder_path.read_text(encoding="utf-8")
        exec(compile(source, str(encoder_path), "exec"), namespace)
        encoder = namespace["LabelEncoder"]()
        normal = encoder.encode_label({"geometry_type": "normal"})
        sharp = encoder.encode_label({"geometry_type": "sharp"})
        self.assertNotEqual(normal, sharp)
        runner = RUN_PATH.read_text(encoding="utf-8")
        self.assertIn('"geometry_type": "sharp"', runner)
        self.assertNotIn('"edge_type": "sharp"', runner)

    def test_cupy_cuda13_arm64_wheels_are_exact_and_import_smoked(self) -> None:
        dockerfile = DOCKERFILE_PATH.read_text(encoding="utf-8")
        expected = {
            "cupy_cuda13x-13.6.0-cp312-cp312-manylinux2014_aarch64.whl": (
                "a3bb49fb023757bfaf0b82c5a1740739a2108ea46d944d699bcff92963c7b87f"
            ),
            "fastrlock-0.8.3-cp312-cp312-manylinux_2_17_aarch64."
            "manylinux2014_aarch64.manylinux_2_28_aarch64.whl": (
                "85a49a1f1e020097d087e1963e42cea6f307897d5ebe2cb6daf4af47ffdd3eed"
            ),
        }
        for filename, digest in expected.items():
            self.assertIn(filename, dockerfile)
            self.assertIn(f"#sha256={digest}", dockerfile)
        self.assertIn(
            "from step1x3d_geometry.models.pipelines.pipeline import "
            "Step1X3DGeometryPipeline",
            dockerfile,
        )
        self.assertIn(
            "from step1x3d_texture.pipelines."
            "step1x_3d_texture_synthesis_pipeline import Step1X3DTexturePipeline",
            dockerfile,
        )

        self.assertIn(
            'org.opencontainers.image.revision="cb5ac944709c6c913109070c7b90c3447f57f3d4"',
            dockerfile,
        )

    def test_triposg_diso_license_matches_pypi_metadata(self) -> None:
        recipe = json.loads((ROOT / "recipes/triposg-pytorch-single.json").read_text())
        model = json.loads((ROOT / "models/triposg.json").read_text())
        self.assertEqual(recipe["models"][0]["model"]["slug"], "triposg")
        self.assertEqual(model["license"]["spdx"], "MIT")

    def test_texture_validates_input_before_importing_upstream(self) -> None:
        source = RUN_PATH.read_text(encoding="utf-8")
        validation = source.index("mesh_path = _validated_input_glb(")
        upstream_import = source.index(
            "from step1x3d_texture.pipelines.step1x_3d_texture_synthesis_pipeline import"
        )
        pipeline_construction = source.index(
            "pipeline = Step1X3DTexturePipeline(config)"
        )

        self.assertLess(validation, upstream_import)
        self.assertLess(validation, pipeline_construction)
        self.assertIn('validate_mesh_glb(path, profile="geometry")', source)

    def test_step1x_model_and_auxiliary_files_form_complete_offline_closure(
        self,
    ) -> None:
        models = _model_definitions()
        cases = (
            (
                "step1x-3d-geometry-pytorch-single.json",
                "step1x-3d-geometry",
                "Step1X-3D-Geometry-1300m/",
                {
                    "primary": "/models/target",
                    "dinov2-registers": "/models/dinov2-registers",
                },
            ),
            (
                "step1x-3d-label-geometry-pytorch-single.json",
                "step1x-3d-label-geometry",
                "Step1X-3D-Geometry-Label-1300m/",
                {
                    "primary": "/models/target",
                    "dinov2-registers": "/models/dinov2-registers",
                    "clip": "/models/clip",
                },
            ),
            (
                "step1x-3d-texture-pytorch-single.json",
                "step1x-3d-texture",
                "Step1X-3D-Texture/",
                {
                    "primary": "/models/target",
                    "sdxl": "/models/sdxl",
                    "sdxl-vae": "/models/sdxl-vae",
                },
            ),
        )
        adapter_sources = RUN_PATH.read_text(encoding="utf-8") + PREPARE_PATH.read_text(
            encoding="utf-8"
        )

        for recipe_name, primary_slug, subfolder, expected_mounts in cases:
            with self.subTest(recipe=recipe_name):
                recipe = RecipeDefinition.model_validate(
                    json.loads(
                        (ROOT / "recipes" / recipe_name).read_text(encoding="utf-8")
                    )
                )
                selections = {selection.id: selection for selection in recipe.models}
                self.assertEqual(set(selections), set(expected_mounts))
                selected_models = [
                    models[(selection.model.publisher, selection.model.slug)]
                    for selection in recipe.models
                ]
                validate_recipe_models(recipe, selected_models)

                primary = selections["primary"]
                primary_model = models[("stepfun-ai", primary_slug)]
                self.assertEqual(
                    primary.model.content_sha256, content_sha256(primary_model)
                )
                self.assertEqual(
                    {selector.file_id for selector in primary.files},
                    {model_file.id for model_file in primary_model.files},
                    "the installed Step1X subfolder must include every manifest file",
                )
                self.assertTrue(
                    all(
                        model_file.path.startswith(subfolder)
                        for model_file in primary_model.files
                    )
                )
                self.assertNotIn(
                    "snapshot", {item.path for item in primary_model.files}
                )

                selected_dependencies = {
                    (
                        selection.model.publisher,
                        selection.model.slug,
                        selection.model.content_sha256,
                    )
                    for selection_id, selection in selections.items()
                    if selection_id != "primary"
                }
                declared_dependencies = {
                    (item.publisher, item.slug, item.content_sha256)
                    for item in primary_model.dependencies
                }
                self.assertEqual(selected_dependencies, declared_dependencies)

                for selection_id, expected_target in expected_mounts.items():
                    selection = selections[selection_id]
                    self.assertEqual(
                        {selector.mount.target for selector in selection.files},
                        {expected_target},
                    )
                    self.assertIn(expected_target, adapter_sources)
                    model = models[(selection.model.publisher, selection.model.slug)]
                    self.assertEqual(
                        {selector.file_id for selector in selection.files},
                        {model_file.id for model_file in model.files},
                    )

        dinov2 = models[("facebook", "dinov2-with-registers-large")]
        self.assertEqual(
            {item.path for item in dinov2.files},
            {"config.json", "preprocessor_config.json"},
        )
        self.assertNotIn(
            "weights", {role for item in dinov2.files for role in item.roles}
        )

        clip = models[("openai", "clip-vit-large-patch14")]
        self.assertEqual({item.path for item in clip.files}, {"config.json"})
        self.assertNotIn(
            "weights", {role for item in clip.files for role in item.roles}
        )
        self.assertEqual(clip.license.spdx, "MIT")
        self.assertIn("cautions against deployed use", clip.metadata.description)

        sdxl = models[("stabilityai", "stable-diffusion-xl-base-1-0")]
        self.assertEqual(
            {item.path for item in sdxl.files},
            {
                "model_index.json",
                "scheduler/scheduler_config.json",
                "text_encoder/config.json",
                "text_encoder/model.safetensors",
                "text_encoder_2/config.json",
                "text_encoder_2/model.safetensors",
                "tokenizer/merges.txt",
                "tokenizer/special_tokens_map.json",
                "tokenizer/tokenizer_config.json",
                "tokenizer/vocab.json",
                "tokenizer_2/merges.txt",
                "tokenizer_2/special_tokens_map.json",
                "tokenizer_2/tokenizer_config.json",
                "tokenizer_2/vocab.json",
                "unet/config.json",
                "unet/diffusion_pytorch_model.safetensors",
            },
        )
        self.assertTrue(sdxl.license.operator_acceptance_required)
        self.assertFalse(sdxl.access.gated)
        self.assertEqual(
            {item.path for item in models[("madebyollin", "sdxl-vae-fp16-fix")].files},
            {"config.json", "diffusion_pytorch_model.safetensors"},
        )

    def test_auxiliary_model_evidence_hashes_immutable_source_files(self) -> None:
        models = _model_definitions()
        evidence_files = {
            ("facebook", "dinov2-with-registers-large"): "config.json",
            ("openai", "clip-vit-large-patch14"): "config.json",
            ("stabilityai", "stable-diffusion-xl-base-1-0"): "model_index.json",
            ("madebyollin", "sdxl-vae-fp16-fix"): "config.json",
        }

        for identity, evidence_path in evidence_files.items():
            with self.subTest(model=identity):
                model = models[identity]
                source_file = next(
                    item for item in model.files if item.path == evidence_path
                )
                source_url = (
                    f"{model.source.repository}/blob/{model.source.revision}/"
                    f"{evidence_path}"
                )
                self.assertEqual(
                    model.provenance.source_revision, model.source.revision
                )
                self.assertEqual(model.provenance.source_url, source_url)
                self.assertEqual(model.provenance.evidence_digest, source_file.sha256)
                self.assertEqual(
                    model.capabilities.provenance.source_revision,
                    model.source.revision,
                )
                self.assertEqual(model.capabilities.provenance.source_url, source_url)
                self.assertEqual(
                    model.capabilities.provenance.evidence_digest, source_file.sha256
                )


if __name__ == "__main__":
    unittest.main()
