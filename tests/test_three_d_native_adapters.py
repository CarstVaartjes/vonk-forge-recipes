from __future__ import annotations

import ast
import hashlib
import importlib.machinery
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
CATALOG_TOOL = ROOT / "tools/build-catalog-index"
LOADER = importlib.machinery.SourceFileLoader("three_d_catalog_tool", str(CATALOG_TOOL))
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
assert SPEC is not None
CATALOG = importlib.util.module_from_spec(SPEC)
sys.modules[LOADER.name] = CATALOG
LOADER.exec_module(CATALOG)

CASES = {
    "skintokens-pytorch-single": (
        "adapters/three-d/skintokens",
        "runtime-distributions/skintokens-tokenrig-arm64.json",
    ),
    "triposg-pytorch-single": (
        "adapters/three-d/triposg",
        "runtime-distributions/triposg-native-arm64.json",
    ),
    "hunyuan3d-omni-pytorch-single": (
        "adapters/three-d/hunyuan3d-omni",
        "runtime-distributions/hunyuan3d-omni-native-arm64.json",
    ),
}
SLOTTED_RECIPES = (
    "hunyuan3d-omni-pytorch-single",
    "pixal3d-pytorch-single",
    "skintokens-pytorch-single",
    "step1x-3d-geometry-pytorch-single",
    "step1x-3d-label-geometry-pytorch-single",
    "step1x-3d-texture-pytorch-single",
    "trellis-2-4b-pytorch-single",
    "triposg-pytorch-single",
)


class NativeThreeDAdapterTests(unittest.TestCase):
    def test_trellis_and_pixal_reject_unsafe_or_oversized_images_before_model_load(self) -> None:
        for adapter in ("pixal3d_job.py", "trellis2_job.py"):
            with self.subTest(adapter=adapter), tempfile.TemporaryDirectory() as value:
                source_path = ROOT / "adapters/three-d/trellis2-native" / adapter
                module = {"__name__": "adapter_test"}
                validation = type(sys)("glb_validation")
                validation.normalize_glb_json_padding = lambda _path: None
                validation.validate_mesh_glb = lambda _path, *, profile: None
                with mock.patch.dict(sys.modules, {"glb_validation": validation}):
                    exec(compile(source_path.read_bytes(), str(source_path), "exec"), module)  # noqa: S102
                inputs = Path(value)
                image = inputs / "source.png"
                image.write_bytes(b"png")
                self.assertEqual(module["one_input"](inputs), image)
                image.unlink()
                target = inputs / "target.bin"
                target.write_bytes(b"png")
                (inputs / "source.png").symlink_to(target)
                with self.assertRaisesRegex(SystemExit, "found 0"):
                    module["one_input"](inputs)
                (inputs / "source.png").unlink()
                target.unlink()
                oversized = inputs / "large.png"
                with oversized.open("wb") as stream:
                    stream.truncate(16 * 1024 * 1024 + 1)
                with self.assertRaisesRegex(SystemExit, "16 MiB"):
                    module["one_input"](inputs)

        trellis = (ROOT / "adapters/three-d/trellis2-native/trellis2_job.py").read_text()
        self.assertLess(
            trellis.index("input_image = validated_input"),
            trellis.index("    import o_voxel"),
        )

    def test_three_d_job_contracts_have_truthful_bounded_input_and_glb_output_slots(self) -> None:
        for slug in SLOTTED_RECIPES:
            with self.subTest(slug=slug):
                recipe = json.loads((ROOT / f"recipes/{slug}.json").read_text())
                interface = recipe["interfaces"][0]
                input_contract = interface["input"]
                output_contract = interface["output"]
                self.assertEqual(input_contract["path"], "/inputs")
                self.assertTrue(input_contract["slots"])
                self.assertLessEqual(
                    sum(slot["max_total_bytes"] for slot in input_contract["slots"]),
                    input_contract["max_bytes"],
                )
                self.assertEqual(output_contract["path"], "/outputs")
                self.assertEqual(len(output_contract["slots"]), 1)
                output = output_contract["slots"][0]
                self.assertEqual(output["media_types"], ["model/gltf-binary"])
                self.assertEqual(output["extensions"], [".glb"])
                self.assertEqual((output["min_files"], output["max_files"]), (1, 1))
                self.assertLessEqual(
                    output["max_total_bytes"], output_contract["max_total_bytes"]
                )

    def test_recipes_bind_exact_native_context_and_runtime(self) -> None:
        for slug, (context_name, runtime_name) in CASES.items():
            with self.subTest(slug=slug):
                recipe = json.loads((ROOT / f"recipes/{slug}.json").read_text())
                context = ROOT / context_name
                archive, _metadata, digest = CATALOG.source_bundle(context)
                self.assertEqual(recipe["execution"]["build"]["context"]["path"], context_name)
                expected_context = {
                    "adapters/three-d/skintokens": "cce641d72ef235966186c5ddc0ab89c65309cbe9dec4653d972027f81f751253",
                    "adapters/three-d/triposg": "5710927e467944ef1a9419fc87b33f896241c415900d62fcf16f9b90f0b030e3",
                    "adapters/three-d/hunyuan3d-omni": "f458c1fe53b940fc943bc49fa00bbaccb7842a009072c2ba290b1bc09ab1dec5",
                }[context_name]
                self.assertEqual(digest, expected_context)
                tags = set(recipe["metadata"]["tags"])
                self.assertIn("candidate", tags)
                self.assertFalse(tags.intersection({"metadata-only", "non-executable", "integration-required"}))

    def test_runtime_is_offline_and_source_authorities_are_immutable(self) -> None:
        revisions = {
            "adapters/three-d/skintokens": "273b691d35989d71cd17ff2895fdc735097b92d1",
            "adapters/three-d/triposg": "fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c",
            "adapters/three-d/hunyuan3d-omni": "4d47c0cc2bd0c4281963a7314ab330a5af36bfa8",
        }
        for context_name, _runtime_name in CASES.values():
            with self.subTest(context=context_name):
                dockerfile = (ROOT / context_name / "Dockerfile").read_text()
                self.assertIn(f'org.opencontainers.image.revision="{revisions[context_name]}"', dockerfile)
                self.assertIn("HF_HUB_OFFLINE=1", dockerfile)
                self.assertIn("TRANSFORMERS_OFFLINE=1", dockerfile)

    def test_entrypoints_are_syntax_valid_and_have_no_runtime_downloads(self) -> None:
        forbidden = ("snapshot_download", "hf_hub_download", "requests.get", "requests.post", "urlopen(", "curl ")
        for context_name, _runtime_name in CASES.values():
            with self.subTest(context=context_name):
                source = (ROOT / context_name / "run.py").read_text()
                ast.parse(source)
                for marker in forbidden:
                    self.assertNotIn(marker, source)
                self.assertIn("output.glb", source)
                self.assertIn("validate_mesh_glb(", source)

    def test_hunyuan_declares_and_forces_exact_local_dinov2(self) -> None:
        recipe = json.loads((ROOT / "recipes/hunyuan3d-omni-pytorch-single.json").read_text())
        model_version = json.loads(
            (ROOT / "models/hunyuan3d-omni.json").read_text()
        )
        self.assertEqual(model_version["license"]["spdx"], "LicenseRef-Tencent-Hunyuan-3D-Omni-Community-License")
        self.assertTrue(model_version["license"]["url"].startswith("https://"))
        self.assertFalse(model_version["license"]["operator_acceptance_required"])
        self.assertEqual(
            recipe["models"][0]["model"]["content_sha256"],
            hashlib.sha256(CATALOG.canonical(model_version)).hexdigest(),
        )
        self.assertEqual(recipe["models"][0]["files"][0]["mount"]["target"], "/models/target")
        self.assertEqual(recipe["models"][0]["files"][0]["file_id"], "snapshot")
        source = (ROOT / "adapters/three-d/hunyuan3d-omni/run.py").read_text()
        self.assertIn('identifier != "facebook/dinov2-large"', source)
        self.assertIn('loader_kwargs["local_files_only"] = True', source)
        self.assertIn("np.random.seed(args.seed)", source)
        self.assertIn("sample_surface(loaded, 81920, seed=seed)", source)

    def test_pixal_validates_input_before_model_load_and_disables_cache_writes(self) -> None:
        source = (ROOT / "adapters/three-d/trellis2-native/pixal3d_job.py").read_text()
        dockerfile = (ROOT / "adapters/three-d/trellis2-native/Dockerfile").read_text()
        self.assertLess(source.index("input_image = validated_input"), source.index("    import torch"))
        self.assertIn('FLEX_GEMM_AUTOTUNE_CACHE_PATH="/opt/vonk/flexgemm-source/autotune_cache.json"', dockerfile)
        self.assertIn('FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE="0"', dockerfile)
        self.assertIn('validate_mesh_glb(temporary, profile="textured-pbr")', source)
        self.assertLess(
            source.index('validate_mesh_glb(temporary, profile="textured-pbr")'),
            source.index("os.replace(temporary"),
        )

    def test_all_native_adapters_validate_before_atomic_publication(self) -> None:
        cases = {
            "hunyuan3d-omni/run.py": 'validate_mesh_glb(temporary, profile="geometry")',
            "trellis2-native/pixal3d_job.py": 'validate_mesh_glb(temporary, profile="textured-pbr")',
            "trellis2-native/trellis2_job.py": 'validate_mesh_glb(temporary, profile="textured-pbr")',
            "step1x-3d/run.py": '_publish_glb(temporary, output_dir / "step1x-textured.glb", profile="textured")',
            "triposg/run.py": 'validate_mesh_glb(temporary, profile="geometry")',
            "skintokens/run.py": 'validate_mesh_glb(temporary_output, profile="skinned")',
        }
        for relative, validation in cases.items():
            with self.subTest(adapter=relative):
                source = (ROOT / "adapters/three-d" / relative).read_text()
                self.assertIn(validation, source)
                self.assertIn("os.replace(", source)

    def test_mesh_conditioned_adapters_validate_inputs_before_model_work(self) -> None:
        step = (ROOT / "adapters/three-d/step1x-3d/run.py").read_text()
        skin = (ROOT / "adapters/three-d/skintokens/run.py").read_text()

        self.assertIn('validate_mesh_glb(path, profile="geometry")', step)
        self.assertLess(
            step.index("mesh_path = _validated_input_glb("),
            step.index("pipeline = Step1X3DTexturePipeline(config)"),
        )
        self.assertIn('validate_mesh_glb(candidate, profile="geometry")', skin)
        self.assertLess(
            skin.index('validate_mesh_glb(candidate, profile="geometry")'),
            skin.index("model = get_model(str(CHECKPOINT))"),
        )


if __name__ == "__main__":
    unittest.main()
