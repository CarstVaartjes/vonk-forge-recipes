from __future__ import annotations

import ast
import hashlib
import importlib.machinery
import importlib.util
import json
import sys
import tarfile
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
    "skintokens-pytorch-single": {
        "context": "adapters/three-d/skintokens",
        "model": "models/skintokens.json",
        "source_repository": "VAST-AI-Research/SkinTokens",
        "source_revision": "273b691d35989d71cd17ff2895fdc735097b92d1",
        "source_archive": "skintokens.tar.gz",
        "source_archive_sha256": "f886ce830f8f6ed5a3eabebb9399244812ac17a44d5b51fe8853c381a214e334",
        "base_image": "nvcr.io/nvidia/cuda:13.2.1-devel-ubuntu24.04@sha256:0e1392f431f89f143d0d6d0fa397a2b9a6a236f8b3628cfd3afbf21e15ab4a98",
    },
    "triposg-pytorch-single": {
        "context": "adapters/three-d/triposg",
        "model": "models/triposg.json",
        "source_repository": "VAST-AI-Research/TripoSG",
        "source_revision": "fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c",
        "source_archive": "triposg.tar.gz",
        "source_archive_sha256": "3d06f11eb795bcabea7863e670e9cea02f96bfc6ec3e6db20e015e5710653682",
        "base_image": "nvcr.io/nvidia/cuda:13.2.1-devel-ubuntu24.04@sha256:0e1392f431f89f143d0d6d0fa397a2b9a6a236f8b3628cfd3afbf21e15ab4a98",
    },
    "hunyuan3d-omni-pytorch-single": {
        "context": "adapters/three-d/hunyuan3d-omni",
        "model": "models/hunyuan3d-omni.json",
        "source_repository": "Tencent-Hunyuan/Hunyuan3D-Omni",
        "source_revision": "4d47c0cc2bd0c4281963a7314ab330a5af36bfa8",
        "source_archive": "hunyuan3d-omni.tar.gz",
        "source_archive_sha256": "1191700188114ac9fd257ed617c3a46bb523adc90a05316c2bbed433063e32d3",
        "base_image": "nvcr.io/nvidia/cuda:13.2.1-runtime-ubuntu24.04@sha256:a52783d8d73ace53998d4e740515e9942e73072dc7fbd5322917eb382a0bc7fb",
    },
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


def read_recipe(slug: str) -> dict[str, object]:
    return json.loads((ROOT / f"recipes/{slug}.json").read_text())


def read_package(slug: str) -> tuple[dict[str, object], dict[str, bytes]]:
    package = ROOT / f"packages/{slug}.tar.gz"
    with tarfile.open(package, mode="r:gz") as archive:
        payloads = {
            member.name: archive.extractfile(member).read()
            for member in archive.getmembers()
            if member.isfile()
        }
    return json.loads(payloads["manifest.json"]), payloads


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
        for slug, case in CASES.items():
            with self.subTest(slug=slug):
                recipe = read_recipe(slug)
                build = recipe["execution"]["build"]
                context_name = case["context"]
                self.assertEqual(build["context"]["path"], context_name)
                self.assertEqual(build["dockerfile"], f"{context_name}/Dockerfile")
                self.assertEqual(build["patches"], [])
                repository, digest = case["base_image"].split("@sha256:", 1)
                self.assertEqual(build["base_image"]["repository"], repository.split(":", 1)[0])
                self.assertEqual(build["base_image"]["digest"], digest)

                manifest, payloads = read_package(slug)
                self.assertEqual(manifest["schema_version"], 2)
                self.assertEqual(manifest["kind"], "recipe-package")
                self.assertEqual(manifest["package_type"], "recipe")
                self.assertEqual(
                    manifest["recipe_content_sha256"],
                    hashlib.sha256(CATALOG.canonical(recipe)).hexdigest(),
                )
                self.assertEqual(json.loads(payloads["recipe.json"]), recipe)
                declared = {entry["path"]: entry for entry in manifest["files"]}
                self.assertEqual(set(declared), set(payloads) - {"manifest.json"})
                for path, content in payloads.items():
                    if path == "manifest.json":
                        continue
                    self.assertEqual(declared[path]["size"], len(content))
                    self.assertEqual(
                        declared[path]["sha256"], hashlib.sha256(content).hexdigest()
                    )

                context_files = {
                    path.relative_to(ROOT).as_posix(): path.read_bytes()
                    for path in (ROOT / context_name).rglob("*")
                    if (
                        path.is_file()
                        and not path.is_symlink()
                        and "__pycache__" not in path.parts
                        and path.suffix != ".pyc"
                    )
                }
                packaged_context = {
                    path: content
                    for path, content in payloads.items()
                    if path.startswith(f"{context_name}/")
                }
                self.assertEqual(packaged_context, context_files)
                model_path = case["model"]
                model = json.loads((ROOT / model_path).read_text())
                packaged_model = json.loads(payloads[model_path])
                self.assertEqual(
                    recipe["models"][0]["model"]["content_sha256"],
                    hashlib.sha256(payloads[model_path]).hexdigest(),
                )
                self.assertEqual(packaged_model["identity"], model["identity"])
                self.assertEqual(packaged_model["source"], model["source"])
                self.assertEqual(packaged_model["files"], model["files"])
                self.assertEqual(
                    manifest["build_inputs"],
                    [{"kind": "oci-image", "platform": "linux/arm64", "reference": case["base_image"]}],
                )
                tags = set(recipe["metadata"]["tags"])
                self.assertIn("candidate", tags)
                self.assertFalse(tags.intersection({"metadata-only", "non-executable", "integration-required"}))

    def test_runtime_is_offline_and_source_authorities_are_immutable(self) -> None:
        for slug, case in CASES.items():
            with self.subTest(slug=slug):
                recipe = read_recipe(slug)
                environment = {
                    item["name"]: item["value"]
                    for item in recipe["runtime"]["environment"]
                }
                self.assertEqual(environment, {"HF_HUB_OFFLINE": "1"})

                _manifest, payloads = read_package(slug)
                dockerfile = payloads[f'{case["context"]}/Dockerfile'].decode()
                revision = case["source_revision"]
                archive = case["source_archive"]
                self.assertIn(
                    f'org.opencontainers.image.revision="{revision}"', dockerfile
                )
                self.assertIn(
                    f'https://github.com/{case["source_repository"]}/archive/{revision}.tar.gz',
                    dockerfile,
                )
                self.assertIn(
                    f'echo "{case["source_archive_sha256"]}  /tmp/{archive}" | sha256sum --check --strict',
                    dockerfile,
                )
                self.assertNotIn("HF_HUB_OFFLINE=1", dockerfile)
                self.assertNotIn("TRANSFORMERS_OFFLINE=1", dockerfile)

    def test_entrypoints_are_syntax_valid_and_have_no_runtime_downloads(self) -> None:
        forbidden = ("snapshot_download", "hf_hub_download", "requests.get", "requests.post", "urlopen(", "curl ")
        for case in CASES.values():
            context_name = case["context"]
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
