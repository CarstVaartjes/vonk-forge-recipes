from __future__ import annotations

import ast
import hashlib
import importlib.machinery
import importlib.util
import json
import re
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from typing import Protocol, cast
from unittest import mock

import pytest
from documents import (
    json_document,
    model_display,
    model_document,
    recipe_display,
    recipe_document,
)
from generated_catalog import GENERATED
from repo_root import ROOT

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
        "source_revision": "273b691d35989d71cd17ff2895fdc735097b92d1",
        "source_archive_url": "https://codeload.github.com/VAST-AI-Research/SkinTokens/tar.gz/273b691d35989d71cd17ff2895fdc735097b92d1",
        "source_archive": "skintokens.tar.gz",
        "source_archive_sha256": "f886ce830f8f6ed5a3eabebb9399244812ac17a44d5b51fe8853c381a214e334",
        # Upstream reads models/ and experiments/ relative to the working
        # directory, which the platform's image stage sets to /tmp.
        "upstream_cwd": True,
        "base_image": "nvcr.io/nvidia/cuda:13.4.1-devel-ubuntu24.04@sha256:d44d6dc249c2c8330d0b380c595a514f0e86137fcc717fe7a4389a25720ade37",
    },
    "triposg-pytorch-single": {
        "context": "adapters/three-d/triposg",
        "model": "models/triposg.json",
        "source_revision": "fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c",
        "source_archive_url": "https://github.com/VAST-AI-Research/TripoSG/archive/fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c.tar.gz",
        "source_archive": "triposg.tar.gz",
        "source_archive_sha256": "3d06f11eb795bcabea7863e670e9cea02f96bfc6ec3e6db20e015e5710653682",
        "base_image": "nvcr.io/nvidia/cuda:13.4.1-devel-ubuntu24.04@sha256:d44d6dc249c2c8330d0b380c595a514f0e86137fcc717fe7a4389a25720ade37",
    },
    "hunyuan3d-omni-pytorch-single": {
        "context": "adapters/three-d/hunyuan3d-omni",
        "model": "models/hunyuan3d-omni.json",
        "source_revision": "4d47c0cc2bd0c4281963a7314ab330a5af36bfa8",
        "source_archive_url": "https://github.com/Tencent-Hunyuan/Hunyuan3D-Omni/archive/4d47c0cc2bd0c4281963a7314ab330a5af36bfa8.tar.gz",
        "source_archive": "hunyuan3d-omni.tar.gz",
        "source_archive_sha256": "1191700188114ac9fd257ed617c3a46bb523adc90a05316c2bbed433063e32d3",
        "base_image": "nvcr.io/nvidia/cuda:13.4.1-runtime-ubuntu24.04@sha256:2eaf346843c93ae617a718818f4a804e2c46c0d97a85392212ec5b310ee57962",
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
    package = GENERATED / f"packages/{slug}.tar.gz"
    with tarfile.open(package, mode="r:gz") as archive:
        payloads = {
            member.name: archive.extractfile(member).read()  # pyright: ignore[reportOptionalMemberAccess]
            for member in archive.getmembers()
            if member.isfile()
        }
    return json.loads(payloads["manifest.json"]), payloads


class NativeThreeDAdapterTests(unittest.TestCase):
    def test_omni_and_skintokens_do_not_install_unused_torchaudio(self) -> None:
        for context in (
            "adapters/three-d/hunyuan3d-omni",
            "adapters/three-d/skintokens",
        ):
            with self.subTest(context=context):
                dockerfile = (ROOT / context / "Dockerfile").read_text()
                assert (
                    "--index-url https://download.pytorch.org/whl/cu132" in dockerfile
                )
                assert "torchaudio" not in dockerfile

    def test_triposg_builds_diso_cuda_kernels_for_spark(self) -> None:
        # diso ships only an sdist whose setup.py builds CUDA kernels only when
        # a GPU is visible or FORCE_CUDA=1, and imports the installed torch.
        dockerfile = (ROOT / "adapters/three-d/triposg/Dockerfile").read_text()
        assert "TORCH_CUDA_ARCH_LIST=12.1" in dockerfile
        install = dockerfile.index("--no-build-isolation diso==0.1.4")
        assert dockerfile.index("torch==2.14.0") < install
        assert "FORCE_CUDA=1 python3 -m pip install" in dockerfile
        assert "grep --quiet sm_121" in dockerfile

    def test_trellis_and_pixal_reject_unsafe_or_oversized_images_before_model_load(
        self,
    ) -> None:
        for adapter in ("pixal3d_job.py", "trellis2_job.py"):
            with self.subTest(adapter=adapter), tempfile.TemporaryDirectory() as value:
                source_path = ROOT / "adapters/three-d/trellis2-native" / adapter
                module = {"__name__": "adapter_test"}
                validation = type(sys)("glb_validation")
                validation.normalize_glb_json_padding = lambda _path: None
                validation.validate_mesh_glb = lambda _path, *, profile: None
                with mock.patch.dict(sys.modules, {"glb_validation": validation}):
                    exec(  # noqa: S102 - load the adapter source without importing heavy dependencies.
                        compile(source_path.read_bytes(), str(source_path), "exec"),
                        module,
                    )
                inputs = Path(value)
                image = inputs / "source.png"
                image.write_bytes(b"png")
                assert module["one_input"](inputs) == image  # pyright: ignore[reportCallIssue]
                image.unlink()
                target = inputs / "target.bin"
                target.write_bytes(b"png")
                (inputs / "source.png").symlink_to(target)
                with pytest.raises(SystemExit, match="found 0"):
                    module["one_input"](inputs)  # pyright: ignore[reportCallIssue]
                (inputs / "source.png").unlink()
                target.unlink()
                oversized = inputs / "large.png"
                with oversized.open("wb") as stream:
                    stream.truncate(16 * 1024 * 1024 + 1)
                with pytest.raises(SystemExit, match="16 MiB"):
                    module["one_input"](inputs)  # pyright: ignore[reportCallIssue]

        trellis = (
            ROOT / "adapters/three-d/trellis2-native/trellis2_job.py"
        ).read_text()
        assert trellis.index("input_image = validated_input") < trellis.index(
            "    import o_voxel"
        )

    def test_three_d_job_contracts_have_truthful_bounded_input_and_glb_output_slots(
        self,
    ) -> None:
        for slug in SLOTTED_RECIPES:
            with self.subTest(slug=slug):
                recipe = recipe_document(slug)
                interface = recipe["interfaces"][0]
                input_contract = interface["input"]
                output_contract = interface["output"]
                assert input_contract["slots"]
                assert (
                    sum(slot["max_total_bytes"] for slot in input_contract["slots"])
                    <= input_contract["max_bytes"]
                )
                assert len(output_contract["slots"]) == 1
                output = output_contract["slots"][0]
                assert output["media_types"] == ["model/gltf-binary"]
                assert output["extensions"] == [".glb"]
                assert (output["min_files"], output["max_files"]) == (1, 1)
                assert output["max_total_bytes"] <= output_contract["max_total_bytes"]

    def test_recipes_bind_exact_native_context_and_runtime(self) -> None:
        for slug, case in CASES.items():
            with self.subTest(slug=slug):
                recipe = read_recipe(slug)
                build = recipe["execution"]["build"]  # pyright: ignore[reportIndexIssue]
                context_name = case["context"]
                assert build["context"]["path"] == context_name
                assert build["dockerfile"] == f"{context_name}/Dockerfile"
                assert build["patches"] == []
                repository, digest = case["base_image"].split("@sha256:", 1)
                assert build["base_image"]["repository"] == repository.split(":", 1)[0]
                assert build["base_image"]["digest"] == digest

                manifest, payloads = read_package(slug)
                assert manifest["schema_version"] == 2
                assert manifest["kind"] == "recipe-package"
                assert manifest["package_type"] == "recipe"
                assert (
                    manifest["recipe_content_sha256"]
                    == hashlib.sha256(CATALOG.canonical(recipe)).hexdigest()
                )
                assert json_document(payloads["recipe.json"]) == recipe
                declared = {entry["path"]: entry for entry in manifest["files"]}  # pyright: ignore[reportGeneralTypeIssues]
                assert set(declared) == set(payloads) - {"manifest.json"}
                for path, content in payloads.items():
                    if path == "manifest.json":
                        continue
                    assert declared[path]["size"] == len(content)
                    assert (
                        declared[path]["sha256"] == hashlib.sha256(content).hexdigest()
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
                assert packaged_context == context_files
                model_path = case["model"]
                model = json.loads((ROOT / model_path).read_text())
                packaged_model = json.loads(payloads[model_path])
                assert (
                    recipe["models"][0]["model"]["content_sha256"]  # pyright: ignore[reportIndexIssue]
                    == hashlib.sha256(payloads[model_path]).hexdigest()
                )
                assert packaged_model["identity"] == model["identity"]
                assert packaged_model["source"] == model["source"]
                assert packaged_model["files"] == model["files"]
                assert manifest["build_inputs"] == [
                    {
                        "kind": "oci-image",
                        "platform": "linux/arm64",
                        "reference": case["base_image"],
                    }
                ]
                tags = set(recipe_display(recipe["identity"]["slug"]).tags)  # pyright: ignore[reportIndexIssue]
                assert not tags.intersection(
                    {"metadata-only", "non-executable", "integration-required"}
                )

    def test_runtime_is_offline_and_source_authorities_are_immutable(self) -> None:
        for slug, case in CASES.items():
            with self.subTest(slug=slug):
                recipe = read_recipe(slug)
                environment = {
                    item["name"]: item["value"]
                    for item in recipe["runtime"]["environment"]  # pyright: ignore[reportIndexIssue]
                }
                assert environment == {"HF_HUB_OFFLINE": "1"}

                _manifest, payloads = read_package(slug)
                dockerfile = payloads[f"{case['context']}/Dockerfile"].decode()
                revision = case["source_revision"]
                archive = case["source_archive"]
                assert f'org.opencontainers.image.revision="{revision}"' in dockerfile
                assert case["source_archive_url"] in dockerfile
                assert (
                    f'echo "{case["source_archive_sha256"]}  /tmp/{archive}" | sha256sum --check --strict'
                    in dockerfile
                )
                assert "HF_HUB_OFFLINE=1" not in dockerfile
                assert "TRANSFORMERS_OFFLINE=1" not in dockerfile

    def test_entrypoints_are_syntax_valid_and_have_no_runtime_downloads(self) -> None:
        forbidden = (
            "snapshot_download",
            "hf_hub_download",
            "requests.get",
            "requests.post",
            "urlopen(",
            "curl ",
        )
        for case in CASES.values():
            context_name = case["context"]
            with self.subTest(context=context_name):
                source = (ROOT / context_name / "run.py").read_text()
                ast.parse(source)
                for marker in forbidden:
                    assert marker not in source
                assert "output.glb" in source
                assert "validate_mesh_glb(" in source

    def test_entrypoints_put_the_extracted_upstream_tree_on_sys_path_first(
        self,
    ) -> None:
        # The image runs run.py as a script, so only /opt/vonk/source is on
        # sys.path; every deferred upstream import needs the tree the
        # Dockerfile extracted inserted before it (SkinTokens lacked this).
        # The platform's image stage ends with WORKDIR /tmp, so an adapter
        # whose upstream reads cwd-relative files also changes into SOURCE.
        for case in CASES.values():
            context = ROOT / case["context"]
            with self.subTest(context=case["context"]):
                dockerfile = (context / "Dockerfile").read_text()
                extracted = re.search(
                    rf"--file /tmp/{re.escape(case['source_archive'])}\b.*?"
                    r"--directory (\S+)",
                    dockerfile,
                )
                assert extracted is not None
                tree = ast.parse((context / "run.py").read_text())
                roots = [
                    node.value
                    for node in tree.body
                    if isinstance(node, ast.Assign)
                    and [ast.unparse(target) for target in node.targets] == ["SOURCE"]
                ]
                assert [ast.unparse(value) for value in roots] == [
                    f"Path({extracted.group(1)!r})"
                ]
                deferred = 0
                for function in ast.walk(tree):
                    if not isinstance(function, ast.FunctionDef):
                        continue
                    required = ["sys.path.insert(0, str(SOURCE))"]
                    if case.get("upstream_cwd"):
                        required.append("os.chdir(SOURCE)")
                    calls = [
                        (ast.unparse(node), node.lineno)
                        for node in ast.walk(function)
                        if isinstance(node, ast.Call)
                    ]
                    firsts = [
                        min(
                            (line for text, line in calls if text == call), default=None
                        )
                        for call in required
                    ]
                    for node in ast.walk(function):
                        if isinstance(node, (ast.Import, ast.ImportFrom)):
                            deferred += 1
                            for call, first in zip(required, firsts, strict=True):
                                assert first is not None, (
                                    f"{ast.unparse(node)} precedes {call}"
                                )
                                assert first < node.lineno, (
                                    f"{ast.unparse(node)} precedes {call}"
                                )
                assert deferred > 0

    def test_hunyuan_declares_and_forces_exact_local_dinov2(self) -> None:
        recipe = recipe_document("hunyuan3d-omni-pytorch-single")
        model_version = model_document("hunyuan3d-omni")
        model_license = model_display("hunyuan3d-omni").license
        assert (
            model_license.spdx == "LicenseRef-Tencent-Hunyuan-3D-Omni-Community-License"
        )
        assert model_license.url.startswith("https://")
        assert (
            recipe["models"][0]["model"]["content_sha256"]
            == hashlib.sha256(CATALOG.canonical(model_version)).hexdigest()
        )
        primary_files = recipe["models"][0]["files"]
        assert {selection["file_id"] for selection in primary_files} == {
            file["id"] for file in model_version["files"]
        }
        assert all(
            selection["mount"]["target"] == "/models/target"
            and selection["roles"] == ["entrypoint"]
            for selection in primary_files
        )
        dinov2 = recipe["models"][1]
        dinov2_model = model_document("dinov2-large-47b73eef")
        assert dinov2["model"]["slug"] == "dinov2-large-47b73eef"
        assert {selection["file_id"] for selection in dinov2["files"]} == {
            file["id"] for file in dinov2_model["files"]
        }
        assert all(
            selection["mount"]["target"] == "/models/dinov2-large"
            and selection["roles"] == ["entrypoint"]
            for selection in dinov2["files"]
        )
        source = (ROOT / "adapters/three-d/hunyuan3d-omni/run.py").read_text()
        assert 'identifier != "facebook/dinov2-large"' in source
        assert 'loader_kwargs["local_files_only"] = True' in source
        assert "np.random.seed(args.seed)" in source
        assert "sample_surface(loaded, 81920, seed=seed)" in source

    def test_pixal_validates_input_before_model_load_and_uses_writable_cache(
        self,
    ) -> None:
        source = (ROOT / "adapters/three-d/trellis2-native/pixal3d_job.py").read_text()
        dockerfile = (ROOT / "adapters/three-d/trellis2-native/Dockerfile").read_text()
        assert source.index("input_image = validated_input") < source.index(
            "    import torch"
        )
        assert (
            'FLEX_GEMM_AUTOTUNE_CACHE_PATH="/tmp/flexgemm-autotune-cache.json"'
            in dockerfile
        )
        assert 'FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE="0"' in dockerfile
        assert 'validate_mesh_glb(temporary, profile="textured-pbr")' in source
        assert source.index(
            'validate_mesh_glb(temporary, profile="textured-pbr")'
        ) < source.index("os.replace(temporary")

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
                assert validation in source
                assert "os.replace(" in source

    def test_mesh_conditioned_adapters_validate_inputs_before_model_work(self) -> None:
        step = (ROOT / "adapters/three-d/step1x-3d/run.py").read_text()
        skin = (ROOT / "adapters/three-d/skintokens/run.py").read_text()

        assert 'validate_mesh_glb(path, profile="geometry")' in step
        assert step.index("mesh_path = _validated_input_glb(") < step.index(
            "pipeline = Step1X3DTexturePipeline(config)"
        )
        assert 'validate_mesh_glb(candidate, profile="geometry")' in skin
        assert skin.index(
            'validate_mesh_glb(candidate, profile="geometry")'
        ) < skin.index("model = get_model(str(CHECKPOINT))")


class _OmniRun(Protocol):
    """The Hunyuan3D-Omni adapter's job reading, as this test drives it."""

    INPUTS: Path

    def job(self, kind: str, bbox: list[float]) -> tuple[str, Path, object]: ...


class HunyuanOmniControlTests(unittest.TestCase):
    """The control-type knob must agree with the files in the control slot.

    Sweep 2026-10-16: six min/max numbers became a (1, 1, 12) input after the
    encoder's own repeat, and the run failed with "mat1 and mat2 shapes cannot
    be multiplied (1x204 and 102x1024)". Upstream's demos/bbox/data.json gives
    three sizes in (0, 1]; they are now three typed knobs with that range.
    """

    def _job(
        self, inputs: Path, kind: str = "bbox", **slots: list[str]
    ) -> tuple[str, Path, object]:
        files = [
            {"slot": slot, "name": name}
            for slot, names in slots.items()
            for name in names
        ]
        (inputs / "manifest.json").write_text(json.dumps({"files": files}))
        heavy = ("numpy", "torch", "trimesh", "PIL", "glb_validation")
        stubs = {name: mock.MagicMock() for name in heavy}
        with mock.patch.dict(sys.modules, stubs):
            spec = importlib.util.spec_from_file_location(
                "hunyuan3d_omni_run", ROOT / "adapters/three-d/hunyuan3d-omni/run.py"
            )
            assert spec is not None
            assert spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            run = cast(_OmniRun, module)
            run.INPUTS = inputs
            return run.job(kind, [0.8, 0.64, 1.0])

    def test_bbox_takes_the_three_sizes_and_no_control_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            inputs = Path(directory)
            (inputs / "object.png").write_bytes(b"png")
            assert self._job(inputs, image=["object.png"])[2] == [0.8, 0.64, 1.0]
            (inputs / "box.glb").write_bytes(b"glb")
            with pytest.raises(SystemExit, match="takes no control file"):
                self._job(inputs, image=["object.png"], control=["box.glb"])

    def test_point_and_voxel_need_a_geometry_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            inputs = Path(directory)
            (inputs / "object.png").write_bytes(b"png")
            (inputs / "cloud.ply").write_bytes(b"ply")
            kind, _image, control = self._job(
                inputs, "point", image=["object.png"], control=["cloud.ply"]
            )
            assert (kind, control) == ("point", inputs / "cloud.ply")
            with pytest.raises(SystemExit, match="needs a control file"):
                self._job(inputs, "voxel", image=["object.png"])
            (inputs / "pose.json").write_text("[[0, 0, 0]]")
            with pytest.raises(SystemExit, match="GLB, glTF, OBJ or PLY"):
                self._job(inputs, "point", image=["object.png"], control=["pose.json"])

    def test_pose_reads_xyz_triples_from_a_json_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            inputs = Path(directory)
            (inputs / "object.png").write_bytes(b"png")
            (inputs / "pose.json").write_text("[[0, 0.5, 1], [1, 1, 1]]")
            assert self._job(
                inputs, "pose", image=["object.png"], control=["pose.json"]
            )[2] == [[0, 0.5, 1], [1, 1, 1]]
            (inputs / "pose.json").write_text("[[0, 0]]")
            with pytest.raises(SystemExit, match="xyz triples"):
                self._job(inputs, "pose", image=["object.png"], control=["pose.json"])


if __name__ == "__main__":
    unittest.main()
