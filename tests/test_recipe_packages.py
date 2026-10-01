from __future__ import annotations

import copy
import hashlib
import io
import json
import runpy
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = runpy.run_path(str(ROOT / "tools/build-catalog-index"))
PLATFORM_OWNED_ENVIRONMENT = {
    "FLASHINFER_WORKSPACE_BASE",
    "TILELANG_CACHE_DIR",
    "TRITON_CACHE_DIR",
    "B12X_CUTE_COMPILE_CACHE_DIR",
    "TORCH_FR_DUMP_TEMP_FILE",
    "TORCH_NCCL_DEBUG_INFO_PIPE_FILE",
}


def test_full_catalog_packages_are_self_contained_and_deterministic(
    tmp_path: Path, built_catalog: tuple[dict, Path]
) -> None:
    first, first_dir = built_catalog
    second_dir = tmp_path / "second"
    second = TOOL["build"](package_dir=second_dir)
    assert first["kind"] == second["kind"] == "recipe-library-index"
    assert first["schema_version"] == second["schema_version"] == 2
    # The generated index as a whole, not only each package, must reproduce.
    assert first == second
    source_models = {
        (
            json.loads(path.read_text())["identity"]["publisher"],
            json.loads(path.read_text())["identity"]["slug"],
        )
        for path in ROOT.joinpath("models").glob("*.json")
    }
    source_recipes = {
        (
            json.loads(path.read_text())["identity"]["publisher"],
            json.loads(path.read_text())["identity"]["slug"],
        )
        for path in ROOT.joinpath("recipes").glob("*.json")
    }
    assert source_models and source_recipes
    assert len(source_models) == len(list(ROOT.joinpath("models").glob("*.json")))
    assert len(source_recipes) == len(list(ROOT.joinpath("recipes").glob("*.json")))
    for catalog in (first, second):
        assert {
            (
                item["document"]["identity"]["publisher"],
                item["document"]["identity"]["slug"],
            )
            for item in catalog["catalog_entities"]
        } == source_models
        assert {
            (
                item["document"]["identity"]["publisher"],
                item["document"]["identity"]["slug"],
            )
            for item in catalog["recipes"]
        } == source_recipes
        package_names = {
            Path(str(item["package"]["path"])).name for item in catalog["recipes"]
        }
        assert package_names == {f"{slug}.tar.gz" for _, slug in source_recipes}
    # The output directory holds exactly the packages the index names: a
    # stale archive from a removed recipe must not survive a rebuild.
    expected_package_names = {f"{slug}.tar.gz" for _, slug in source_recipes}
    assert {path.name for path in first_dir.glob("*.tar.gz")} == expected_package_names

    for first_row, second_row in zip(first["recipes"], second["recipes"], strict=True):
        first_package = first_row["package"]
        second_package = second_row["package"]
        filename = Path(str(first_package["path"])).name
        first_bytes = (first_dir / filename).read_bytes()
        second_bytes = (second_dir / filename).read_bytes()
        assert first_bytes == second_bytes
        assert hashlib.sha256(first_bytes).hexdigest() == first_package["sha256"]
        assert first_package == second_package
        with tarfile.open(fileobj=io.BytesIO(first_bytes), mode="r:gz") as archive:
            names = archive.getnames()
            assert len(names) == len(set(names))
            assert all(
                not name.startswith("/") and ".." not in name.split("/")
                for name in names
            )
            manifest = json.load(archive.extractfile("manifest.json"))
            assert manifest["schema_version"] == 2
            assert manifest["kind"] == "recipe-package"
            assert manifest["recipe_content_sha256"] == first_row["content_sha256"]
            assert manifest["package_type"] == "recipe"
            for entry in manifest["files"]:
                payload = archive.extractfile(entry["path"]).read()
                assert len(payload) == entry["size"]
                assert hashlib.sha256(payload).hexdigest() == entry["sha256"]


def test_source_bundle_ignores_generated_python_cache_files(tmp_path: Path) -> None:
    context = tmp_path / "context"
    context.mkdir()
    (context / "Dockerfile").write_text("FROM scratch\n")
    cache = context / "__pycache__"
    cache.mkdir()
    (cache / "generated.cpython-313.pyc").write_bytes(b"generated")
    (context / "standalone.pyc").write_bytes(b"generated")
    _archive, files, _digest = TOOL["source_bundle"](context)
    assert [entry["path"] for entry in files] == ["Dockerfile"]


def test_editing_one_recipe_changes_only_that_package(
    built_catalog: tuple[dict, Path],
) -> None:
    catalog, package_dir = built_catalog
    rows = catalog["recipes"]
    original = {
        Path(str(row["package"]["path"])).name: (
            package_dir / Path(str(row["package"]["path"])).name
        ).read_bytes()
        for row in rows
    }
    target = rows[0]
    edited = copy.deepcopy(target["document"])
    edited["metadata"]["description"] += " edited"
    entities = TOOL["_catalog_entity_documents"]()
    package_bytes, package = TOOL["recipe_package"](
        edited,
        recipe_path=ROOT / str(target["source_path"]),
        entity_documents=entities,
    )
    assert package_bytes != original[Path(str(target["package"]["path"])).name]
    assert (
        package["media_type"] == "application/vnd.vonk-forge.recipe-package.v2+tar+gzip"
    )
    # `recipe_package` renders exactly the one document it is handed, so editing
    # a recipe cannot reach another one: re-rendering every untouched recipe has
    # to reproduce the bytes `build()` wrote for it. Reading the built archive
    # back would only compare a file with itself.
    for row in rows[1:]:
        filename = Path(str(row["package"]["path"])).name
        rebuilt, _metadata = TOOL["recipe_package"](
            row["document"],
            recipe_path=ROOT / str(row["source_path"]),
            entity_documents=entities,
        )
        assert rebuilt == original[filename], filename


def test_supplied_source_commit_only_changes_index_metadata(tmp_path: Path) -> None:
    # This field records Git provenance only; package contents do not consume
    # platform harness files. Use this checkout to exercise revision lookup.
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first = TOOL["build"](
        package_dir=first_dir,
        platform_root=ROOT,
        source_commit="a" * 40,
    )
    second = TOOL["build"](
        package_dir=second_dir,
        platform_root=ROOT,
        source_commit="b" * 40,
    )
    assert first["source_commit"] == "a" * 40
    assert second["source_commit"] == "b" * 40
    assert first["platform_commit"] == second["platform_commit"]
    for row in first["recipes"]:
        filename = Path(str(row["package"]["path"])).name
        assert (first_dir / filename).read_bytes() == (
            second_dir / filename
        ).read_bytes()


def test_platform_owned_cache_variables_are_not_recipe_inputs() -> None:
    for path in sorted((ROOT / "recipes").glob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        runtime = document.get("runtime")
        environment = (
            runtime.get("environment", []) if isinstance(runtime, dict) else []
        )
        names = {
            item.get("name")
            for item in environment
            if isinstance(item, dict) and isinstance(item.get("name"), str)
        }
        assert not PLATFORM_OWNED_ENVIRONMENT & names, path.name


def test_catalog_excludes_provider_gated_models() -> None:
    assert all(
        not json.loads(path.read_text())["requires_token"]
        for path in ROOT.joinpath("models").glob("*.json")
    )


def test_model_territorial_restrictions_preserve_all_published_records() -> None:
    expected = {
        "hunyuan-video-15-distilled": (
            ["EU", "GB", "KR"],
            "The Tencent Hunyuan Community License Agreement does not apply in the European Union, United Kingdom, or South Korea.",
        ),
        "hunyuan-video-15-i2v-step-distilled": (
            ["EU", "GB", "KR"],
            "The Tencent Hunyuan Community License Agreement does not apply in the European Union, United Kingdom, or South Korea.",
        ),
        "hunyuan-video-15-t2v": (
            ["EU", "GB", "KR"],
            "The Tencent Hunyuan Community License Agreement does not apply in the European Union, United Kingdom, or South Korea.",
        ),
        "hunyuan-video-foley-xl": (
            ["EU", "GB", "KR"],
            "The Tencent Hunyuan Community License Agreement does not apply in the European Union, United Kingdom, or South Korea.",
        ),
        "hunyuan-video-foley-xxl": (
            ["EU", "GB", "KR"],
            "The Tencent Hunyuan Community License Agreement does not apply in the European Union, United Kingdom, or South Korea.",
        ),
        "hunyuan3d-omni": (
            ["EU", "GB", "KR"],
            "The upstream Hunyuan3D-Omni Community License does not apply in the European Union, United Kingdom, or South Korea.",
        ),
        "hunyuanocr-1-5-47644ecc": (
            ["EU", "GB", "KR"],
            "The Tencent Hunyuan Community License Agreement does not apply in the European Union, United Kingdom, or South Korea.",
        ),
        "minimax-h3": (
            ["EU", "GB", "KR", "US"],
            "The MiniMax H3 Community License Agreement excludes the European Union, United Kingdom, Republic of Korea, and United States of America from its Applicable Territory.",
        ),
        "minimax-h3-fl2va-42ed227e": (
            ["EU", "GB", "KR", "US"],
            "The MiniMax H3 Community License Agreement excludes the European Union, United Kingdom, Republic of Korea, and United States of America from its Applicable Territory.",
        ),
    }
    actual = {}
    for path in ROOT.joinpath("models").glob("*.json"):
        document = json.loads(path.read_text())
        restriction = document["license"].get("territorial_restrictions")
        if restriction is not None:
            actual[document["identity"]["slug"]] = (
                restriction["denied_jurisdictions"],
                restriction["notice"],
            )
    assert actual == expected


def test_packages_contain_metadata_and_sources_but_no_model_or_oci_payloads(
    built_catalog: tuple[dict, Path],
) -> None:
    """Model weights and image layers remain separately cached artifacts."""

    catalog, package_dir = built_catalog
    payload_suffixes = {
        ".safetensors",
        ".safetensors.index.json",
        ".bin",
        ".pt",
        ".pth",
        ".ckpt",
        ".onnx",
    }
    for row in catalog["recipes"]:
        package_path = package_dir / Path(str(row["package"]["path"])).name
        with tarfile.open(package_path, mode="r:gz") as archive:
            names = archive.getnames()
        build = row["document"]["execution"].get("build")
        source_context = build["context"]["path"] if build is not None else None
        assert all(
            not any(name.endswith(suffix) for suffix in payload_suffixes)
            for name in names
            if source_context is None or not name.startswith(f"{source_context}/")
        )
        assert all(
            not name.startswith(
                (
                    "image/",
                    "oci/",
                    "weights/",
                    "runtime-distributions/",
                    "patch-bundles/",
                    "execution-harnesses/",
                )
            )
            for name in names
        )


def test_ds4_multistage_package_manifests_both_digest_pinned_base_images(
    built_catalog: tuple[dict, Path],
) -> None:
    catalog, package_dir = built_catalog
    row = next(
        item
        for item in catalog["recipes"]
        if item["document"]["identity"]["slug"] == "deepseek-v4-flash-0731-ds4-single"
    )
    with tarfile.open(
        package_dir / Path(str(row["package"]["path"])).name, mode="r:gz"
    ) as archive:
        manifest = json.load(archive.extractfile("manifest.json"))
    assert manifest["build_inputs"] == [
        {
            "kind": "oci-image",
            "reference": "nvcr.io/nvidia/cuda:13.4.1-devel-ubuntu24.04@sha256:d44d6dc249c2c8330d0b380c595a514f0e86137fcc717fe7a4389a25720ade37",
            "platform": "linux/arm64",
        },
        {
            "kind": "oci-image",
            "reference": "nvcr.io/nvidia/cuda:13.4.1-runtime-ubuntu24.04@sha256:2eaf346843c93ae617a718818f4a804e2c46c0d97a85392212ec5b310ee57962",
            "platform": "linux/arm64",
        },
    ]
