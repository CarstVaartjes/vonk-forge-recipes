from __future__ import annotations

import base64
import copy
import hashlib
import struct
import io
import json
import os
import runpy
import tarfile
from unittest import SkipTest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOL = runpy.run_path(str(ROOT / "tools/build-catalog-index"))
PLATFORM_ROOT = Path(os.environ.get("VONK_FORGE_PLATFORM_ROOT", "/opt/vonk-forge"))
PLATFORM_OWNED_ENVIRONMENT = {
    "FLASHINFER_WORKSPACE_BASE",
    "TILELANG_CACHE_DIR",
    "TRITON_CACHE_DIR",
    "B12X_CUTE_COMPILE_CACHE_DIR",
    "TORCH_FR_DUMP_TEMP_FILE",
    "TORCH_NCCL_DEBUG_INFO_PIPE_FILE",
}


def _platform_root() -> Path:
    if not (PLATFORM_ROOT / "config" / "execution-harnesses").is_dir():
        raise SkipTest("authoritative Vonk Forge platform checkout is unavailable")
    return PLATFORM_ROOT


def test_full_catalog_packages_are_self_contained_and_deterministic(tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    platform_root = _platform_root()
    first = TOOL["build"](package_dir=first_dir, platform_root=platform_root)
    second = TOOL["build"](package_dir=second_dir, platform_root=platform_root)
    assert first["kind"] == second["kind"] == "recipe-library-index"
    assert first["schema_version"] == second["schema_version"] == 2
    source_models = {
        (json.loads(path.read_text())["identity"]["publisher"], json.loads(path.read_text())["identity"]["slug"])
        for path in ROOT.joinpath("models").glob("*.json")
    }
    source_recipes = {
        (json.loads(path.read_text())["identity"]["publisher"], json.loads(path.read_text())["identity"]["slug"])
        for path in ROOT.joinpath("recipes").glob("*.json")
    }
    assert source_models and source_recipes
    assert len(source_models) == len(list(ROOT.joinpath("models").glob("*.json")))
    assert len(source_recipes) == len(list(ROOT.joinpath("recipes").glob("*.json")))
    for catalog in (first, second):
        assert {
            (item["document"]["identity"]["publisher"], item["document"]["identity"]["slug"])
            for item in catalog["catalog_entities"]
        } == source_models
        assert {
            (item["document"]["identity"]["publisher"], item["document"]["identity"]["slug"])
            for item in catalog["recipes"]
        } == source_recipes
        package_names = {Path(str(item["package"]["path"])).name for item in catalog["recipes"]}
        assert package_names == {f"{slug}.tar.gz" for _, slug in source_recipes}
    checked_index = json.loads((ROOT / "catalog-index.json").read_text())
    assert {
        (item["document"]["identity"]["publisher"], item["document"]["identity"]["slug"])
        for item in checked_index["catalog_entities"]
    } == source_models
    assert {
        (item["document"]["identity"]["publisher"], item["document"]["identity"]["slug"])
        for item in checked_index["recipes"]
    } == source_recipes
    expected_package_names = {f"{slug}.tar.gz" for _, slug in source_recipes}
    assert {path.name for path in ROOT.joinpath("packages").glob("*.tar.gz")} == expected_package_names

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
            assert all(not name.startswith("/") and ".." not in name.split("/") for name in names)
            manifest = json.load(archive.extractfile("manifest.json"))
            assert manifest["schema_version"] == 2
            assert manifest["kind"] == "recipe-package"
            assert manifest["recipe_content_sha256"] == first_row["content_sha256"]
            assert manifest["package_type"] == "recipe"
            for entry in manifest["files"]:
                payload = archive.extractfile(entry["path"]).read()
                assert len(payload) == entry["size"]
                assert hashlib.sha256(payload).hexdigest() == entry["sha256"]


def test_vision_serving_uses_a_real_png_payload() -> None:
    recipe_path = ROOT / "recipes/deepseek-v4-flash-vision-exp-mia-dual.json"
    recipe = json.loads(recipe_path.read_text())
    checks = recipe["validation"]["serving"]["checks"]
    vision = next(check for check in checks if check["kind"] == "openai.vision")
    parts = vision["request"]["body"]["messages"][0]["content"]
    image = next(part["image_url"]["url"] for part in parts if part["type"] == "image_url")
    prefix, encoded = image.split(",", 1)
    assert prefix == "data:image/png;base64"
    payload = base64.b64decode(encoded, validate=True)
    assert payload.startswith(b"\x89PNG\r\n\x1a\n")
    assert payload[12:16] == b"IHDR"
    assert struct.unpack(">II", payload[16:24]) == (64, 64)


def test_release_history_is_typed_recipe_metadata_without_self_digest() -> None:
    recipes = [json.loads(path.read_text()) for path in ROOT.joinpath("recipes").glob("*.json")]
    assert recipes
    entries = [entry for recipe in recipes for entry in recipe["release"]["history"]]
    assert entries
    assert all("recipe_content_sha256" not in entry for entry in entries)
    assert all(
        entry.get("prior_recipe_content_sha256") is None
        or len(entry["prior_recipe_content_sha256"]) == 64
        for entry in entries
    )
    assert {entry["upgrade_effect"] for entry in entries} <= {"none", "restart", "reprepare", "rebuild"}
    assert all(recipe["release"]["history"][0]["version"] == recipe["release"]["version"] for recipe in recipes)


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


def test_editing_one_recipe_changes_only_that_package(tmp_path: Path) -> None:
    platform_root = _platform_root()
    catalog = TOOL["build"](package_dir=tmp_path, platform_root=platform_root)
    rows = catalog["recipes"]
    original = {
        Path(str(row["package"]["path"])).name: (
            tmp_path / Path(str(row["package"]["path"])).name
        ).read_bytes()
        for row in rows
    }
    target = rows[0]
    edited = copy.deepcopy(target["document"])
    edited["metadata"]["description"] += " edited"
    entities = TOOL["_catalog_entity_documents"]()
    entities.update(TOOL["_platform_harness_documents"](platform_root))
    package_bytes, package = TOOL["recipe_package"](
        edited,
        recipe_path=ROOT / str(target["source_path"]),
        entity_documents=entities,
    )
    assert package_bytes != original[Path(str(target["package"]["path"])).name]
    assert package["media_type"] == "application/vnd.vonk-forge.recipe-package.v2+tar+gzip"
    for row in rows[1:]:
        filename = Path(str(row["package"]["path"])).name
        assert (tmp_path / filename).read_bytes() == original[filename]


def test_supplied_source_commit_only_changes_index_metadata(tmp_path: Path) -> None:
    platform_root = _platform_root()
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first = TOOL["build"](
        package_dir=first_dir,
        platform_root=platform_root,
        source_commit="a" * 40,
    )
    second = TOOL["build"](
        package_dir=second_dir,
        platform_root=platform_root,
        source_commit="b" * 40,
    )
    assert first["source_commit"] == "a" * 40
    assert second["source_commit"] == "b" * 40
    assert first["platform_commit"] == second["platform_commit"]
    for row in first["recipes"]:
        filename = Path(str(row["package"]["path"])).name
        assert (first_dir / filename).read_bytes() == (second_dir / filename).read_bytes()


def test_platform_owned_cache_variables_are_not_recipe_inputs() -> None:
    for path in sorted((ROOT / "recipes").glob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        runtime = document.get("runtime")
        environment = runtime.get("environment", []) if isinstance(runtime, dict) else []
        names = {
            item.get("name")
            for item in environment
            if isinstance(item, dict) and isinstance(item.get("name"), str)
        }
        assert not PLATFORM_OWNED_ENVIRONMENT & names, path.name


def test_model_capability_authority_is_external_and_canonical() -> None:
    evidence = json.loads(
        (ROOT / "docs/model-capability-evidence-2026-09-05.json").read_text(
            encoding="utf-8"
        )
    )
    assert evidence["schema_version"] == 2
    evidence_digest = evidence["evidence_digest"]
    entries = {
        (item["model_version"]["publisher"], item["model_version"]["slug"]): item
        for item in evidence["entries"]
    }
    model_versions = sorted((ROOT / "models").glob("*.json"))
    model_keys = {
        (json.loads(path.read_text(encoding="utf-8"))["identity"]["publisher"],
         json.loads(path.read_text(encoding="utf-8"))["identity"]["slug"])
        for path in model_versions
    }
    evidence_keys = set(entries)
    assert evidence_keys <= model_keys
    unknown_keys = model_keys - evidence_keys
    assert evidence_keys | unknown_keys == model_keys
    assert not evidence_keys & unknown_keys
    unknown = []
    for path in model_versions:
        document = json.loads(path.read_text(encoding="utf-8"))
        capabilities = document.get("capabilities")
        key = (document["identity"]["publisher"], document["identity"]["slug"])
        assert capabilities is not None
        if not capabilities["facts"]:
            unknown.append(key)
            assert key in unknown_keys
            continue
        assert key in evidence_keys
        assert capabilities["schema_version"] == 2
        assert capabilities["provenance"]["evidence_digest"] == evidence_digest
        assert capabilities["provenance"]["source_url"].startswith("https://")
        assert len(capabilities["provenance"]["source_revision"]) == 40
        facts = capabilities["facts"]
        assert facts == sorted(facts, key=lambda item: item["capability"])
        assert len({item["capability"] for item in facts}) == len(facts)
        assert all(item["support"] == "supported" for item in facts)
        assert all(item["evidence_status"] == "declared" for item in facts)
        assert all(
            item["evidence_digest"] in {None, evidence_digest} for item in facts
        )
        assert all("vision" != item["capability"] for item in facts)
    assert set(unknown) == unknown_keys


def test_model_access_lineage_and_related_model_references_are_preserved() -> None:
    restricted = {}
    dependency_count = 0
    supersedes = []
    for path in ROOT.joinpath("models").glob("*.json"):
        document = json.loads(path.read_text())
        access = document["access"]
        assert set(access) == {"visibility", "gated", "authentication"}
        if access["visibility"] == "restricted":
            restricted[document["identity"]["slug"]] = access
        dependency_count += len(document["dependencies"])
        if document["supersedes"] is not None:
            supersedes.append(document["identity"]["slug"])
        lineage = document["lineage"]
        assert set(lineage) == {"publisher", "relation", "source_model", "derivation"}
        assert set(lineage["source_model"]) == {"kind", "publisher", "slug"}
    assert set(restricted) == {
        "glm-5-3-flash-nvfp4-ablit-l15-43-mtp-l45-80b6d18d",
        "glm-5-3-flash-nvfp4-abliterated-d7f8afa8",
        "ltx-2-5-22b-distilled-bf16-diffusers",
    }
    assert all(value == {"visibility": "restricted", "gated": True, "authentication": "token"} for value in restricted.values())
    assert dependency_count == 7
    assert supersedes == ["hunyuanocr-1-5-47644ecc"]


def test_model_territorial_restrictions_preserve_all_published_records() -> None:
    expected = {
        "hunyuan-video-15-distilled": (["EU", "GB", "KR"], "The Tencent Hunyuan Community License Agreement does not apply in the European Union, United Kingdom, or South Korea."),
        "hunyuan-video-15-i2v-step-distilled": (["EU", "GB", "KR"], "The Tencent Hunyuan Community License Agreement does not apply in the European Union, United Kingdom, or South Korea."),
        "hunyuan-video-15-t2v": (["EU", "GB", "KR"], "The Tencent Hunyuan Community License Agreement does not apply in the European Union, United Kingdom, or South Korea."),
        "hunyuan-video-foley-xl": (["EU", "GB", "KR"], "The Tencent Hunyuan Community License Agreement does not apply in the European Union, United Kingdom, or South Korea."),
        "hunyuan-video-foley-xxl": (["EU", "GB", "KR"], "The Tencent Hunyuan Community License Agreement does not apply in the European Union, United Kingdom, or South Korea."),
        "hunyuan3d-omni": (["EU", "GB", "KR"], "The upstream Hunyuan3D-Omni Community License does not apply in the European Union, United Kingdom, or South Korea."),
        "hunyuanocr-1-5-449e7d47": (["EU", "GB", "KR"], "The Tencent Hunyuan Community License Agreement does not apply in the European Union, United Kingdom, or South Korea."),
        "hunyuanocr-1-5-47644ecc": (["EU", "GB", "KR"], "The Tencent Hunyuan Community License Agreement does not apply in the European Union, United Kingdom, or South Korea."),
        "minimax-h3": (["EU", "GB", "KR", "US"], "The MiniMax H3 Community License Agreement excludes the European Union, United Kingdom, Republic of Korea, and United States of America from its Applicable Territory."),
        "minimax-h3-fl2va-42ed227e": (["EU", "GB", "KR", "US"], "The MiniMax H3 Community License Agreement excludes the European Union, United Kingdom, Republic of Korea, and United States of America from its Applicable Territory."),
    }
    actual = {}
    for path in ROOT.joinpath("models").glob("*.json"):
        document = json.loads(path.read_text())
        restriction = document["license"].get("territorial_restrictions")
        if restriction is not None:
            actual[document["identity"]["slug"]] = (restriction["denied_jurisdictions"], restriction["notice"])
    assert actual == expected


def test_packages_contain_metadata_and_sources_but_no_model_or_oci_payloads(
    tmp_path: Path,
) -> None:
    """Model weights and image layers remain separately cached artifacts."""

    catalog = TOOL["build"](
        package_dir=tmp_path,
        platform_root=_platform_root(),
    )
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
        package_path = tmp_path / Path(str(row["package"]["path"])).name
        with tarfile.open(package_path, mode="r:gz") as archive:
            names = archive.getnames()
        assert all(
            not any(name.endswith(suffix) for suffix in payload_suffixes)
            for name in names
        )
        assert all(not name.startswith(("image/", "oci/")) for name in names)


def test_ds4_multistage_package_manifests_both_digest_pinned_base_images(tmp_path: Path) -> None:
    catalog = TOOL["build"](package_dir=tmp_path, platform_root=_platform_root())
    row = next(item for item in catalog["recipes"] if item["document"]["identity"]["slug"] == "deepseek-v4-flash-0731-ds4-single")
    with tarfile.open(tmp_path / Path(str(row["package"]["path"])).name, mode="r:gz") as archive:
        manifest = json.load(archive.extractfile("manifest.json"))
    assert manifest["build_inputs"] == [
        {"kind": "oci-image", "reference": "nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04@sha256:5c36750138dc1447a17dafbb397674f167d3b44ce18d9160d769df114577b35d", "platform": "linux/arm64"},
        {"kind": "oci-image", "reference": "nvcr.io/nvidia/cuda:13.0.1-runtime-ubuntu24.04@sha256:36050649ad1acc5d3de2c26620191c25850fb12a5771b6c22996033003d952e4", "platform": "linux/arm64"},
    ]
