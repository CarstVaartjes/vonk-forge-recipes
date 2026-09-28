from __future__ import annotations

import base64
import hashlib
import importlib.machinery
import importlib.util
import json
import os
import subprocess
import sys
import urllib.error
from email.message import Message
from pathlib import Path
from typing import Any, cast

import pytest
from generated_catalog import GENERATED
from vonk_forge_contracts import RecipeDefinition, content_sha256

from qualification import catalog_source
from qualification.catalog_source import package_tree, release_asset
from qualification.coverage_identity import execution_stack_identity

ROOT = Path(__file__).resolve().parents[1]
QUALIFICATION_ROOT = ROOT / "qualification"
PLAN_PATH = ROOT / "docs/recipe-qualification-plan-2026-09-24.md"
AUTHORITY_PATH = QUALIFICATION_ROOT / "authorities/nl-family-aware-20260924.json"
CAMPAIGN_PATH = QUALIFICATION_ROOT / "campaigns/nl-family-aware-20260924.json"


def _authority_namespace() -> dict[str, Any]:
    """Load the extensionless entry point and return its real module namespace.

    ``runpy.run_path`` returns a *copy* of the module globals, so a test that
    patched that copy would leave the functions under test unchanged and pass
    for the wrong reason. The module's own ``__dict__`` is what its functions
    close over, so patching it reaches the code under test.
    """

    loader = importlib.machinery.SourceFileLoader(
        "build_qualification_authority",
        str(ROOT / "tools/build-qualification-authority"),
    )
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module.__dict__


AUTHORITY_TOOL = _authority_namespace()


def _document(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _git_blob(commit: str, relative_path: str, *, root: Path = ROOT) -> bytes:
    environment = os.environ.copy()
    environment["GIT_NO_REPLACE_OBJECTS"] = "1"
    environment["GIT_NO_LAZY_FETCH"] = "1"
    result = subprocess.run(
        [
            "git",
            "--no-replace-objects",
            "cat-file",
            "blob",
            f"{commit}:{relative_path}",
        ],
        cwd=root,
        env=environment,
        check=True,
        capture_output=True,
        timeout=10,
    )
    return result.stdout


def _bindings(document: dict[str, object]) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for field in ("recipes", "service_recipes", "special_fixtures"):
        values = document[field]
        assert isinstance(values, dict)
        for key, value in values.items():
            assert key not in result
            assert isinstance(value, dict)
            result[key] = value
    return result


def test_recipe_digests_are_generated_locally_and_cover_supported_topologies() -> None:
    definitions = _document(QUALIFICATION_ROOT / "definitions.json")
    generated = _document(GENERATED / "qualification" / "qualification-index.json")
    source_bindings = _bindings(definitions)
    generated_bindings = _bindings(generated)

    assert all("content_sha256" not in value for value in source_bindings.values())
    expected: dict[str, str] = {}
    for path in sorted((ROOT / "recipes").glob("*.json")):
        recipe = RecipeDefinition.model_validate_json(path.read_bytes())
        if recipe.topology.node_count <= 2:
            key = f"{recipe.identity.publisher}/{recipe.identity.slug}"
            expected[key] = content_sha256(recipe)

    assert set(source_bindings) == set(generated_bindings) == set(expected)
    assert {
        key: value["content_sha256"] for key, value in generated_bindings.items()
    } == expected


def test_qualification_assets_are_owned_and_digest_checked_here() -> None:
    definitions = _document(QUALIFICATION_ROOT / "definitions.json")
    fixtures = definitions["fixtures"]
    assert isinstance(fixtures, dict)
    assert fixtures
    for fixture_id, value in fixtures.items():
        assert isinstance(value, dict), fixture_id
        path = (QUALIFICATION_ROOT / str(value["path"])).resolve()
        assert path.is_relative_to(QUALIFICATION_ROOT.resolve())
        content = path.read_bytes()
        if value["encoding"] == "base64":
            content = base64.b64decode(b"".join(content.split()), validate=True)
        assert len(content) == value["size_bytes"], fixture_id
        assert hashlib.sha256(content).hexdigest() == value["sha256"], fixture_id
        assert isinstance(value.get("provenance"), dict), fixture_id


def test_deepseek_vision_smoke_contract_is_recipe_owned() -> None:
    definitions = _document(QUALIFICATION_ROOT / "definitions.json")
    services = definitions["service_recipes"]
    assert isinstance(services, dict)
    contract = services["vonk-forge/deepseek-v4-flash-vision-exp-mia-dual"]
    assert contract["alias"] == "deepseek-v4-flash-vision-exp"
    assert contract["smoke_cases"] == ["M0", "A391", "T_REPORT", "V_RED"]
    assert "content_sha256" not in contract


def test_cube_fixture_generator_is_byte_identical(tmp_path: Path) -> None:
    output = tmp_path / "cube.glb"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/generate-qualification-cube-glb"),
            str(output),
        ],
        check=True,
    )
    definitions = _document(QUALIFICATION_ROOT / "definitions.json")
    fixtures = definitions["fixtures"]
    assert isinstance(fixtures, dict)
    fixture = fixtures["generic-mesh-glb"]
    assert isinstance(fixture, dict)
    fixture_path = fixture["path"]
    assert isinstance(fixture_path, str)
    encoded = (QUALIFICATION_ROOT / fixture_path).read_bytes()
    assert output.read_bytes() == base64.b64decode(
        b"".join(encoded.split()), validate=True
    )


def test_skintokens_derivation_pins_upstream_and_transform() -> None:
    source = (ROOT / "tools/derive-skintokens-rigged-figure").read_text(
        encoding="utf-8"
    )
    assert "d6be85417d3e256861ee733eea6916093a7af7c79c16366181fd8abcaeb38cf5" in source
    assert 'trimesh.__version__ != "5.0.0"' in source
    assert 'force="mesh", process=True' in source


def test_release_asset_that_differs_from_its_pin_fails_closed(
    monkeypatch, tmp_path: Path
) -> None:
    """A downloaded or cached asset is never trusted without its pinned digest."""

    monkeypatch.setattr(catalog_source, "RELEASE_CACHE", tmp_path)
    (tmp_path / "v9.9.9").mkdir()
    (tmp_path / "v9.9.9" / "catalog-index.json").write_bytes(b"stale cache")
    monkeypatch.setattr(catalog_source, "_download", lambda tag, name: b"tampered")
    with pytest.raises(ValueError, match="differs from its pinned digest"):
        release_asset("v9.9.9", "catalog-index.json", "0" * 64)
    assert (tmp_path / "v9.9.9" / "catalog-index.json").read_bytes() == b"stale cache"

    good = b"published bytes"
    digest = hashlib.sha256(good).hexdigest()
    monkeypatch.setattr(catalog_source, "_download", lambda tag, name: good)
    assert release_asset("v9.9.9", "catalog-index.json", digest) == good
    # The verified download is cached, so later reads need no network.
    monkeypatch.setattr(catalog_source, "_download", None)
    assert release_asset("v9.9.9", "catalog-index.json", digest) == good


def test_unreachable_release_asset_is_unavailable_after_bounded_retries(
    monkeypatch, tmp_path: Path
) -> None:
    """A transient GitHub outage is retried, then reported as unavailable."""

    attempts: list[str] = []

    def outage(url: str, timeout: float) -> Any:
        attempts.append(url)
        raise urllib.error.HTTPError(url, 500, "Internal Server Error", Message(), None)

    monkeypatch.setattr(catalog_source, "RELEASE_CACHE", tmp_path)
    monkeypatch.setattr(catalog_source.urllib.request, "urlopen", outage)
    monkeypatch.setattr(catalog_source.time, "sleep", lambda seconds: None)
    with pytest.raises(catalog_source.ReleaseAssetUnavailable):
        release_asset("v9.9.9", "catalog-index.json", "0" * 64)
    assert len(attempts) == catalog_source._DOWNLOAD_ATTEMPTS > 1


def test_authority_fails_closed_on_a_malformed_pin(monkeypatch) -> None:
    """Accepting a release pin that is not a full Git SHA must fail closed."""

    real_document = AUTHORITY_TOOL["_document"]

    def fake_document(path: Path) -> dict[str, Any]:
        document = real_document(path)
        if path.name == "catalog-release.json":
            document["commit"] = "not-a-commit"
        return document

    monkeypatch.setitem(AUTHORITY_TOOL, "_document", fake_document)
    with pytest.raises(
        ValueError, match="catalog release commit must be a full Git SHA"
    ):
        AUTHORITY_TOOL["_build"]()


def test_authority_writes_nothing_when_the_build_fails(
    monkeypatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Writing an output before the build succeeds must fail."""

    outputs = (
        AUTHORITY_PATH,
        CAMPAIGN_PATH,
        PLAN_PATH,
    )
    before = {path: path.read_bytes() for path in outputs}

    def unavailable(catalog_root: Path | None = None) -> dict[Path, bytes]:
        raise ValueError("pinned catalog is unavailable")

    monkeypatch.setitem(AUTHORITY_TOOL, "_build", unavailable)
    monkeypatch.setattr(sys, "argv", ["build-qualification-authority", "--check"])
    assert AUTHORITY_TOOL["main"]() == 1
    assert "pinned catalog is unavailable" in capsys.readouterr().err
    assert {path: path.read_bytes() for path in outputs} == before


def test_published_stack_identity_uses_exact_package_not_stale_source_commit(
    tmp_path: Path,
) -> None:
    """The release-bound package, not its stale source pointer, owns source bytes."""

    # Keep the reproducer on the release with the stale pointer; the active
    # campaign pin must be free to advance to corrected future releases.
    catalog_commit = "7b23f1a1569e16f8f2728ddd6846cf4f7c58f0f7"
    catalog = json.loads(_git_blob(catalog_commit, "catalog-index.json"))
    source_commit = catalog["source_commit"]
    assert isinstance(source_commit, str)
    entry = next(
        item
        for item in catalog["recipes"]
        if item["document"]["identity"]["slug"]
        == "ltx-2-19b-distilled-diffusers-single"
    )
    recipe = RecipeDefinition.model_validate(entry["document"])
    assert recipe.execution.mode == "build"
    payload = _git_blob(catalog_commit, entry["package"]["path"])
    with package_tree(entry["package"], recipe, payload) as package_root:
        package_wheel = (
            package_root
            / "adapters/video/ltx2-sync-native/vonk_agent_protocol-3.0.0-py3-none-any.whl"
        )
        package_wheel_bytes = package_wheel.read_bytes()
        assert hashlib.sha256(package_wheel_bytes).hexdigest() == (
            "519484690b626f03e27efabad787ad1040a7d527404f2dd4faa3e2d84c40d116"
        )
        package_identity = execution_stack_identity(
            package_root, recipe, recipe.execution.build
        )

    stale_source_wheel = _git_blob(
        source_commit,
        "adapters/video/ltx2-sync-native/vonk_agent_protocol-2.2.0-py3-none-any.whl",
    )
    assert hashlib.sha256(stale_source_wheel).hexdigest() == (
        "7555df9ec0f576ac0530d1e6abdd2f3845614cf397a7bc2c8dcd01bc86967565"
    )
    assert stale_source_wheel != package_wheel_bytes

    key = "vonk-forge/ltx-2-19b-distilled-diffusers-single"
    # The historical release had no assets; the byte-identical package under
    # catalog_root stands in for the release asset, so nothing is downloaded.
    (tmp_path / "packages").mkdir()
    (tmp_path / entry["package"]["path"]).write_bytes(payload)
    generated = AUTHORITY_TOOL["_pinned_stack_identities"](
        "v0.0.0", {key: entry}, tmp_path
    )
    assert generated[key] == package_identity

    mismatched_package = {**entry["package"], "sha256": "0" * 64}
    with (
        pytest.raises(ValueError, match="package digest differs"),
        package_tree(mismatched_package, recipe, payload),
    ):
        pass


def _authority_document() -> dict[str, Any]:
    return cast(Any, _document(AUTHORITY_PATH))


def test_plan_prose_changes_never_invalidate_the_generated_inventory() -> None:
    """A prose edit, renamed heading or moved paragraph must not go stale."""

    text = PLAN_PATH.read_text(encoding="utf-8")
    authority = _authority_document()
    refresh = AUTHORITY_TOOL["_refresh_plan"]
    assert refresh(text, authority) == text

    edited = text.replace(
        "Keep this inventory complete.",
        "Keep this inventory complete and reviewed by its owner.",
        1,
    )
    assert edited != text
    assert refresh(edited, authority) == edited

    renamed = edited.replace(
        "## Complete inventory and current batch assignments",
        "## Inventory and campaign order",
        1,
    )
    assert renamed != edited
    assert refresh(renamed, authority) == renamed


def test_plan_without_the_generated_inventory_block_fails_closed() -> None:
    """A missing, duplicated or reversed marker must fail closed."""

    text = PLAN_PATH.read_text(encoding="utf-8")
    authority = _authority_document()
    begin = AUTHORITY_TOOL["_INVENTORY_BEGIN"]
    end = AUTHORITY_TOOL["_INVENTORY_END"]
    assert text.count(begin) == 1
    assert text.count(end) == 1
    swapped = text.replace(begin, "<!-- swap -->", 1)
    swapped = swapped.replace(end, begin, 1).replace("<!-- swap -->", end, 1)

    for broken in (
        text.replace(begin, "", 1),
        text.replace(end, "", 1),
        text.replace(begin, f"{begin}\n{begin}", 1),
        text.replace(begin, "", 1).replace(end, "", 1),
        swapped,
    ):
        with pytest.raises(ValueError):
            AUTHORITY_TOOL["_plan_rows"](broken)
        with pytest.raises(ValueError):
            AUTHORITY_TOOL["_refresh_plan"](broken, authority)
