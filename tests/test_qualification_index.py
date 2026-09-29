from __future__ import annotations

import base64
import hashlib
import importlib.machinery
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest
from generated_catalog import GENERATED
from vonk_forge_contracts import RecipeDefinition, document_sha256

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
        document = json.loads(path.read_bytes())
        recipe = RecipeDefinition.model_validate(document)
        if recipe.topology.node_count <= 2:
            key = f"{recipe.identity.publisher}/{recipe.identity.slug}"
            expected[key] = document_sha256(document)

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


def test_authority_requires_the_generated_catalog(tmp_path: Path) -> None:
    """Without a generated catalog there is nothing to bind: fail closed."""

    with pytest.raises(ValueError, match="run tools/build-catalog-index"):
        AUTHORITY_TOOL["_build"](tmp_path)


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
