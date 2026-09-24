from __future__ import annotations

import base64
import hashlib
import importlib.machinery
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest
from vonk_forge_contracts import RecipeDefinition, content_sha256

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


def test_generated_qualification_index_is_current() -> None:
    subprocess.run(
        [sys.executable, "tools/build-catalog-index", "--check"],
        cwd=ROOT,
        check=True,
    )


def test_recipe_digests_are_generated_locally_and_cover_supported_topologies() -> None:
    definitions = _document(QUALIFICATION_ROOT / "definitions.json")
    generated = _document(QUALIFICATION_ROOT / "qualification-index.json")
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


def test_authority_reads_both_indexes_from_the_release_commit(monkeypatch) -> None:
    """Reading a working-tree index instead of the pinned commit must fail."""

    release = _document(QUALIFICATION_ROOT / "catalog-release.json")
    commit = release["commit"]
    assert isinstance(commit, str)
    calls: list[tuple[str, str]] = []
    real_blob = AUTHORITY_TOOL["_git_blob"]

    def spy(commit_arg: str, relative_path: str) -> bytes:
        calls.append((commit_arg, relative_path))
        return real_blob(commit_arg, relative_path)

    monkeypatch.setitem(AUTHORITY_TOOL, "_git_blob", spy)
    AUTHORITY_TOOL["_build"]()
    assert calls == [
        (commit, "catalog-index.json"),
        (commit, "qualification/qualification-index.json"),
    ]


def test_authority_fails_closed_on_an_unreadable_pin() -> None:
    """Following an object that is not in this repository must fail closed."""

    with pytest.raises(ValueError, match="cannot read pinned Git object"):
        AUTHORITY_TOOL["_git_blob"]("0" * 40, "catalog-index.json")


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

    def unavailable() -> dict[Path, bytes]:
        raise ValueError("pinned catalog is unavailable")

    monkeypatch.setitem(AUTHORITY_TOOL, "_build", unavailable)
    monkeypatch.setattr(sys, "argv", ["build-qualification-authority", "--check"])
    assert AUTHORITY_TOOL["main"]() == 1
    assert "pinned catalog is unavailable" in capsys.readouterr().err
    assert {path: path.read_bytes() for path in outputs} == before


def test_batched_authority_binds_every_recipe_with_at_most_two_nodes() -> None:
    subprocess.run(
        [sys.executable, "tools/build-qualification-authority", "--check"],
        cwd=ROOT,
        check=True,
    )
    authority_path = AUTHORITY_PATH
    campaign_path = CAMPAIGN_PATH
    authority = cast(Any, _document(authority_path))
    campaign = cast(Any, _document(campaign_path))
    review_gates = cast(Any, _document(QUALIFICATION_ROOT / "review-gates.json"))
    release = cast(Any, _document(QUALIFICATION_ROOT / "catalog-release.json"))
    catalog_commit = release["commit"]
    assert isinstance(catalog_commit, str)
    catalog_index_bytes = _git_blob(catalog_commit, "catalog-index.json")
    qualification_bytes = _git_blob(
        catalog_commit, "qualification/qualification-index.json"
    )
    catalog_index = cast(Any, json.loads(catalog_index_bytes))
    qualification = cast(Any, json.loads(qualification_bytes))

    from vonk_forge_contracts.qualification_authority import (
        QualificationAuthority,
        QualificationCampaignManifest,
    )

    # Validate the actual generated bytes through the same consumer contract used
    # by the platform; checking a dictionary assembled by the producer is not enough.
    authority_bytes = authority_path.read_bytes()
    campaign_bytes = campaign_path.read_bytes()
    authority_model = QualificationAuthority.model_validate_json(authority_bytes)
    campaign_model = QualificationCampaignManifest.model_validate_json(campaign_bytes)
    assert authority_model.authority_id == "nl-family-aware-20260924"
    assert campaign_model.qualification_authority == (
        "../authorities/nl-family-aware-20260924.json"
    )
    assert authority["schema_version"] == 4
    assert campaign["schema_version"] == 2
    assert authority["catalog"]["repository"] == "CarstVaartjes/vonk-forge-recipes"
    assert (
        authority["catalog"]["catalog_index_sha256"]
        == hashlib.sha256(catalog_index_bytes).hexdigest()
    )
    assert (
        authority["catalog"]["qualification_index_sha256"]
        == hashlib.sha256(qualification_bytes).hexdigest()
    )
    assert authority["catalog"]["source_commit"] == catalog_index["source_commit"]
    assert authority["catalog"]["recipe_count"] == len(catalog_index["recipes"]) == 85
    assert authority["scope"]["maximum_node_count"] == 2

    assert authority["catalog"]["commit"] == release["commit"]
    assert authority["catalog"]["release_tag"] == release["release_tag"]

    recipes_by_key: dict[str, Any] = {}
    models_by_digest: dict[str, Any] = {}
    for entry in catalog_index["recipes"]:
        document = entry["document"]
        identity = document["identity"]
        recipes_by_key[f"{identity['publisher']}/{identity['slug']}"] = entry
    for entity in catalog_index["catalog_entities"]:
        if entity["document"].get("kind") == "model":
            models_by_digest[entity["content_sha256"]] = entity

    expected_in_scope = {
        key
        for key, entry in recipes_by_key.items()
        if entry["document"]["topology"]["node_count"] <= 2
    }
    expected_excluded = {
        key
        for key, entry in recipes_by_key.items()
        if entry["document"]["topology"]["node_count"] > 2
    }
    rows = authority["recipes"]
    assert len(rows) == authority["scope"]["recipe_count"] == 81
    assert [row["sequence"] for row in rows] == list(range(1, 82))
    assert {row["key"] for row in rows} == expected_in_scope
    assert set(authority["scope"]["excluded_topology_recipe_keys"]) == expected_excluded
    capacity_reviews = review_gates["capacity_reviews"]

    artifact_qualifications = qualification["recipes"]
    service_qualifications = qualification["service_recipes"]
    service_templates = qualification["service_case_templates"]
    fixtures = qualification["fixtures"]

    def fixture_markers(value: object) -> list[str]:
        found: list[str] = []
        if isinstance(value, dict):
            for field, child in value.items():
                if field in {"$fixture_data_uri", "$fixture_base64"}:
                    assert isinstance(child, str)
                    found.append(child)
                else:
                    found.extend(fixture_markers(child))
        elif isinstance(value, list):
            for child in value:
                found.extend(fixture_markers(child))
        return found

    for sequence, row in enumerate(rows, start=1):
        key = row["key"]
        entry = recipes_by_key[key]
        recipe = entry["document"]
        digest = entry["content_sha256"]
        assert row["sequence"] == sequence
        assert row["node_count"] == recipe["topology"]["node_count"]
        assert row["content_sha256"] == digest
        assert row["recipe_version"] == recipe["release"]["version"]
        assert row["package"]["sha256"] == entry["package"]["sha256"]
        assert len(row["runtime_stack_sha256"]) == 64
        assert len(row["topology_sha256"]) == 64
        assert row["recovery_coverage_refs"]

        if key in artifact_qualifications:
            qualification_row = artifact_qualifications[key]
            assert qualification_row["content_sha256"] == digest
            cases = qualification_row.get("cases", [])
            expected_cases = ["default", *[case["id"] for case in cases]]
            expected_inputs = [
                fixture_input["fixture"]
                for case in [qualification_row, *cases]
                for fixture_input in case["inputs"]
            ]
        else:
            qualification_row = service_qualifications[key]
            assert qualification_row["content_sha256"] == digest
            expected_cases = qualification_row["smoke_cases"]
            expected_inputs = [
                fixture_id
                for case_id in expected_cases
                for section in ("body", "assertions")
                for fixture_id in fixture_markers(service_templates[case_id][section])
            ]
        assert row["smoke_cases"] == expected_cases
        assert row["qualification_inputs"] == list(dict.fromkeys(expected_inputs))
        assert all(fixture_id in fixtures for fixture_id in row["qualification_inputs"])

        expected_model_refs = []
        for selected in recipe["models"]:
            reference = selected["model"]
            model = models_by_digest[reference["content_sha256"]]["document"]
            license_data = model["license"]
            identity = model["identity"]
            model_ref = {
                "key": f"{identity['publisher']}/{identity['slug']}",
                "content_sha256": reference["content_sha256"],
                "spdx": license_data["spdx"],
                "url": license_data["url"],
                "attribution": license_data["attribution"],
                "operator_acceptance_required": license_data[
                    "operator_acceptance_required"
                ],
            }
            if license_data.get("territorial_restrictions") is not None:
                model_ref["territorial_restrictions"] = license_data[
                    "territorial_restrictions"
                ]
            expected_model_refs.append(model_ref)
        assert row["model_license_refs"] == expected_model_refs

        requires_acceptance = any(
            model_ref["operator_acceptance_required"]
            for model_ref in expected_model_refs
        )
        expected_gate_kinds = []
        if requires_acceptance:
            expected_gate_kinds.append("operator-acceptance-required")
        if key in capacity_reviews:
            expected_gate_kinds.append("capacity-review")
        assert row["operator_acceptance_required"] is requires_acceptance
        assert [gate["kind"] for gate in row["review_gates"]] == expected_gate_kinds
        assert row["disposition"] == (
            "operator-acceptance-required"
            if requires_acceptance
            else "capacity-review"
            if key in capacity_reviews
            else "actionable"
        )
        if expected_gate_kinds:
            assert row["disposition_reason"]
            assert all(gate["reason"] for gate in row["review_gates"])

    assert campaign["options"]["cleanup"] == "stop"
    authority_reference = campaign["qualification_authority"]
    fixture_reference = campaign["fixture_manifest"]
    assert isinstance(authority_reference, str)
    assert isinstance(fixture_reference, str)
    assert (
        campaign_path.parent / authority_reference
    ).resolve() == authority_path.resolve()
    assert (campaign_path.parent / fixture_reference).resolve() == (
        QUALIFICATION_ROOT / "qualification-index.json"
    ).resolve()
    batch_assignments = [
        assignment["recipe"]
        for batch in authority["batches"]
        for assignment in batch["assignments"]
    ]
    assert len(authority["batches"]) == 45
    assert sum(batch["mode"] == "paired-single" for batch in authority["batches"]) == 36
    assert sum(batch["mode"] == "exclusive-dual" for batch in authority["batches"]) == 9
    assert len(batch_assignments) == len(set(batch_assignments)) == 81
    assert set(batch_assignments) == expected_in_scope
    assert len(expected_in_scope) + len(expected_excluded) == 85
    assert (
        sum(definition["shared"] for definition in authority["recovery_coverage"]) == 0
    )
    assert not (
        QUALIFICATION_ROOT / "authorities/nl-single-spark-7173cb48.json"
    ).exists()
    assert not (QUALIFICATION_ROOT / "campaigns/nl-single-spark-7173cb48.json").exists()
    assert not (QUALIFICATION_ROOT / "authorities/nl-sequential-2c118a99.json").exists()
    assert not (QUALIFICATION_ROOT / "campaigns/nl-sequential-2c118a99.json").exists()


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
