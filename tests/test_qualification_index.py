from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
QUALIFICATION_ROOT = ROOT / "qualification"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


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
        recipe = _document(path)
        identity = recipe["identity"]
        topology = recipe["topology"]
        assert isinstance(identity, dict)
        assert isinstance(topology, dict)
        if topology["node_count"] <= 2:
            key = f"{identity['publisher']}/{identity['slug']}"
            expected[key] = hashlib.sha256(_canonical(recipe)).hexdigest()

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


def test_sequential_authority_binds_every_recipe_with_at_most_two_nodes() -> None:
    subprocess.run(
        [sys.executable, "tools/build-qualification-authority", "--check"],
        cwd=ROOT,
        check=True,
    )
    authority_path = QUALIFICATION_ROOT / "authorities/nl-sequential-2c118a99.json"
    campaign_path = QUALIFICATION_ROOT / "campaigns/nl-sequential-2c118a99.json"
    authority = cast(Any, _document(authority_path))
    campaign = cast(Any, _document(campaign_path))
    catalog_index = cast(Any, _document(ROOT / "catalog-index.json"))
    qualification = cast(
        Any, _document(QUALIFICATION_ROOT / "qualification-index.json")
    )
    review_gates = cast(Any, _document(QUALIFICATION_ROOT / "review-gates.json"))
    release = cast(Any, _document(QUALIFICATION_ROOT / "catalog-release.json"))

    assert authority["schema_version"] == 3
    assert campaign["schema_version"] == 2
    assert authority["catalog"]["repository"] == "CarstVaartjes/vonk-forge-recipes"
    assert (
        authority["catalog"]["catalog_index_sha256"]
        == hashlib.sha256((ROOT / "catalog-index.json").read_bytes()).hexdigest()
    )
    assert (
        authority["catalog"]["qualification_index_sha256"]
        == hashlib.sha256(
            (QUALIFICATION_ROOT / "qualification-index.json").read_bytes()
        ).hexdigest()
    )
    assert authority["catalog"]["source_commit"] == catalog_index["source_commit"]
    assert authority["catalog"]["recipe_count"] == len(catalog_index["recipes"]) == 85
    assert authority["scope"]["maximum_node_count"] == 2

    assert authority["catalog"]["commit"] == release["commit"]
    assert authority["catalog"]["release_tag"] == release["release_tag"] == "v1.0.8"

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
    assert "lanes" not in campaign
    assert "node_id" not in json.dumps(campaign)
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
    assert not (
        QUALIFICATION_ROOT / "authorities/nl-single-spark-7173cb48.json"
    ).exists()
    assert not (QUALIFICATION_ROOT / "campaigns/nl-single-spark-7173cb48.json").exists()
