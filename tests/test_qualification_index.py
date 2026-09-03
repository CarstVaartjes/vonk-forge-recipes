from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import sys
from pathlib import Path

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
    fixture = definitions["fixtures"]["generic-mesh-glb"]
    encoded = (QUALIFICATION_ROOT / fixture["path"]).read_bytes()
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


def test_campaign_authority_and_partition_are_recipe_owned() -> None:
    authority_path = QUALIFICATION_ROOT / "authorities/nl-single-spark-7173cb48.json"
    campaign_path = QUALIFICATION_ROOT / "campaigns/nl-single-spark-7173cb48.json"
    authority = _document(authority_path)
    campaign = _document(campaign_path)

    assert (
        authority["catalog"]["catalog_index_sha256"]
        == hashlib.sha256((ROOT / "catalog-index.json").read_bytes()).hexdigest()
    )
    authority_keys = set(authority["actionable_recipe_keys"])
    lane_keys = [recipe for lane in campaign["lanes"] for recipe in lane["recipes"]]
    assert len(lane_keys) == len(set(lane_keys))
    assert set(lane_keys) == authority_keys
    assert (
        campaign_path.parent / campaign["qualification_authority"]
    ).resolve() == authority_path.resolve()
    assert (campaign_path.parent / campaign["fixture_manifest"]).resolve() == (
        QUALIFICATION_ROOT / "qualification-index.json"
    ).resolve()
