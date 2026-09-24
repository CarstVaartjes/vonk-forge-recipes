"""Guards for the derived family-aware coverage matrix.

Each guard names the wrong implementation it rejects: a hand-edited or stale
matrix, a dropped or duplicated recipe, a schedule that loses a lane, a vendor
fork image graded as generic engine evidence, shared recovery evidence promoted
into an enforceable claim, and a representative that is not the cheapest member
of its own risk group.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from vonk_forge_contracts import RecipeDefinition

from qualification.coverage_identity import execution_stack_identity

ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = ROOT / "qualification/coverage/family-aware-coverage-2026-09-24.json"
REPORT_PATH = ROOT / "qualification/coverage/family-aware-coverage-2026-09-24.md"
TOOL = "tools/build-family-aware-coverage"

# The stock upstream image per engine. Any other image is a declared fork.
STOCK_IMAGES = {
    "vllm": "docker.io/vllm/vllm-openai",
    "sglang": "docker.io/lmsysorg/sglang",
}


def _matrix() -> dict[str, Any]:
    value = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _rows() -> dict[str, dict[str, Any]]:
    return {row["key"]: row for row in _matrix()["rows"]}


def _catalog() -> dict[str, RecipeDefinition]:
    recipes: dict[str, RecipeDefinition] = {}
    for path in sorted((ROOT / "recipes").glob("*.json")):
        recipe = RecipeDefinition.model_validate_json(path.read_text(encoding="utf-8"))
        recipes[f"{recipe.identity.publisher}/{recipe.identity.slug}"] = recipe
    return recipes


def test_committed_matrix_matches_the_derivation() -> None:
    """A stale or hand-edited matrix must fail instead of passing review."""
    result = subprocess.run(
        [sys.executable, TOOL, "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    assert REPORT_PATH.exists()


def test_every_catalog_recipe_appears_once_with_its_topology() -> None:
    """A dropped, duplicated or mislabelled recipe must fail."""
    matrix = _matrix()
    rows = _rows()
    catalog = _catalog()
    assert set(rows) == set(catalog)
    assert len(rows) == 85
    assert matrix["catalog"]["recipe_count"] == len(catalog)
    assert matrix["catalog"]["in_scope_recipe_count"] == 81
    assert matrix["catalog"]["out_of_scope_recipe_count"] == 4
    groups = set(matrix["coverage_group_order"])
    assert sorted(row["matrix_row"] for row in rows.values()) == list(range(1, 86))
    for key, recipe in catalog.items():
        row = rows[key]
        assert row["node_count"] == recipe.topology.node_count
        assert row["in_scope"] is (recipe.topology.node_count <= 2)
        assert row["coverage_group"] in groups
        assert len(row["runtime_stack_sha256"]) == 64
        assert len(row["topology_sha256"]) == 64
        assert len(row["authored_tree_stack_sha256"]) == 64
        assert len(row["published_build_source_sha256"]) == 64
        assert row["published_build_source_file_count"] > 0
        assert isinstance(row["authored_tree_matches_published"], bool)
    assert sorted(row["node_count"] for row in rows.values()) == (
        [1] * 72 + [2] * 9 + [3, 4, 4, 8]
    )


def test_schedule_covers_every_in_scope_recipe_exactly_once() -> None:
    """A batch plan that drops, duplicates or mis-sizes a lane must fail."""
    matrix = _matrix()
    rows = _rows()
    assignments = [
        lane["recipe"] for batch in matrix["batches"] for lane in batch["assignments"]
    ]
    lanes = [
        lane["recipe"]
        for batch in matrix["batches"]
        if batch["mode"] == "paired-single"
        for lane in batch["assignments"]
    ]
    singles = sorted(
        row["key"]
        for row in rows.values()
        if row["in_scope"] and row["node_count"] == 1
    )
    assert sorted(lanes) == singles
    assert len(assignments) == len(set(assignments)) == 81
    batches = matrix["batches"]
    assert len(batches) == 45
    assert sum(batch["mode"] == "paired-single" for batch in batches) == 36
    assert sum(batch["mode"] == "exclusive-dual" for batch in batches) == 9
    assert [batch["sequence"] for batch in batches] == list(range(1, 46))
    for batch in batches:
        if batch["mode"] == "paired-single":
            assert len(batch["assignments"]) == 2
            assert {lane["lane"] for lane in batch["assignments"]} == {1, 2}
            assert {lane["node_count"] for lane in batch["assignments"]} == {1}
        elif batch["mode"] == "exclusive-dual":
            assert len(batch["assignments"]) == 1
            assert batch["assignments"][0]["node_count"] == 2
        else:
            raise AssertionError(f"unexpected current batch mode: {batch['mode']}")
    duals = sorted(
        lane["recipe"]
        for batch in batches
        if batch["mode"] == "exclusive-dual"
        for lane in batch["assignments"]
    )
    assert duals == sorted(
        row["key"] for row in rows.values() if row["in_scope"] and row["node_count"] > 1
    )
    outside = sorted(entry["recipe"] for entry in matrix["excluded_schedule"])
    assert outside == sorted(row["key"] for row in rows.values() if not row["in_scope"])


def test_declared_fork_images_are_never_graded_as_generic_engines() -> None:
    """A vendor fork image graded as generic engine evidence must fail."""
    for row in _rows().values():
        repository = row["stack"]["base_image"]["repository"]
        stock = STOCK_IMAGES.get(row["engine"])
        if stock is None or repository == stock:
            continue
        assert f"image:{repository}" in row["specialized_forks"]
        assert row["coverage_group"] == "specialized-fork-engines"


def test_recovery_coverage_is_typed_and_exactly_referenced() -> None:
    """A recovery ref outside its exact member and failure mode must fail."""
    matrix = _matrix()
    rows = _rows()
    definitions = matrix["recovery_coverage"]["definitions"]
    assert definitions
    definitions_by_id = {item["coverage_id"]: item for item in definitions}
    assert len(definitions_by_id) == len(definitions)
    refs_by_recipe: dict[str, list[dict[str, Any]]] = {
        key: row["recovery_coverage_refs"]
        for key, row in rows.items()
        if row["in_scope"]
    }
    for key, row in rows.items():
        if not row["in_scope"]:
            assert row["recovery_coverage_refs"] == []
            continue
        required_modes = (
            {"single-host-restart"}
            if row["node_count"] == 1
            else {"dual-rank-loss-recovery", "dual-host-restart"}
        )
        refs = refs_by_recipe[key]
        assert {ref["failure_mode"] for ref in refs} == required_modes
        for ref in refs:
            definition = definitions_by_id[ref["coverage_id"]]
            assert definition["failure_mode"] == ref["failure_mode"]
            assert key in {member["recipe"] for member in definition["members"]}
            assert ref["representative_recipe"] == definition["representative_recipe"]
    for definition in definitions:
        member_keys = [member["recipe"] for member in definition["members"]]
        assert member_keys == sorted(set(member_keys))
        assert definition["representative_recipe"] in member_keys
        assert definition["shared"] is (len(member_keys) > 1)
        if definition["failure_mode"] == "dual-rank-loss-recovery":
            assert definition["shared"] is False
        identity_pairs = {
            (member["runtime_stack_sha256"], member["topology_sha256"])
            for member in definition["members"]
        }
        if definition["shared"]:
            assert len(identity_pairs) == 1
            assert definition["physical_receipts_required"] is True


def test_build_source_bytes_at_the_same_path_change_stack_identity(
    tmp_path: Path,
) -> None:
    """A changed build input must invalidate the exact execution-stack identity."""
    context = tmp_path / "adapters" / "fixture"
    context.mkdir(parents=True)
    source_file = context / "Dockerfile"
    source_file.write_bytes(b"FROM example@sha256:one\n")

    def model_dump(value: dict[str, Any]):
        return lambda **_: value

    recipe = SimpleNamespace(
        runtime=SimpleNamespace(
            model_dump=model_dump({"engine": "test"}),
            lifecycle=SimpleNamespace(failure=None),
        ),
        topology=SimpleNamespace(model_dump=model_dump({"mode": "single"})),
    )
    build = SimpleNamespace(
        context=SimpleNamespace(path="adapters/fixture"),
        dockerfile="adapters/fixture/Dockerfile",
        patches=(),
        model_dump=model_dump(
            {
                "context": {"path": "adapters/fixture"},
                "dockerfile": "adapters/fixture/Dockerfile",
                "patches": [],
            }
        ),
    )

    before = execution_stack_identity(tmp_path, recipe, build)
    source_file.write_bytes(b"FROM example@sha256:two\n")
    after = execution_stack_identity(tmp_path, recipe, build)

    assert before["build_source_sha256"] != after["build_source_sha256"]
    assert before["runtime_stack_sha256"] != after["runtime_stack_sha256"]


def test_representatives_are_the_cheapest_member_of_their_risk_group() -> None:
    """A representative that is not the cheapest member of its own group must fail."""
    matrix = _matrix()
    rows = _rows()
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows.values():
        if row["in_scope"]:
            grouped.setdefault(row["risk_group_id"], []).append(row)
    assert len(matrix["representatives"]) == len(grouped)
    for entry in matrix["representatives"]:
        members = grouped[entry["risk_group_id"]]
        marked = sorted(row["key"] for row in members if row["representative"])
        assert marked == [entry["representative"]]
        cheapest = min(
            members, key=lambda row: (row["declared_artifact_bytes"], row["key"])
        )
        assert cheapest["key"] == entry["representative"]
        for row in members:
            if row["key"] != entry["representative"]:
                assert row["representative_for"] == entry["representative"]
