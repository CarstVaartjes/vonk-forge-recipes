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
from typing import Any

from vonk_forge_contracts import RecipeDefinition

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
    assert sorted(row["node_count"] for row in rows.values()) == (
        [1] * 72 + [2] * 9 + [3, 4, 4, 8]
    )


def test_schedule_covers_every_in_scope_recipe_exactly_once() -> None:
    """A scheduler that drops, duplicates or refills a lane must fail."""
    matrix = _matrix()
    rows = _rows()
    lanes = [
        lane["recipe"] for batch in matrix["paired_batches"] for lane in batch["lanes"]
    ]
    singles = sorted(
        row["key"]
        for row in rows.values()
        if row["in_scope"] and row["node_count"] == 1
    )
    assert sorted(lanes) == singles
    assert len(lanes) == len(set(lanes)) == 72
    for batch in matrix["paired_batches"]:
        assert len(batch["lanes"]) == 2
        assert {lane["lane"] for lane in batch["lanes"]} == {1, 2}
    duals = sorted(
        entry["recipe"]
        for entry in matrix["exclusive_schedule"]
        if entry["kind"] == "dual-spark"
    )
    assert duals == sorted(
        row["key"] for row in rows.values() if row["in_scope"] and row["node_count"] > 1
    )
    outside = sorted(
        entry["recipe"]
        for entry in matrix["exclusive_schedule"]
        if entry["kind"] == "outside-available-capacity"
    )
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


def test_shared_recovery_coverage_stays_a_proposal() -> None:
    """Promoting proposed shared recovery evidence into a claim must fail."""
    matrix = _matrix()
    rows = _rows()
    coverage = matrix["recovery_coverage"]
    assert coverage["enforceable"] is False
    assert coverage["signatures"]
    for entry in coverage["signatures"]:
        assert entry["enforceable"] is False
        assert entry["blocked_reason"]
        assert set(entry["recipes"]) <= set(rows)
        forked = {key for key in entry["recipes"] if rows[key]["specialized_forks"]}
        assert forked == set(entry["dedicated_required"])
        if entry["shared_coverage_candidate"]:
            assert len(entry["candidate_recipes"]) > 1
            assert set(entry["candidate_recipes"]) <= set(entry["recipes"])
        else:
            assert len(entry["candidate_recipes"]) < 2


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
