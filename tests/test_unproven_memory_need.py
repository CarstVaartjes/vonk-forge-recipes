"""Historical catalog presence is not hardware proof of a memory reservation."""

import json
import runpy
from pathlib import Path

import pytest
from catalog_documents import catalog_models
from vonk_forge_contracts import RecipeDefinition, document_sha256

from qualification.memory_need import (
    GUARD_BYTES,
    calculate_need,
    load_inputs,
    require_memory_need,
)
from spark_sweep.removal_contract import RecipeVerdict

ROOT = Path(__file__).resolve().parents[1]
MOSAIC = "glm-5-3-flash-exl3-mosaic-12l-0xsero-sglang-single"


def test_unchanged_never_passed_mosaic_cannot_bypass_the_catalog_guard() -> None:
    recipe = RecipeDefinition.model_validate_json(
        (ROOT / "recipes" / f"{MOSAIC}.json").read_text()
    )
    recipe.runtime.arguments = [
        arg for arg in recipe.runtime.arguments if arg.name != "max-total-tokens"
    ]
    # Original on-main identity: the old historical-document waiver accepted it
    # before ever calculating need or checking an absolute pool.
    with pytest.raises((ValueError, TypeError)):
        require_memory_need(
            recipe,
            catalog_models(),
            load_inputs(),
            document_digest="6e5f713f3487415a3e3081e186ac9afd07d4346a7f0a43e3fe1a0f4ad474f653",
        )


def test_historical_multi_spark_reserve_cannot_bypass_host_headroom() -> None:
    slug = "deepseek-v4-flash-vision-exp-eugr-vllm-dual"
    recipe = RecipeDefinition.model_validate_json(
        (ROOT / "recipes" / f"{slug}.json").read_text()
    )
    # The old utilization guard allowed this peak and ignored the reserve.
    for role in recipe.topology.roles:
        role.resources.memory.peak_bytes = 120_000_000_000
        role.resources.memory.reserve_bytes = GUARD_BYTES - 1
    with pytest.raises(ValueError):
        require_memory_need(
            recipe,
            catalog_models(),
            load_inputs(),
            document_digest="9866ab75030511fe8263ddc79ccbbc04cdbc771e4c98065a523707e865ef2f06",
        )


@pytest.mark.parametrize(
    "path", sorted((ROOT / "recipes").glob("*.json")), ids=lambda p: p.stem
)
def test_every_catalog_recipe_satisfies_memory_policy(path: Path) -> None:
    document = json.loads(path.read_text())
    require_memory_need(
        RecipeDefinition.model_validate(document),
        catalog_models(),
        load_inputs(),
        document_digest=document_sha256(document),
    )


def test_mosaic_capped_need_is_honestly_nonfitting() -> None:
    recipe = RecipeDefinition.model_validate_json(
        (ROOT / "recipes" / f"{MOSAIC}.json").read_text()
    )
    need = calculate_need(recipe, catalog_models(), load_inputs().geometries)
    assert need.total_bytes == 119_029_287_744 > need.limit_bytes
    assert "does-not-fit" in recipe.metadata.tags
    assert all(
        role.resources.memory.peak_bytes >= need.total_bytes
        for role in recipe.topology.roles
    )
    recipe.metadata.tags.remove("does-not-fit")
    with pytest.raises(ValueError, match="memory need"):
        require_memory_need(recipe, catalog_models(), load_inputs(), document_digest="")


def test_pass_generator_uses_all_history_and_only_real_passes(tmp_path: Path) -> None:
    generator = runpy.run_path(str(ROOT / "tools/update-memory-hardware-passes"))[
        "generate"
    ]
    (tmp_path / "recipes").mkdir()
    document = {"identity": {"publisher": "vonk-forge", "slug": "example"}}
    (tmp_path / "recipes/example.json").write_text(json.dumps(document))
    passed = {
        "status": RecipeVerdict.PASSED,
        "recipe": "vonk-forge/example",
        "content_sha256": "a" * 64,
        "recorded_at": "2026-10-20T00:00:00Z",
        "library_commit": "b" * 40,
        "recipe_revision_id": "revision",
    }
    newer = {**passed, "recorded_at": "2026-10-21T00:00:00Z"}
    history = tmp_path / "results.jsonl"
    rotated = tmp_path / "results.jsonl.rotated"
    history.write_text(
        json.dumps(newer)
        + "\n"
        + json.dumps({**passed, "status": RecipeVerdict.FAILED})
        + "\n"
    )
    rotated.write_text(json.dumps(passed) + "\n")
    result = generator([rotated, history], root=tmp_path)
    assert set(result) == {"vonk-forge/example"}
    record = result["vonk-forge/example"]
    assert record.model_dump(mode="json")["recorded_at"] == newer["recorded_at"]
    assert record.content_sha256 == passed["content_sha256"]
    assert record.catalog_content_sha256 == document_sha256(document)
    history.write_text(json.dumps({**passed, "status": RecipeVerdict.DEFERRED}) + "\n")
    assert generator([history], root=tmp_path) == {}


def test_per_rank_cache_rounds_up_heads_and_pipeline_layers() -> None:
    recipe = RecipeDefinition.model_validate_json(
        (
            ROOT / "recipes/llama-3-1-8b-instruct-nvfp4-nvidia-vllm-single.json"
        ).read_text()
    )
    recipe.topology.node_count = 15
    recipe.topology.parallelism.tensor = 3
    recipe.topology.parallelism.pipeline = 5
    need = calculate_need(recipe, catalog_models(), load_inputs().geometries)
    # 8 heads / TP3 -> 3; 32 layers / PP5 -> 7. Neither may round down.
    assert need.kv_bytes == 7 * 3 * 256 * 131072


def test_nonfit_label_does_not_waive_pool_or_peak_bounds() -> None:
    recipe = RecipeDefinition.model_validate_json(
        (ROOT / "recipes" / f"{MOSAIC}.json").read_text()
    )
    need = calculate_need(recipe, catalog_models(), load_inputs().geometries)
    recipe.topology.roles[0].resources.memory.peak_bytes = need.total_bytes - 1
    with pytest.raises(ValueError, match="peak understates"):
        require_memory_need(recipe, catalog_models(), load_inputs(), document_digest="")


def test_small_workload_cannot_declare_a_reserve_above_its_peak() -> None:
    recipe = RecipeDefinition.model_validate_json(
        (ROOT / "recipes/d3-edge-vllm-single.json").read_text()
    )
    recipe.topology.roles[0].resources.memory.reserve_bytes = GUARD_BYTES
    with pytest.raises(ValueError, match="reserve exceeds peak"):
        require_memory_need(recipe, catalog_models(), load_inputs(), document_digest="")
