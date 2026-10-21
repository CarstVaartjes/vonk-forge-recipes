"""Exercise actual catalog rejection and bounded KV without retuning known recipes."""

import json
import runpy
from pathlib import Path

import pytest
from catalog_documents import catalog_models
from vonk_forge_contracts import RecipeDefinition, document_sha256
from vonk_forge_contracts.recipe import RecipeRuntimeArgument

from qualification.memory_need import (
    GUARD_BYTES,
    IDLE_USE_BYTES,
    LIMIT_BYTES,
    TOTAL_BYTES,
    CacheGeometry,
    argument_value,
    cache_bytes,
    calculate_need,
    load_inputs,
    require_memory_need,
)

ROOT = Path(__file__).resolve().parents[1]
LLAMA = "llama-3-1-8b-instruct-nvfp4-nvidia-vllm-single"
INPUTS = load_inputs()


def llama() -> RecipeDefinition:
    return RecipeDefinition.model_validate_json(
        (ROOT / "recipes" / f"{LLAMA}.json").read_text()
    )


def checked(recipe: RecipeDefinition) -> None:
    require_memory_need(
        recipe,
        catalog_models(),
        INPUTS,
        document_digest=document_sha256(recipe.model_dump(mode="json")),
    )


def test_llama_need_is_small_and_the_pool_matches_the_declared_workload() -> None:
    recipe = llama()
    need = calculate_need(recipe, catalog_models(), INPUTS.geometries)
    assert need.kv_bytes == 32 * 8 * (128 + 128) * 131_072 == 8 * 1024**3
    assert need.total_bytes == 22_617_795_968
    assert need.total_bytes < need.limit_bytes == 108_470_788_096
    assert argument_value(recipe, "kv-cache-memory-bytes") == need.kv_bytes
    assert argument_value(recipe, "gpu-memory-utilization") is None
    checked(recipe)


def test_limit_subtracts_idle_use_and_guard_exactly_once() -> None:
    assert IDLE_USE_BYTES == 5_012_574_208
    assert GUARD_BYTES == 17_179_869_184
    assert TOTAL_BYTES - IDLE_USE_BYTES - GUARD_BYTES == LIMIT_BYTES


def test_every_hardware_pass_remains_accepted() -> None:
    assert len(INPUTS.hardware_passes) == 180
    assert (
        sum(
            p.recorded_at < "2026-10-21T01:33:00Z"
            for p in INPUTS.hardware_passes.values()
        )
        == 174
    )
    for key in INPUTS.hardware_passes:
        path = ROOT / "recipes" / f"{key.split('/')[1]}.json"
        document = json.loads(path.read_text())
        recipe = RecipeDefinition.model_validate(document)
        require_memory_need(
            recipe, catalog_models(), INPUTS, document_digest=document_sha256(document)
        )


def test_proven_identity_is_not_an_exemption_for_a_mutated_recipe() -> None:
    recipe = llama()
    recipe.identity.slug = "gpt-oss-20b-nvidia-vllm-single"
    assert recipe.settings.kind == "generation"
    recipe.settings.context_tokens.value *= 100
    with pytest.raises(
        ValueError,
        match="reduce declared context/concurrency.*smaller quantization.*multi-Spark recipe id",
    ):
        checked(recipe)


@pytest.mark.parametrize(
    "name,value",
    [
        ("kv-cache-memory-bytes", "120G"),
        ("kv-cache-memory-bytes", "1G"),
        ("max-num-seqs", 256),
    ],
)
def test_guard_rejects_pool_or_scheduler_drift(name: str, value: str | int) -> None:
    recipe = llama()
    argument = next(item for item in recipe.runtime.arguments if item.name == name)
    argument.setting = None
    argument.value = value
    with pytest.raises(ValueError, match="absolute kv-cache|memory need"):
        checked(recipe)


def test_fraction_is_not_a_substitute_for_an_absolute_pool() -> None:
    recipe = llama()
    recipe.runtime.arguments = [
        item
        for item in recipe.runtime.arguments
        if item.name != "kv-cache-memory-bytes"
    ]
    with pytest.raises(ValueError, match="absolute kv-cache-memory-bytes"):
        checked(recipe)


def test_failed_recipe_cannot_reuse_its_historical_pass_to_bypass_need() -> None:
    recipe = llama()
    recipe.runtime.arguments = [
        item
        for item in recipe.runtime.arguments
        if item.name != "kv-cache-memory-bytes"
    ]
    key = f"{recipe.identity.publisher}/{recipe.identity.slug}"
    with pytest.raises(ValueError, match="absolute kv-cache-memory-bytes"):
        require_memory_need(
            recipe,
            catalog_models(),
            INPUTS,
            document_digest=INPUTS.hardware_passes[key].content_sha256,
        )


def test_draft_weights_and_cache_are_counted_not_a_flat_allowance() -> None:
    recipe = llama()
    draft = recipe.models[0].model_copy(deep=True)
    draft.id = "draft"
    recipe.models.append(draft)
    need = calculate_need(recipe, catalog_models(), INPUTS.geometries)
    assert need.auxiliary_weights_bytes == need.weights_bytes
    assert need.kv_bytes == 16 * 1024**3
    assert need.overhead_bytes == 8_000_000_000


def test_need_scales_with_context_and_concurrency_and_kv_dtype() -> None:
    recipe = llama()
    assert recipe.settings.kind == "generation"
    assert recipe.settings.concurrency is not None
    initial = calculate_need(recipe, catalog_models(), INPUTS.geometries)
    recipe.settings.context_tokens.value //= 2
    recipe.settings.concurrency.value *= 4
    assert (
        calculate_need(recipe, catalog_models(), INPUTS.geometries).kv_bytes
        == initial.kv_bytes * 2
    )
    next(
        item for item in recipe.runtime.arguments if item.name == "kv-cache-dtype"
    ).value = "float16"
    assert (
        calculate_need(recipe, catalog_models(), INPUTS.geometries).kv_bytes
        == initial.kv_bytes * 4
    )


def test_unknown_new_model_requires_source_backed_geometry() -> None:
    with pytest.raises(ValueError, match="source-backed cache geometry"):
        calculate_need(llama(), catalog_models(), {})


def test_sglang_requires_an_absolute_token_pool() -> None:
    recipe = llama()
    recipe.runtime.engine = "sglang"
    next(
        item for item in recipe.runtime.arguments if item.name == "max-model-len"
    ).name = "context-length"
    next(
        item for item in recipe.runtime.arguments if item.name == "max-num-seqs"
    ).name = "max-running-requests"
    with pytest.raises(ValueError, match="absolute max-total-tokens=131072"):
        checked(recipe)
    next(
        item
        for item in recipe.runtime.arguments
        if item.name == "kv-cache-memory-bytes"
    ).name = "max-total-tokens"
    next(
        item for item in recipe.runtime.arguments if item.name == "max-total-tokens"
    ).value = 131072
    checked(recipe)
    recipe.runtime.arguments.append(
        RecipeRuntimeArgument(name="mem-fraction-static", value="0.01")
    )
    with pytest.raises(ValueError, match="static memory budget cannot hold weights"):
        checked(recipe)
    recipe.runtime.arguments[-1].value = "0.90"
    checked(recipe)


def test_sliding_layers_use_their_window_and_recurrent_state_uses_concurrency() -> None:
    recipe = llama()
    slug = recipe.models[0].model.slug
    geometry = INPUTS.geometries[slug].model_copy(
        update={
            "sliding_layers": 16,
            "sliding_window": 1024,
            "state_bytes_per_request": 100,
        }
    )
    assert isinstance(geometry, CacheGeometry)
    need = calculate_need(recipe, catalog_models(), {slug: geometry})
    assert need.kv_bytes == (131072 * 16 + 1024 * 16) * 8 * 256
    assert need.state_bytes == 100


@pytest.mark.parametrize(
    "value,expected",
    [("8G", 8 * 1024**3), ("8g", 8_000_000_000), (8589934592, 8589934592)],
)
def test_absolute_cache_units(value: str | int, expected: int) -> None:
    assert cache_bytes(value) == expected


@pytest.mark.parametrize("value", ["0", "-1", "NaN", "0.5"])
def test_invalid_absolute_cache_bytes(value: str) -> None:
    with pytest.raises(ValueError):
        cache_bytes(value)


def test_real_catalog_build_rejects_an_over_limit_new_recipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool = runpy.run_path(str(ROOT / "tools/build-catalog-index"))
    build = tool["build"]
    # Use the actual publisher entry point and the original pinned Model. Only
    # the synthetic Recipe is new, so this fails on the old unguarded builder.
    recipe = llama()
    recipe.identity.slug = "synthetic-over-limit-new"
    assert recipe.settings.kind == "generation"
    recipe.settings.context_tokens.value *= 100
    (tmp_path / "recipes").mkdir()
    (tmp_path / "recipes/synthetic-over-limit-new.json").write_text(
        recipe.model_dump_json()
    )
    (tmp_path / "models").symlink_to(ROOT / "models", target_is_directory=True)
    monkeypatch.setitem(build.__globals__, "ROOT", tmp_path)

    def forbidden_package(*args: object, **kwargs: object) -> None:
        pytest.fail("publisher reached packaging without rejecting unsafe memory need")

    monkeypatch.setitem(build.__globals__, "recipe_package", forbidden_package)
    with pytest.raises(
        SystemExit,
        match=r"memory need \d+ bytes > limit 108470788096 bytes.*reduce declared context/concurrency",
    ):
        build(package_dir=tmp_path / "packages", source_commit="0" * 40)
