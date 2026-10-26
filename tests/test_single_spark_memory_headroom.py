"""Exercise actual catalog rejection and bounded KV without retuning known recipes."""

import json
import runpy
from pathlib import Path

import pytest
from catalog_documents import catalog_models
from vonk_forge_contracts import RecipeDefinition, document_sha256
from vonk_forge_contracts.recipe import RecipeRuntimeArgument

from qualification.memory_need import (
    DRAFT_MODEL_OVERHEAD_BYTES,
    GUARD_BYTES,
    IDLE_USE_BYTES,
    LIMIT_BYTES,
    OVERHEAD_BYTES,
    TOTAL_BYTES,
    CacheGeometry,
    argument_value,
    cache_bytes,
    calculate_need,
    hardware_proven,
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
    # Identity belongs to the authored JSON, not defaults added by parsing it.
    # The guard separately verifies that this parsed recipe still matches it.
    document = json.loads(
        (ROOT / "recipes" / f"{recipe.identity.slug}.json").read_text()
    )
    require_memory_need(
        recipe,
        catalog_models(),
        INPUTS,
        document_digest=document_sha256(document),
    )


def test_llama_need_is_small_and_the_pool_matches_the_declared_workload() -> None:
    recipe = llama()
    need = calculate_need(recipe, catalog_models(), INPUTS)
    assert need.kv_bytes == 32 * 8 * (128 + 128) * 131_072 == 8 * 1024**3
    assert need.total_bytes == 22_617_795_968
    assert need.total_bytes < need.limit_bytes == 108_470_788_096
    assert argument_value(recipe, "kv-cache-memory-bytes") == need.kv_bytes
    assert argument_value(recipe, "gpu-memory-utilization") is None
    checked(recipe)


def test_limit_subtracts_idle_use_and_guard_exactly_once() -> None:
    assert IDLE_USE_BYTES == 5_012_574_208
    assert GUARD_BYTES > IDLE_USE_BYTES
    assert TOTAL_BYTES - IDLE_USE_BYTES - GUARD_BYTES == LIMIT_BYTES


def test_hardware_pass_status_comes_from_qualification_evidence() -> None:
    document = json.loads(
        (ROOT / "recipes/gpt-oss-20b-nvidia-vllm-single.json").read_text()
    )
    recipe = RecipeDefinition.model_validate(document)
    assert hardware_proven(recipe, document_sha256(document))
    assert not hardware_proven(recipe, "0" * 64)


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
    with pytest.raises(ValueError, match="absolute kv-cache-memory-bytes"):
        require_memory_need(
            recipe,
            catalog_models(),
            INPUTS,
            document_digest="stale-passed-digest",
        )


def test_auxiliary_weights_and_cache_are_counted_without_speculation() -> None:
    recipe = llama()
    draft = recipe.models[0].model_copy(deep=True)
    draft.id = "draft"
    recipe.models.append(draft)
    need = calculate_need(recipe, catalog_models(), INPUTS)
    assert need.auxiliary_weights_bytes == need.weights_bytes
    assert need.kv_bytes == 16 * 1024**3
    assert need.overhead_bytes == OVERHEAD_BYTES


def test_need_scales_with_context_and_concurrency_and_kv_dtype() -> None:
    recipe = llama()
    assert recipe.settings.kind == "generation"
    assert recipe.settings.concurrency is not None
    initial = calculate_need(recipe, catalog_models(), INPUTS)
    recipe.settings.context_tokens.value //= 2
    recipe.settings.concurrency.value *= 4
    assert (
        calculate_need(recipe, catalog_models(), INPUTS).kv_bytes
        == initial.kv_bytes * 2
    )
    next(
        item for item in recipe.runtime.arguments if item.name == "kv-cache-dtype"
    ).value = "float16"
    assert (
        calculate_need(recipe, catalog_models(), INPUTS).kv_bytes
        == initial.kv_bytes * 4
    )


def test_unknown_new_model_requires_source_backed_geometry() -> None:
    with pytest.raises(ValueError, match="source-backed cache geometry"):
        calculate_need(
            llama(), catalog_models(), INPUTS.model_copy(update={"geometries": {}})
        )


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
    need = calculate_need(
        recipe,
        catalog_models(),
        INPUTS.model_copy(update={"geometries": {slug: geometry}}),
    )
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


@pytest.mark.parametrize(
    "config,separate",
    [
        ({"method": "mtp", "num_speculative_tokens": 2}, False),
        ({"method": "deepseek_mtp", "model": "/models"}, False),
        ({"method": "dflash", "model": "/models/draft"}, True),
    ],
)
def test_need_keeps_measured_overhead_only_for_a_separate_draft(
    config: dict[str, str | int], separate: bool
) -> None:
    recipe = llama()
    draft = recipe.models[0].model_copy(deep=True)
    draft.id = "draft"
    recipe.models.append(draft)
    recipe.runtime.arguments.append(
        RecipeRuntimeArgument(name="speculative-config", value=json.dumps(config))
    )
    need = calculate_need(recipe, catalog_models(), INPUTS)
    assert need.auxiliary_weights_bytes == need.weights_bytes
    assert need.overhead_bytes == OVERHEAD_BYTES + (
        DRAFT_MODEL_OVERHEAD_BYTES if separate else 0
    )


def test_need_rejects_a_pool_that_only_fits_without_measured_draft_overhead() -> None:
    recipe = llama()
    draft = recipe.models[0].model_copy(deep=True)
    draft.id = "draft"
    recipe.models.append(draft)
    assert recipe.settings.kind == "generation"
    assert recipe.settings.concurrency is not None
    recipe.settings.concurrency.value = 5
    recipe.runtime.arguments.append(
        RecipeRuntimeArgument(
            name="speculative-config",
            value='{"method":"dflash","model":"/models/draft"}',
        )
    )
    need = calculate_need(recipe, catalog_models(), INPUTS)
    assert need.total_bytes - DRAFT_MODEL_OVERHEAD_BYTES <= need.limit_bytes
    assert need.total_bytes > need.limit_bytes
    next(
        item
        for item in recipe.runtime.arguments
        if item.name == "kv-cache-memory-bytes"
    ).value = need.kv_bytes
    with pytest.raises(ValueError, match="memory need"):
        checked(recipe)


@pytest.mark.parametrize(
    "slug",
    [
        "deepseek-v4-flash-vision-exp-eugr-vllm-dual",
        "glm-5-2-aqlm-vllm-triple",
        "step-5-preview-exl3-vllm-four",
        "muse-spark-open-nvfp4-vllm-eight",
    ],
)
def test_multispark_delegates_every_role_even_for_historical_identity(
    slug: str,
) -> None:
    document = json.loads((ROOT / "recipes" / f"{slug}.json").read_text())
    recipe = RecipeDefinition.model_validate(document)
    digest = document_sha256(document)
    assert (
        require_memory_need(recipe, catalog_models(), INPUTS, document_digest=digest)
        is None
    )
    worker = recipe.topology.roles[-1]
    worker.resources.memory.peak_bytes = 0
    with pytest.raises(ValueError, match=f"{slug}:.*{worker.name}:.*understates"):
        require_memory_need(recipe, catalog_models(), INPUTS, document_digest=digest)


def test_glm_hybrid_cache_profiles_without_autosizing_the_physical_pools() -> None:
    import logging
    import math
    import runpy
    from types import SimpleNamespace

    recipe = RecipeDefinition.model_validate_json(
        (
            ROOT / "recipes/glm-5-3-flash-exl3-a0-2-05bpw-0xsero-sglang-single.json"
        ).read_text()
    )
    args = {item.name: item.value for item in recipe.runtime.arguments}
    schedule = SimpleNamespace(
        mem_fraction_static=float(str(args["mem-fraction-static"])),
        max_total_tokens=args.get("max-total-tokens"),
        max_mamba_cache_size=args.get("max-mamba-cache-size"),
        max_running_requests=1,
        mamba_full_memory_ratio=1.0,
    )

    def override(_reason: str, **values: int) -> None:
        for name, value in values.items():
            setattr(schedule, name, value)

    namespace = runpy.run_path(
        str(ROOT / "tests/fixtures/glm-a0-cache-sizing.py.txt"),
        init_globals={
            "math": math,
            "logging": logging,
            "get_schedule": lambda: schedule,
            "get_available_gpu_memory": lambda *a, **kw: 30 - 6.32,
            "get_world_group": lambda: SimpleNamespace(world_size=1, cpu_group=None),
            "get_mm": lambda: SimpleNamespace(mm_feature_transport=None),
            "get_memory": lambda: SimpleNamespace(disable_radix_cache=False),
            "mm_runtime_reservation_gb": lambda **kw: 0,
            "get_exec": lambda: SimpleNamespace(
                mamba=SimpleNamespace(enable_linear_replayssm_spec=False)
            ),
            "get_context": lambda: SimpleNamespace(override=override),
            "get_parallel": lambda: SimpleNamespace(config=SimpleNamespace(pp_size=1)),
        },
    )
    sizer = namespace["CacheSizer"]()
    sizer.device, sizer.gpu_id = "cuda", 0
    sizer.post_capture_kv_active = False
    sizer.model_config = SimpleNamespace(is_multimodal=True)
    sizer.ps = SimpleNamespace(pp_size=1, attn_dp_size=1)
    sizer.spec_algorithm = SimpleNamespace(is_none=lambda: True)
    sizer.mambaish_config = SimpleNamespace(
        mamba2_cache_params=SimpleNamespace(
            layers=list(range(34)), mamba_cache_per_req=int(140.78 * 1024**2)
        )
    )
    # Same -6.32 GiB residual as operation 3d2ed06e at 0.75. The exact
    # pre-load free memory was truncated; 120 GiB is a synthetic test input.
    budget = sizer._profile_available_bytes(120)
    assert budget > 0
    assert schedule.max_mamba_cache_size == 2
    assert sizer._apply_token_constraints(1_000_000) == 8192
    checked(recipe)
    need = calculate_need(recipe, catalog_models(), INPUTS)
    assert need.kv_bytes == 8192 * 65536
    assert need.state_bytes == 3 * 150 * 1024**2
    assert need.overhead_bytes == 8_000_000_000
    assert need.resident_expansion_bytes == 14_000_000_000
    assert need.total_bytes <= recipe.topology.roles[0].resources.memory.peak_bytes
    assert recipe.topology.roles[0].resources.memory.peak_bytes <= LIMIT_BYTES

    # Replay the rejected profile through the image's actual sizing functions.
    schedule.mem_fraction_static = 0.75
    schedule.max_mamba_cache_size = None
    with pytest.raises(RuntimeError, match="Not enough GPU memory for hybrid"):
        sizer._profile_available_bytes(120)


@pytest.mark.parametrize(
    "name,value", [("max-total-tokens", 262144), ("max-mamba-cache-size", 16)]
)
def test_capped_sglang_guard_rejects_restored_large_pools(
    name: str, value: int
) -> None:
    recipe = RecipeDefinition.model_validate_json(
        (
            ROOT / "recipes/glm-5-3-flash-exl3-a0-2-05bpw-0xsero-sglang-single.json"
        ).read_text()
    )
    next(item for item in recipe.runtime.arguments if item.name == name).value = value
    with pytest.raises(ValueError):
        checked(recipe)


@pytest.mark.parametrize("defect", ["graphs", "image", "state-cap", "peak", "fraction"])
def test_hybrid_need_rejects_uncalibrated_runtime_or_hidden_demand(defect: str) -> None:
    recipe = RecipeDefinition.model_validate_json(
        (
            ROOT / "recipes/glm-5-3-flash-exl3-a0-2-05bpw-0xsero-sglang-single.json"
        ).read_text()
    )
    if defect == "graphs":
        next(
            a for a in recipe.runtime.arguments if a.name == "disable-cuda-graph"
        ).value = False
    elif defect == "image":
        recipe.execution.build.base_image.digest = "0" * 64
    elif defect == "fraction":
        next(
            a for a in recipe.runtime.arguments if a.name == "mem-fraction-static"
        ).value = "0.75"
    elif defect == "state-cap":
        recipe.runtime.arguments = [
            a for a in recipe.runtime.arguments if a.name != "max-mamba-cache-size"
        ]
    else:
        need = calculate_need(recipe, catalog_models(), INPUTS)
        recipe.topology.roles[0].resources.memory.peak_bytes = need.total_bytes - 1
    with pytest.raises((ValueError, TypeError)):
        checked(recipe)
