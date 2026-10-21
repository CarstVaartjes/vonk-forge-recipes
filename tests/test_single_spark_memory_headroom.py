"""GB10 engine reservations must leave the host pressure guard room to recover."""

from decimal import Decimal
from pathlib import Path

import pytest
from catalog_documents import catalog_models
from test_vllm_memory_fit import (
    OVERHEAD_BYTES,
    ROOT,
    SPARK_3542_IDLE_AVAILABLE_BYTES,
    SPARK_MEMORY_BYTES,
    non_utilization_bytes,
)
from vonk_forge_contracts import RecipeDefinition

# vonk-forge 040c51b8531523bebe542b5e1f7b9403e0767f72:
# rust/crates/vonk-agent-protocol/src/host_memory_guard_policy.rs, re-exported
# by rust/crates/vonk-agent-helper/src/host_memory_guard.rs. Full PSI >30% for
# 3 seconds below 16 GiB triggers the guard (also with an NVIDIA allocation
# failure). The old 6 GB OS floor in test_vllm_memory_fit is insufficient.
STALL_AVAILABLE_BYTES = 16 * 1024**3
MAX_WORKLOAD_BYTES = SPARK_3542_IDLE_AVAILABLE_BYTES - STALL_AVAILABLE_BYTES


def cache_bytes(value: object) -> int:
    """vLLM human-readable integers: G is binary, lowercase g is decimal."""
    text = str(value)
    multipliers = {
        "k": 1000,
        "m": 1000**2,
        "g": 1000**3,
        "K": 1024,
        "M": 1024**2,
        "G": 1024**3,
    }
    if text[-1:] in multipliers:
        return int(Decimal(text[:-1]) * multipliers[text[-1]])
    return int(text)


def selected_weight_bytes(recipe: RecipeDefinition) -> int:
    models = catalog_models()
    weights: dict[str, int] = {}
    for selection in recipe.models:
        selected = {item.file_id for item in selection.files}
        for item in models[selection.model.content_sha256].files:
            if item.id in selected and "weights" in item.roles:
                weights[item.sha256] = item.size_bytes
    return sum(weights.values())


def assert_host_headroom(recipe: RecipeDefinition) -> None:
    if recipe.topology.node_count != 1 or recipe.runtime.engine not in {
        "vllm",
        "sglang",
    }:
        return
    name = (
        "gpu-memory-utilization"
        if recipe.runtime.engine == "vllm"
        else "mem-fraction-static"
    )
    arguments = [item for item in recipe.runtime.arguments if item.name == name]
    assert arguments, (
        f"{recipe.identity.slug}: set {name} explicitly; engine defaults change"
    )
    argument = arguments[-1]
    assert argument.setting is None, "Memory reservation must be a fixed budget"
    fraction = Decimal(str(argument.value))
    assert fraction.is_finite() and 0 < fraction <= 1
    # Absolute SGLang token/state caps bound the allocations independently of
    # the profiling fraction. The EXL3 bound includes resident expansion and is
    # tied to the exact checkpoint geometry; it is not a general fraction waiver.
    if (
        recipe.runtime.engine == "sglang"
        and any(item.name == "max-total-tokens" for item in recipe.runtime.arguments)
        and fraction * SPARK_MEMORY_BYTES + OVERHEAD_BYTES > MAX_WORKLOAD_BYTES
    ):
        assert_capped_exl3_headroom(recipe)
        return
    overhead = (
        non_utilization_bytes(recipe)
        if recipe.runtime.engine == "vllm"
        else OVERHEAD_BYTES
    )
    reserved = fraction * SPARK_MEMORY_BYTES + overhead
    assert reserved <= MAX_WORKLOAD_BYTES, (
        f"{recipe.identity.slug}: {name}={fraction} plus overhead reserves {reserved}; "
        f"maximum {MAX_WORKLOAD_BYTES} leaves the host guard's 16 GiB floor"
    )
    # Explicit vLLM KV bytes bypass gpu-memory-utilization. Count the selected
    # weights (including any drafter) instead of accepting a cosmetic fraction.
    for item in recipe.runtime.arguments:
        if item.name == "kv-cache-memory-bytes":
            assert item.setting is None
            reserved = (
                selected_weight_bytes(recipe) + cache_bytes(item.value) + OVERHEAD_BYTES
            )
            assert reserved <= MAX_WORKLOAD_BYTES, (
                f"{recipe.identity.slug}: explicit KV cache plus weights/overhead "
                f"reserves {reserved}, above {MAX_WORKLOAD_BYTES}"
            )


def assert_capped_exl3_headroom(recipe: RecipeDefinition) -> None:
    args = {item.name: item for item in recipe.runtime.arguments}
    assert len(recipe.models) == 1
    assert recipe.models[0].model.slug == "glm-5-3-flash-exl3-2-05bpw-51058cd5"
    assert recipe.models[0].model.content_sha256 == (
        "0f92e7cfb75689d7b51aef8072df572ce8360e8f8034793a88b0290178602f07"
    )
    assert recipe.execution.build.base_image.digest == (
        "85cb3fa86d31a781b94dcf10ee168adf096cfeaac14d2f1e6c560504e58e4eed"
    )
    assert args["disable-cuda-graph"].value is True
    assert args["max-running-requests"].setting == "concurrency"
    assert args["context-length"].setting == "context_tokens"
    assert args["kv-cache-dtype"].value == "fp8_e4m3"
    assert not any("speculative" in name for name in args)
    tokens = args["max-total-tokens"]
    states = args["max-mamba-cache-size"]
    assert tokens.setting is None and states.setting is None
    assert isinstance(tokens.value, int) and 0 < tokens.value <= 8192
    assert isinstance(states.value, int) and 0 < states.value <= 2
    assert recipe.settings.kind == "generation"
    assert recipe.settings.context_tokens is not None
    assert recipe.settings.context_tokens.value <= tokens.value
    assert recipe.settings.concurrency is not None
    assert recipe.settings.concurrency.value == 1
    # 45 layers * (512 MLA + 128 index) * 2 bytes < 64 KiB/token.
    # The trace prices all hybrid layers at 140.78 MiB/state; round up to
    # 150 MiB and count the padding state. 22 GB resident/runtime expansion is
    # an estimate above the selected weight bytes, not a measured peak.
    bound = (
        selected_weight_bytes(recipe)
        + 22_000_000_000
        + tokens.value * 65536
        + (states.value + 1) * 150 * 1024**2
    )
    peak = recipe.topology.roles[0].resources.memory.peak_bytes
    assert bound <= peak <= MAX_WORKLOAD_BYTES


@pytest.mark.parametrize(
    "path", sorted((ROOT / "recipes").glob("*.json")), ids=lambda path: path.stem
)
def test_single_spark_engine_reservation_leaves_host_headroom(path: Path) -> None:
    recipe = RecipeDefinition.model_validate_json(path.read_text())
    assert_host_headroom(recipe)
    for option in recipe.options:
        for choice in option.choices:
            assert_host_headroom(
                recipe.with_option_choices({option.name: choice.value})
            )


@pytest.mark.parametrize(
    "engine,name",
    [("vllm", "gpu-memory-utilization"), ("sglang", "mem-fraction-static")],
)
@pytest.mark.parametrize("value", ["0.80", "0.85", "0.90", "0.95", None])
def test_guard_rejects_unsafe_and_implicit_reservations(
    engine: str, name: str, value: str | None
) -> None:
    recipe = RecipeDefinition.model_validate_json(
        (
            ROOT / f"recipes/llama-3-1-8b-instruct-nvfp4-nvidia-{engine}-single.json"
        ).read_text()
    )
    if value is None:
        recipe.runtime.arguments = [
            item for item in recipe.runtime.arguments if item.name != name
        ]
    else:
        next(
            item for item in recipe.runtime.arguments if item.name == name
        ).value = value
    with pytest.raises(AssertionError):
        assert_host_headroom(recipe)


def test_guard_rejects_absolute_kv_cache_bypassing_fraction() -> None:
    recipe = RecipeDefinition.model_validate_json(
        (
            ROOT / "recipes/nemotron-3-5-lightning-dspark-lowmem-canary-single.json"
        ).read_text()
    )
    next(
        item
        for item in recipe.runtime.arguments
        if item.name == "kv-cache-memory-bytes"
    ).value = "120G"
    with pytest.raises(AssertionError, match="explicit KV cache"):
        assert_host_headroom(recipe)


def test_bound_counts_guard_floor_os_and_runtime_overhead() -> None:
    assert Decimal("0.75") * SPARK_MEMORY_BYTES + OVERHEAD_BYTES <= MAX_WORKLOAD_BYTES
    assert Decimal("0.80") * SPARK_MEMORY_BYTES + OVERHEAD_BYTES > MAX_WORKLOAD_BYTES


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
    assert_capped_exl3_headroom(recipe)

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
    with pytest.raises(AssertionError):
        assert_host_headroom(recipe)
