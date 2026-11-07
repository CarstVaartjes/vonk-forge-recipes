"""Exercise actual catalog rejection and bounded KV without retuning known recipes."""

import json
import runpy
from decimal import Decimal
from pathlib import Path

import pytest
from catalog_documents import catalog_models
from vonk_forge_contracts import RecipeDefinition, document_sha256
from vonk_forge_contracts.memory_admission import evaluate_memory_admission
from vonk_forge_contracts.recipe import RecipeRuntimeArgument
from warning_asserts import assert_not_warns, assert_warns, declare_peak

from qualification.memory_need import (
    ADMISSION_IDLE_USE_BYTES,
    GUARD_BYTES,
    IDLE_USE_BYTES,
    LIMIT_BYTES,
    OVERHEAD_BYTES,
    SGLANG_MAMBA_SLOTS_PER_REQUEST,
    SPARK_MEMORY_BYTES,
    TOTAL_BYTES,
    CacheGeometry,
    argument_value,
    budget_margin_bytes,
    cache_bytes,
    calculate_need,
    hardware_proven,
    load_inputs,
    memory_estimate_warnings,
    rank_weight_bytes,
    sglang_capped_static_need,
    sglang_demand,
    sglang_static_need,
    vllm_utilization_envelope,
)

ROOT = Path(__file__).resolve().parents[1]
LLAMA = "llama-3-1-8b-instruct-nvfp4-nvidia-vllm-single"
INPUTS = load_inputs()


def llama() -> RecipeDefinition:
    return RecipeDefinition.model_validate_json(
        (ROOT / "recipes" / f"{LLAMA}.json").read_text()
    )


def checked(recipe: RecipeDefinition) -> list[str]:
    # Identity belongs to the authored JSON, not defaults added by parsing it.
    # The guard separately verifies that this parsed recipe still matches it.
    document = json.loads(
        (ROOT / "recipes" / f"{recipe.identity.slug}.json").read_text()
    )
    return memory_estimate_warnings(
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


def test_limit_subtracts_idle_use_and_guard_exactly_once() -> None:
    assert IDLE_USE_BYTES == 5_012_574_208
    assert GUARD_BYTES > IDLE_USE_BYTES
    assert TOTAL_BYTES - IDLE_USE_BYTES - GUARD_BYTES == LIMIT_BYTES


@pytest.mark.live_evidence
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
    assert_warns(
        checked(recipe),
        "cannot hold one full-context request",
    )


@pytest.mark.parametrize(
    "value,match",
    [("120G", "peak understates need"), ("1G", "one full-context request")],
)
def test_guard_rejects_pool_drift(value: str, match: str) -> None:
    recipe = llama()
    argument = next(
        item
        for item in recipe.runtime.arguments
        if item.name == "kv-cache-memory-bytes"
    )
    argument.value = value
    assert_warns(
        checked(recipe),
        "",
    )


def test_pool_holds_one_request_at_any_engine_concurrency() -> None:
    # vLLM schedules sequences within the pool: concurrency does not size it.
    recipe = llama()
    declare_peak(recipe)
    argument = next(
        item for item in recipe.runtime.arguments if item.name == "max-num-seqs"
    )
    argument.setting = None
    argument.value = 256
    assert_not_warns(checked(recipe), "full-context request")
    assert_not_warns(
        memory_estimate_warnings(
            recipe, catalog_models(), INPUTS, document_digest="", touched=True
        ),
        "full-context request",
    )


def declare_envelope(recipe: RecipeDefinition) -> None:
    envelope = vllm_utilization_envelope(recipe)
    assert envelope is not None
    for role in recipe.topology.roles:
        role.resources.memory.peak_bytes = envelope


def test_utilization_sized_peak_covers_the_vllm_budget_on_one_spark() -> None:
    # The Controller reserves the declared peak: it must cover the budget.
    recipe = llama()
    recipe.runtime.arguments = [
        item
        for item in recipe.runtime.arguments
        if item.name != "kv-cache-memory-bytes"
    ] + [RecipeRuntimeArgument(name="gpu-memory-utilization", value="0.70")]
    for role in recipe.topology.roles:
        role.resources.memory.peak_bytes = 1_000_000_000
    assert_warns(
        checked(recipe),
        "peak understates vLLM budget",
    )
    declare_envelope(recipe)
    assert_not_warns(checked(recipe), "peak understates")


def test_touched_utilization_budget_must_be_admissible_on_one_spark() -> None:
    # With the 2 GiB platform floor the largest admitted peak is 124,452,516,352
    # bytes (utilization up to 0.9050 beside the 6.2 GB single-Spark margin);
    # vLLM's 0.90 default is admissible, 0.91 is not.
    recipe = llama()
    recipe.runtime.arguments = [
        item
        for item in recipe.runtime.arguments
        if item.name != "kv-cache-memory-bytes"
    ]

    def touched() -> list[str]:
        declare_envelope(recipe)
        return memory_estimate_warnings(
            recipe, catalog_models(), INPUTS, document_digest="", touched=True
        )

    assert touched() == []
    recipe.runtime.arguments.append(
        RecipeRuntimeArgument(name="gpu-memory-utilization", value="0.91")
    )
    assert_warns(
        touched(),
        "not admissible",
    )
    recipe.runtime.arguments[-1].value = "0.88"
    assert touched() == []


def test_failed_recipe_cannot_reuse_its_historical_pass_to_bypass_need() -> None:
    recipe = llama()
    recipe.runtime.arguments = [
        item
        for item in recipe.runtime.arguments
        if item.name != "kv-cache-memory-bytes"
    ]
    for role in recipe.topology.roles:
        role.resources.memory.peak_bytes = 1_000_000_000
    assert_warns(
        memory_estimate_warnings(
            recipe,
            catalog_models(),
            INPUTS,
            document_digest="stale-passed-digest",
            touched=True,
        ),
        "peak understates vLLM budget",
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


def test_single_spark_sglang_token_cap_holds_one_request() -> None:
    recipe = llama()
    declare_peak(recipe)
    recipe.runtime.engine = "sglang"
    next(
        item for item in recipe.runtime.arguments if item.name == "max-model-len"
    ).name = "context-length"
    concurrency = next(
        item for item in recipe.runtime.arguments if item.name == "max-num-seqs"
    )
    concurrency.name = "max-running-requests"
    cap = next(
        item
        for item in recipe.runtime.arguments
        if item.name == "kv-cache-memory-bytes"
    )
    cap.name = "max-total-tokens"
    # One context at any concurrency: SGLang schedules within the cap.
    cap.value = 131072
    concurrency.setting = None
    concurrency.value = 8
    assert_not_warns(checked(recipe), "full-context request")
    cap.value = 131071
    assert_warns(
        checked(recipe),
        "one full-context request",
    )
    recipe.runtime.arguments.remove(cap)
    # SGLang derives an omitted fraction at launch; a declared one is bounded.
    assert_not_warns(
        memory_estimate_warnings(
            recipe, catalog_models(), INPUTS, document_digest="", touched=True
        ),
        "mem-fraction-static",
    )
    recipe.runtime.arguments.append(
        RecipeRuntimeArgument(name="mem-fraction-static", value="0.90")
    )
    assert_warns(
        memory_estimate_warnings(
            recipe, catalog_models(), INPUTS, document_digest="", touched=True
        ),
        "mem-fraction-static 0.90 share.*not admissible",
    )
    recipe.runtime.arguments[-1].value = "0.85"
    assert_not_warns(
        memory_estimate_warnings(
            recipe, catalog_models(), INPUTS, document_digest="", touched=True
        ),
        "mem-fraction-static",
    )


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


def test_real_catalog_build_warns_about_a_new_recipe_whose_pool_cannot_hold_a_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tool = runpy.run_path(str(ROOT / "tools/build-catalog-index"))
    build = tool["build"]
    # Use the actual publisher entry point and the original pinned Model. Only
    # the synthetic Recipe is new, so this fails on the old unguarded builder.
    recipe = llama()
    recipe.identity.slug = "synthetic-pool-below-request-new"
    assert recipe.settings.kind == "generation"
    recipe.settings.context_tokens.value *= 100
    (tmp_path / "recipes").mkdir()
    (tmp_path / "recipes/synthetic-pool-below-request-new.json").write_text(
        recipe.model_dump_json()
    )
    (tmp_path / "models").symlink_to(ROOT / "models", target_is_directory=True)
    (tmp_path / "display/recipes").mkdir(parents=True)
    (tmp_path / "display/recipes/synthetic-pool-below-request-new.json").write_text(
        json.dumps(
            {
                "kind": "recipe-display",
                "identity": {
                    "publisher": recipe.identity.publisher,
                    "slug": recipe.identity.slug,
                },
                "title": "Synthetic",
                "description": "A synthetic recipe.",
                "tags": [],
            }
        )
    )
    (tmp_path / "display/models").symlink_to(
        ROOT / "display/models", target_is_directory=True
    )
    monkeypatch.setitem(build.__globals__, "ROOT", tmp_path)

    class PackagingReached(Exception):
        pass

    def reached_packaging(*args: object, **kwargs: object) -> None:
        raise PackagingReached

    # The recipes are estimates until tried: a pool below one request warns and
    # the build goes on to packaging.
    monkeypatch.setitem(build.__globals__, "recipe_package", reached_packaging)
    with pytest.raises(PackagingReached):
        build(
            package_dir=tmp_path / "packages",
            source_commit="0" * 40,
            source_date_epoch=0,
        )
    assert "cannot hold one full-context request" in capsys.readouterr().err


@pytest.mark.parametrize(
    "config",
    [
        {"method": "mtp", "num_speculative_tokens": 2},
        {"method": "deepseek_mtp", "model": "/models"},
        {"method": "dflash", "model": "/models/draft"},
    ],
)
def test_need_counts_a_draft_by_its_weights_however_it_is_loaded(
    config: dict[str, str | int],
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
    assert need.overhead_bytes == OVERHEAD_BYTES


def test_a_drafter_is_charged_inside_the_vllm_budget() -> None:
    # vLLM loads the drafter after its free-memory snapshot and profiles it
    # inside gpu-memory-utilization (vllm_memory_fit.py): its weights must fit
    # the budget, and nothing is added beside it.
    recipe = RecipeDefinition.model_validate_json(
        (ROOT / "recipes/qwen3-8-27b-nvfp4-dflash2-eugr-vllm-single.json").read_text()
    )
    declare_peak(recipe)
    models = catalog_models()

    def touched() -> list[str]:
        return memory_estimate_warnings(
            recipe, models, INPUTS, document_digest="", touched=True
        )

    assert touched() == []
    target_only = recipe.model_copy(deep=True)
    target_only.models = target_only.models[:1]
    target_only.runtime.arguments = [
        item
        for item in target_only.runtime.arguments
        if item.name != "speculative-config"
    ]
    weights = rank_weight_bytes(recipe, models, INPUTS)
    draft_bytes = weights - rank_weight_bytes(target_only, models, INPUTS)
    assert draft_bytes > 0
    assert vllm_utilization_envelope(recipe) == vllm_utilization_envelope(target_only)
    # A budget that holds the target and the overhead, but not the drafter too.
    share = (
        Decimal(weights + OVERHEAD_BYTES - draft_bytes // 2) / TOTAL_BYTES
    ).quantize(Decimal("0.0001"))
    next(
        item
        for item in recipe.runtime.arguments
        if item.name == "gpu-memory-utilization"
    ).value = str(share)
    declare_envelope(recipe)
    assert_warns(
        touched(),
        "cannot hold the selected weights",
    )


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
    worker = recipe.topology.roles[-1]
    worker.resources.memory.peak_bytes = 0
    assert_warns(
        memory_estimate_warnings(
            recipe, catalog_models(), INPUTS, document_digest=digest
        ),
        "",
    )


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
    declare_peak(recipe)
    args = {item.name: item.value for item in recipe.runtime.arguments}
    concurrency = int(str(argument_value(recipe, "max-running-requests")))
    context = int(str(argument_value(recipe, "context-length")))
    schedule = SimpleNamespace(
        mem_fraction_static=float(str(args["mem-fraction-static"])),
        max_total_tokens=args.get("max-total-tokens"),
        max_mamba_cache_size=args.get("max-mamba-cache-size"),
        max_running_requests=concurrency,
        mamba_full_memory_ratio=1.0,
        disable_overlap_schedule=False,
    )

    def override(_reason: str, **values: int) -> None:
        for name, value in values.items():
            setattr(schedule, name, value)

    namespace = runpy.run_path(
        str(ROOT / "tests/fixtures/glm-a0-cache-sizing.py.txt"),
        init_globals={
            "math": math,
            "logging": logging,
            "logger": logging.getLogger("sglang"),
            "envs": SimpleNamespace(
                SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=SimpleNamespace(get=lambda: False)
            ),
            "get_schedule": lambda: schedule,
            "get_available_gpu_memory": lambda *a, **kw: 30 - 6.32,
            "get_world_group": lambda: SimpleNamespace(world_size=1, cpu_group=None),
            "get_mm": lambda: SimpleNamespace(mm_feature_transport=None),
            "get_memory": lambda: SimpleNamespace(disable_radix_cache=False),
            "mm_runtime_reservation_gb": lambda **kw: 0,
            "get_exec": lambda: SimpleNamespace(
                mamba=SimpleNamespace(
                    enable_linear_replayssm_spec=False,
                    enable_mamba_extra_buffer=True,
                    enable_mamba_extra_buffer_lazy=False,
                )
            ),
            "get_context": lambda: SimpleNamespace(override=override),
            "get_parallel": lambda: SimpleNamespace(config=SimpleNamespace(pp_size=1)),
        },
    )
    sizer = namespace["CacheSizer"]()
    sizer.device, sizer.gpu_id = "cuda", 0
    sizer.post_capture_kv_active = False
    sizer.model_config = SimpleNamespace(is_multimodal=True, context_len=context)
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
    # Five state slots serve the recipe's one running request; the pool is the
    # recipe's own value, not autosized.
    assert schedule.max_mamba_cache_size == 5 * concurrency
    assert sizer._apply_token_constraints(1_000_000) == 8192
    assert sizer._calculate_mamba_ratio() == SGLANG_MAMBA_SLOTS_PER_REQUEST
    assert sizer.resolve_max_num_reqs(8192) == concurrency
    # 2026-10-28 operation 31e3de45: two slots floor to zero requests.
    schedule.max_mamba_cache_size = 2
    with pytest.raises(RuntimeError, match="too small to serve any requests"):
        sizer.resolve_max_num_reqs(8192)
    schedule.max_mamba_cache_size = 5 * concurrency
    assert checked(recipe) == []
    need = calculate_need(recipe, catalog_models(), INPUTS)
    assert need.kv_bytes == 8192 * 65536
    assert need.state_bytes == 6 * 150 * 1024**2
    assert need.overhead_bytes == 8_000_000_000
    assert need.resident_expansion_bytes == 14_000_000_000
    assert need.total_bytes <= recipe.topology.roles[0].resources.memory.peak_bytes
    # Honest peak with five state slots is 158 MB above the 108.47 GB planning
    # limit, so only the Controller's idle admission rule bounds it.
    assert evaluate_memory_admission(
        recipe.topology.roles[0].resources.memory.peak_bytes,
        SPARK_MEMORY_BYTES,
        occupied_bytes=ADMISSION_IDLE_USE_BYTES,
    ).admissible

    # Replay the rejected profile through the image's actual sizing functions.
    schedule.mem_fraction_static = 0.75
    schedule.max_mamba_cache_size = None
    with pytest.raises(RuntimeError, match="Not enough GPU memory for hybrid"):
        sizer._profile_available_bytes(120)


GLM_A0_PEAK = 108_629_500_270


def glm_a0() -> RecipeDefinition:
    return RecipeDefinition.model_validate_json(
        (
            ROOT / "recipes/glm-5-3-flash-exl3-a0-2-05bpw-0xsero-sglang-single.json"
        ).read_text()
    )


@pytest.mark.parametrize(
    "name,value,match",
    [
        # 116.8 GB of weights, expansion and capped pools: beyond the declared
        # peak and the 106.8 GB static budget of 0.85 x 125.65 GB.
        ("max-total-tokens", 262144, "peak understates SGLang static need"),
        ("max-mamba-cache-size", 16, "peak understates SGLang static need"),
        # The OOM-killed fa86311a fraction.
        ("mem-fraction-static", "0.75", "cannot hold the selected weights"),
    ],
)
def test_capped_sglang_guard_rejects_restored_large_pools(
    name: str, value: int | str, match: str
) -> None:
    recipe = glm_a0()
    declare_peak(recipe, GLM_A0_PEAK)
    assert (
        memory_estimate_warnings(
            recipe, catalog_models(), INPUTS, document_digest="", touched=True
        )
        == []
    )
    next(item for item in recipe.runtime.arguments if item.name == name).value = value
    assert_warns(
        memory_estimate_warnings(
            recipe, catalog_models(), INPUTS, document_digest="", touched=True
        ),
        "",
    )


def test_capped_sglang_pools_are_admitted_by_need_not_by_fraction() -> None:
    """The fraction gates only rest > 0 when both pools are capped (SGLang
    kv_cache_configurator): a capped recipe at 0.95 takes need + overhead."""
    recipe = glm_a0()
    declare_peak(recipe, GLM_A0_PEAK)
    next(
        item for item in recipe.runtime.arguments if item.name == "mem-fraction-static"
    ).value = "0.95"
    assert (
        memory_estimate_warnings(
            recipe, catalog_models(), INPUTS, document_digest="", touched=True
        )
        == []
    )
    # Uncapping the token pool restores the fraction model: 0.95 is refused.
    recipe.runtime.arguments.remove(
        next(
            item for item in recipe.runtime.arguments if item.name == "max-total-tokens"
        )
    )
    assert_warns(
        memory_estimate_warnings(
            recipe, catalog_models(), INPUTS, document_digest="", touched=True
        ),
        "",
    )


def test_uncapped_sglang_fraction_stays_the_calibrated_proxy() -> None:
    recipe = glm_a0()
    fraction = argument_value(recipe, "mem-fraction-static")
    assert sglang_demand(recipe, None) == Decimal(str(fraction)) * TOTAL_BYTES + (
        budget_margin_bytes(recipe)
    )
    assert budget_margin_bytes(recipe) == 8_500_000_000
    assert sglang_demand(recipe, 40_000_000_000) == Decimal(48_000_000_000)


def test_hybrid_without_mamba_cap_stays_on_the_fraction_model() -> None:
    """nemotron-3-5-lightning is a proven 0.85 launch with no max-mamba-cache-size:
    the Mamba pool grows with the fraction, so it is not capped and 119.06e9
    (0.85 x total + 8e9) is the demand; 0.95 is refused whatever the peak."""
    recipe = RecipeDefinition.model_validate_json(
        (
            ROOT / "recipes/nemotron-3-5-lightning-r0b0tlab-sglang-eagle-single.json"
        ).read_text()
    )
    declare_peak(recipe, 100_000_000_000)
    assert argument_value(recipe, "max-total-tokens") is not None
    assert (
        sglang_capped_static_need(recipe, catalog_models(), INPUTS, "nemotron") is None
    )
    assert_not_warns(
        memory_estimate_warnings(
            recipe, catalog_models(), INPUTS, document_digest="", touched=True
        ),
        "not admissible on an idle Spark",
    )
    next(
        item for item in recipe.runtime.arguments if item.name == "mem-fraction-static"
    ).value = "0.95"
    for role in recipe.topology.roles:
        role.resources.memory.peak_bytes = 40_000_000_000
    assert_warns(
        memory_estimate_warnings(
            recipe, catalog_models(), INPUTS, document_digest="", touched=True
        ),
        "not admissible on an idle Spark",
    )


@pytest.mark.parametrize("defect", ["graphs", "image", "state-cap"])
def test_hybrid_estimate_rejects_uncalibrated_runtime(defect: str) -> None:
    # The informative full-workload estimate refuses geometry it cannot apply.
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
    else:
        recipe.runtime.arguments = [
            a for a in recipe.runtime.arguments if a.name != "max-mamba-cache-size"
        ]
    with pytest.raises((ValueError, TypeError)):
        calculate_need(recipe, catalog_models(), INPUTS)


@pytest.mark.parametrize("slots", [2, 4])
def test_state_padding_geometry_needs_five_state_slots_per_request(slots: int) -> None:
    recipe = glm_a0()
    next(
        item for item in recipe.runtime.arguments if item.name == "max-mamba-cache-size"
    ).value = slots
    with pytest.raises(ValueError, match="cannot hold the 5 state slots"):
        calculate_need(recipe, catalog_models(), INPUTS)
    with pytest.raises(ValueError, match="cannot hold the 5 state slots"):
        sglang_static_need(recipe, catalog_models(), INPUTS, "glm", pinned=False)
