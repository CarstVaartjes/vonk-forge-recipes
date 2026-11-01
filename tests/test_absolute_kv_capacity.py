"""Reject fixed cache pools that cannot admit the advertised context."""

import json
import runpy
import subprocess
from decimal import Decimal
from pathlib import Path

import pytest
from catalog_documents import catalog_models
from documents import recipe as document_recipe
from documents import recipe_display
from vonk_forge_contracts import ModelDefinition, RecipeDefinition, document_sha256
from vonk_forge_contracts.memory_admission import evaluate_memory_admission
from vonk_forge_contracts.recipe import RecipeRuntimeArgument

from qualification.memory_need import (
    ADMISSION_IDLE_USE_BYTES,
    SPARK_MEMORY_BYTES,
    absolute_kv_need,
    argument_value,
    cache_bytes,
    hardware_proven,
    load_inputs,
    memory_variants,
    raise_peaks_to_envelope,
    require_absolute_kv_capacity,
    require_memory_need,
    require_sglang_rank_budget,
    vllm_utilization_envelope,
)

ROOT = Path(__file__).resolve().parents[1]
VISION = "deepseek-v4-flash-vision-exp-mia-dual"
LAGUNA = "laguna-m-1-nvfp4-r0b0tlab-vllm-dual"
QWEN_NEXT = "qwen3-8-flash-next-w4a16-r0b0tlab-sglang-dual"
JASL = "deepseek-v4-flash-jasl-tonyd2wild-vllm-dual"
GLM_DFLASH2 = "glm-5-3-flash-nvidia-nvfp4-dflash2-vllm-dual"


def recipe_for(slug: str = VISION) -> RecipeDefinition:
    return RecipeDefinition.model_validate_json(
        (ROOT / "recipes" / f"{slug}.json").read_text()
    )


@pytest.mark.parametrize(
    "slug",
    [VISION, "deepseek-v4-flash-vision-exp-dspark-tonyd2wild-vllm-dual"],
)
def test_reported_startup_fault_is_rejected_before_catalog_publication(
    slug: str,
) -> None:
    recipe = recipe_for(slug)
    assert recipe.settings.kind == "generation"
    recipe.settings.context_tokens.value = 1_048_576
    next(
        arg for arg in recipe.runtime.arguments if arg.name == "kv-cache-memory-bytes"
    ).value = 495_976_448
    with pytest.raises(ValueError, match="cannot hold one full-context request"):
        require_memory_need(recipe, catalog_models(), load_inputs(), document_digest="")


@pytest.mark.parametrize("context", [None, "auto", -1, "-1", 1_048_576])
def test_explicit_and_model_default_context_have_the_same_need(context) -> None:
    recipe = recipe_for()
    recipe.runtime.arguments = [
        arg for arg in recipe.runtime.arguments if arg.name != "max-model-len"
    ]
    if context is not None:
        recipe.runtime.arguments.append(
            RecipeRuntimeArgument(name="max-model-len", value=context)
        )
    need = absolute_kv_need(recipe, catalog_models(), load_inputs())
    assert need == 1_048_576 * 8192 + 64 * 1024**2
    # MLA is replicated on the two tensor ranks, never divided by TP=2.
    assert need > 7.54 * 1024**3


def test_pool_boundary_and_wrapper_alias_are_checked() -> None:
    recipe = recipe_for()
    need = absolute_kv_need(recipe, catalog_models(), load_inputs())
    pool = next(
        arg for arg in recipe.runtime.arguments if arg.name == "kv-cache-memory-bytes"
    )
    pool.name = "kv-cache-memory"
    pool.value = need
    require_absolute_kv_capacity(recipe, catalog_models(), load_inputs())
    pool.value = need - 1
    with pytest.raises(ValueError, match="cannot hold one full-context request"):
        require_absolute_kv_capacity(recipe, catalog_models(), load_inputs())


def test_option_dtype_can_invalidate_a_base_pool() -> None:
    recipe = recipe_for("hy3-295b-nvfp4-vllm-mia-dual")
    need = absolute_kv_need(recipe, catalog_models(), load_inputs())
    next(
        arg for arg in recipe.runtime.arguments if arg.name == "kv-cache-memory-bytes"
    ).value = need
    with pytest.raises(ValueError, match="cannot hold one full-context request"):
        require_absolute_kv_capacity(recipe, catalog_models(), load_inputs())


def test_gqa_sharding_and_nvfp4_block_scales() -> None:
    recipe = recipe_for("hy3-295b-nvfp4-a4q-drowzeys-vllm-dual")
    assert recipe.settings.kind == "generation"
    recipe.settings.context_tokens.value = 16
    assert absolute_kv_need(recipe, catalog_models(), load_inputs()) == (
        16 * 80 * 4 * (128 // 2 + 128 // 16) * 2
    )
    recipe.topology.parallelism.tensor = 1
    assert absolute_kv_need(recipe, catalog_models(), load_inputs()) == (
        16 * 80 * 8 * (128 // 2 + 128 // 16) * 2
    )


def test_pipeline_layers_shard_while_mla_stays_tensor_replicated() -> None:
    recipe = recipe_for()
    assert recipe.settings.kind == "generation"
    recipe.settings.context_tokens.value = 256
    recipe.topology.parallelism.pipeline = 2
    # The largest stage has 22 of the 43 layers; round its per-token bound up.
    assert absolute_kv_need(recipe, catalog_models(), load_inputs()) == (
        256 * 4192 + 64 * 1024**2
    )


def test_geometry_must_match_selected_configuration() -> None:
    inputs = load_inputs()
    recipe = recipe_for()
    inputs.geometries[recipe.models[0].model.slug].configuration_sha256 = "0" * 64
    with pytest.raises(ValueError, match="pinned model configuration"):
        absolute_kv_need(recipe, catalog_models(), inputs)


def test_sglang_token_pool_must_hold_one_context() -> None:
    recipe = recipe_for("deepseek-v4-1-flash-1017-mia-sglang-triple")
    context = argument_value(recipe, "context-length")
    assert isinstance(context, int)
    pool = next(
        arg for arg in recipe.runtime.arguments if arg.name == "max-total-tokens"
    )
    pool.value = context
    require_absolute_kv_capacity(recipe, catalog_models(), load_inputs())
    pool.value = context - 1
    with pytest.raises(ValueError, match="cannot hold one full-context request"):
        require_absolute_kv_capacity(recipe, catalog_models(), load_inputs())


def test_recurrent_state_counted_once_is_rejected() -> None:
    # Operation c2940bb5 (2026-10-27): vLLM refused this 1M-context pool, sized
    # for 48 GDN layers' raw state counted once; its startup check alone needed
    # 34,681,651,200 bytes (padded pages, two per layer with prefix caching).
    recipe = recipe_for("qwen3-8-27b-aday777-abliterated-drowzeys-vllm-single")
    pool = next(
        arg for arg in recipe.runtime.arguments if arg.name == "kv-cache-memory-bytes"
    )
    pool.value = 34_527_510_528
    with pytest.raises(ValueError, match="cannot hold one full-context request"):
        require_absolute_kv_capacity(recipe, catalog_models(), load_inputs())
    assert absolute_kv_need(recipe, catalog_models(), load_inputs()) > 34_681_651_200


def test_recurrent_pages_beyond_a_hand_bound_are_rejected() -> None:
    # vLLM pads Super VL's 40 Mamba layers to the attention page and holds
    # two plus seven speculative pages per recurrent group with prefix
    # caching: 3,019,898,880 bytes per request, above the 256 MiB hand bound.
    recipe = recipe_for("nemotron-3-5-super-vl-lilicorr-sfxnz-vllm-single")
    pool = next(
        arg for arg in recipe.runtime.arguments if arg.name == "kv-cache-memory-bytes"
    )
    pool.value = 2_800_000_000
    with pytest.raises(ValueError, match="cannot hold one full-context request"):
        require_absolute_kv_capacity(recipe, catalog_models(), load_inputs())


def test_catalog_absolute_pools_hold_one_full_context_for_every_option() -> None:
    for path in sorted((ROOT / "recipes").glob("*.json")):
        recipe = RecipeDefinition.model_validate_json(path.read_text())
        require_absolute_kv_capacity(recipe, catalog_models(), load_inputs())


def test_corrected_pools_cover_all_declared_concurrency() -> None:
    # The source records the operator-visible correction; no pass/failure ledger.
    for path in sorted((ROOT / "recipes").glob("*.json")):
        document = json.loads(path.read_text())
        recipe = RecipeDefinition.model_validate(document)
        if (
            "Absolute KV correction:"
            not in recipe_display(recipe.identity.slug).description
        ):
            continue
        for variant in memory_variants(recipe):
            concurrency = argument_value(variant, "max-num-seqs")
            assert isinstance(concurrency, int)
            pool = argument_value(variant, "kv-cache-memory-bytes")
            assert pool is not None
            assert cache_bytes(pool) >= absolute_kv_need(
                variant, catalog_models(), load_inputs(), concurrency=concurrency
            )


def utilization_argument(recipe: RecipeDefinition) -> RecipeRuntimeArgument:
    return next(
        arg for arg in recipe.runtime.arguments if arg.name == "gpu-memory-utilization"
    )


def admissible(recipe: RecipeDefinition) -> bool:
    envelope = vllm_utilization_envelope(recipe)
    assert envelope is not None
    return evaluate_memory_admission(
        envelope, SPARK_MEMORY_BYTES, occupied_bytes=ADMISSION_IDLE_USE_BYTES
    ).admissible


def smallest_inadmissible_utilization(recipe: RecipeDefinition) -> Decimal:
    utilization = utilization_argument(recipe)
    for hundredths in range(50, 100):
        candidate = Decimal(hundredths) / 100
        utilization.value = str(candidate)
        if not admissible(recipe):
            return candidate
    raise AssertionError("every two-decimal utilization is admissible")


def declare_envelope(recipe: RecipeDefinition) -> None:
    """A touched pool-less dual declares its vLLM budget envelope."""
    envelope = vllm_utilization_envelope(recipe)
    assert envelope is not None
    for role in recipe.topology.roles:
        role.resources.memory.peak_bytes = envelope


def test_touched_multi_spark_vllm_recipe_bounds_kv_like_the_proven_set() -> None:
    recipe = recipe_for(LAGUNA)
    models, inputs = catalog_models(), load_inputs()
    require_memory_need(recipe, models, inputs, document_digest="", touched=True)
    pool = next(
        arg for arg in recipe.runtime.arguments if arg.name == "kv-cache-memory-bytes"
    )
    # One full-context request, not context x max-num-seqs, is the floor.
    one = absolute_kv_need(recipe, models, inputs)
    pool.value = one
    require_memory_need(recipe, models, inputs, document_digest="", touched=True)
    pool.value = one - 1
    with pytest.raises(ValueError, match="one full-context request"):
        require_memory_need(recipe, models, inputs, document_digest="", touched=True)
    # Fault injection: a pool-less budget the Spark cannot admit. Laguna's
    # 0.82 stop on 2026-10-26 was the removed per-sample 16 GiB bar, so 0.82
    # itself is no longer refused; the smallest two-decimal utilization whose
    # envelope is inadmissible still is.
    recipe.runtime.arguments.remove(pool)
    utilization = utilization_argument(recipe)
    utilization.value = "0.82"
    declare_envelope(recipe)
    require_memory_need(recipe, models, inputs, document_digest="", touched=True)
    utilization.value = str(smallest_inadmissible_utilization(recipe))
    declare_envelope(recipe)
    with pytest.raises(ValueError, match="admissible on an idle Spark"):
        require_memory_need(recipe, models, inputs, document_digest="", touched=True)
    # The highest proven utilization-sized dual budget (0.80) is accepted.
    next(
        arg for arg in recipe.runtime.arguments if arg.name == "gpu-memory-utilization"
    ).value = "0.80"
    declare_envelope(recipe)
    require_memory_need(recipe, models, inputs, document_digest="", touched=True)


@pytest.mark.parametrize("single", [True, False], ids=["single", "multi"])
@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_touching_any_proven_document_keeps_it_valid(engine: str, single: bool) -> None:
    # Calibration: every hardware-proven document passes the touched rules.
    models, inputs = catalog_models(), load_inputs()
    checked, rejected = 0, []
    for path in sorted((ROOT / "qualification/evidence").rglob("*.json")):
        key = json.loads(path.read_text())["recipe_key"]
        document = json.loads(
            (ROOT / "recipes" / f"{key.split('/')[1]}.json").read_text()
        )
        recipe = RecipeDefinition.model_validate(document)
        if (
            recipe.runtime.engine != engine
            or (recipe.topology.node_count == 1) != single
            or not hardware_proven(recipe, document_sha256(document))
        ):
            continue
        checked += 1
        # A touched document is a refresh, which raises its peaks to the envelope.
        recipe = raise_peaks_to_envelope(recipe) or recipe
        try:
            require_memory_need(
                recipe, models, inputs, document_digest="", touched=True
            )
        except (TypeError, ValueError) as error:
            rejected.append(str(error))
    assert checked > 0
    assert not rejected, f"{len(rejected)} of {checked} rejected:\n" + "\n".join(
        rejected
    )


FAILURES = ROOT / "tests/fixtures/single-spark-memory-failures"


@pytest.mark.parametrize(
    "name,match,image_of",
    [
        # Exhausted host memory 2026-10-21: 0.85 beside a 10 GB Marlin repack.
        # Judged on the image its geometry now pins (same Marlin expansion).
        (
            "gpt-oss-120b-nvidia-vllm-single-4036b68d",
            "admissible on an idle Spark",
            "gpt-oss-120b-nvidia-vllm-single",
        ),
        # OOM-killed (exit 137) 2026-10-21: 0.75 below weights + expansion.
        (
            "glm-5-3-flash-exl3-a0-2-05bpw-0xsero-sglang-single-fa86311a",
            "cannot hold the selected weights",
            None,
        ),
        # 2026-10-21: 0.95, not admissible on an idle Spark.
        (
            "glm-5-3-flash-exl3-mosaic-12l-0xsero-sglang-single-6e5f713f",
            "admissible on an idle Spark",
            None,
        ),
        # 2026-10-19 and 2026-10-27: 0.6 left 4.25 GB beside 74.15 GB of
        # target and draft weights; vLLM found no memory for KV cache blocks.
        (
            "laguna-s-2-1-nvfp4-r0b0tlab-vllm-single-7b2c74ef",
            "cannot hold the selected weights",
            None,
        ),
        # 2026-10-27: 0.85 left 1.12 GB beside 101.94 GB of 826aacdf target and
        # draft weights, below one 262,144-token BF16 request (13.0 GB); vLLM
        # reported -23.95 GiB of KV cache memory and found no cache blocks.
        (
            "laguna-s-2-1-nvfp4-dflash-vllm-mia-single-b72a8ee1",
            "one full-context request",
            None,
        ),
        # 2026-10-21: 0.60 (78.40 GB) was below even the 80.32 GB of weights
        # vLLM loads; vLLM found no memory for KV cache blocks.
        (
            "nemotron-3-super-120b-a12b-vllm-single-4c7855fb",
            "cannot hold the selected weights",
            None,
        ),
    ],
)
def test_recorded_single_spark_memory_failures_are_rejected_when_touched(
    name: str, match: str, image_of: str | None
) -> None:
    # With the proven-set calibration above, this proves both directions.
    models = dict(catalog_models())
    for path in sorted((FAILURES / "models").glob("*.json")):
        document = json.loads(path.read_text())
        models[document_sha256(document)] = ModelDefinition.model_validate(document)
    document = json.loads((FAILURES / f"{name}.json").read_text())
    assert document_sha256(document).startswith(name.rsplit("-", 1)[1])
    recipe = RecipeDefinition.model_validate(document)
    if image_of is not None:
        recipe.execution.build.base_image = recipe_for(
            image_of
        ).execution.build.base_image
    with pytest.raises(ValueError, match=match):
        require_memory_need(
            recipe, models, load_inputs(), document_digest="", touched=True
        )


def test_recorded_fraction_is_rejected_on_the_current_capped_glm_a0() -> None:
    # fa86311a's 0.75 cannot hold the weights and 14 GB expansion even with
    # today's 8,192-token and 2-slot caps; the document's 0.85 can.
    recipe = recipe_for("glm-5-3-flash-exl3-a0-2-05bpw-0xsero-sglang-single")
    models, inputs = catalog_models(), load_inputs()
    require_memory_need(recipe, models, inputs, document_digest="", touched=True)
    next(
        arg for arg in recipe.runtime.arguments if arg.name == "mem-fraction-static"
    ).value = "0.75"
    with pytest.raises(ValueError, match="cannot hold the selected weights"):
        require_memory_need(recipe, models, inputs, document_digest="", touched=True)


def test_multi_spark_sglang_token_cap_holds_one_request_within_the_budget() -> None:
    # Proven (9b60b9a7): one 262,144-token context capped at concurrency 4.
    recipe = recipe_for(QWEN_NEXT)
    models, inputs = catalog_models(), load_inputs()
    assert recipe.settings.kind == "generation"
    assert recipe.settings.concurrency is not None
    assert recipe.settings.concurrency.value == 4
    context = recipe.settings.context_tokens.value
    assert argument_value(recipe, "max-total-tokens") == context
    require_memory_need(recipe, models, inputs, document_digest="", touched=True)
    cap = next(
        arg for arg in recipe.runtime.arguments if arg.name == "max-total-tokens"
    )
    cap.value = context - 1
    with pytest.raises(ValueError, match="one full-context request"):
        require_memory_need(recipe, models, inputs, document_digest="")
    # Uncapped, the fraction bounds the pools and the Controller admits the
    # declared peak: only SGLang's startup room is checked (see
    # test_inkling_recipes.py for the failing side).
    recipe.runtime.arguments.remove(cap)
    require_memory_need(recipe, models, inputs, document_digest="")
    # The state cap holds five slots for each of the four running requests.
    recipe = recipe_for(QWEN_NEXT)
    slots = next(
        arg for arg in recipe.runtime.arguments if arg.name == "max-mamba-cache-size"
    )
    slots.value = 20
    require_memory_need(recipe, models, inputs, document_digest="")
    slots.value = 19
    with pytest.raises(ValueError, match="state slots of each of 4 running"):
        require_memory_need(recipe, models, inputs, document_digest="")

    # Fault injection at the budget edge: without speculation 0.584 holds the
    # sharded weights, the capped token pool and the 21 state slots; the
    # default NEXTN option adds (min(4, 20 // 5) + 1) x 4 intermediate slot
    # states, which need 0.594.
    def rank_budget(
        fraction: str, draft_tokens: int | None = None, expect_refused: bool = False
    ) -> None:
        for variant in memory_variants(recipe_for(QWEN_NEXT)):
            for arg in variant.runtime.arguments:
                if arg.name == "mem-fraction-static":
                    arg.value = fraction
                elif arg.name == "speculative-num-draft-tokens" and draft_tokens:
                    arg.value = draft_tokens
            if argument_value(variant, "speculative-algorithm") is not None:
                speculative.append(variant)
            elif expect_refused:
                with pytest.raises(ValueError, match="static budget"):
                    require_sglang_rank_budget(variant, models, inputs, "k")
            else:
                require_sglang_rank_budget(variant, models, inputs, "k")

    speculative: list[RecipeDefinition] = []
    rank_budget("0.583", expect_refused=True)
    speculative.clear()
    rank_budget("0.593")
    assert speculative
    for variant in speculative:
        with pytest.raises(ValueError, match="static budget"):
            require_sglang_rank_budget(variant, models, inputs, "k")
    speculative.clear()
    rank_budget("0.594")
    for variant in speculative:
        require_sglang_rank_budget(variant, models, inputs, "k")
    # Eight draft tokens double the intermediate state past that edge.
    speculative.clear()
    rank_budget("0.594", draft_tokens=8)
    for variant in speculative:
        with pytest.raises(ValueError, match="static budget"):
            require_sglang_rank_budget(variant, models, inputs, "k")


def test_geometry_bound_to_an_image_rejects_another_image() -> None:
    recipe = recipe_for(QWEN_NEXT)
    models, inputs = catalog_models(), load_inputs()
    recipe.execution.build.base_image.digest = "0" * 64
    with pytest.raises(ValueError, match="does not match the pinned image"):
        require_memory_need(recipe, models, inputs, document_digest="")


def test_only_recipe_files_changed_since_the_merge_base_are_touched(
    tmp_path: Path,
) -> None:
    def git(*args: str) -> None:
        subprocess.run(
            ["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t"]
            + list(args),
            check=True,
            capture_output=True,
        )

    (tmp_path / "recipes").mkdir()
    for name in ("kept", "edited"):
        (tmp_path / f"recipes/{name}.json").write_text("{}")
    git("init", "-q", "-b", "main")
    git("add", "-A")
    git("commit", "-q", "-m", "base")
    (tmp_path / "recipes/edited.json").write_text('{"x": 1}')
    (tmp_path / "recipes/new.json").write_text("{}")
    touched = runpy.run_path(str(ROOT / "tools/check-recipe-memory-need"))[
        "touched_recipes"
    ]
    assert touched(tmp_path, "HEAD") == [
        tmp_path / "recipes/edited.json",
        tmp_path / "recipes/new.json",
    ]


def creator_dual(slug: str, utilization: str) -> RecipeDefinition:
    """The recipe at its creator's utilization, every role declaring the envelope."""
    recipe = recipe_for(slug)
    utilization_argument(recipe).value = utilization
    envelope = vllm_utilization_envelope(recipe)
    assert envelope is not None
    for role in recipe.topology.roles:
        role.resources.memory.peak_bytes = envelope
    return recipe


@pytest.mark.parametrize(
    ("slug", "utilization", "reserve_bytes"),
    [
        # sparkfix.yaml#L16 at 050f1ba3.
        (JASL, "0.825", None),
        # boot_dflash2.sh#L14 at 22269731; passed on hardware, pool-less (7cb340f9).
        (GLM_DFLASH2, "0.86", 8_000_000_000),
    ],
)
def test_creator_dual_utilizations_are_bounded_by_admission(
    slug: str, utilization: str, reserve_bytes: int | None
) -> None:
    recipe = creator_dual(slug, utilization)
    if reserve_bytes is not None:
        for role in recipe.topology.roles:
            role.resources.memory.reserve_bytes = reserve_bytes
    assert admissible(recipe)
    require_memory_need(
        recipe, catalog_models(), load_inputs(), document_digest="", touched=True
    )


def test_a_dual_utilization_the_spark_cannot_admit_is_still_refused() -> None:
    recipe = creator_dual(JASL, "0.90")
    assert not admissible(recipe)
    with pytest.raises(ValueError, match="admissible on an idle Spark"):
        require_memory_need(
            recipe, catalog_models(), load_inputs(), document_digest="", touched=True
        )


# Creator-logged pools (GLM DSA, fp8/nvfp4 sparse-MLA): recipe, dtype, context,
# pool bytes and the tokens the creator's vLLM logged for that pool.
GLM_DSA_CREATOR_POOLS = [
    pytest.param(
        "glm-5-2-quanttrio-200k-mtp-tonyd2wild-vllm-four",
        "fp8_ds_mla",
        200_000,
        10_950_000_000,
        200_064,
        id="glm-5.2-quanttrio-fp8",
    ),
    pytest.param(
        "glm-5-3-int4-int8mix-mtp-tonyd2wild-vllm-four",
        "fp8_ds_mla",
        200_000,
        10_950_000_000,
        200_064,
        id="glm-5.3-int4-fp8",
    ),
    pytest.param(
        "glm-5-2-quanttrio-nvfp4-kv-tonyd2wild-vllm-four",
        "nvfp4_ds_mla",
        316_000,
        10_950_000_000,
        317_279,
        id="glm-5.2-nvfp4-kv",
    ),
    pytest.param(
        "glm-5-2-aqlm-vllm-triple",
        "nvfp4_ds_mla",
        348_160,
        11_811_160_064,
        354_496,
        id="glm-5.2-aqlm",
    ),
]


def creator_glm_recipe(
    slug: str, dtype: str, context: int, pool: int
) -> RecipeDefinition:
    recipe = document_recipe(slug)
    assert recipe.settings.kind == "generation"
    recipe.settings.context_tokens.value = context
    set_pool(recipe, pool)
    assert argument_value(recipe, "kv-cache-dtype") == dtype
    return recipe


def set_pool(recipe: RecipeDefinition, pool: int) -> None:
    next(
        arg for arg in recipe.runtime.arguments if arg.name == "kv-cache-memory-bytes"
    ).value = pool


@pytest.mark.parametrize("slug,dtype,context,pool,logged", GLM_DSA_CREATOR_POOLS)
def test_glm_dsa_creator_pools_hold_their_logged_context(
    slug: str, dtype: str, context: int, pool: int, logged: int
) -> None:
    recipe = creator_glm_recipe(slug, dtype, context, pool)
    require_absolute_kv_capacity(recipe, catalog_models(), load_inputs())
    # The derived per-token rate also predicts at least the tokens the
    # creator's vLLM logged for this pool.
    need = absolute_kv_need(recipe, catalog_models(), load_inputs())
    assert pool // (need // context) >= logged


@pytest.mark.parametrize("slug,dtype,context,pool,logged", GLM_DSA_CREATOR_POOLS)
def test_glm_dsa_pool_one_byte_below_the_need_is_still_refused(
    slug: str, dtype: str, context: int, pool: int, logged: int
) -> None:
    recipe = creator_glm_recipe(slug, dtype, context, pool)
    need = absolute_kv_need(recipe, catalog_models(), load_inputs())
    set_pool(recipe, need)
    require_absolute_kv_capacity(recipe, catalog_models(), load_inputs())
    set_pool(recipe, need - 1)
    with pytest.raises(ValueError, match="cannot hold one full-context request"):
        require_absolute_kv_capacity(recipe, catalog_models(), load_inputs())
