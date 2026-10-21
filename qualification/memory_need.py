"""Offline memory demand; only hardware-proven recipe identities are exempt."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal
from itertools import product
from pathlib import Path

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field
from vonk_forge_contracts import ModelDefinition, RecipeDefinition

from qualification.vllm_memory_fit import (
    DRAFT_MODEL_OVERHEAD_BYTES,
    OVERHEAD_BYTES,
    SPARK_3542_IDLE_AVAILABLE_BYTES,
    SPARK_MEMORY_BYTES,
    VLLM_DEFAULT_UTILIZATION,
)
from qualification.vllm_memory_fit import (
    loads_draft_model as configured_draft_model,
)

ROOT = Path(__file__).resolve().parents[1]
# GB10 total and lower idle MemAvailable from spark-3542, 2026-10-15/16;
# retained from tests/test_vllm_memory_fit.py at the historical source commit.
TOTAL_BYTES = SPARK_MEMORY_BYTES
IDLE_AVAILABLE_BYTES = SPARK_3542_IDLE_AVAILABLE_BYTES
IDLE_USE_BYTES = TOTAL_BYTES - IDLE_AVAILABLE_BYTES
# vonk-forge 040c51b8531523bebe542b5e1f7b9403e0767f72:
# rust/crates/vonk-agent-helper/src/host_memory_guard.rs re-exports
# host_memory_guard_policy.rs::STALL_AVAILABLE_BYTES. This is a planning floor;
# the live guard also requires sustained pressure (30% full PSI, 3 seconds).
GUARD_BYTES = 16 * 1024**3
LIMIT_BYTES = TOTAL_BYTES - GUARD_BYTES - IDLE_USE_BYTES


class CacheGeometry(BaseModel):
    """Pinned configuration dimensions; never guessed from the model's name."""

    model_config = ConfigDict(extra="forbid")
    configuration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source: str
    layers: int = Field(gt=0)
    kv_heads: int = Field(gt=0)
    key_dim: int = Field(gt=0)
    value_dim: int = Field(ge=0)
    sliding_layers: int = Field(default=0, ge=0)
    sliding_window: int = Field(default=0, ge=0)
    state_bytes_per_request: int = Field(default=0, ge=0)
    # Compressed/hybrid layouts may use a conservative packed-token bound.
    token_bytes: int | None = Field(default=None, gt=0)
    state_padding: int = Field(default=0, ge=0)
    resident_expansion_bytes: int = Field(default=0, ge=0)
    derived_draft_weight_bytes: int = Field(default=0, ge=0)
    bundled_draft_prefix: str | None = None
    runtime_image_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class HardwarePass(BaseModel):
    model_config = ConfigDict(extra="forbid")
    recorded_at: AwareDatetime
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    catalog_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    library_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    recipe_revision_id: str


class MemoryInputs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hardware_passes: dict[str, HardwarePass]
    geometries: dict[str, CacheGeometry]


def load_inputs(root: Path = ROOT) -> MemoryInputs:
    return MemoryInputs.model_validate_json(
        (root / "qualification/memory-inputs.json").read_text()
    )


@dataclass(frozen=True)
class MemoryNeed:
    weights_bytes: int
    auxiliary_weights_bytes: int
    kv_bytes: int
    state_bytes: int
    resident_expansion_bytes: int = 0
    overhead_bytes: int = OVERHEAD_BYTES
    limit_bytes: int = LIMIT_BYTES

    @property
    def total_bytes(self) -> int:
        return (
            self.weights_bytes
            + self.auxiliary_weights_bytes
            + self.kv_bytes
            + self.state_bytes
            + self.resident_expansion_bytes
            + self.overhead_bytes
        )


def cache_bytes(value: object) -> int:
    """vLLM human-readable bytes: G is binary, g decimal; preserve exact integers."""
    text = str(value)
    multipliers = {
        "k": 1000,
        "m": 1000**2,
        "g": 1000**3,
        "K": 1024,
        "M": 1024**2,
        "G": 1024**3,
    }
    number = (
        Decimal(text[:-1]) * multipliers[text[-1]]
        if text[-1:] in multipliers
        else Decimal(text)
    )
    if not number.is_finite() or number <= 0 or number != number.to_integral_value():
        raise ValueError(f"invalid absolute cache byte count: {text}")
    return int(number)


def argument_value(recipe: RecipeDefinition, name: str) -> object:
    for item in reversed(recipe.runtime.arguments):
        if item.name == name:
            if item.setting is None:
                return item.value
            setting = getattr(recipe.settings, item.setting, None)
            if setting is None:
                raise ValueError(
                    f"{recipe.identity.slug}: {name} has no declared value"
                )
            return setting.value
    return None


def positive_argument(recipe: RecipeDefinition, name: str) -> int:
    value = argument_value(recipe, name)
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise TypeError(
            f"{recipe.identity.slug}: bind {name} to declared engine context/concurrency"
        )
    number = int(value)
    if number < 1:
        raise ValueError(f"{recipe.identity.slug}: {name} must be positive")
    return number


def loads_draft_model(recipe: RecipeDefinition) -> bool:
    wrapper = ROOT / recipe.execution.build.context.path / "vllm-wrapper.sh"
    if wrapper.is_file() and "export SPEC_MODEL_PATH=" in wrapper.read_text():
        return True
    return configured_draft_model(recipe) or (
        argument_value(recipe, "speculative-draft-model-path") is not None
        and argument_value(recipe, "speculative-draft-model-path")
        != argument_value(recipe, "model-path")
    )


def calculate_need(
    recipe: RecipeDefinition,
    models: Mapping[str, ModelDefinition],
    geometries: Mapping[str, CacheGeometry],
) -> MemoryNeed:
    """Full declared context times engine concurrency, including selected drafts.

    Supports per-rank ordinary attention and supplied compressed
    geometry. A new topology/automatic scheduler needs explicit inputs rather
    than borrowing a benchmark concurrency or a fraction of physical memory.
    Auxiliary drafts, recurrent state and resident expansion stay replicated.
    """
    label = recipe.identity.slug
    parallel = recipe.topology.parallelism
    tensor, pipeline = parallel.tensor, parallel.pipeline
    shards = tensor * pipeline
    settings = recipe.settings
    if settings.kind not in {"generation", "decision"} or settings.concurrency is None:
        raise ValueError(
            f"{label}: declare engine context and concurrency for memory sizing"
        )
    derived_draft = loads_draft_model(recipe) and len(recipe.models) == 1
    primary_geometry = geometries.get(recipe.models[0].model.slug)
    if derived_draft and (
        primary_geometry is None
        or not (
            primary_geometry.derived_draft_weight_bytes
            or primary_geometry.bundled_draft_prefix
        )
    ):
        raise ValueError(
            f"{label}: include the wrapper's draft weights in the need model"
        )
    context_flag, concurrency_flag = (
        ("max-model-len", "max-num-seqs")
        if recipe.runtime.engine == "vllm"
        else ("context-length", "max-running-requests")
    )
    context = positive_argument(recipe, context_flag)
    concurrency = positive_argument(recipe, concurrency_flag)
    # Each choice can override a setting-bound argument through the contract's
    # option merger; calculate the actual variant, not just the base settings.
    block_flag = "block-size" if recipe.runtime.engine == "vllm" else "page-size"
    block = (
        positive_argument(recipe, block_flag)
        if argument_value(recipe, block_flag) is not None
        else (16 if recipe.runtime.engine == "vllm" else 1)
    )
    context = ((context + block - 1) // block) * block
    dtype = argument_value(recipe, "kv-cache-dtype")
    if dtype in (None, "auto", "float16", "bfloat16"):
        element_bits = 16
    elif dtype in ("fp8", "fp8_e4m3", "fp8_e5m2"):
        element_bits = 8
    elif all(
        geometries.get(selection.model.slug) is not None
        and geometries[selection.model.slug].token_bytes is not None
        for selection in recipe.models
    ):
        element_bits = 16  # explicit packed-token upper bound supplies storage size
    else:
        raise ValueError(f"{label}: supply cache layout calibration for dtype {dtype}")
    weights: list[int] = []
    kv = state = expansion = 0
    for selection in recipe.models:
        model = models[selection.model.content_sha256]
        selected = {item.file_id for item in selection.files}
        weights.append(
            sum(
                item.size_bytes
                for item in model.files
                if item.id in selected and "weights" in item.roles
            )
        )
        geometry = geometries.get(selection.model.slug)
        if geometry is None:
            raise ValueError(
                f"{label}: add source-backed cache geometry for {selection.model.slug} to qualification/memory-inputs.json"
            )
        if not any(
            item.sha256 == geometry.configuration_sha256
            and "configuration" in item.roles
            for item in model.files
        ):
            raise ValueError(
                f"{label}: cache geometry does not match the pinned model configuration"
            )
        if geometry.sliding_layers > geometry.layers or (
            geometry.sliding_layers and not geometry.sliding_window
        ):
            raise ValueError(f"{label}: invalid sliding-window cache geometry")
        if (
            geometry.runtime_image_digest is not None
            and recipe.execution.build.base_image.digest
            != geometry.runtime_image_digest
        ):
            raise ValueError(f"{label}: cache geometry does not match the pinned image")
        if (
            geometry.resident_expansion_bytes
            and argument_value(
                recipe,
                "disable-cuda-graph"
                if recipe.runtime.engine == "sglang"
                else "enforce-eager",
            )
            is not True
        ):
            raise ValueError(
                f"{label}: calibrated resident bound requires disabled graphs"
            )
        expansion += geometry.resident_expansion_bytes
        token_layers = context * (
            (geometry.layers - geometry.sliding_layers + pipeline - 1) // pipeline
        ) + min(context, geometry.sliding_window) * (
            (geometry.sliding_layers + pipeline - 1) // pipeline
        )
        token_need = (
            token_layers
            * concurrency
            * ((geometry.kv_heads + tensor - 1) // tensor)
            * (geometry.key_dim + geometry.value_dim)
            * element_bits
            + 7
        ) // 8
        rank_tokens = (
            context
            * concurrency
            * (
                (
                    geometry.token_bytes
                    * ((geometry.layers + pipeline - 1) // pipeline)
                    + geometry.layers
                    - 1
                )
                // geometry.layers
            )
            if geometry.token_bytes is not None
            else token_need
        )
        # MLA/token bounds are conservatively replicated across tensor ranks;
        # recurrent state and resident expansion are never divided away.
        kv += rank_tokens
        slots = concurrency
        if geometry.state_bytes_per_request and recipe.runtime.engine == "sglang":
            slots = positive_argument(recipe, "max-mamba-cache-size")
            if slots < concurrency:
                raise ValueError(
                    f"{label}: absolute state pool cannot hold concurrency"
                )
        state += geometry.state_bytes_per_request * (slots + geometry.state_padding)
    # Selected draft weights/cache do not replace the measured outside-budget
    # floor: no replacement calibration exists for the 2026-10-13 Spark freeze.
    if derived_draft:
        assert primary_geometry is not None
        bundled_bytes = 0
        if primary_geometry.bundled_draft_prefix is not None:
            selection = recipe.models[0]
            selected = {item.file_id for item in selection.files}
            bundled_bytes = sum(
                item.size_bytes
                for item in models[selection.model.content_sha256].files
                if item.id in selected
                and "weights" in item.roles
                and item.path.startswith(primary_geometry.bundled_draft_prefix)
            )
            if not bundled_bytes:
                raise ValueError(f"{label}: include the bundled draft weights")
            weights[0] -= bundled_bytes
        weights.append(primary_geometry.derived_draft_weight_bytes + bundled_bytes)
        kv *= 2
        state *= 2
    overhead = OVERHEAD_BYTES
    if loads_draft_model(recipe):
        overhead += DRAFT_MODEL_OVERHEAD_BYTES
    return MemoryNeed(
        (weights[0] + shards - 1) // shards,
        sum(weights[1:]),
        kv,
        state,
        resident_expansion_bytes=expansion,
        overhead_bytes=overhead,
    )


def memory_variants(recipe: RecipeDefinition) -> list[RecipeDefinition]:
    """Base launch and the joint option product; independent choices can overflow."""
    variants = [recipe]
    if recipe.options:
        variants.extend(
            recipe.with_option_choices(
                {
                    option.name: choice.value
                    for option, choice in zip(recipe.options, choices, strict=True)
                }
            )
            for choices in product(*(option.choices for option in recipe.options))
        )
    return variants


def require_memory_need(
    recipe: RecipeDefinition,
    models: Mapping[str, ModelDefinition],
    inputs: MemoryInputs,
    *,
    document_digest: str,
) -> MemoryNeed | None:
    """Check every unproven vLLM/SGLang document, including existing catalog rows."""
    if recipe.runtime.engine not in {"vllm", "sglang"}:
        return None
    key = f"{recipe.identity.publisher}/{recipe.identity.slug}"
    proven = inputs.hardware_passes.get(key)
    if proven is not None and document_digest == proven.catalog_content_sha256:
        return None
    result = None
    for variant in memory_variants(recipe):
        need = calculate_need(variant, models, inputs.geometries)
        if (
            need.total_bytes > need.limit_bytes
            and "does-not-fit" not in recipe.metadata.tags
        ):
            raise ValueError(
                f"{key}: memory need {need.total_bytes} bytes > limit {need.limit_bytes} bytes "
                f"(weights={need.weights_bytes}, draft/auxiliary={need.auxiliary_weights_bytes}, "
                f"KV={need.kv_bytes}, state={need.state_bytes}, "
                f"resident={need.resident_expansion_bytes}, overhead={need.overhead_bytes}); "
                "reduce declared context/concurrency, select a smaller quantization, "
                "or author a multi-Spark recipe id"
            )
        if variant.runtime.engine == "vllm":
            pool = argument_value(variant, "kv-cache-memory-bytes")
            if pool is None or cache_bytes(pool) != need.kv_bytes + need.state_bytes:
                raise ValueError(
                    f"{key}: set absolute kv-cache-memory-bytes={need.kv_bytes + need.state_bytes}; a utilization fraction does not bound KV by need"
                )
        else:
            fraction = argument_value(variant, "mem-fraction-static")
            if fraction is not None:
                static_bytes = Decimal(str(fraction)) * TOTAL_BYTES
                pool_need = need.total_bytes - need.overhead_bytes
                if (
                    not static_bytes.is_finite()
                    or not 0 < Decimal(str(fraction)) <= 1
                    or (
                        static_bytes < pool_need
                        and need.total_bytes <= need.limit_bytes
                    )
                ):
                    raise ValueError(
                        f"{key}: static memory budget cannot hold weights and absolute cache ({pool_need} bytes); increase mem-fraction-static while retaining the absolute token cap"
                    )
            tokens = positive_argument(variant, "context-length") * positive_argument(
                variant, "max-running-requests"
            )
            if (
                argument_value(variant, "max-total-tokens") is None
                or positive_argument(variant, "max-total-tokens") != tokens
            ):
                raise ValueError(
                    f"{key}: set absolute max-total-tokens={tokens}; a static fraction does not bound KV by need"
                )
        for role in variant.topology.roles:
            if (
                variant.topology.node_count > 1
                and role.resources.memory.reserve_bytes < GUARD_BYTES
            ):
                raise ValueError(
                    f"{key}/{role.name}: reserve understates host headroom {GUARD_BYTES}"
                )
            if role.resources.memory.peak_bytes < need.total_bytes:
                raise ValueError(
                    f"{key}/{role.name}: peak understates need {need.total_bytes}"
                )
            if role.resources.memory.reserve_bytes > role.resources.memory.peak_bytes:
                raise ValueError(f"{key}/{role.name}: reserve exceeds peak")
        if result is None or need.total_bytes > result.total_bytes:
            result = need
    return result


@dataclass(frozen=True)
class PerRankMemoryNeed:
    peak_bytes: int
    safe_limit_bytes: int
    node_count: int

    @property
    def fits(self) -> bool:
        return self.peak_bytes <= self.safe_limit_bytes

    @property
    def verdict(self) -> str:
        return f"{'fits' if self.fits else 'does not fit'} {self.node_count} Sparks"


def per_rank_need(
    recipe: RecipeDefinition, models: Mapping[str, ModelDefinition]
) -> PerRankMemoryNeed:
    """Lower bound, not a measurement: engine budget and selected weight/KV floor.

    Dividing all weights evenly is deliberately optimistic: replicated tensors,
    quantization expansion and startup peaks can only increase the loading floor.
    A separate vLLM drafter adds the calibrated 17 GB outside the GPU budget.
    """
    if recipe.runtime.engine not in {"vllm", "sglang"}:
        raise ValueError("memory need requires a vLLM/SGLang recipe")
    fraction_flag = (
        "gpu-memory-utilization"
        if recipe.runtime.engine == "vllm"
        else "mem-fraction-static"
    )
    utilization = VLLM_DEFAULT_UTILIZATION
    kv_bytes = 0
    for argument in recipe.runtime.arguments:
        if argument.name == fraction_flag:
            if argument.setting is not None:
                raise ValueError("memory utilization must be a fixed budget")
            utilization = Decimal(str(argument.value))
        elif argument.name == "kv-cache-memory-bytes":
            if argument.setting is not None:
                raise ValueError("KV bytes must be a fixed budget")
            kv_bytes = cache_bytes(argument.value)
    if not utilization.is_finite() or not 0 < utilization <= 1:
        raise ValueError("memory utilization must be in (0, 1]")
    need = calculate_need(recipe, models, load_inputs().geometries)
    # Absolute vLLM pools supersede utilization; SGLang still profiles its
    # static fraction before applying the bounded token/state pools.
    budget = utilization * SPARK_MEMORY_BYTES + need.overhead_bytes
    if kv_bytes or argument_value(recipe, "max-total-tokens") is not None:
        budget = Decimal(need.total_bytes)
    peak = max(int(budget.to_integral_value(rounding=ROUND_CEILING)), need.total_bytes)
    return PerRankMemoryNeed(peak, LIMIT_BYTES, recipe.topology.node_count)
