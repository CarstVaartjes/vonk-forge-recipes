"""Offline memory demand, separate from historical engine reservation fractions.

Historical catalog documents retain their behavior. A new document must supply
source-backed cache geometry and an absolute pool sized to its declared workload.
The historical set is an identity boundary, not a numeric memory measurement.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from itertools import product
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from vonk_forge_contracts import ModelDefinition, RecipeDefinition

from qualification.vllm_memory_fit import (
    DRAFT_MODEL_OVERHEAD_BYTES,
    OVERHEAD_BYTES,
    PENDING_CREATOR,
    SPARK_3542_IDLE_AVAILABLE_BYTES,
    SPARK_MEMORY_BYTES,
    assert_memory_fit,
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


class HardwarePass(BaseModel):
    model_config = ConfigDict(extra="forbid")
    recorded_at: str
    content_sha256: str
    library_commit: str
    recipe_revision_id: str


class MemoryInputs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hardware_passes: dict[str, HardwarePass]
    historical_sources: list[str]
    historical_documents: dict[str, list[str]]
    failed_recipes: list[str]
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
    overhead_bytes: int = OVERHEAD_BYTES
    limit_bytes: int = LIMIT_BYTES

    @property
    def total_bytes(self) -> int:
        return (
            self.weights_bytes
            + self.auxiliary_weights_bytes
            + self.kv_bytes
            + self.state_bytes
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
    return (
        configured_draft_model(recipe)
        or argument_value(recipe, "speculative-draft-model-path") is not None
    )


def calculate_need(
    recipe: RecipeDefinition,
    models: Mapping[str, ModelDefinition],
    geometries: Mapping[str, CacheGeometry],
) -> MemoryNeed:
    """Full declared context times engine concurrency, including selected drafts.

    Currently supports single-node ordinary attention and supplied compressed
    geometry. A new topology/automatic scheduler needs explicit inputs rather
    than borrowing a benchmark concurrency or a fraction of physical memory.
    """
    label = recipe.identity.slug
    if recipe.topology.node_count != 1:
        raise ValueError(
            f"{label}: supply a per-rank need model for this multi-Spark recipe id"
        )
    settings = recipe.settings
    if settings.kind != "generation" or settings.concurrency is None:
        raise ValueError(
            f"{label}: declare engine context and concurrency for memory sizing"
        )
    if loads_draft_model(recipe) and len(recipe.models) == 1:
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
    else:
        raise ValueError(f"{label}: supply cache layout calibration for dtype {dtype}")
    weights: list[int] = []
    kv = state = 0
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
        token_layers = (
            context * (geometry.layers - geometry.sliding_layers)
            + min(context, geometry.sliding_window) * geometry.sliding_layers
        )
        kv += (
            token_layers
            * concurrency
            * geometry.kv_heads
            * (geometry.key_dim + geometry.value_dim)
            * element_bits
            + 7
        ) // 8
        state += geometry.state_bytes_per_request * concurrency
    # Selected draft weights/cache do not replace the measured outside-budget
    # floor: no replacement calibration exists for the 2026-10-13 Spark freeze.
    overhead = OVERHEAD_BYTES
    if loads_draft_model(recipe):
        overhead += DRAFT_MODEL_OVERHEAD_BYTES
    return MemoryNeed(weights[0], sum(weights[1:]), kv, state, overhead_bytes=overhead)


def require_memory_need(
    recipe: RecipeDefinition,
    models: Mapping[str, ModelDefinition],
    inputs: MemoryInputs,
    *,
    document_digest: str,
) -> MemoryNeed | None:
    """Check new/changed vLLM/SGLang documents without retuning historical ones."""
    if recipe.runtime.engine not in {"vllm", "sglang"}:
        return None
    if recipe.runtime.engine == "vllm" and recipe.topology.node_count != 1:
        # The absolute-need calculator is single-node. Delegate every multi-Spark
        # document (including historical identities) to the calibrated per-role
        # guard, never divide an aggregate reservation by node_count.
        variants = product(*(option.choices for option in recipe.options))
        for choices in ((), *variants):
            variant = (
                recipe.with_option_choices(
                    {
                        option.name: choice.value
                        for option, choice in zip(recipe.options, choices)
                    }
                )
                if choices
                else recipe
            )
            try:
                assert_memory_fit(
                    variant,
                    count_draft_model=recipe.identity.slug not in PENDING_CREATOR,
                )
            except AssertionError as exc:
                raise ValueError(str(exc)) from exc
        return None
    key = f"{recipe.identity.publisher}/{recipe.identity.slug}"
    if (
        key not in inputs.failed_recipes
        and document_digest in inputs.historical_documents.get(key, [])
    ):
        return None
    # Check the joint product: two individually fitting choices can exceed RAM.
    variants = product(*(option.choices for option in recipe.options))
    result = None
    for choices in variants:
        variant = (
            recipe.with_option_choices(
                {
                    option.name: choice.value
                    for option, choice in zip(recipe.options, choices, strict=True)
                }
            )
            if choices
            else recipe
        )
        need = calculate_need(variant, models, inputs.geometries)
        if need.total_bytes > need.limit_bytes:
            raise ValueError(
                f"{key}: memory need {need.total_bytes} bytes > limit {need.limit_bytes} bytes "
                f"(weights={need.weights_bytes}, draft/auxiliary={need.auxiliary_weights_bytes}, "
                f"KV={need.kv_bytes}, state={need.state_bytes}, overhead={need.overhead_bytes}); "
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
                if not static_bytes.is_finite() or static_bytes < pool_need:
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
            if role.resources.memory.peak_bytes < need.total_bytes:
                raise ValueError(
                    f"{key}/{role.name}: peak_bytes must cover need {need.total_bytes}"
                )
        result = need
    return result
