"""Offline memory demand, separate from historical engine reservation fractions.

Unchanged catalog documents retain their behavior. A new document must supply
source-backed cache geometry and an absolute pool sized to its declared workload.
The committed content snapshot is not a qualification status list.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from itertools import product
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from vonk_forge_contracts import ModelDefinition, RecipeDefinition, document_sha256

# Offline policy pinned to vonk-forge cee017ae806f0ac4c0842a3652f8ef7ee57388b8:
# control/src/vonk_control/resource_planning/types.py::PLATFORM_MEMORY_FLOOR_BYTES
# and rust/crates/vonk-agent-protocol/src/host_memory_guard_policy.rs::STALL_AVAILABLE_BYTES.
# These estimates do not redefine live admission or the sustained-pressure guard.
OS_FLOOR_BYTES = 6_000_000_000
# Historical GB10 total, not a substitute for available live node telemetry.
SPARK_MEMORY_BYTES = 130_663_231_488
SPARK_IDLE_AVAILABLE_BYTES = 126_600_000_000
SPARK_3542_IDLE_AVAILABLE_BYTES = 125_650_657_280
OVERHEAD_BYTES = 8_000_000_000
# Separate speculative-model calibration; see vllm_memory_fit.py for the incident.
DRAFT_MODEL_OVERHEAD_BYTES = 17_000_000_000

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
    runtime_image_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class MemoryInputs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    geometries: dict[str, CacheGeometry]
    historical_sources: list[str]
    historical_documents: dict[str, list[str]]


def unchanged_recipe(
    recipe: RecipeDefinition, document_digest: str, inputs: MemoryInputs
) -> bool:
    """Recognize exact content in the committed snapshot or qualification evidence."""
    from spark_sweep.qualification import read_evidence

    if not document_digest:
        return False
    key = f"{recipe.identity.publisher}/{recipe.identity.slug}"
    known = document_digest in inputs.historical_documents.get(key, []) or any(
        record.identity.content_sha256 == document_digest
        for record in read_evidence(ROOT, key).records
    )
    if not known:
        return False
    # Bind the caller's digest AND parsed recipe to the authored bytes. An old
    # digest cannot waive an edited file or an in-memory launch mutation.
    document = json.loads(
        (ROOT / "recipes" / f"{recipe.identity.slug}.json").read_text()
    )
    return (
        document_sha256(document) == document_digest
        and RecipeDefinition.model_validate(document) == recipe
    )


def hardware_proven(recipe: RecipeDefinition, document_digest: str) -> bool:
    """Use the qualification reducer, including later failures and exact reverts."""
    from vonk_forge_contracts import QualificationStatus, derive_status

    from spark_sweep.qualification import read_evidence

    key = f"{recipe.identity.publisher}/{recipe.identity.slug}"
    evidence = read_evidence(ROOT, key)
    for record in evidence.records:
        if record.identity.content_sha256 != document_digest:
            continue
        identity = record.identity.model_copy(
            update={"required_sparks": recipe.topology.node_count}
        )
        if (
            derive_status(evidence, identity, recipe.topology.node_count, False).status
            == QualificationStatus.WORKING
        ):
            return True
    return False


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
    from qualification.vllm_memory_fit import (
        loads_draft_model as configured_draft_model,
    )

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
            and argument_value(recipe, "disable-cuda-graph") is not True
        ):
            raise ValueError(
                f"{label}: calibrated resident bound requires disabled graphs"
            )
        expansion += geometry.resident_expansion_bytes
        token_layers = (
            context * (geometry.layers - geometry.sliding_layers)
            + min(context, geometry.sliding_window) * geometry.sliding_layers
        )
        token_need = (
            token_layers
            * concurrency
            * geometry.kv_heads
            * (geometry.key_dim + geometry.value_dim)
            * element_bits
            + 7
        ) // 8
        kv += (
            context * concurrency * geometry.token_bytes
            if geometry.token_bytes is not None
            else token_need
        )
        slots = concurrency
        if geometry.state_padding:
            slots = positive_argument(recipe, "max-mamba-cache-size")
            if slots < concurrency:
                raise ValueError(
                    f"{label}: absolute state pool cannot hold concurrency"
                )
        state += geometry.state_bytes_per_request * (slots + geometry.state_padding)
    # Selected draft weights/cache do not replace the measured outside-budget
    # floor: no replacement calibration exists for the 2026-10-13 Spark freeze.
    overhead = OVERHEAD_BYTES
    if loads_draft_model(recipe):
        overhead += DRAFT_MODEL_OVERHEAD_BYTES
    return MemoryNeed(
        weights[0],
        sum(weights[1:]),
        kv,
        state,
        resident_expansion_bytes=expansion,
        overhead_bytes=overhead,
    )


def memory_variants(recipe: RecipeDefinition) -> list[RecipeDefinition]:
    """Base launch and the joint option product."""
    return (
        [
            recipe,
            *(
                recipe.with_option_choices(
                    {
                        option.name: choice.value
                        for option, choice in zip(recipe.options, choices, strict=True)
                    }
                )
                for choices in product(*(option.choices for option in recipe.options))
            ),
        ]
        if recipe.options
        else [recipe]
    )


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
    key = f"{recipe.identity.publisher}/{recipe.identity.slug}"
    if unchanged_recipe(recipe, document_digest, inputs):
        return None
    if recipe.topology.node_count != 1:
        for variant in memory_variants(recipe):
            need = per_rank_need(variant, models)
            problems = declaration_problems(variant, need)
            if problems:
                raise ValueError(f"{key}: " + "; ".join(problems))
            if all(
                need.peak_bytes + role.resources.memory.reserve_bytes
                <= IDLE_AVAILABLE_BYTES
                for role in variant.topology.roles
            ):
                require_admissible_declaration(variant)
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
                if not static_bytes.is_finite() or (
                    static_bytes < pool_need and need.total_bytes <= need.limit_bytes
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
            if role.resources.memory.peak_bytes < need.total_bytes:
                raise ValueError(
                    f"{key}/{role.name}: peak_bytes must cover need {need.total_bytes}"
                )
        if need.total_bytes <= need.limit_bytes:
            require_admissible_declaration(variant)
        result = need
    return result


def memory_warnings(
    recipe: RecipeDefinition,
    models: Mapping[str, ModelDefinition],
    inputs: MemoryInputs,
) -> list[str]:
    """Informative catalog estimates; these never change hardware qualification."""
    if recipe.runtime.engine not in {"vllm", "sglang"}:
        return []
    missing = {
        selection.model.content_sha256 for selection in recipe.models
    } - models.keys()
    if missing:
        return [
            (
                f"WARNING {recipe.identity.slug}: memory estimate unavailable; "
                f"missing Model inputs: {', '.join(sorted(missing))}"
            )
        ]
    warnings = []
    for index, variant in enumerate(memory_variants(recipe)):
        prefix = f"WARNING {recipe.identity.slug} [{index}]"
        try:
            need = calculate_need(variant, models, inputs.geometries)
        except (ValueError, TypeError) as error:
            warnings.append(f"{prefix}: full-workload estimate unavailable: {error}")
        else:
            warnings.append(
                f"{prefix}: estimated workload {need.total_bytes} bytes/rank; "
                f"planning limit {LIMIT_BYTES}; hardware test required"
            )
        rank = per_rank_need(variant, models)
        warnings.append(
            f"{prefix}: loading floor {rank.peak_bytes} bytes/rank; "
            f"observed idle capacity {rank.safe_limit_bytes}; no hardware verdict"
        )
    return warnings


def selected_weight_bytes(
    recipe: RecipeDefinition, models: Mapping[str, ModelDefinition]
) -> int:
    weights: dict[str, int] = {}
    for selection in recipe.models:
        selected = {item.file_id for item in selection.files}
        for item in models[selection.model.content_sha256].files:
            if item.id in selected and "weights" in item.roles:
                weights[item.sha256] = item.size_bytes
    return sum(weights.values())


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
    """Optimistic loading floor, never a proof of full-context or runtime fit.

    Selected weights and explicit KV pools are counted once. Adaptive pool
    ceilings and the host pressure planning margin are warnings, not proof that
    a workload cannot run. See the calibration sources in recipe-authoring.md.
    """
    if recipe.runtime.engine not in {"vllm", "sglang"}:
        raise ValueError("memory need requires a vLLM/SGLang recipe")
    kv_bytes = 0
    pool = argument_value(recipe, "kv-cache-memory-bytes")
    if pool is None:
        pool = argument_value(recipe, "kv-cache-memory")
    if pool is not None:
        kv_bytes = cache_bytes(pool)
    parallel = recipe.topology.parallelism
    shards = parallel.tensor * parallel.pipeline
    loading_floor = (
        (selected_weight_bytes(recipe, models) + shards - 1) // shards
        + kv_bytes
        + OVERHEAD_BYTES
    )
    # Utilization sizes an adaptive pool; it is not irreducible workload demand.
    # The measured 17 GB term is relative to that utilization budget, not another
    # allocation beside explicitly counted draft weights. Keep it in the strict
    # workload/legacy envelope guards, not this optimistic loading floor.
    # Physical fit uses observed idle availability. The 16 GiB planning guard
    # remains LIMIT_BYTES and is reported as a warning, not a hardware verdict.
    return PerRankMemoryNeed(
        loading_floor, IDLE_AVAILABLE_BYTES, recipe.topology.node_count
    )


def declaration_problems(
    recipe: RecipeDefinition, need: PerRankMemoryNeed
) -> list[str]:
    """An oversized estimate is valid; hiding demand or headroom is not."""
    problems: list[str] = []
    fraction = argument_value(recipe, "mem-fraction-static")
    static_peak = (
        Decimal(str(fraction)) * TOTAL_BYTES + OVERHEAD_BYTES
        if recipe.runtime.engine == "sglang" and fraction is not None
        else None
    )
    for role in recipe.topology.roles:
        memory = role.resources.memory
        if memory.peak_bytes < need.peak_bytes:
            problems.append(f"{role.name}: peak understates need {need.peak_bytes}")
        if memory.reserve_bytes < OS_FLOOR_BYTES:
            problems.append(
                f"{role.name}: reserve understates host headroom {OS_FLOOR_BYTES}"
            )
        if static_peak is not None and (
            not static_peak.is_finite() or static_peak > memory.peak_bytes
        ):
            problems.append(
                f"{role.name}: peak understates SGLang static allocation plus overhead"
            )
    return problems


def require_admissible_declaration(recipe: RecipeDefinition) -> None:
    """Check the declared reservation, independently of estimated workload fit.

    The 2026-10-21 deferral observed 4,515,061,760 host bytes. The owner's
    lower historical idle availability includes 5,012,574,208 host bytes,
    so use that existing, more conservative baseline without another constant.
    This is an offline declaration check, not a hardware qualification verdict.
    """
    for role in recipe.topology.roles:
        memory = role.resources.memory
        demand = memory.peak_bytes + max(memory.reserve_bytes, OS_FLOOR_BYTES)
        if demand > IDLE_AVAILABLE_BYTES:
            raise ValueError(
                f"{recipe.identity.slug}/{role.name}: declared peak plus reserve "
                f"{demand} bytes exceeds idle Spark capacity {IDLE_AVAILABLE_BYTES} "
                f"bytes (pool={TOTAL_BYTES}, host={IDLE_USE_BYTES}); "
                "reserve by need and bound the engine allocation"
            )
