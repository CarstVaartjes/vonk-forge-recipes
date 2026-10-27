"""Offline memory demand, separate from historical engine reservation fractions.

Unchanged catalog documents retain their behavior. A new document's fixed pool
must hold one full-context request by source-backed cache geometry, and its
declaration must cover the loading floor and be admissible.
The committed content snapshot is not a qualification status list.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal
from enum import StrEnum
from functools import cache
from itertools import product
from math import lcm
from pathlib import Path
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator
from vonk_forge_contracts import ModelDefinition, RecipeDefinition, document_sha256
from vonk_forge_contracts.memory_admission import (
    PLATFORM_MEMORY_FLOOR_BYTES,
    evaluate_memory_admission,
)

# Offline policy pinned to vonk-forge cee017ae806f0ac4c0842a3652f8ef7ee57388b8:
# control/src/vonk_control/resource_planning/types.py::PLATFORM_MEMORY_FLOOR_BYTES
# and rust/crates/vonk-agent-protocol/src/host_memory_guard_policy.rs::STALL_AVAILABLE_BYTES.
# These estimates do not redefine live admission or the sustained-pressure guard.
OS_FLOOR_BYTES = PLATFORM_MEMORY_FLOOR_BYTES
# Historical GB10 total, not a substitute for available live node telemetry.
SPARK_MEMORY_BYTES = 130_663_231_488
SPARK_IDLE_AVAILABLE_BYTES = 126_600_000_000
SPARK_3542_IDLE_AVAILABLE_BYTES = 125_650_657_280
# Idle host use for offline declaration admission, calibrated on the proven
# set: the live Controller admitted and the sweep passed 120,370,379,081-byte
# peaks (DeepSeek V4.1 Flash EXL3 Mia dual, Gemma 4 31B Mia single, Qwen3.6
# 35B r0b0tlab single), so host use was at most 4,292,852,407 bytes there.
ADMISSION_IDLE_USE_BYTES = SPARK_MEMORY_BYTES - SPARK_IDLE_AVAILABLE_BYTES
OVERHEAD_BYTES = 8_000_000_000
# sglang 94602c9c (and the Qwen3.8 Flash Next r0b0tlab image's qwen4_exp copy)
# kv_cache_configurator._calculate_mamba_ratio: at most 3 base plus 2
# extra_buffer overlap state slots per running request.
SGLANG_MAMBA_SLOTS_PER_REQUEST = 5
# The DeepSeek V4.1 Engram row store's resident cache size (GiB per Spark).
ENGRAM_ROW_CACHE_ENVIRONMENT = "DSV41_CACHE_GIB"
# Separate speculative-model calibration; see vllm_memory_fit.py for the incident.
DRAFT_MODEL_OVERHEAD_BYTES = 17_000_000_000
# vLLM's gpu-memory-utilization when a launch omits it.
VLLM_DEFAULT_UTILIZATION = Decimal("0.90")

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


class MambaCacheMode(StrEnum):
    """vLLM's --mamba-cache-mode values (vllm/config/cache.py)."""

    NONE = "none"
    ALIGN = "align"
    ALL = "all"


# vLLM --mamba-cache-dtype / --mamba-ssm-cache-dtype element sizes.
MAMBA_DTYPE_BYTES = {"float32": 4, "float16": 2, "bfloat16": 2}


# Pinned-config architectures (config.json "architectures"[0]; "params.json"
# for Mistral-native checkpoints), classified from the vLLM model class they
# load. Hybrid classes hold Mamba/GDN/KDA/short-conv state that vLLM pads and
# groups with the attention pages; a geometry with an unclassified
# architecture is rejected.
HYBRID_ARCHITECTURES = frozenset(
    {
        "Glm5NextForConditionalGeneration",  # KDA linear + DSA layers
        "InklingForConditionalGeneration",  # use_sconv short convolution
        "NemotronHForCausalLM",
        "NemotronH_Nano_Omni_Reasoning_V3",
        "NemotronH_Omni_Reasoning_V3",
        "Qwen3_5ForConditionalGeneration",  # GDN linear attention
        "Qwen4ExpForConditionalGeneration",  # GDN linear attention
    }
)
ATTENTION_ARCHITECTURES = frozenset(
    {
        "DeepseekV41ForCausalLM",
        "DeepseekV4ForCausalLM",
        "Glm5vForConditionalGeneration",
        "GlmMoeDsaForCausalLM",
        "GptOssForCausalLM",
        "HYV3ForCausalLM",
        "LagunaForCausalLM",
        "LlamaForCausalLM",
        "MiMoV2ForCausalLM",
        "MiniMaxM2ForCausalLM",
        "Step3p7ForConditionalGeneration",
        "Step5ForConditionalGeneration",
        "params.json",
    }
)
# Classes vLLM loads that implement SupportsMambaPrefixCaching, so prefix
# caching defaults to mode "all": NemotronHForCausalLM in vLLM v0.20.0
# (nemotron_h.py:765-773), v0.27.0 and v0.30.0 (nemotron_h.py:711-721). The
# Omni/VL architectures load NemotronH_Nano_VL_V2 (registry.py, v0.20.0 to
# nightly 3f1bcfd8), which does not, so they run "align"; nightly 3f1bcfd8
# defaults every hybrid to "align" (models/config.py:711-713).
MAMBA_PREFIX_CACHING_ARCHITECTURES = frozenset({"NemotronHForCausalLM"})
# vLLM-pinned hybrid geometries whose layout vllm_hybrid_need cannot express,
# judged by their existing attention and hand bounds instead.
VLLM_UNMODELED_HYBRID = {
    # vLLM v0.30.0 Qwen4Exp adds a second tp_replicated PLE MambaSpec, a QSA
    # CircularBufferSpec ring, an MLA compressed-key cache and an MTP QSA
    # layer; its sfxnz dual is proven (9f77f966) on this hand bound.
    "qwen3-8-flash-next-nvfp4-fab0aecb": "Qwen4Exp on vLLM v0.30.0",
    # Sliding-window plus short-convolution layers on creator images whose
    # vLLM Inkling layout is not pinned; SGLang-measured state.
    "inkling-small-nvfp4": "Inkling short convolution on creator images",
    # KDA linear + DSA layers on creator vLLM forks: not modeled (follow-up).
    "glm-5-3-flash-exl3-tr3-4bpw-brandonmusic-1ae6d704": "Glm5Next KDA + DSA not modeled (follow-up)",
    "glm-5-3-flash-nvfp4-caca4e6a": "Glm5Next KDA + DSA not modeled (follow-up)",
    "glm-5-3-flash-nvfp4-spark-a6082410": "Glm5Next KDA + DSA not modeled (follow-up)",
}


class RecurrentCache(BaseModel):
    """Mamba/GDN layer state from the pinned configuration, before TP sharding.

    vLLM v0.27 mamba_utils.MambaStateShapeCalculator: conv state is
    conv_channels / tp x (conv_kernel - 1 + speculative tokens) in the model
    dtype (BF16) unless --mamba-cache-dtype; temporal state is
    state_elements / tp in state_dtype_bytes unless --mamba-ssm-cache-dtype.
    """

    model_config = ConfigDict(extra="forbid")
    layers: int = Field(gt=0)
    # Full-attention layers vLLM groups with them (the geometry's `layers`
    # may over-bound attention for other engines).
    attention_layers: int = Field(gt=0)
    conv_channels: int = Field(gt=0)
    conv_kernel: int = Field(gt=1)
    state_elements: int = Field(gt=0)
    state_dtype_bytes: int = Field(gt=0)
    # Smallest attention kernel block of the pinned image's backend.
    kernel_block_tokens: int = Field(default=16, gt=0)
    # Whether the loaded vLLM class supports prefix-caching mode "all"
    # (MAMBA_PREFIX_CACHING_ARCHITECTURES), and its Mamba chunk size.
    mamba_prefix_caching: bool
    chunk_tokens: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def chunk_for_all_mode(self) -> Self:
        if self.mamba_prefix_caching != (self.chunk_tokens is not None):
            raise ValueError('mode "all" needs the Mamba chunk size, and only it')
        return self


class CacheGeometry(BaseModel):
    """Pinned configuration dimensions; never guessed from the model's name."""

    model_config = ConfigDict(extra="forbid")
    configuration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source: str
    max_context_tokens: int | None = Field(default=None, gt=0)
    tensor_replicated: bool = False
    replicated_token_bytes: int = Field(default=0, ge=0)
    token_bytes_by_dtype: dict[str, Annotated[int, Field(gt=0)]] = Field(
        default_factory=dict
    )
    layers: int = Field(gt=0)
    kv_heads: int = Field(gt=0)
    key_dim: int = Field(gt=0)
    value_dim: int = Field(ge=0)
    sliding_layers: int = Field(default=0, ge=0)
    sliding_window: int = Field(default=0, ge=0)
    architecture: str
    # Mamba/GDN layers on vLLM: padded, grouped and null-blocked
    # (vllm_hybrid_need). Required for a vLLM-pinned hybrid architecture
    # outside VLLM_UNMODELED_HYBRID.
    recurrent: RecurrentCache | None = None
    # Hand-entered extra state counted once per request (sliding windows,
    # compressed or replay caches, SGLang slots, other engines' recurrent
    # state). vLLM ignores it where `recurrent` is set.
    state_bytes_per_request: int = Field(default=0, ge=0)
    # Compressed/hybrid layouts may use a conservative packed-token bound.
    token_bytes: int | None = Field(default=None, gt=0)
    state_padding: int = Field(default=0, ge=0)
    resident_expansion_bytes: int = Field(default=0, ge=0)
    # SGLang HybridSWAPoolConfigurator pools these sliding layers, outside
    # `layers`, at swa-full-tokens-ratio of max-total-tokens.
    sglang_swa_layers: int = Field(default=0, ge=0)
    # SGLang per-rank hybrid state per max-mamba-cache-size slot (+1 padding).
    sglang_state_bytes_per_slot: int = Field(default=0, ge=0)
    runtime_image_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def classified_recurrent_bound(self) -> Self:
        """A classified architecture; typed recurrent state only on a hybrid
        one, with plain attention and the loaded class's prefix caching."""
        if self.architecture not in HYBRID_ARCHITECTURES | ATTENTION_ARCHITECTURES:
            raise ValueError(
                f"classify architecture {self.architecture} as hybrid or attention"
            )
        recurrent = self.recurrent
        if recurrent is None:
            return self
        if (
            self.architecture not in HYBRID_ARCHITECTURES
            or recurrent.mamba_prefix_caching
            != (self.architecture in MAMBA_PREFIX_CACHING_ARCHITECTURES)
            or recurrent.attention_layers > self.layers
            or self.state_padding
            or self.sglang_state_bytes_per_slot
            or self.sliding_layers
            or self.token_bytes is not None
            or self.token_bytes_by_dtype
            or self.replicated_token_bytes
        ):
            raise ValueError(
                "recurrent geometry needs a hybrid architecture, its prefix-caching "
                "support and plain attention layers"
            )
        return self


class MemoryInputs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    geometries: dict[str, CacheGeometry]
    # Engram n-gram table files, by content digest, with their source. Owner
    # rule (2026-10-26): Engram tables are never memory usage.
    engram_files: dict[str, str]
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

    if wrapper_loads_draft(
        ROOT / recipe.execution.build.context.path / "vllm-wrapper.sh"
    ):
        return True
    return (
        configured_draft_model(recipe)
        or argument_value(recipe, "speculative-draft-model-path") is not None
    )


@cache
def wrapper_loads_draft(wrapper: Path) -> bool:
    """Read each committed wrapper once per process, not once per check."""
    return wrapper.is_file() and "export SPEC_MODEL_PATH=" in wrapper.read_text()


def calculate_need(
    recipe: RecipeDefinition,
    models: Mapping[str, ModelDefinition],
    inputs: MemoryInputs,
) -> MemoryNeed:
    """Informative full workload: declared context times engine concurrency,
    including selected drafts. Warnings only; require_memory_need gates.

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
    weights: list[int] = []
    kv = state = expansion = 0
    for selection in recipe.models:
        model = models[selection.model.content_sha256]
        selected = {item.file_id for item in selection.files}
        weights.append(
            sum(
                item.size_bytes
                for item in model.files
                if item.id in selected
                and "weights" in item.roles
                and item.sha256 not in inputs.engram_files
            )
        )
        geometry = pinned_geometry(
            recipe, selection.model.slug, model, inputs.geometries
        )
        if geometry.sliding_layers > geometry.layers or (
            geometry.sliding_layers and not geometry.sliding_window
        ):
            raise ValueError(f"{label}: invalid sliding-window cache geometry")
        graph_flag = (
            "enforce-eager" if recipe.runtime.engine == "vllm" else "disable-cuda-graph"
        )
        if (
            geometry.resident_expansion_bytes
            and argument_value(recipe, graph_flag) is not True
        ):
            raise ValueError(
                f"{label}: calibrated resident bound requires disabled graphs"
            )
        expansion += geometry.resident_expansion_bytes
        recurrent = vllm_recurrent(recipe, selection.model.slug, geometry)
        if recurrent is not None:
            kv += vllm_hybrid_need(recipe, geometry, recurrent, context, concurrency)
            continue
        kv += cache_need_for_context(recipe, geometry, context) * concurrency
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


def pinned_geometry(
    recipe: RecipeDefinition,
    slug: str,
    model: ModelDefinition,
    geometries: Mapping[str, CacheGeometry],
) -> CacheGeometry:
    """Geometry bound to the selected configuration and, if pinned, the image."""
    label = recipe.identity.slug
    geometry = geometries.get(slug)
    if geometry is None:
        raise ValueError(
            f"{label}: add source-backed cache geometry for {slug} to qualification/memory-inputs.json"
        )
    if not any(
        item.sha256 == geometry.configuration_sha256 and "configuration" in item.roles
        for item in model.files
    ):
        raise ValueError(
            f"{label}: cache geometry does not match the pinned model configuration"
        )
    if (
        geometry.runtime_image_digest is not None
        and recipe.execution.build.base_image.digest != geometry.runtime_image_digest
    ):
        raise ValueError(f"{label}: cache geometry does not match the pinned image")
    return geometry


def sglang_static_need(
    recipe: RecipeDefinition,
    models: Mapping[str, ModelDefinition],
    inputs: MemoryInputs,
    key: str,
    *,
    pinned: bool,
) -> int:
    """Per-rank bytes SGLang holds inside mem-fraction-static once profiled.

    Weights, the resident expansion of the selected geometry and, when
    max-total-tokens caps the pools, the capped token and state pools.
    ``pinned`` requires geometry for every Model (multi-Spark); otherwise only
    geometry pinned to this launch is charged and an uncapped pool, sized
    from what remains, adds nothing.
    """
    capped = argument_value(recipe, "max-total-tokens") is not None
    tokens = positive_argument(recipe, "max-total-tokens") if capped else 0
    need = rank_weight_bytes(recipe, models, inputs)
    for selection in recipe.models:
        model = models[selection.model.content_sha256]
        geometry = (
            pinned_geometry(recipe, selection.model.slug, model, inputs.geometries)
            if pinned
            else applicable_geometry(recipe, selection.model.slug, model, inputs)
        )
        if geometry is None:
            continue
        need += geometry.resident_expansion_bytes
        if not capped:
            continue
        need += cache_need_for_context(recipe, geometry, tokens)
        if geometry.sglang_swa_layers:
            ratio = argument_value(recipe, "swa-full-tokens-ratio")
            if ratio is None:
                raise ValueError(f"{key}: declare swa-full-tokens-ratio")
            swa = geometry.model_copy(
                update={
                    "layers": geometry.sglang_swa_layers,
                    "sliding_layers": 0,
                    "token_bytes": None,
                    "token_bytes_by_dtype": {},
                    "replicated_token_bytes": 0,
                }
            )
            swa_tokens = int(
                (Decimal(str(ratio)) * tokens).to_integral_value(ROUND_CEILING)
            )
            need += cache_need_for_context(recipe, swa, swa_tokens)
        if geometry.sglang_state_bytes_per_slot:
            if argument_value(recipe, "max-mamba-cache-size") is None:
                raise ValueError(f"{key}: set absolute max-mamba-cache-size")
            slots = positive_argument(recipe, "max-mamba-cache-size")
            if slots < SGLANG_MAMBA_SLOTS_PER_REQUEST:
                raise ValueError(
                    f"{key}: max-mamba-cache-size {slots} cannot hold the "
                    f"{SGLANG_MAMBA_SLOTS_PER_REQUEST} state slots of one request"
                )
            need += geometry.sglang_state_bytes_per_slot * (slots + 1)
            if argument_value(recipe, "speculative-algorithm") is not None:
                # _handle_max_mamba_cache: per-step intermediate state for the
                # running requests the slots admit, plus the padding slot.
                running = min(
                    positive_argument(recipe, "max-running-requests"),
                    slots // SGLANG_MAMBA_SLOTS_PER_REQUEST,
                )
                need += (
                    geometry.sglang_state_bytes_per_slot
                    * (running + 1)
                    * positive_argument(recipe, "speculative-num-draft-tokens")
                )
        if geometry.state_padding:
            # The pinned image's allocator charges max-mamba-cache-size plus
            # its padding slots (glm-a0-cache-sizing.py.txt).
            if argument_value(recipe, "max-mamba-cache-size") is None:
                raise ValueError(f"{key}: set absolute max-mamba-cache-size")
            need += geometry.state_bytes_per_request * (
                positive_argument(recipe, "max-mamba-cache-size")
                + geometry.state_padding
            )
    return need


def require_sglang_rank_budget(
    recipe: RecipeDefinition,
    models: Mapping[str, ModelDefinition],
    inputs: MemoryInputs,
    key: str,
) -> None:
    """Multi-Spark SGLang profiles its pools after the weights are resident.

    sglang 94602c9c kv_cache_configurator._profile_available_bytes charges
    everything resident against mem-fraction-static times pre-load free memory,
    so that budget on the lowest idle Spark must hold the sharded weights, their
    resident expansion and the capped token and state pools.

    The token cap must hold one full-context request
    (require_absolute_kv_capacity), not context x max-running-requests: SGLang
    schedules requests within it. The proven Qwen3.8 Flash Next W4A16 r0b0tlab
    dual caps one 262,144-token context at concurrency 4. The state cap must
    hold one request's slots: at startup kv_cache_configurator
    .resolve_max_num_reqs lowers max-running-requests to max-mamba-cache-size
    // 5 and fails only below one, so 5 slots serve concurrency 1. With
    speculation, _handle_max_mamba_cache adds intermediate state for those
    running requests plus the padding slot, per draft token.
    """
    if argument_value(recipe, "max-total-tokens") is None:
        raise ValueError(
            f"{key}: set absolute max-total-tokens; a static fraction does not bound KV by need"
        )
    fraction = argument_value(recipe, "mem-fraction-static")
    if fraction is None:
        raise ValueError(f"{key}: declare mem-fraction-static for the static budget")
    need = sglang_static_need(recipe, models, inputs, key, pinned=True)
    static_bytes = Decimal(str(fraction)) * IDLE_AVAILABLE_BYTES
    if not static_bytes.is_finite() or static_bytes < need:
        raise ValueError(
            f"{key}: static budget {static_bytes:.0f} bytes (mem-fraction-static x "
            f"idle available {IDLE_AVAILABLE_BYTES}) cannot hold sharded weights, "
            f"resident expansion and capped pools ({need} bytes/rank); increase "
            "mem-fraction-static while retaining the absolute caps"
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


def cache_need_for_context(
    recipe: RecipeDefinition, geometry: CacheGeometry, context: int
) -> int:
    """One cache-layout calculation for single-node and per-rank workload sizing."""
    dtype = str(argument_value(recipe, "kv-cache-dtype") or "auto")
    parallel = recipe.topology.parallelism
    layers, token_bytes = attention_token_bytes(recipe, geometry)
    needed = context * token_bytes
    if not geometry.token_bytes_by_dtype.get(dtype) and geometry.token_bytes is None:
        sliding_layers = (
            geometry.sliding_layers + parallel.pipeline - 1
        ) // parallel.pipeline
        needed -= (
            (context - min(context, geometry.sliding_window))
            * sliding_layers
            * (token_bytes // layers)
        )
    replicated_bytes = (
        geometry.replicated_token_bytes * layers + geometry.layers - 1
    ) // geometry.layers
    return needed + context * replicated_bytes


def attention_token_bytes(
    recipe: RecipeDefinition, geometry: CacheGeometry
) -> tuple[int, int]:
    """Attention layers on the largest pipeline stage and their bytes per token."""
    dtype = str(argument_value(recipe, "kv-cache-dtype") or "auto")
    parallel = recipe.topology.parallelism
    layers = (geometry.layers + parallel.pipeline - 1) // parallel.pipeline
    heads = (
        geometry.kv_heads
        if geometry.tensor_replicated
        else (geometry.kv_heads + parallel.tensor - 1) // parallel.tensor
    )
    if dtype not in geometry.token_bytes_by_dtype and dtype not in {
        "auto",
        "bfloat16",
        "float16",
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
        "nvfp4",
    }:
        raise ValueError(
            f"{recipe.identity.slug}: supply cache layout calibration for dtype {dtype}"
        )
    if dtype in geometry.token_bytes_by_dtype:
        token_bytes = (
            geometry.token_bytes_by_dtype[dtype] * layers + geometry.layers - 1
        ) // geometry.layers
    elif geometry.token_bytes is not None:
        token_bytes = (
            geometry.token_bytes * layers + geometry.layers - 1
        ) // geometry.layers
    elif dtype in ("auto", "bfloat16", "float16"):
        token_bytes = layers * heads * (geometry.key_dim + geometry.value_dim) * 2
    elif dtype in ("fp8", "fp8_e4m3", "fp8_e5m2"):
        token_bytes = layers * heads * (geometry.key_dim + geometry.value_dim)
    elif dtype == "nvfp4":
        # Packed data plus one FP8 scale per 16 elements, separately for K/V.
        token_bytes = (
            layers
            * heads
            * sum(
                (dim + 1) // 2 + (dim + 15) // 16
                for dim in (geometry.key_dim, geometry.value_dim)
            )
        )
    else:
        raise ValueError(f"{recipe.identity.slug}: supply cache layout for {dtype}")
    return layers, token_bytes


def speculative_config(recipe: RecipeDefinition) -> dict[str, object] | None:
    """vLLM's --speculative-config object (vLLM's structure, not ours).

    None when absent or "None" (speculation off); anything else that is not a
    JSON object is a deterministic recipe error.
    """
    config = argument_value(recipe, "speculative-config")
    if config is None or config == "None":
        return None
    if isinstance(config, str):
        try:
            config = json.loads(config)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"{recipe.identity.slug}: speculative-config is not JSON: {error}"
            ) from error
    if not isinstance(config, dict):
        raise TypeError(
            f"{recipe.identity.slug}: speculative-config must be a JSON object"
        )
    return config


def speculative_tokens(recipe: RecipeDefinition) -> int:
    """vLLM num_speculative_tokens; 0 only without speculation."""
    config = speculative_config(recipe)
    if config is None:
        return 0
    tokens = config.get("num_speculative_tokens")
    if isinstance(tokens, bool) or not isinstance(tokens, int | str):
        raise TypeError(
            f"{recipe.identity.slug}: speculative-config needs num_speculative_tokens"
        )
    return int(tokens)


def mamba_dtype_bytes(recipe: RecipeDefinition, flag: str, default: int) -> int:
    value = argument_value(recipe, flag)
    if value is None or value == "auto":
        return default
    if str(value) not in MAMBA_DTYPE_BYTES:
        raise ValueError(
            f"{recipe.identity.slug}: supply state size for {flag} {value}"
        )
    return MAMBA_DTYPE_BYTES[str(value)]


def recurrent_layer_bytes(recipe: RecipeDefinition, recurrent: RecurrentCache) -> int:
    """One request's unpadded state in one recurrent layer on one tensor rank."""
    tensor = recipe.topology.parallelism.tensor
    conv = -(-recurrent.conv_channels // tensor) * (
        recurrent.conv_kernel - 1 + speculative_tokens(recipe)
    )
    return conv * mamba_dtype_bytes(recipe, "mamba-cache-dtype", 2) + -(
        -recurrent.state_elements // tensor
    ) * mamba_dtype_bytes(recipe, "mamba-ssm-cache-dtype", recurrent.state_dtype_bytes)


def mamba_cache_mode(
    recipe: RecipeDefinition, recurrent: RecurrentCache
) -> MambaCacheMode:
    """vLLM MambaModelConfig: prefix caching selects "all" where the loaded
    class supports it, else "align"; off holds one page per recurrent group.

    An omitted flag counts as on: v0.27 keeps hybrid prefix caching opt-in
    (arg_utils.py default_prefix_caching), but nightly 3f1bcfd8 defaults it
    to is_prefix_caching_supported, and "align" never needs fewer blocks.
    """
    if argument_value(recipe, "enable-prefix-caching") is False:
        return MambaCacheMode.NONE
    mode = MambaCacheMode(
        str(argument_value(recipe, "mamba-cache-mode") or MambaCacheMode.NONE)
    )
    if mode is MambaCacheMode.ALIGN or not recurrent.mamba_prefix_caching:
        return MambaCacheMode.ALIGN
    return MambaCacheMode.ALL


def vllm_recurrent(
    recipe: RecipeDefinition, slug: str, geometry: CacheGeometry
) -> RecurrentCache | None:
    """Typed recurrent state a vLLM launch is sized by; a hybrid architecture
    without it is rejected unless VLLM_UNMODELED_HYBRID names the geometry."""
    if recipe.runtime.engine != "vllm":
        return None
    if (
        geometry.recurrent is None
        and geometry.architecture in HYBRID_ARCHITECTURES
        and slug not in VLLM_UNMODELED_HYBRID
    ):
        raise ValueError(
            f"{recipe.identity.slug}: {slug} is a hybrid {geometry.architecture}; "
            "add its typed recurrent geometry"
        )
    return geometry.recurrent


def vllm_hybrid_need(
    recipe: RecipeDefinition,
    geometry: CacheGeometry,
    recurrent: RecurrentCache,
    context: int,
    concurrency: int,
) -> int:
    """Per-rank pool vLLM v0.27's hybrid allocator needs for the requests.

    Platform._align_hybrid_block_size raises the block to the smallest
    kernel-aligned size (mode "all": aligned to the Mamba chunk) whose
    attention page holds one recurrent state, and pads that state to the page.
    _get_kv_cache_groups_uniform_page_size groups layers by the smallest
    bucket (the largest when within 1.5x), padding each type's last group;
    get_num_blocks sizes pool // page // group blocks and BlockPool takes one
    as the null block. Per request an attention group holds cdiv(context,
    block) blocks and a recurrent group (MambaSpec.max_memory_usage_bytes)
    1 (mode none), 2 (align) or cdiv(context, block) (all), plus one per
    speculative token. The startup check (_max_memory_usage_bytes_from_groups)
    counts the same padded groups without the null block, so this covers it.
    """
    stage_layers, token_bytes = attention_token_bytes(recipe, geometry)
    per_token = token_bytes // stage_layers
    state = recurrent_layer_bytes(recipe, recurrent)
    align = recurrent.kernel_block_tokens
    if argument_value(recipe, "block-size") is not None:
        align = max(align, positive_argument(recipe, "block-size"))
    mode = mamba_cache_mode(recipe, recurrent)
    if mode is MambaCacheMode.ALL:
        assert recurrent.chunk_tokens is not None
        chunk = lcm(recurrent.chunk_tokens, align)
        tokens = -(-state // per_token)
        block = chunk * -(-tokens // chunk)
    else:
        block = align * -(-state // (align * per_token))
    stages = recipe.topology.parallelism.pipeline
    attention_layers = -(-recurrent.attention_layers // stages)
    recurrent_layers = -(-recurrent.layers // stages)
    smallest, largest = sorted((attention_layers, recurrent_layers))
    group = largest if 2 * largest < 3 * smallest else smallest
    attention_blocks = -(-context // block)
    recurrent_blocks = (
        speculative_tokens(recipe)
        + {
            MambaCacheMode.NONE: 1,
            MambaCacheMode.ALIGN: 2,
            MambaCacheMode.ALL: attention_blocks,
        }[mode]
    )
    blocks = (
        concurrency
        * (
            -(-attention_layers // group) * attention_blocks
            + -(-recurrent_layers // group) * recurrent_blocks
        )
        + 1
    )
    return blocks * group * block * per_token


def absolute_kv_need(
    recipe: RecipeDefinition,
    models: Mapping[str, ModelDefinition],
    inputs: MemoryInputs,
    *,
    concurrency: int = 1,
) -> int:
    """Per-rank pool for full context, including block rounding and packed scales.

    MLA latents replicate over tensor ranks; ordinary GQA heads shard, with at
    least one head per rank. Geometry is bound to the selected configuration.
    Padded/hybrid layouts use a source-backed conservative allocation bound.
    This is an allocation estimate, not a hardware qualification verdict.
    """
    selection = recipe.models[0]
    model = models[selection.model.content_sha256]
    geometry = pinned_geometry(recipe, selection.model.slug, model, inputs.geometries)
    context_flag = (
        "max-model-len" if recipe.runtime.engine == "vllm" else "context-length"
    )
    context_value = argument_value(recipe, context_flag)
    context = (
        geometry.max_context_tokens
        if context_value in (None, "auto", -1, "-1")
        else positive_argument(recipe, context_flag)
    )
    if context is None:
        raise ValueError(f"{recipe.identity.slug}: declare default model context")
    block_flag = "block-size" if recipe.runtime.engine == "vllm" else "page-size"
    block = (
        positive_argument(recipe, block_flag)
        if argument_value(recipe, block_flag) is not None
        else 16
    )
    context = ((context + block - 1) // block) * block
    recurrent = vllm_recurrent(recipe, selection.model.slug, geometry)
    if recurrent is not None:
        return vllm_hybrid_need(recipe, geometry, recurrent, context, concurrency)
    return (
        cache_need_for_context(recipe, geometry, context)
        + geometry.state_bytes_per_request
    ) * concurrency


def require_absolute_kv_capacity(
    recipe: RecipeDefinition,
    models: Mapping[str, ModelDefinition],
    inputs: MemoryInputs,
) -> None:
    """A fixed pool must hold at least one full-context request, even on TP/PP."""
    if recipe.runtime.engine not in {"vllm", "sglang"}:
        return
    for variant in memory_variants(recipe):
        if variant.runtime.engine == "sglang":
            tokens = argument_value(variant, "max-total-tokens")
            if tokens is not None and positive_argument(
                variant, "max-total-tokens"
            ) < positive_argument(variant, "context-length"):
                raise ValueError(
                    f"{recipe.identity.slug}: absolute KV token pool cannot hold one full-context request"
                )
        elif variant.runtime.engine == "vllm":
            pool = argument_value(variant, "kv-cache-memory-bytes")
            if pool is None:
                pool = argument_value(variant, "kv-cache-memory")
            if pool is not None:
                need = absolute_kv_need(variant, models, inputs)
                if cache_bytes(pool) < need:
                    raise ValueError(
                        f"{recipe.identity.slug}: absolute KV pool {cache_bytes(pool)} bytes cannot hold one full-context request ({need} bytes/rank required)"
                    )


def require_memory_need(
    recipe: RecipeDefinition,
    models: Mapping[str, ModelDefinition],
    inputs: MemoryInputs,
    *,
    document_digest: str,
    touched: bool = False,
) -> None:
    """Check new/changed documents without retuning historical ones.

    Calibrated on the proven set: a fixed pool holds one full-context request,
    not context x concurrency, and the declared peak covers the loading floor
    (and on one Spark a vLLM utilization budget with what vLLM holds beside
    it) and is admissible. ``touched`` marks a recipe file the change under
    review edits; only those must bound a utilization-sized KV pool.
    """
    require_absolute_kv_capacity(recipe, models, inputs)
    # Recorded content is judged by hardware, not by this offline check.
    if unchanged_recipe(recipe, document_digest, inputs):
        return
    require_admissible_declaration(recipe)
    if recipe.runtime.engine not in {"vllm", "sglang"}:
        return
    key = f"{recipe.identity.publisher}/{recipe.identity.slug}"
    single = recipe.topology.node_count == 1
    for variant in memory_variants(recipe):
        need = per_rank_need(variant, models, inputs)
        # Proven single-Spark documents run below the multi-Spark reserve and
        # SGLang static-peak declarations; the loading floor, vLLM's budget and
        # SGLang's weights, expansion and capped pools hold there, because the
        # Controller reserves the declared peak.
        vllm = variant.runtime.engine == "vllm"
        envelope = (
            None
            if not single
            else vllm_utilization_envelope(variant)
            if vllm
            else sglang_static_need(variant, models, inputs, key, pinned=False)
            + OVERHEAD_BYTES
        )
        engine = "vLLM budget" if vllm else "SGLang static need"
        problems = (
            [
                f"{role.name}: peak understates need {need.peak_bytes}"
                for role in variant.topology.roles
                if role.resources.memory.peak_bytes < need.peak_bytes
            ]
            + [
                f"{role.name}: peak understates {engine} plus overhead {envelope}"
                for role in variant.topology.roles
                if envelope is not None and role.resources.memory.peak_bytes < envelope
            ]
            if single
            else declaration_problems(variant, need)
        )
        if problems:
            raise ValueError(f"{key}: " + "; ".join(problems))
        if not single and variant.runtime.engine == "sglang":
            require_sglang_rank_budget(variant, models, inputs, key)
        elif touched:
            require_bounded_kv_pool(variant, models, inputs, key)
        if not single:
            require_vllm_startup_budget(variant)


def vllm_utilization_envelope(recipe: RecipeDefinition) -> int | None:
    """vLLM's utilization budget plus what it holds beside it, if pool-less.

    The 8 GB overhead and the measured 17 GB floor of a separately loaded
    draft (qualification/vllm_memory_fit.py, the 2026-10-13 freeze).
    """
    if recipe.runtime.engine != "vllm" or any(
        argument_value(recipe, name) is not None
        for name in ("kv-cache-memory-bytes", "kv-cache-memory")
    ):
        return None
    fraction = argument_value(recipe, "gpu-memory-utilization")
    share = Decimal(str(fraction)) if fraction is not None else VLLM_DEFAULT_UTILIZATION
    envelope = share * TOTAL_BYTES + OVERHEAD_BYTES
    if loads_draft_model(recipe):
        envelope += DRAFT_MODEL_OVERHEAD_BYTES
    return int(envelope.to_integral_value(ROUND_CEILING))


def applicable_geometry(
    recipe: RecipeDefinition,
    slug: str,
    model: ModelDefinition,
    inputs: MemoryInputs,
) -> CacheGeometry | None:
    """Source-backed geometry pinned to this launch, if any."""
    if slug not in inputs.geometries:
        return None
    try:
        return pinned_geometry(recipe, slug, model, inputs.geometries)
    except ValueError:
        return None  # Bound to another configuration or image.


def require_bounded_kv_pool(
    recipe: RecipeDefinition,
    models: Mapping[str, ModelDefinition],
    inputs: MemoryInputs,
    key: str,
) -> None:
    """A touched recipe bounds KV, calibrated on the proven set.

    A utilization-sized pool grows into what remains: on 2026-10-26 Laguna M.1
    at TP=2 and 0.82 took 475,392 tokens for a 4 x 8,192-token workload and
    left its entrypoint Spark at the 16 GiB host memory guard floor. Its
    budget plus overhead exceeded the Spark total minus that guard; every
    hardware-proven utilization-sized dual runs at 0.80 or below, inside it.
    Proven single-Spark documents run up to 0.86 (SGLang 0.85), whose share
    plus overhead is the largest admitted peak, 120,370,379,081 bytes: on one
    Spark the budget, plus the measured 17 GB floor of a draft vLLM loads
    beside it and the source-backed resident expansion of geometry pinned to
    the launch, must be admissible on an idle Spark: gpt-oss-120b 4036b68d at
    0.85 beside its 10 GB Marlin repack exhausted host memory on 2026-10-21.
    A declared SGLang single fraction, capped or not, must be admissible and
    hold its weights, expansion and capped pools: GLM a0 fa86311a at 0.75 was
    OOM-killed on 2026-10-21.

    An absolute pool must hold one full-context request
    (require_absolute_kv_capacity), not context x max-num-seqs: vLLM schedules
    sequences within the pool. Proven duals hold fewer: DeepSeek V4 Flash 0731
    r0b0tlab (16.0 GB for 16 x 327,680 tokens), GLM 5.3 Flash NVFP4 Spark eugr
    (14.69 GB for 4 x 500,000) and the EXL3 Mia dual (2.5 GiB, 774,400 tokens
    measured upstream, for 2 x 600,000). Sizing the DSpark vLLM 0.25 dual pool
    for 16 x 200,000 tokens (27.31 GB) exhausted host memory on both Sparks.
    SGLang derives an omitted mem-fraction-static at launch; the proven
    Nemotron 3 Nano NVFP4 NVIDIA SGLang single runs on it, so none is required.
    """
    vllm = recipe.runtime.engine == "vllm"
    name = "gpu-memory-utilization" if vllm else "mem-fraction-static"
    fraction = argument_value(recipe, name)
    if not vllm and recipe.topology.node_count == 1:
        if fraction is None:
            return
        # SGLang profiles after loading: the static budget must hold the
        # weights, their expansion and any capped pools (GLM a0 fa86311a at
        # 0.75 was OOM-killed), and a cap bounds the pools, not the budget the
        # fraction reserves.
        share = Decimal(str(fraction))
        resident = sglang_static_need(recipe, models, inputs, key, pinned=False)
        if share * IDLE_AVAILABLE_BYTES < resident:
            raise ValueError(
                f"{key}: mem-fraction-static {fraction} x idle available "
                f"{IDLE_AVAILABLE_BYTES} cannot hold the selected weights, resident "
                f"expansion and capped pools ({resident} bytes)"
            )
        if not evaluate_memory_admission(
            int(share * TOTAL_BYTES) + OVERHEAD_BYTES,
            SPARK_MEMORY_BYTES,
            occupied_bytes=ADMISSION_IDLE_USE_BYTES,
        ).admissible:
            raise ValueError(
                f"{key}: mem-fraction-static {fraction} share of {TOTAL_BYTES} bytes "
                f"plus {OVERHEAD_BYTES} overhead is not admissible on an idle Spark"
            )
        return
    pools = (
        ("kv-cache-memory-bytes", "kv-cache-memory") if vllm else ("max-total-tokens",)
    )
    if any(argument_value(recipe, name) is not None for name in pools):
        return
    if fraction is None and not vllm:
        return  # Multi-Spark SGLang uses require_sglang_rank_budget.
    share = Decimal(str(fraction)) if fraction is not None else VLLM_DEFAULT_UTILIZATION
    if recipe.topology.node_count != 1:
        ceiling = f"leaves the {GUARD_BYTES}-byte host memory guard"
        bounded = share * TOTAL_BYTES + OVERHEAD_BYTES <= TOTAL_BYTES - GUARD_BYTES
    else:
        budget = (vllm_utilization_envelope(recipe) or 0) + sum(
            geometry.resident_expansion_bytes
            for selection in recipe.models
            if (
                geometry := applicable_geometry(
                    recipe,
                    selection.model.slug,
                    models[selection.model.content_sha256],
                    inputs,
                )
            )
            is not None
        )
        ceiling = "is admissible on an idle Spark"
        bounded = evaluate_memory_admission(
            budget, SPARK_MEMORY_BYTES, occupied_bytes=ADMISSION_IDLE_USE_BYTES
        ).admissible
    if not bounded:
        raise ValueError(
            f"{key}: set absolute {pools[0]}, or a {name} whose share of "
            f"{TOTAL_BYTES} bytes plus {OVERHEAD_BYTES} overhead {ceiling}"
        )


def require_vllm_startup_budget(recipe: RecipeDefinition) -> None:
    """Absolute KV overrides cache sizing, not vLLM's earlier startup check."""
    if recipe.runtime.engine != "vllm" or (
        argument_value(recipe, "kv-cache-memory-bytes") is None
        and argument_value(recipe, "kv-cache-memory") is None
    ):
        return
    fraction = argument_value(recipe, "gpu-memory-utilization")
    if fraction is None:
        raise ValueError(
            f"{recipe.identity.slug}: absolute KV needs an explicit startup "
            "gpu-memory-utilization budget"
        )
    utilization = Decimal(str(fraction))
    if not utilization.is_finite() or not 0 < utilization <= 1:
        raise ValueError(f"{recipe.identity.slug}: invalid vLLM startup budget")
    for role in recipe.topology.roles:
        if (
            utilization * TOTAL_BYTES + OVERHEAD_BYTES
            > role.resources.memory.peak_bytes
        ):
            raise ValueError(
                f"{recipe.identity.slug}/{role.name}: vLLM startup budget plus "
                "overhead exceeds declared peak"
            )


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
            need = calculate_need(variant, models, inputs)
        except (ValueError, TypeError) as error:
            warnings.append(f"{prefix}: full-workload estimate unavailable: {error}")
        else:
            warnings.append(
                f"{prefix}: estimated workload {need.total_bytes} bytes/rank; "
                f"planning limit {LIMIT_BYTES}; hardware test required"
            )
        rank = per_rank_need(variant, models, inputs)
        warnings.append(
            f"{prefix}: loading floor {rank.peak_bytes} bytes/rank; "
            f"observed idle capacity {rank.safe_limit_bytes}; no hardware verdict"
        )
    return warnings


def rank_weight_bytes(
    recipe: RecipeDefinition,
    models: Mapping[str, ModelDefinition],
    inputs: MemoryInputs,
) -> int:
    """Selected weights per rank, Engram tables excluded, plus an explicit row cache.

    Engram files are identified by content digest, for every recipe and engine.
    A row cache counts only when the recipe sets a positive DSV41_CACHE_GIB
    (GiB per Spark, shared by the ranks on that Spark).
    """
    weights: dict[str, int] = {}
    for selection in recipe.models:
        selected = {item.file_id for item in selection.files}
        for item in models[selection.model.content_sha256].files:
            if (
                item.id in selected
                and "weights" in item.roles
                and item.sha256 not in inputs.engram_files
            ):
                weights[item.sha256] = item.size_bytes
    parallel = recipe.topology.parallelism
    shards = parallel.tensor * parallel.pipeline
    cache = 0
    for item in recipe.runtime.environment:
        if item.name == ENGRAM_ROW_CACHE_ENVIRONMENT and Decimal(str(item.value)) > 0:
            per_spark = int(
                (Decimal(str(item.value)) * 2**30).to_integral_value(ROUND_CEILING)
            )
            cache = -(-per_spark * recipe.topology.node_count // shards)
    return (sum(weights.values()) + shards - 1) // shards + cache


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
    recipe: RecipeDefinition,
    models: Mapping[str, ModelDefinition],
    inputs: MemoryInputs,
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
    loading_floor = (
        rank_weight_bytes(recipe, models, inputs) + kv_bytes + OVERHEAD_BYTES
    )
    # Utilization sizes an adaptive pool; it is not irreducible workload demand.
    # The measured 17 GB term is relative to that utilization budget, not another
    # allocation beside explicitly counted draft weights. Keep it in the strict
    # workload/legacy envelope guards, not this optimistic loading floor.
    # Physical fit uses observed idle availability; a changed document's peak
    # must cover this floor and be admissible on an idle Spark. For multi-Spark
    # recipes the 16 GiB planning guard (LIMIT_BYTES) is only a warning: the
    # hardware-proven set runs above it.
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


def require_admissible_declaration(
    recipe: RecipeDefinition, *, pool_bytes: int = SPARK_MEMORY_BYTES
) -> None:
    """Every launch variant must fit an idle Spark by the Controller's rule.

    An idle Spark's host already uses ADMISSION_IDLE_USE_BYTES; a declaration
    that only fits with no host use can never be admitted (2026-10-21 deferral).
    """
    for variant in memory_variants(recipe):
        for role in variant.topology.roles:
            if not evaluate_memory_admission(
                role.resources.memory.peak_bytes,
                pool_bytes,
                occupied_bytes=ADMISSION_IDLE_USE_BYTES,
            ).admissible:
                raise ValueError(
                    f"{recipe.identity.slug}/{role.name}: declared peak "
                    f"{role.resources.memory.peak_bytes} bytes is not admissible "
                    f"on an idle Spark (pool {pool_bytes} bytes, host "
                    f"{ADMISSION_IDLE_USE_BYTES} bytes, floor "
                    f"{PLATFORM_MEMORY_FLOOR_BYTES} bytes); reserve by need"
                )
