"""Cache geometry: attention, recurrent state and KV bytes."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from vonk_forge_contracts import (
    RecipeDefinition,
)


class MambaCacheMode(StrEnum):
    """vLLM's --mamba-cache-mode values (vllm/config/cache.py)."""

    NONE = "none"
    ALIGN = "align"


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
        "MiniMaxM3SparseForConditionalGeneration",
        "Step3p7ForConditionalGeneration",
        "Step5ForConditionalGeneration",
        "params.json",
    }
)
# vLLM-pinned hybrid geometries whose layout vllm_hybrid_need cannot express,
# judged by their existing attention and hand bounds instead.
VLLM_UNMODELED_HYBRID = {
    # vLLM v0.30.0 Qwen4Exp adds a second tp_replicated PLE MambaSpec, a QSA
    # CircularBufferSpec ring, an MLA compressed-key cache and an MTP QSA
    # layer; its sfxnz dual is proven (9f77f966) on this hand bound.
    "qwen3-8-flash-next-nvfp4-fab0aecb": "Qwen4Exp on vLLM v0.30.0",
    "qwen3-8-flash-next-fp4-local-b9ce18a6": "Qwen4Exp on the eugr b12x vLLM image",
    # Sliding-window plus short-convolution layers on creator images whose
    # vLLM Inkling layout is not pinned; SGLang-measured state.
    "inkling-small-nvfp4": "Inkling short convolution on creator images",
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
    # Their KV heads (config swa_num_key_value_heads, sglang ModelConfig
    # get_swa_num_kv_heads) when they differ from `kv_heads`.
    sglang_swa_kv_heads: int | None = Field(default=None, gt=0)
    # SGLang per-rank hybrid state per max-mamba-cache-size slot (+1 padding).
    sglang_state_bytes_per_slot: int = Field(default=0, ge=0)
    # SGLang's pre-load free memory measured on this runtime image, when known;
    # otherwise SGLANG_PRELOAD_FREE_BYTES.
    sglang_preload_free_bytes: int | None = Field(default=None, gt=0)
    runtime_image_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def swa_heads_with_swa_layers(self) -> Self:
        if self.sglang_swa_kv_heads is not None and not self.sglang_swa_layers:
            raise ValueError("sglang_swa_kv_heads needs sglang_swa_layers")
        return self

    @model_validator(mode="after")
    def classified_recurrent_bound(self) -> Self:
        """A classified architecture; typed recurrent state only on a hybrid
        one, with plain attention. A hand
        per-token bound is allowed only when spread over the layers vLLM groups."""
        if self.architecture not in HYBRID_ARCHITECTURES | ATTENTION_ARCHITECTURES:
            raise ValueError(
                f"classify architecture {self.architecture} as hybrid or attention"
            )
        recurrent = self.recurrent
        if recurrent is None:
            return self
        if self.architecture not in HYBRID_ARCHITECTURES:
            raise ValueError("typed recurrent state needs a hybrid architecture")
        if recurrent.attention_layers > self.layers:
            raise ValueError("recurrent.attention_layers cannot exceed layers")
        if (
            self.state_padding
            or self.sglang_state_bytes_per_slot
            or self.sliding_layers
            or self.token_bytes is not None
            or self.replicated_token_bytes
        ):
            raise ValueError(
                "typed recurrent state needs plain attention layers: no "
                "state_padding, sglang_state_bytes_per_slot, sliding_layers, "
                "token_bytes or replicated_token_bytes"
            )
        if self.token_bytes_by_dtype and self.layers != recurrent.attention_layers:
            raise ValueError(
                "a hand token bound with typed recurrent needs layers == "
                "recurrent.attention_layers"
            )
        return self


class MemoryInputs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    geometries: dict[str, CacheGeometry]
    # Engram n-gram table files, by content digest, with their source. Owner
    # rule (2026-10-26): Engram tables are never memory usage.
    engram_files: dict[str, str]
    # Bytes of a weights file that a launch reading its table from disk on demand
    # (QWEN4EXP_PLE_MMAP=1, see reads_table_from_disk) never holds resident; the
    # rest of the file still counts. By content digest.
    # Qwen3.8 Flash Next: model-fp8-mtp-ple.safetensors carries the 320,001,536 x
    # 160 B PLE n-gram table (51,200,245,760 B) that tonyd2wild's adapter reads
    # with preadv; the MTP weights in the same file stay counted.
    disk_resident_bytes: dict[str, int] = Field(default_factory=dict)
    historical_sources: list[str]
    historical_documents: dict[str, list[str]]
    # Qwen4Exp PLE table storage dtype by model configuration digest. A
    # configuration naming none is bfloat16 (b12x getattr default), which b12x
    # resolves to the io_uring disk table unless VLLM_PLE_TABLE_MEMORY is set.
    ple_embedding_dtypes: dict[str, str] = Field(default_factory=dict)


PLE_DISK_READ_ENVIRONMENT = "QWEN4EXP_PLE_MMAP"


def reads_table_from_disk(recipe: RecipeDefinition) -> bool:
    """Residency is a property of the launch: tonyd2wild's ple_mmap reads the PLE
    table with preadv only when QWEN4EXP_PLE_MMAP=1 in the merged environment
    (an option variant may turn it off and keep the table in memory)."""
    return any(
        item.name == PLE_DISK_READ_ENVIRONMENT and str(item.value) == "1"
        for item in recipe.runtime.environment
    )


def resident_weight_bytes(
    size_bytes: int, sha256: str, recipe: RecipeDefinition, inputs: MemoryInputs
) -> int:
    """Bytes of a weights file that sit in memory: none for an Engram file, less
    the disk-resident part when this launch reads that table from disk."""
    if sha256 in inputs.engram_files:
        return 0
    if reads_table_from_disk(recipe):
        return size_bytes - inputs.disk_resident_bytes.get(sha256, 0)
    return size_bytes
