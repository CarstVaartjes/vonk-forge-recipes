"""Calibrated memory constants and their provenance."""

from __future__ import annotations

from decimal import Decimal

from recipe_tooling.paths import REPO_ROOT
from vonk_forge_contracts import (
    RecipeDefinition,
)
from vonk_forge_contracts.memory_admission import (
    PLATFORM_MEMORY_FLOOR_BYTES,
)

# The platform floor is owned by vonk_forge_contracts.memory_admission (contracts 2.8.1).
# GUARD_BYTES below is this estimator's own headroom; the live host memory guard
# (vonk-forge #1936 fc6059e42edb1b26416fd78493138201f427de96,
# rust/crates/vonk-agent-protocol/src/host_memory_guard_policy.rs) has no
# per-sample MemAvailable bar. These estimates do not redefine live admission
# or the guard.
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
# The repo's calibrated OFFLINE idle maximum: the largest peak
# admissible_declaration_warnings accepts (evaluate_memory_admission with
# ADMISSION_IDLE_USE_BYTES), i.e. the same derivation. The live Controller works
# from telemetry and tries an over-maximum estimate alone on an empty Spark. A
# peak above this only blocks co-tenants and is never reserved in full, so
# estimates are capped here; a bigger need is the hardware try's to measure.
ADMISSION_MAXIMUM_PEAK_BYTES = (
    SPARK_MEMORY_BYTES - ADMISSION_IDLE_USE_BYTES - PLATFORM_MEMORY_FLOOR_BYTES
)
OVERHEAD_BYTES = 8_000_000_000
# Data-backed use beside the engine budget (p90 of 277 sweep samples, audit
# 2026-11-01). The single margin below is the only owner; OVERHEAD_BYTES stays
# the loading floor, the inside-budget check and the SGLang static need.
VLLM_SINGLE_MARGIN_BYTES = 6_200_000_000
# Measured on 2-Spark runs only (56 samples, 28 contents); borrowed for 3, 4, 8.
VLLM_MULTI_MARGIN_BYTES = 10_400_000_000
SGLANG_SINGLE_MARGIN_BYTES = 8_500_000_000


def budget_margin_bytes(recipe: RecipeDefinition) -> int:
    """Bytes a launch holds beside its engine budget, by engine and topology.

    Applies only where the constant means "beside the budget": the vLLM
    utilization envelope and the SGLang fraction demand. Multi-Spark SGLang
    keeps OVERHEAD_BYTES.
    """
    single = recipe.topology.node_count == 1
    if recipe.runtime.engine == "vllm":
        return VLLM_SINGLE_MARGIN_BYTES if single else VLLM_MULTI_MARGIN_BYTES
    return SGLANG_SINGLE_MARGIN_BYTES if single else OVERHEAD_BYTES


# Offline copy of the swap policy pinned to vonk-forge fc6059e42edb1b26416fd78493138201f427de96
# (#1936 swap-aware memory guard): rust/crates/vonk-agent-protocol/src/
# host_memory_guard_policy.rs::SWAP_EXHAUSTED_FREE_FLOOR_BYTES (2 GiB),
# SWAP_EXHAUSTED_FREE_PERCENT (10) and SWAP_FILLING_USED_PERCENT (90). R1
# (vonk-agent-helper/src/host_memory_guard/decision.rs:185) keeps max(floor, total / 100 * 10) free;
# R2 (decision.rs:191) allows at most total / 100 * 90 used. Also
# agent_protocol/src/vonk_agent_protocol/swap.py::SwapPolicy maximum_gib 128,
# default_gib 64. usable/needed are proven against vonk-forge's shared vector
# (agent_protocol/src/vonk_agent_protocol/vectors/swap-v1.json). These do not redefine live admission or the guard.
GIB = 1024**3
SWAP_FREE_FLOOR_BYTES = 2 * GIB
SWAP_FREE_PERCENT = 10
SWAP_FILLING_USED_PERCENT = 90
SWAP_MAXIMUM_BYTES = 128 * 1024**3
SWAP_DEFAULT_BYTES = 64 * 1024**3
# sglang 94602c9c (and the Qwen3.8 Flash Next r0b0tlab image's qwen4_exp copy)
# kv_cache_configurator._calculate_mamba_ratio: at most 3 base plus 2
# extra_buffer overlap state slots per running request.
SGLANG_MAMBA_SLOTS_PER_REQUEST = 5
# server_args.mamba_full_memory_ratio default (sglang b7252cc6 and 94602c9c).
SGLANG_MAMBA_FULL_MEMORY_RATIO = Decimal("0.9")
# The DeepSeek V4.1 Engram row store's resident cache size (GiB per Spark).
ENGRAM_ROW_CACHE_ENVIRONMENT = "DSV41_CACHE_GIB"
# vLLM's gpu-memory-utilization when a launch omits it.
VLLM_DEFAULT_UTILIZATION = Decimal("0.90")
ROOT = REPO_ROOT
# GB10 total and lower idle MemAvailable from spark-3542, 2026-10-15/16;
# retained from tests/qualification/test_vllm_memory_fit.py at the historical source commit.
TOTAL_BYTES = SPARK_MEMORY_BYTES
IDLE_AVAILABLE_BYTES = SPARK_3542_IDLE_AVAILABLE_BYTES
IDLE_USE_BYTES = TOTAL_BYTES - IDLE_AVAILABLE_BYTES
# SGLang's pre-load free memory on GB10 is MemAvailable after CUDA init: the
# CUDA context takes about 4.2e9 bytes on the 2026-10-28 Inkling TP2 start
# (113.11 GiB before load, SGLang prints GiB, against 125,650,657,280 idle) and
# 4.70e9 on the GLM a0 single-Spark start (114.89 GiB before load against
# 128.06e9 MemAvailable). The larger one is the planning figure; this is the one
# owner of SGLang's base. Capped and uncapped launches profile the same way
# (_profile_available_bytes); a cap enters only in _apply_token_constraints.
SGLANG_CUDA_CONTEXT_BYTES = 4_700_000_000
SGLANG_PRELOAD_FREE_BYTES = IDLE_AVAILABLE_BYTES - SGLANG_CUDA_CONTEXT_BYTES
# The estimator's own headroom: the informative workload limit subtracts it;
# it bounds no touched recipe. It began as the host memory
# guard's per-sample MemAvailable bar (vonk-forge 040c51b8531523bebe542b5e1f7b9403e0767f72,
# host_memory_guard_policy.rs::STALL_AVAILABLE_BYTES), which the swap-aware
# guard (vonk-forge #1936) removed; the live guard's rules are R1-R5 there.
GUARD_BYTES = 16 * 1024**3
LIMIT_BYTES = TOTAL_BYTES - GUARD_BYTES - IDLE_USE_BYTES
