"""Swap declarations."""

from __future__ import annotations

from recipe_tooling.qualification.memory_need import args as mn_args
from recipe_tooling.qualification.memory_need.constants import (
    ADMISSION_IDLE_USE_BYTES,
    GIB,
    SPARK_MEMORY_BYTES,
    SWAP_DEFAULT_BYTES,
    SWAP_FILLING_USED_PERCENT,
    SWAP_FREE_FLOOR_BYTES,
    SWAP_FREE_PERCENT,
    SWAP_MAXIMUM_BYTES,
)
from vonk_forge_contracts import (
    RecipeDefinition,
)
from vonk_forge_contracts.memory_admission import (
    PLATFORM_MEMORY_FLOOR_BYTES,
    evaluate_memory_admission,
)
from vonk_forge_contracts.recipe import RecipeMemoryResources


def usable_swap_bytes(total_bytes: int) -> int:
    """Swap a load may fill before the host guard's R1 or R2 applies."""
    kept_free = max(SWAP_FREE_FLOOR_BYTES, total_bytes // 100 * SWAP_FREE_PERCENT)
    return max(
        0,
        min(total_bytes - kept_free, total_bytes // 100 * SWAP_FILLING_USED_PERCENT),
    )


def swap_needed_bytes(load_bytes: int) -> int:
    """The smallest whole-GiB swap total whose usable part holds the load."""
    if load_bytes <= 0:
        return 0
    size_gib = -(-load_bytes // GIB)
    while usable_swap_bytes(size_gib * GIB) < load_bytes:
        size_gib += 1
    return size_gib * GIB


def load_overshoot_bytes(memory: RecipeMemoryResources) -> int:
    """Load use beyond the steady peak, which the host serves from swap."""
    if memory.load_peak_bytes is None:
        return 0
    return memory.load_peak_bytes - memory.peak_bytes


def swap_warnings(recipe: RecipeDefinition) -> list[str]:
    """Roles whose load needs more swap than the host default.

    An overshoot beyond the usable maximum is reported by swap_declaration_warnings;
    skipping it here keeps swap_needed_bytes bounded for any declared value.
    """
    warnings = []
    for variant in mn_args.memory_variants(recipe):
        for role in variant.topology.roles:
            overshoot = load_overshoot_bytes(role.resources.memory)
            if overshoot > usable_swap_bytes(SWAP_MAXIMUM_BYTES):
                continue
            needed = swap_needed_bytes(overshoot)
            if needed > SWAP_DEFAULT_BYTES:
                warnings.append(
                    f"WARNING {recipe.identity.slug}/{role.name}: load needs "
                    f"{needed} bytes of swap, above the {SWAP_DEFAULT_BYTES}-byte default"
                )
    return warnings


def swap_declaration_warnings(recipe: RecipeDefinition) -> list[str]:
    """A load that overshoots the steady peak should fit, and name its swap."""
    warnings: list[str] = []
    for variant in mn_args.memory_variants(recipe):
        for role in variant.topology.roles:
            memory = role.resources.memory
            overshoot = load_overshoot_bytes(memory)
            if overshoot == 0:
                continue
            where = f"{recipe.identity.slug}/{role.name}"
            if overshoot > usable_swap_bytes(SWAP_MAXIMUM_BYTES):
                warnings.append(
                    f"{where}: load overshoot {overshoot} bytes exceeds the "
                    f"usable part of the {SWAP_MAXIMUM_BYTES}-byte swap maximum "
                    f"({usable_swap_bytes(SWAP_MAXIMUM_BYTES)}); it can never fit"
                )
                continue
            needed = swap_needed_bytes(overshoot)
            if memory.minimum_swap_bytes < needed:
                warnings.append(
                    f"{where}: minimum_swap_bytes {memory.minimum_swap_bytes} "
                    f"is below the {needed} bytes the {overshoot}-byte load "
                    "overshoot needs"
                )
            if memory.minimum_swap_bytes > SWAP_MAXIMUM_BYTES:
                warnings.append(
                    f"{where}: minimum_swap_bytes {memory.minimum_swap_bytes} "
                    f"exceeds the {SWAP_MAXIMUM_BYTES}-byte swap maximum"
                )
    return warnings


def admissible_declaration_warnings(
    recipe: RecipeDefinition, *, pool_bytes: int = SPARK_MEMORY_BYTES
) -> list[str]:
    """Every launch variant should fit an idle Spark by the Controller's rule.

    An idle Spark's host already uses ADMISSION_IDLE_USE_BYTES; a declaration
    that only fits with no host use is an estimate the Controller would not
    admit (2026-10-21 deferral). The hardware try decides.
    """
    warnings = [
        f"{recipe.identity.slug}/{role.name}: declared peak "
        f"{role.resources.memory.peak_bytes} bytes is not admissible "
        f"on an idle Spark (pool {pool_bytes} bytes, host "
        f"{ADMISSION_IDLE_USE_BYTES} bytes, floor "
        f"{PLATFORM_MEMORY_FLOOR_BYTES} bytes); reserve by need"
        for variant in mn_args.memory_variants(recipe)
        for role in variant.topology.roles
        if not evaluate_memory_admission(
            role.resources.memory.peak_bytes,
            pool_bytes,
            occupied_bytes=ADMISSION_IDLE_USE_BYTES,
        ).admissible
    ]
    warnings.extend(swap_declaration_warnings(recipe))
    return warnings
