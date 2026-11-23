"""Reading engine arguments and the geometry of a recipe."""

from __future__ import annotations

import json
from collections.abc import Mapping
from decimal import ROUND_CEILING, Decimal
from functools import cache
from itertools import product
from pathlib import Path

from vonk_forge_contracts import (
    ModelDefinition,
    RecipeDefinition,
)

from recipe_tooling.qualification.memory_need import constants as mn_constants
from recipe_tooling.qualification.memory_need.constants import (
    ENGRAM_ROW_CACHE_ENVIRONMENT,
)
from recipe_tooling.qualification.memory_need.geometry import (
    CacheGeometry,
    MemoryInputs,
    resident_weight_bytes,
)
from recipe_tooling.qualification.memory_need.records import MissingEstimateInput


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


def configured_draft_model(recipe: RecipeDefinition) -> bool:
    """Whether the speculative config names a model other than the served target."""
    served = set(recipe.runtime.entrypoint) | {
        str(item.value) for item in recipe.runtime.arguments if item.name == "model"
    }
    config = speculative_config(recipe)
    draft = config.get("model") if config is not None else None
    return draft is not None and str(draft) not in served


def loads_draft_model(recipe: RecipeDefinition) -> bool:
    if wrapper_loads_draft(
        mn_constants.ROOT / recipe.execution.build.context.path / "vllm-wrapper.sh"
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


def pinned_geometry(
    recipe: RecipeDefinition,
    slug: str,
    model: ModelDefinition,
    geometries: Mapping[str, CacheGeometry],
) -> CacheGeometry:
    """Geometry bound to the selected configuration and, if pinned, the image."""
    label = recipe.identity.slug
    # An entry keyed "<slug>@<image digest>" is the one for that runtime image.
    geometry = geometries.get(
        f"{slug}@{recipe.execution.build.base_image.digest}"
    ) or geometries.get(slug)
    if geometry is None:
        raise MissingEstimateInput(
            f"{label}: add source-backed cache geometry for {slug} to qualification/memory-inputs.json"
        )
    if not any(
        item.sha256 == geometry.configuration_sha256 and "configuration" in item.roles
        for item in model.files
    ):
        raise MissingEstimateInput(
            f"{label}: cache geometry does not match the pinned model configuration"
        )
    if (
        geometry.runtime_image_digest is not None
        and recipe.execution.build.base_image.digest != geometry.runtime_image_digest
    ):
        raise MissingEstimateInput(
            f"{label}: cache geometry does not match the pinned image"
        )
    return geometry


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


def applicable_geometry(
    recipe: RecipeDefinition,
    slug: str,
    model: ModelDefinition,
    inputs: MemoryInputs,
) -> CacheGeometry | None:
    """Source-backed geometry pinned to this launch, if any."""
    try:
        return pinned_geometry(recipe, slug, model, inputs.geometries)
    except ValueError:
        return None  # Bound to another configuration or image.


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
            if item.id in selected and "weights" in item.roles:
                weights[item.sha256] = resident_weight_bytes(
                    item.size_bytes, item.sha256, recipe, inputs
                )
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
