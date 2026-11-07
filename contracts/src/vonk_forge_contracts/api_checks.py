"""Facts a RecipeApi may not state against its recipe or the recipe's Models.

``check_recipe_api`` is the one structural check the producer guard and the
Controller's loader call. It compares the base block and every option choice's block with the
runtime the recipe really compiles to (``with_option_choices``), so a choice
that flips a runtime argument without carrying its own block is refused too.
``check_api_models`` is the cross-document half: input modalities need a Model
display that declares them, and a speech route needs one that produces audio.
Whether a route really serves tools or a reasoning parser
is owned by ``qualification.capabilities.serving_capabilities`` (it reads
adapter wrappers and kit flags this package cannot see); ``tools/check-recipe-api``
applies it to every variant from ``api_variants``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable

from .api import (
    ApiBlock,
    ApiInputModalities,
    ApiTokenArgument,
    ApiTokenValue,
    AudioSpeechApi,
    AudioTranscriptionApi,
    ChatApi,
    RecipeApi,
    require_api_of,
)
from .capability import CapabilityName
from .display import ModelDisplay
from .recipe import RecipeDecisionSettings, RecipeDefinition, RecipeGenerationSettings

# The argument each engine takes its context window from; the one table, also
# used by the memory estimator. It must be present and bound to the recipe's
# ``context_tokens`` setting, or the declared context and the served one can
# drift apart; a route whose launch fixes the window elsewhere says its prompt
# limit with an explicit ``max_input_tokens`` instead of an argument.
CONTEXT_ARGUMENTS: dict[str, tuple[str, ...]] = {
    "vllm": ("max-model-len",),
    "sglang": ("context-length",),
    "tensorrt-llm": ("max_seq_len",),
    "llama-cpp": ("ctx-size",),
    "tensorfold": ("context",),
    "ds4": ("ctx",),
    "exllamav3": ("cache_size", "max-seq-len", "window"),
}
# The capability a Model declares for each non-text input modality it accepts: the
# one table the API check and the Model modality guard share.
INPUT_MODALITY_CAPABILITIES: dict[str, CapabilityName] = {
    "image": CapabilityName.IMAGE_UNDERSTANDING,
    "audio": CapabilityName.AUDIO_UNDERSTANDING,
    "video": CapabilityName.VIDEO_UNDERSTANDING,
}
# vLLM's own transcription upload and decode limits (environment variables,
# in MiB and seconds). A recipe that sets one states the same limit in its block.
VLLM_AUDIO_LIMITS: dict[str, tuple[str, int]] = {
    "VLLM_MAX_AUDIO_CLIP_FILESIZE_MB": ("max_file_bytes", 1 << 20),
    "VLLM_MAX_AUDIO_DECODE_DURATION_S": ("max_audio_seconds", 1),
}
_AUDIO_IN = ApiInputModalities(
    text=False, image=False, video=False, audio=True, pdf=False
)
_TEXT_IN = ApiInputModalities(
    text=True, image=False, video=False, audio=False, pdf=False
)


class RecipeApiError(ValueError):
    """The API facts contradict the recipe's runtime or its Models."""


def accepted_modalities(block: ApiBlock) -> ApiInputModalities:
    """What the route takes in; an audio mode fixes it by the endpoint."""

    if isinstance(block, AudioTranscriptionApi):
        return _AUDIO_IN
    if isinstance(block, AudioSpeechApi):
        return _TEXT_IN
    return block.input_modalities


def api_variants(
    recipe: RecipeDefinition, api: RecipeApi
) -> Iterable[tuple[str, RecipeDefinition, ApiBlock]]:
    """The base route, then every single option choice, with its block."""

    yield "base", recipe.with_option_choices({}), api.for_choices(recipe, {})
    for option in recipe.options:
        for choice in option.choices:
            choices = {option.name: choice.value}
            yield (
                f"option {option.name}={choice.value}",
                recipe.with_option_choices(choices),
                api.for_choices(recipe, choices),
            )


def join_problems(found: dict[str, list[str]], variants: int) -> str:
    """Each distinct problem once, naming its variants unless all share it."""

    return "; ".join(
        message if len(labels) == variants else f"{message} ({', '.join(labels)})"
        for message, labels in found.items()
    )


def _argument(recipe: RecipeDefinition, name: str) -> object | None:
    """A runtime argument's literal value, ``"<setting>"`` markers aside."""

    for argument in recipe.runtime.arguments:
        if argument.name == name:
            return argument.value if argument.setting is None else argument.setting
    return None


def _is_set(recipe: RecipeDefinition, name: str) -> bool:
    value = _argument(recipe, name)
    return value is True or (isinstance(value, str) and value != "")


def _mm_limits(recipe: RecipeDefinition) -> dict[str, object]:
    value = _argument(recipe, "limit-mm-per-prompt")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return {}
    return value if isinstance(value, dict) else {}


def integer_argument(
    recipe: RecipeDefinition,
    name: str,
    settings: RecipeGenerationSettings | RecipeDecisionSettings,
) -> int | None:
    """The integer a runtime argument carries, resolving a setting binding."""

    for argument in recipe.runtime.arguments:
        if argument.name != name:
            continue
        if argument.setting is None:
            value = argument.value
        else:
            bound = settings.knobs.get(argument.setting) or getattr(
                settings, argument.setting, None
            )
            value = bound.value if bound is not None else None
        return value if isinstance(value, int) and not isinstance(value, bool) else None
    return None


def _audio_limit_problems(
    recipe: RecipeDefinition, block: AudioTranscriptionApi
) -> list[str]:
    if recipe.runtime.engine != "vllm":
        return []
    found: list[str] = []
    for item in recipe.runtime.environment:
        if item.name not in VLLM_AUDIO_LIMITS:
            continue
        field, unit = VLLM_AUDIO_LIMITS[item.name]
        try:
            limit = int(str(item.value)) * unit
        except ValueError:
            found.append(f"{item.name} is not an integer")
            continue
        if getattr(block, field) != limit:
            found.append(f"{field} must be {limit}, as {item.name} sets")
    return found


def _audio_problems(
    recipe: RecipeDefinition, block: AudioTranscriptionApi | AudioSpeechApi
) -> list[str]:
    """An audio route has no token facts: only its modality and engine limits."""

    found: list[str] = []
    settings = recipe.settings
    if not isinstance(settings, RecipeGenerationSettings | RecipeDecisionSettings):
        return ["the recipe does not serve a generation or decision route"]
    if settings.kind == "decision":
        found.append(f"a decision recipe cannot serve a {block.mode} block")
    if isinstance(block, AudioTranscriptionApi):
        if recipe.runtime.engine == "vllm" and _is_set(recipe, "language-model-only"):
            found.append("audio input conflicts with --language-model-only")
        if recipe.runtime.engine == "vllm" and _mm_limits(recipe).get("audio") == 0:
            found.append("audio input conflicts with --limit-mm-per-prompt 0")
        found.extend(_audio_limit_problems(recipe, block))
    return found


def _problems(recipe: RecipeDefinition, block: ApiBlock) -> list[str]:
    if isinstance(block, AudioTranscriptionApi | AudioSpeechApi):
        return _audio_problems(recipe, block)
    engine = recipe.runtime.engine
    found: list[str] = []
    names = CONTEXT_ARGUMENTS.get(engine, ())
    present = [a for a in recipe.runtime.arguments if a.name in names]
    for argument in present:
        if argument.setting != "context_tokens":
            found.append(
                f"--{argument.name} must be bound to the context_tokens setting"
            )
    if names and not present and block.max_input_tokens is None:
        found.append(
            f"no context argument (--{names[0]}) is bound to context_tokens; "
            "bind one or declare max_input_tokens"
        )
    settings = recipe.settings
    if not isinstance(settings, RecipeGenerationSettings | RecipeDecisionSettings):
        return [*found, "the recipe does not serve a generation or decision route"]
    if (settings.kind == "decision") != (block.mode == "decision"):
        found.append(f"a {settings.kind} recipe cannot serve a {block.mode} block")
    context = settings.context_tokens.value
    if block.max_input_tokens is not None and block.max_input_tokens > context:
        found.append("max_input_tokens exceeds context_tokens")
    for modality in ("image", "video", "audio"):
        if not getattr(block.input_modalities, modality):
            continue
        if engine == "vllm" and _is_set(recipe, "language-model-only"):
            found.append(f"{modality} input conflicts with --language-model-only")
        if engine == "vllm" and _mm_limits(recipe).get(modality) == 0:
            found.append(f"{modality} input conflicts with --limit-mm-per-prompt 0")
    if not isinstance(block, ChatApi):
        return found
    if (
        isinstance(block.max_output_tokens, ApiTokenValue)
        and block.max_output_tokens.value > context
    ):
        found.append("max_output_tokens exceeds context_tokens")
    if isinstance(block.default_output_tokens, ApiTokenValue) and (
        block.default_output_tokens.value > context
    ):
        found.append("default_output_tokens exceeds context_tokens")
    if isinstance(block.default_output_tokens, ApiTokenArgument):
        name = block.default_output_tokens.argument
        default = integer_argument(recipe, name, settings)
        if default is None:
            found.append(
                f"default_output_tokens names --{name}, which has no integer value"
            )
        elif default > context:
            found.append("default_output_tokens exceeds context_tokens")
    slots = _argument(recipe, "parallel")
    if engine == "llama-cpp":
        if slots == "concurrency" and settings.concurrency is not None:
            slots = settings.concurrency.value
        if isinstance(slots, int) and not isinstance(slots, bool) and slots > 1:
            share = context // slots
            if block.max_input_tokens is None or block.max_input_tokens > share:
                found.append(
                    f"--parallel {slots} gives each request {share} tokens; "
                    "max_input_tokens must say so"
                )
    return found


def check_recipe_api(recipe: RecipeDefinition, api: RecipeApi) -> None:
    """Raise ``RecipeApiError`` for every fact that contradicts the recipe."""

    found: dict[str, list[str]] = {}
    count = 0
    try:
        require_api_of(api, recipe.identity)
        for label, effective, block in api_variants(recipe, api):
            count += 1
            for item in _problems(effective, block):
                found.setdefault(item, []).append(label)
    except ValueError as error:  # RecipeOptionError is a ValueError
        raise RecipeApiError(str(error)) from error
    if found:
        raise RecipeApiError(join_problems(found, count))


def check_api_models(
    recipe: RecipeDefinition, api: RecipeApi, models: Iterable[ModelDisplay]
) -> None:
    """Raise unless some selected Model declares each modality the API accepts.

    The modalities and capabilities are Model display facts, so the caller
    passes the displays of the Models the recipe selects.
    """

    selected = list(models)
    found: dict[str, list[str]] = {}
    count = 0
    for label, _, block in api_variants(recipe, api):
        count += 1
        wanted = accepted_modalities(block)
        if isinstance(block, AudioSpeechApi) and not any(
            "audio" in m.modalities
            and CapabilityName.AUDIO_GENERATION in m.capabilities
            for m in selected
        ):
            found.setdefault(
                "audio output needs a Model with the audio modality "
                f"and the {CapabilityName.AUDIO_GENERATION} capability",
                [],
            ).append(label)
        for modality, capability in INPUT_MODALITY_CAPABILITIES.items():
            if getattr(wanted, modality) and not any(
                modality in m.modalities and capability in m.capabilities
                for m in selected
            ):
                found.setdefault(
                    f"{modality} input needs a Model with the {modality} modality "
                    f"and the {capability} capability",
                    [],
                ).append(label)
    if found:
        raise RecipeApiError(join_problems(found, count))
