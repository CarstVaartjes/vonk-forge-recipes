"""The short chat the sweep sends, derived from a recipe's own ``recipe-api`` facts.

A recipe declares a bounded chat check (``hello``, a few dozen tokens). Sent as written to a
route whose reasoning defaults on, the whole reply is spent thinking and no answer arrives.
The facts the recipe publishes beside it (``api/recipes/<slug>.json``) say how to ask for a
short answer, so this one helper turns the declared request into the one that fits them. The
facts are those of the launch that ran: ``RecipeApi.for_choices`` with the run's option
choices, and the recipe's runtime arguments under the same choices.

* ``switch`` route: its declared OFF value under every declared key, at the declared
  placement (``chat_template_kwargs`` or top level), unless the request already names the key;
* ``always-on`` route (thinking cannot be turned off): the request takes the route's own
  output allowance per ``default_output_tokens``: a number, the value of the runtime
  argument it names, or none at all when the route's default is the remaining context
  (the case is then bounded by its timeout and response size). The allowance never
  exceeds ``max_output_tokens``;
* a route without reasoning: the request is sent as declared.

No per-recipe case and no key the facts do not state. A recipe without published facts is
sent as declared (``None`` facts): the stricter ``chat.nonempty`` assertion needs facts.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from vonk_forge_contracts.api import (
    ApiReasoningAlwaysOn,
    ApiReasoningSwitch,
    ApiRemainingContext,
    ApiTokenArgument,
    ApiTokenSetting,
    ApiTokenValue,
    ChatApi,
    RecipeApi,
)
from vonk_forge_contracts.recipe import RecipeDefinition, RecipeGenerationSettings


@dataclass(frozen=True)
class ChatFacts:
    """The chat block in force for one launch, and the recipe as that launch ran it."""

    api: ChatApi
    launch: RecipeDefinition


def chat_facts(
    definition: RecipeDefinition, api: RecipeApi, choices: dict[str, str]
) -> ChatFacts | None:
    """The facts of the launch with these option choices; None for a non-chat route."""
    block = api.for_choices(definition, choices)
    if not isinstance(block, ChatApi):
        return None
    return ChatFacts(block, definition.with_option_choices(choices))


def _context_tokens(launch: RecipeDefinition) -> int | None:
    settings = launch.settings
    if isinstance(settings, RecipeGenerationSettings):
        return settings.context_tokens.value
    return None


def _argument_tokens(launch: RecipeDefinition, name: str) -> int | None:
    """The integer a runtime argument carries, resolving a setting binding."""
    settings = launch.settings
    for argument in launch.runtime.arguments:
        if argument.name != name:
            continue
        if argument.setting is None:
            value = argument.value
        else:
            bound = getattr(settings, "knobs", {}).get(argument.setting) or getattr(
                settings, argument.setting, None
            )
            value = bound.value if bound is not None else None
        return value if isinstance(value, int) and not isinstance(value, bool) else None
    return None


def _allowance(facts: ChatFacts) -> int | None:
    """The route's own default output allowance; None when it is the remaining context."""
    api = facts.api
    default = api.default_output_tokens
    if isinstance(default, ApiRemainingContext):
        return None
    if isinstance(default, ApiTokenValue):
        tokens = default.value
    elif isinstance(default, ApiTokenArgument):
        tokens = _argument_tokens(facts.launch, default.argument)
        if tokens is None:
            return None
    else:
        return None
    ceiling = api.max_output_tokens
    if isinstance(ceiling, ApiTokenValue):
        return min(tokens, ceiling.value)
    if isinstance(ceiling, ApiTokenSetting):
        context = _context_tokens(facts.launch)
        return tokens if context is None else min(tokens, context)
    return tokens


def bounded_chat_body(body: dict[str, Any], facts: ChatFacts | None) -> dict[str, Any]:
    """The declared chat request, adjusted so a short answer fits the route's reasoning."""
    request = copy.deepcopy(body)
    if facts is None:
        return request
    reasoning = facts.api.reasoning
    if isinstance(reasoning, ApiReasoningSwitch):
        target = (
            request.setdefault("chat_template_kwargs", {})
            if reasoning.via == "chat_template_kwargs"
            else request
        )
        if isinstance(target, dict):
            for key in reasoning.keys:
                target.setdefault(key, reasoning.off)
    elif isinstance(reasoning, ApiReasoningAlwaysOn):
        allowance = _allowance(facts)
        declared = request.get("max_tokens")
        if allowance is None:
            request.pop("max_tokens", None)
        elif not isinstance(declared, int) or declared < allowance:
            request["max_tokens"] = allowance
    return request
