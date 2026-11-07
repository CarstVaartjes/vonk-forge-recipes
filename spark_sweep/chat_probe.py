"""The short chat the sweep sends, derived from a recipe's own ``recipe-api`` facts.

A recipe declares a bounded chat check (``hello``, a few dozen tokens). Sent as written to a
route whose reasoning defaults on, the whole reply is spent thinking and no answer arrives.
The facts the recipe publishes beside it (``api/recipes/<slug>.json``, ``reasoning``) say how
to ask for a short answer, so this one helper turns the declared request into the one that
fits those facts:

* a ``switch`` route: its declared OFF value under every declared key, at its declared
  placement (``chat_template_kwargs`` or top level), unless the request already names the key;
* an ``always-on`` route cannot turn thinking off: the token budget rises to what its
  ``default_output_tokens`` / ``max_output_tokens`` imply (``ALWAYS_ON_TOKENS`` when they
  name no number);
* a route without reasoning: the request is sent as declared.

No per-recipe case and no key that the facts do not state.
"""

from __future__ import annotations

import copy
from typing import Any

from vonk_forge_contracts.api import (
    ApiReasoningAlwaysOn,
    ApiReasoningSwitch,
    ApiTokenValue,
    ChatApi,
)

#: The budget for a route that always thinks and states no token number of its own.
ALWAYS_ON_TOKENS = 4096


def _always_on_budget(api: ChatApi) -> int:
    ceiling = (
        api.max_output_tokens.value
        if isinstance(api.max_output_tokens, ApiTokenValue)
        else None
    )
    budget = (
        api.default_output_tokens.value
        if isinstance(api.default_output_tokens, ApiTokenValue)
        else ALWAYS_ON_TOKENS
    )
    return min(budget, ceiling) if ceiling is not None else budget


def bounded_chat_body(body: dict[str, Any], api: ChatApi | None) -> dict[str, Any]:
    """The declared chat request, adjusted so a short answer fits the route's reasoning."""
    request = copy.deepcopy(body)
    if api is None:
        return request
    reasoning = api.reasoning
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
        declared = request.get("max_tokens")
        request["max_tokens"] = max(
            declared if isinstance(declared, int) else 0, _always_on_budget(api)
        )
    return request
