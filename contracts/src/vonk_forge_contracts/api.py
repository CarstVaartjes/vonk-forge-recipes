"""The API facts of a Recipe's served route, as its own small contract.

A ``RecipeApi`` states what a client may rely on when it calls the route a
recipe serves: token limits, input modalities, reasoning control, tool use,
structured output and endpoints. It never carries execution data, and it has no
digest of its own: the signed index that transports it is verified at ingress.

The document lives in ``api/recipes/<slug>.json``, never in the Recipe, so the
recipe's ``content_sha256`` (its execution identity) and every qualification
record keyed by it stay as they are. A document is optional at contract level;
a route without one publishes no gateway facts.

Every field of a block is required: "not applicable" is written down (``false``,
``{"kind": "none"}``, ``null``), never left out. A block is discriminated by
``mode``. An option choice that changes any API fact carries its own complete
block in ``option_choices``; it replaces the base block as a whole, the way the
choice's runtime arguments replace base arguments (``RecipeApi.for_choices``).

Strict ``model_validate`` is for this repository's files and output;
``read_recipe_api`` ignores unknown fields and is only for consumers of a
published index. ``require_api_of`` is the one identity rule. The facts that
must agree with a recipe's runtime or its Models are checked in ``api_checks``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from .recipe import Identifier, RecipeDefinition, RecipeIdentity, Scalar


class _ApiContract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


Endpoint = Literal[
    "/v1/chat/completions", "/v1/completions", "/v1/responses", "/v1/messages"
]
ToolChoice = Literal["auto", "none", "required", "named"]
ReasoningDefault = Literal["on", "off", "model"]
RequestPlacement = Literal["chat_template_kwargs", "top_level"]
_Key = Annotated[StrictStr, Field(min_length=1, max_length=64)]


class ApiTokenValue(_ApiContract):
    value: StrictInt = Field(ge=1)


class ApiTokenSetting(_ApiContract):
    """Bound to the recipe setting of the same name, as ``RecipeRuntimeArgument``."""

    setting: Literal["context_tokens"]


ApiTokenBound = ApiTokenValue | ApiTokenSetting


class ApiTokenArgument(_ApiContract):
    """The server default is the value of this runtime argument.

    The number is never copied here: it is read from the recipe's arguments (as
    the option choice in force sets them), so it cannot drift from the launch.
    """

    argument: StrictStr = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")


class ApiRemainingContext(_ApiContract):
    """No server default: the request may use what the context still holds."""

    remaining: Literal["context", "slot_context"]


class ApiInputModalities(_ApiContract):
    text: StrictBool
    image: StrictBool
    video: StrictBool
    audio: StrictBool
    pdf: StrictBool


class ApiReasoningEffort(_ApiContract):
    via: RequestPlacement
    key: _Key
    levels: list[_Key] = Field(min_length=1, max_length=16)
    default: _Key

    @model_validator(mode="after")
    def default_is_a_level(self) -> ApiReasoningEffort:
        if len(self.levels) != len(set(self.levels)):
            raise ValueError("effort levels must be unique")
        if self.default not in self.levels:
            raise ValueError("effort default must be one of its levels")
        return self


class ApiReasoningNone(_ApiContract):
    kind: Literal["none"]


class ApiReasoningAlwaysOn(_ApiContract):
    kind: Literal["always-on"]
    effort: ApiReasoningEffort | None


class ApiReasoningSwitch(_ApiContract):
    """Thinking can be turned on and off per request.

    ``keys`` are the alternative request keys the route reads (any one works);
    ``default`` is what the route does when the request says nothing, and
    ``model`` when the checkpoint decides.
    """

    kind: Literal["switch"]
    via: RequestPlacement
    keys: list[_Key] = Field(min_length=1, max_length=4)
    on: Scalar
    off: Scalar
    default: ReasoningDefault
    effort: ApiReasoningEffort | None

    @model_validator(mode="after")
    def distinct_values(self) -> ApiReasoningSwitch:
        if len(self.keys) != len(set(self.keys)):
            raise ValueError("switch keys must be unique")
        if self.on == self.off:
            raise ValueError("switch on and off values must differ")
        return self


ApiReasoning = Annotated[
    ApiReasoningNone | ApiReasoningAlwaysOn | ApiReasoningSwitch,
    Field(discriminator="kind"),
]


class ApiToolsNone(_ApiContract):
    kind: Literal["none"]


class ApiToolsSupported(_ApiContract):
    """Function calling works.

    ``tool_choice`` lists the choices the route enforces. ``none`` is always
    honoured, so it is implied and never listed. An empty list means non-auto
    choices are accepted but not enforced (a prompt nudge, or ignored). An
    unlisted value may be refused (HTTP 400).
    """

    kind: Literal["supported"]
    tool_choice: list[ToolChoice] = Field(max_length=4)
    parallel_calls: StrictBool

    @field_validator("tool_choice")
    @classmethod
    def unique_choices(cls, value: list[ToolChoice]) -> list[ToolChoice]:
        if len(value) != len(set(value)):
            raise ValueError("tool_choice values must be unique")
        return value


ApiTools = Annotated[ApiToolsNone | ApiToolsSupported, Field(discriminator="kind")]


class ChatApi(_ApiContract):
    mode: Literal["chat"]
    max_output_tokens: ApiTokenBound
    default_output_tokens: ApiTokenValue | ApiTokenArgument | ApiRemainingContext
    # Null: the prompt may fill the context. A number is the engine's own lower
    # prompt cap (TRT-LLM without chunked prefill, a llama.cpp slot's share).
    max_input_tokens: StrictInt | None = Field(ge=1)
    input_modalities: ApiInputModalities
    audio_output: StrictBool
    reasoning: ApiReasoning
    tools: ApiTools
    structured_output: StrictBool
    system_messages: StrictBool
    streaming: StrictBool
    endpoints: list[Endpoint] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def text_in_and_unique_endpoints(self) -> ChatApi:
        if not self.input_modalities.text:
            raise ValueError("a chat route takes text input")
        if len(self.endpoints) != len(set(self.endpoints)):
            raise ValueError("endpoints must be unique")
        return self


class DecisionApi(_ApiContract):
    """A route serving ``POST /v1/systemone``; it has no chat-shaped facts."""

    mode: Literal["decision"]
    max_input_tokens: StrictInt | None = Field(ge=1)
    input_modalities: ApiInputModalities

    @model_validator(mode="after")
    def text_in(self) -> DecisionApi:
        if not self.input_modalities.text:
            raise ValueError("a decision route takes text input")
        return self


ApiBlock = Annotated[ChatApi | DecisionApi, Field(discriminator="mode")]


class RecipeApiOptionChoice(_ApiContract):
    """The complete block that applies when ``option`` is set to ``value``."""

    option: Identifier
    value: Identifier
    api: ApiBlock


class RecipeApi(_ApiContract):
    kind: Literal["recipe-api"] = "recipe-api"
    identity: RecipeIdentity
    api: ApiBlock
    option_choices: list[RecipeApiOptionChoice] = Field(max_length=64)

    @model_validator(mode="after")
    def consistent_choices(self) -> RecipeApi:
        keys = [(c.option, c.value) for c in self.option_choices]
        if len(keys) != len(set(keys)):
            raise ValueError("option choices must be unique")
        if len({option for option, _ in keys}) > 1:
            raise ValueError("at most one option may carry API blocks")
        if any(c.api.mode != self.api.mode for c in self.option_choices):
            raise ValueError("an option choice cannot change the API mode")
        return self

    def for_choices(
        self, recipe: RecipeDefinition, choices: Mapping[str, str] | None = None
    ) -> ApiBlock:
        """The block in force for these option choices (unnamed options default).

        The same resolution as ``RecipeDefinition.with_option_choices``: it
        raises ``RecipeOptionError`` for an unknown option or value, and a
        choice's block replaces the base block as a whole.
        """

        resolved = recipe.resolve_options(choices)
        for entry in self.option_choices:
            if resolved.get(entry.option) == entry.value:
                return entry.api
        return self.api


def read_recipe_api(document: Mapping[str, object]) -> RecipeApi:
    """Parse a published RecipeApi, ignoring fields a newer minor added."""

    return RecipeApi.model_validate(document, extra="ignore")


def require_api_of(api: RecipeApi, document_identity: RecipeIdentity) -> None:
    """Raise unless ``api`` names exactly the recipe it travels with."""

    if (api.identity.publisher, api.identity.slug) != (
        document_identity.publisher,
        document_identity.slug,
    ):
        raise ValueError("api identity differs from its recipe")
