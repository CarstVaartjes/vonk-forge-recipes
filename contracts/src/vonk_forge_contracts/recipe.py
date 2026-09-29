"""Strict public recipe authoring contract.

Only author intent is represented here.  Runtime resolution, image receipts,
engine compatibility, and placement plans remain platform-owned concerns.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from datetime import date
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from .model import ModelReference


class _RecipeContract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


Identifier = Annotated[
    StrictStr, Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_-]*$")
]
Sha256 = Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]
_SEGMENT = r"(?:[A-Za-z0-9_-][A-Za-z0-9._-]*|\.[A-Za-z0-9_-][A-Za-z0-9._-]*)"
AbsolutePath = Annotated[
    StrictStr, Field(max_length=256, pattern=rf"^/{_SEGMENT}(?:/{_SEGMENT})*$")
]
RelativePath = Annotated[
    StrictStr, Field(max_length=256, pattern=rf"^{_SEGMENT}(?:/{_SEGMENT})*$")
]
Scalar = StrictStr | StrictInt | StrictBool | StrictFloat
type JsonValue = Scalar | None | list[JsonValue] | dict[StrictStr, JsonValue]
type RuntimeArgumentValue = Scalar | list[JsonValue] | dict[StrictStr, JsonValue]
ChangeEffect = Literal["none", "restart", "reprepare", "rebuild"]

# Every Spark shares one fabric bandwidth floor for multi-node topologies.
DISTRIBUTED_FABRIC_MINIMUM_MBPS = 200_000


# Runtime options are trusted recipe data, but still cross a process boundary.
# Keep that boundary bounded without trying to predict every engine's option
# vocabulary or requiring shell syntax for argv tokens.
MAX_RUNTIME_ARGUMENTS = 128
MAX_RUNTIME_ARGV_TOKEN_BYTES = 65_536
MAX_RUNTIME_ARGV_BYTES = 1_048_576
_MAX_RUNTIME_ARGUMENT_DEPTH = 8
_MAX_RUNTIME_ARGUMENT_ITEMS = 256


def _reject_nul(value: str, *, label: str) -> str:
    if "\x00" in value:
        raise ValueError(f"{label} must not contain NUL")
    return value


def _validate_runtime_argument_value(
    value: RuntimeArgumentValue | None,
) -> RuntimeArgumentValue | None:
    """Validate bounded JSON data without normalizing trusted engine options."""

    if value is None:
        return value

    def visit(node: JsonValue, depth: int) -> None:
        if depth > _MAX_RUNTIME_ARGUMENT_DEPTH:
            raise ValueError("runtime argument value exceeds maximum nesting depth")
        if isinstance(node, str):
            _reject_nul(node, label="runtime argument value")
            if len(node.encode("utf-8")) > MAX_RUNTIME_ARGV_TOKEN_BYTES:
                raise ValueError(
                    "runtime argument value string exceeds maximum UTF-8 size"
                )
        elif type(node) is float and not math.isfinite(node):
            raise ValueError("runtime argument value contains a non-finite number")
        elif isinstance(node, list):
            if len(node) > _MAX_RUNTIME_ARGUMENT_ITEMS:
                raise ValueError("runtime argument value has too many items")
            for item in node:
                visit(item, depth + 1)
        elif isinstance(node, dict):
            if len(node) > _MAX_RUNTIME_ARGUMENT_ITEMS:
                raise ValueError("runtime argument value has too many items")
            for key, item in node.items():
                _reject_nul(key, label="runtime argument object key")
                if len(key.encode("utf-8")) > MAX_RUNTIME_ARGV_TOKEN_BYTES:
                    raise ValueError(
                        "runtime argument object key exceeds maximum UTF-8 size"
                    )
                visit(item, depth + 1)

    visit(value, 0)
    serialized = (
        value
        if isinstance(value, str)
        else json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
    )
    try:
        encoded = serialized.encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("runtime argument value must be finite JSON") from error
    if len(encoded) > MAX_RUNTIME_ARGV_TOKEN_BYTES:
        raise ValueError("runtime argument value exceeds maximum UTF-8 size")
    return value


def _validate_argv(tokens: list[str]) -> list[str]:
    for token in tokens:
        _reject_nul(token, label="argv token")
        if len(token.encode("utf-8")) > MAX_RUNTIME_ARGV_TOKEN_BYTES:
            raise ValueError("argv token exceeds maximum UTF-8 size")
    if sum(len(token.encode("utf-8")) for token in tokens) > MAX_RUNTIME_ARGV_BYTES:
        raise ValueError("argv exceeds maximum rendered size")
    return tokens


def _serialize_runtime_argument_value(value: JsonValue) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _runtime_argument_tokens(argument: RecipeRuntimeArgument) -> list[str]:
    """Render one argument without shell parsing or option-name rewriting."""

    flag = f"--{argument.name}"
    value = argument.value
    if value is None:
        return [flag]
    if type(value) is bool:
        return [flag] if value else []
    return [flag, _serialize_runtime_argument_value(value)]


class RecipeIdentity(_RecipeContract):
    publisher: StrictStr = Field(
        min_length=2, max_length=64, pattern=r"^[a-z0-9][a-z0-9-]{1,62}$"
    )
    slug: StrictStr = Field(
        min_length=2, max_length=64, pattern=r"^[a-z0-9][a-z0-9-]{1,62}$"
    )


class RecipeMetadata(_RecipeContract):
    title: StrictStr = Field(min_length=1, max_length=120)
    description: StrictStr = Field(min_length=1, max_length=4000)
    tags: list[StrictStr] = Field(max_length=20)
    alignment: (
        Literal["standard", "abliterated", "derisked", "other-modified", "unspecified"]
        | None
    ) = None


class RecipeMount(_RecipeContract):
    # Model files are always mounted read-only.
    target: AbsolutePath


class RecipeModelFile(_RecipeContract):
    id: StrictStr = Field(
        min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_-]{0,63}$"
    )
    file_id: StrictStr = Field(
        min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_-]{0,63}$"
    )
    roles: list[StrictStr] = Field(min_length=1, max_length=32)
    mount: RecipeMount


class RecipeModelSelection(_RecipeContract):
    id: Identifier
    model: ModelReference
    # Large sharded manifests are valid exact model snapshots.  Keep a
    # bounded upper limit while allowing the catalog's largest current
    # manifests to be represented without truncating evidence.
    files: list[RecipeModelFile] = Field(min_length=1, max_length=4096)


class BuildContext(_RecipeContract):
    path: RelativePath


class RecipeImage(_RecipeContract):
    repository: StrictStr = Field(
        min_length=1, max_length=512, pattern=r"^[a-z0-9][a-z0-9._/-]*$"
    )
    # Always the linux/arm64 image for DGX Spark.
    digest: Sha256


class BuildPatch(_RecipeContract):
    path: RelativePath


class BuildNetwork(_RecipeContract):
    # The build reaches only these hosts; an empty list builds offline.
    hosts: list[StrictStr] = Field(max_length=64)

    @model_validator(mode="after")
    def unique_hosts(self) -> BuildNetwork:
        if len(self.hosts) != len(set(self.hosts)):
            raise ValueError("build network hosts must be unique")
        if any(not host for host in self.hosts):
            raise ValueError("build network hosts must be nonempty")
        return self


class RecipeBuildDefinition(_RecipeContract):
    base_image: RecipeImage
    context: BuildContext
    dockerfile: RelativePath
    patches: list[BuildPatch] = Field(max_length=64)
    network: BuildNetwork


class RecipeExecution(_RecipeContract):
    """The platform builds every recipe image from its pinned base and context."""

    build: RecipeBuildDefinition


class _RecipeSettings(_RecipeContract):
    knobs: dict[Identifier, RecipeSetting] = Field(default_factory=dict, max_length=64)


class RecipeSetting(_RecipeContract):
    value: Scalar
    change_effect: ChangeEffect


class RecipeGenerationSettings(_RecipeSettings):
    kind: Literal["generation"]
    context_tokens: RecipeIntegerSetting
    # Some engines leave scheduler capacity automatic. Null records that
    # runtime fact without mistaking a benchmark request count for a limit.
    concurrency: RecipeIntegerSetting | None = None
    max_batch_tokens: RecipeIntegerSetting | None = None


class RecipeEmbeddingSettings(_RecipeSettings):
    kind: Literal["embedding"]
    concurrency: RecipeIntegerSetting | None = None
    max_batch_tokens: RecipeIntegerSetting | None = None


class RecipeJobSettings(_RecipeSettings):
    kind: Literal["job"]
    concurrency: RecipeIntegerSetting | None = None


class RecipeIntegerSetting(RecipeSetting):
    value: StrictInt = Field(ge=1)


RecipeSettings = Annotated[
    RecipeGenerationSettings | RecipeEmbeddingSettings | RecipeJobSettings,
    Field(discriminator="kind"),
]


class RecipeRuntimeArgument(_RecipeContract):
    # This is an engine keyword, not a shell token or an exhaustive option
    # enum.  Keep its shape structural so the compiler can form a flag safely;
    # the pinned engine remains the authority for whether the name is known.
    name: StrictStr = Field(
        min_length=1, max_length=64, pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$"
    )
    value: RuntimeArgumentValue | None = Field(
        default=None,
        description="A literal process value; null is reserved for the setting-bound placeholder.",
    )
    setting: Identifier | None = None

    @field_validator("name", mode="before")
    @classmethod
    def name_has_no_nul(cls, value: object) -> object:
        if isinstance(value, str):
            _reject_nul(value, label="runtime argument name")
            if len(value.encode("utf-8")) > MAX_RUNTIME_ARGV_TOKEN_BYTES:
                raise ValueError("runtime argument name exceeds maximum UTF-8 size")
        return value

    @field_validator("value")
    @classmethod
    def value_is_bounded_json(
        cls, value: RuntimeArgumentValue | None
    ) -> RuntimeArgumentValue | None:
        return _validate_runtime_argument_value(value)

    @model_validator(mode="after")
    def one_source(self) -> RecipeRuntimeArgument:
        if (self.value is None) == (self.setting is None):
            raise ValueError("exactly one of value or setting is required")
        return self


class RecipeRuntimeEnvironment(_RecipeContract):
    name: StrictStr = Field(min_length=1, max_length=128)
    value: Scalar

    @field_validator("name")
    @classmethod
    def name_has_no_nul(cls, value: str) -> str:
        return _reject_nul(value, label="runtime environment name")

    @field_validator("value")
    @classmethod
    def value_has_no_nul(cls, value: Scalar) -> Scalar:
        _validate_runtime_argument_value(value)
        return value


Argv = Annotated[
    list[
        Annotated[
            StrictStr, Field(min_length=1, max_length=MAX_RUNTIME_ARGV_TOKEN_BYTES)
        ]
    ],
    Field(min_length=1, max_length=64),
]


class RecipeOptionError(ValueError):
    """An option choice does not name a declared option or one of its values."""


class RecipeOptionChoice(_RecipeContract):
    """One named value of an option, with the runtime changes it selects.

    ``args`` replace a base runtime argument of the same name in place, or are
    appended after the base arguments; ``env`` does the same for environment
    variables. Both go through the checks of ``runtime.arguments`` and
    ``runtime.environment``, and the platform applies its own security, mount
    and port rules to the merged result.
    """

    value: Identifier
    label: StrictStr = Field(min_length=1, max_length=64)
    help: StrictStr = Field(min_length=1, max_length=500)
    default: StrictBool = False
    args: list[RecipeRuntimeArgument] = Field(default_factory=list, max_length=32)
    env: dict[StrictStr, Scalar] = Field(default_factory=dict, max_length=32)

    @model_validator(mode="after")
    def literal_runtime_changes(self) -> RecipeOptionChoice:
        for argument in self.args:
            if argument.setting is not None:
                raise ValueError("option arguments must carry a literal value")
        names = [argument.name for argument in self.args]
        if len(names) != len(set(names)):
            raise ValueError("option choice arguments must be unique")
        for name, value in self.env.items():
            RecipeRuntimeEnvironment(name=name, value=value)
        return self


class RecipeOption(_RecipeContract):
    """A recipe-declared setting with a fixed set of named values.

    Users pick one of the enumerated choices; there are no free-form values.
    Exactly one choice is the default and applies whenever nothing is chosen.
    """

    name: Identifier
    label: StrictStr = Field(min_length=1, max_length=64)
    help: StrictStr = Field(min_length=1, max_length=500)
    choices: list[RecipeOptionChoice] = Field(min_length=2, max_length=16)

    @model_validator(mode="after")
    def one_default_unique_values(self) -> RecipeOption:
        values = [choice.value for choice in self.choices]
        if len(values) != len(set(values)):
            raise ValueError("option choice values must be unique")
        if sum(choice.default for choice in self.choices) != 1:
            raise ValueError("an option needs exactly one default choice")
        return self

    @property
    def default_value(self) -> str:
        return next(choice.value for choice in self.choices if choice.default)


class RecipeLifecycle(_RecipeContract):
    stop_timeout_seconds: StrictInt = Field(ge=1, le=600)


class RecipeRuntime(_RecipeContract):
    engine: Identifier
    entrypoint: Argv
    arguments: list[RecipeRuntimeArgument] = Field(max_length=MAX_RUNTIME_ARGUMENTS)
    environment: list[RecipeRuntimeEnvironment] = Field(max_length=128)
    lifecycle: RecipeLifecycle

    @field_validator("entrypoint")
    @classmethod
    def entrypoint_has_no_nul(cls, value: list[str]) -> list[str]:
        return _validate_argv(value)

    @model_validator(mode="after")
    def rendered_argument_argv_is_bounded(self) -> RecipeRuntime:
        tokens = list(self.entrypoint)
        for argument in self.arguments:
            tokens.extend(_runtime_argument_tokens(argument))
        _validate_argv(tokens)
        return self


class RecipeMemoryResources(_RecipeContract):
    """Unified (DGX Spark) memory one role needs."""

    # The most the workload uses at any time: startup or steady state.
    peak_bytes: StrictInt = Field(ge=1)
    # Memory left to the host operating system and runtime.
    reserve_bytes: StrictInt = Field(ge=0)


class RecipeDiskResources(_RecipeContract):
    image_bytes: StrictInt = Field(ge=0)
    artifact_bytes: StrictInt = Field(ge=0)
    # Scratch space the workload writes: staging plus caches.
    working_bytes: StrictInt = Field(ge=0)
    safety_margin_bytes: StrictInt = Field(ge=0)


class RecipeRoleResources(_RecipeContract):
    memory: RecipeMemoryResources
    disk: RecipeDiskResources


class RecipeTopologyRole(_RecipeContract):
    name: StrictStr = Field(min_length=1, max_length=64)
    count: StrictInt = Field(ge=1)
    endpoint_owner: StrictBool
    resources: RecipeRoleResources


class RecipeParallelism(_RecipeContract):
    tensor: StrictInt = Field(ge=1)
    pipeline: StrictInt = Field(ge=1)
    data: StrictInt = Field(ge=1)
    backend: StrictStr = Field(min_length=1, max_length=64)


class RecipeTopology(_RecipeContract):
    """Roles and their start order; everything else follows from node_count.

    One node runs alone. More nodes share one connected fabric: losing a rank
    withdraws the endpoint, recovery restarts the workers and then the
    entrypoint, and stopping always starts with the endpoint owner.
    """

    name: StrictStr = Field(min_length=1, max_length=64)
    node_count: StrictInt = Field(ge=1)
    roles: list[RecipeTopologyRole] = Field(min_length=1, max_length=32)
    parallelism: RecipeParallelism
    start_order: list[StrictStr] = Field(min_length=1, max_length=32)

    @property
    def distributed(self) -> bool:
        return self.node_count > 1

    @property
    def mode(self) -> Literal["single", "distributed"]:
        return "distributed" if self.distributed else "single"

    @property
    def world_size(self) -> int:
        return self.node_count

    @property
    def fabric_connectivity(self) -> Literal["none", "connected"]:
        return "connected" if self.distributed else "none"

    @property
    def fabric_minimum_bandwidth_mbps(self) -> int:
        return DISTRIBUTED_FABRIC_MINIMUM_MBPS if self.distributed else 0

    @property
    def stop_order(self) -> list[str]:
        """The endpoint owner stops first, then the other roles in order."""

        owners = [role.name for role in self.roles if role.endpoint_owner]
        return owners + [role.name for role in self.roles if not role.endpoint_owner]


class RecipeFileSlot(_RecipeContract):
    id: StrictStr = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,31}$")
    label: StrictStr = Field(min_length=1, max_length=64)
    description: StrictStr = Field(min_length=1, max_length=256)
    media_types: list[StrictStr] = Field(min_length=1, max_length=16)
    extensions: list[StrictStr] = Field(max_length=16)
    min_files: StrictInt = Field(ge=0, le=32)
    max_files: StrictInt = Field(ge=1, le=32)
    max_file_bytes: StrictInt = Field(ge=1)
    max_total_bytes: StrictInt = Field(ge=1)

    @model_validator(mode="after")
    def consistent(self) -> RecipeFileSlot:
        if (
            self.min_files > self.max_files
            or self.max_file_bytes > self.max_total_bytes
        ):
            raise ValueError("file slot limits are inconsistent")
        if len(self.media_types) != len(set(self.media_types)) or len(
            self.extensions
        ) != len(set(self.extensions)):
            raise ValueError("file slot media types and extensions must be unique")
        return self


class RecipeInputSlot(RecipeFileSlot):
    max_file_bytes: StrictInt = Field(ge=1, le=536870912)
    max_total_bytes: StrictInt = Field(ge=1, le=1073741824)


class RecipeOutputSlot(RecipeFileSlot):
    extensions: list[StrictStr] = Field(min_length=1, max_length=16)
    max_file_bytes: StrictInt = Field(ge=1, le=1073741824)
    max_total_bytes: StrictInt = Field(ge=1, le=2147483648)


class RecipeJobInput(_RecipeContract):
    # Inputs are staged read-only under /inputs.
    required: StrictBool
    media_types: list[StrictStr] = Field(min_length=1, max_length=16)
    max_bytes: StrictInt = Field(ge=1, le=1073741824)
    slots: list[RecipeInputSlot] | None = Field(
        default=None, min_length=1, max_length=32
    )


class RecipeJobOutput(_RecipeContract):
    # The job writes its outputs under /outputs.
    max_total_bytes: StrictInt = Field(ge=1, le=2147483648)
    slots: list[RecipeOutputSlot] = Field(min_length=1, max_length=32)


class RecipeOpenAIInterface(_RecipeContract):
    adapter: Literal["openai"]
    port: StrictInt = Field(ge=1024, le=65535)
    model_aliases: list[Annotated[StrictStr, Field(min_length=1, max_length=120)]] = (
        Field(min_length=1, max_length=16)
    )
    health_path: AbsolutePath


class RecipeJobInterface(_RecipeContract):
    adapter: Literal["image-job", "audio-job", "video-job", "mesh-job", "artifact-job"]
    input: RecipeJobInput | None = None
    output: RecipeJobOutput


RecipeInterface = Annotated[
    RecipeOpenAIInterface | RecipeJobInterface, Field(discriminator="adapter")
]


ServingKind = Literal[
    "openai.health",
    "openai.chat",
    "openai.vision",
    "openai.tools",
    "openai.completion",
    "openai.embedding",
    "image-job.output",
    "audio-job.output",
    "video-job.output",
    "mesh-job.output",
    "artifact-job.output",
]
ServingAssertion = Literal[
    "endpoint.healthy",
    "chat.nonempty",
    "chat.output-cap",
    "tools.called",
    "completion.nonempty",
    "completion.output-cap",
    "embedding.nonempty",
    "inference.completed",
    "artifact.output",
]

_ASSERTIONS_BY_KIND: dict[str, frozenset[str]] = {
    "openai.health": frozenset({"endpoint.healthy"}),
    "openai.chat": frozenset({"chat.nonempty", "chat.output-cap"}),
    "openai.vision": frozenset({"chat.nonempty", "chat.output-cap"}),
    "openai.tools": frozenset({"chat.nonempty", "chat.output-cap", "tools.called"}),
    "openai.completion": frozenset({"completion.nonempty", "completion.output-cap"}),
    "openai.embedding": frozenset({"embedding.nonempty"}),
    "image-job.output": frozenset({"inference.completed", "artifact.output"}),
    "audio-job.output": frozenset({"inference.completed", "artifact.output"}),
    "video-job.output": frozenset({"inference.completed", "artifact.output"}),
    "mesh-job.output": frozenset({"inference.completed", "artifact.output"}),
    "artifact-job.output": frozenset({"inference.completed", "artifact.output"}),
}


class RecipeHttpServingRequest(_RecipeContract):
    transport: Literal["http"]
    method: Literal["GET", "POST"]
    path: AbsolutePath
    body: dict[StrictStr, JsonValue] | None = Field(
        default=None, min_length=1, max_length=32
    )

    @model_validator(mode="after")
    def method_body(self) -> RecipeHttpServingRequest:
        if (self.method == "POST") != (self.body is not None):
            raise ValueError("POST requires a body; GET must omit the body")
        return self


class RecipeJobServingRequest(_RecipeContract):
    """A job check stages its fixture as the input when the interface has one."""

    transport: Literal["job"]
    fixture: RelativePath
    input_slots: dict[Identifier, RelativePath] = Field(
        default_factory=dict, max_length=32
    )
    output_slot: Identifier


ServingRequest = Annotated[
    RecipeHttpServingRequest | RecipeJobServingRequest, Field(discriminator="transport")
]


class RecipeValidationCheck(_RecipeContract):
    name: Identifier
    kind: ServingKind
    request: ServingRequest
    assertions: list[ServingAssertion] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def executable(self) -> RecipeValidationCheck:
        if len(self.assertions) != len(set(self.assertions)):
            raise ValueError("serving assertions must be unique")
        if not set(self.assertions) <= _ASSERTIONS_BY_KIND[self.kind]:
            raise ValueError("serving assertion is not applicable to its check kind")
        is_job = self.kind.endswith(".output")
        if is_job != isinstance(self.request, RecipeJobServingRequest):
            raise ValueError("serving request transport must match serving kind")
        if self.kind == "openai.health":
            if (
                not isinstance(self.request, RecipeHttpServingRequest)
                or self.request.method != "GET"
                or self.assertions != ["endpoint.healthy"]
            ):
                raise ValueError(
                    "health checks require an HTTP GET and endpoint.healthy"
                )
        elif not is_job:
            if (
                not isinstance(self.request, RecipeHttpServingRequest)
                or self.request.method != "POST"
            ):
                raise ValueError("representative OpenAI checks require an HTTP POST")
            body = self.request.body or {}
            if self.kind in {"openai.chat", "openai.vision", "openai.tools"}:
                messages = body.get("messages")
                if (
                    self.request.path != "/v1/chat/completions"
                    or not isinstance(messages, list)
                    or not messages
                ):
                    raise ValueError(
                        "chat checks require messages at /v1/chat/completions"
                    )
                if self.kind == "openai.vision" and not any(
                    isinstance(message, dict)
                    and isinstance(content := message.get("content"), list)
                    and any(
                        isinstance(part, dict)
                        and part.get("type") == "image_url"
                        and isinstance(image_url := part.get("image_url"), dict)
                        and isinstance(url := image_url.get("url"), str)
                        and bool(url)
                        for part in content
                    )
                    for message in messages
                ):
                    raise ValueError("vision checks require image_url content")
                required = (
                    "tools.called" if self.kind == "openai.tools" else "chat.nonempty"
                )
                if (
                    required not in self.assertions
                    or self.kind == "openai.tools"
                    and not body.get("tools")
                ):
                    raise ValueError(
                        "OpenAI check does not exercise its declared behavior"
                    )
            elif self.kind == "openai.completion":
                if (
                    self.request.path != "/v1/completions"
                    or not body.get("prompt")
                    or "completion.nonempty" not in self.assertions
                ):
                    raise ValueError(
                        "completion checks require a prompt and completion.nonempty"
                    )
            elif self.kind == "openai.embedding":
                if (
                    self.request.path != "/v1/embeddings"
                    or not body.get("input")
                    or "embedding.nonempty" not in self.assertions
                ):
                    raise ValueError(
                        "embedding checks require input and embedding.nonempty"
                    )
            output_cap = body.get("max_tokens")
            if any(item.endswith("output-cap") for item in self.assertions) and (
                type(output_cap) is not int or output_cap <= 0
            ):
                raise ValueError(
                    "output-cap requires a positive max_tokens request limit"
                )
        else:
            if (
                not isinstance(self.request, RecipeJobServingRequest)
                or not self.request.output_slot
            ):
                raise ValueError("job checks require an output slot")
            if "artifact.output" not in self.assertions:
                raise ValueError("job checks require an artifact.output assertion")
        return self


class RecipeServingValidation(_RecipeContract):
    interface: Literal[
        "openai", "image-job", "audio-job", "video-job", "mesh-job", "artifact-job"
    ]
    checks: list[RecipeValidationCheck] = Field(min_length=1, max_length=32)


class RecipeValidation(_RecipeContract):
    serving: RecipeServingValidation


class RecipeProvenance(_RecipeContract):
    source_reference: StrictStr | None = Field(default=None, max_length=2048)
    attribution: list[StrictStr] = Field(max_length=32)


class RecipeRelease(_RecipeContract):
    """The version this recipe runs.

    When the upstream project publishes versions, this is the upstream version
    and its release date (for example ``1.6`` released 2026-09-17). A recipe
    whose upstream has no versions carries its own semantic version instead.
    The recipe library's own version follows the contract, not recipe content.
    """

    version: StrictStr = Field(
        min_length=1, max_length=64, pattern=r"^[0-9A-Za-z][0-9A-Za-z.+_-]{0,63}$"
    )
    released_at: Annotated[StrictStr, Field(pattern=r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")]

    @field_validator("released_at")
    @classmethod
    def valid_date(cls, value: str) -> str:
        try:
            parsed = date.fromisoformat(value)
        except ValueError as error:
            raise ValueError("released_at must be an ISO 8601 calendar date") from error
        if parsed.isoformat() != value:
            raise ValueError("released_at must use YYYY-MM-DD form")
        return value


class RecipeDefinition(_RecipeContract):
    """The sole public recipe authoring contract."""

    kind: Literal["recipe"] = "recipe"
    identity: RecipeIdentity
    metadata: RecipeMetadata
    models: list[RecipeModelSelection] = Field(min_length=1, max_length=32)
    execution: RecipeExecution
    runtime: RecipeRuntime
    topology: RecipeTopology
    interfaces: list[RecipeInterface] = Field(min_length=1, max_length=1)
    validation: RecipeValidation
    provenance: RecipeProvenance
    settings: RecipeSettings
    release: RecipeRelease
    options: list[RecipeOption] = Field(default_factory=list, max_length=16)

    def resolve_options(
        self, choices: Mapping[str, str] | None = None
    ) -> dict[str, str]:
        """Return the effective ``{option: value}`` for every declared option.

        Options not named in ``choices`` take their default. An unknown option
        name or value raises :class:`RecipeOptionError` listing what is valid.
        """

        supplied = dict(choices or {})
        declared = {option.name: option for option in self.options}
        unknown = sorted(set(supplied) - set(declared))
        if unknown:
            raise RecipeOptionError(
                f"unknown recipe option {unknown[0]!r}; options: "
                + ", ".join(sorted(declared) or ["none"])
            )
        resolved: dict[str, str] = {}
        for option in self.options:
            value = supplied.get(option.name, option.default_value)
            if value not in {choice.value for choice in option.choices}:
                raise RecipeOptionError(
                    f"unknown value {value!r} for option {option.name!r}; choices: "
                    + ", ".join(choice.value for choice in option.choices)
                )
            resolved[option.name] = value
        return resolved

    def with_option_choices(
        self, choices: Mapping[str, str] | None = None
    ) -> RecipeDefinition:
        """Return this recipe with the chosen options merged into its runtime.

        The result has the same runtime shape as an authored recipe, so every
        downstream check and compile step applies to it unchanged.
        """

        resolved = self.resolve_options(choices)
        arguments = list(self.runtime.arguments)
        environment = list(self.runtime.environment)
        for option in self.options:
            chosen = next(c for c in option.choices if c.value == resolved[option.name])
            for argument in chosen.args:
                index = next(
                    (i for i, a in enumerate(arguments) if a.name == argument.name),
                    None,
                )
                if index is None:
                    arguments.append(argument)
                else:
                    arguments[index] = argument
            for name, value in chosen.env.items():
                item = RecipeRuntimeEnvironment(name=name, value=value)
                index = next(
                    (i for i, e in enumerate(environment) if e.name == name), None
                )
                if index is None:
                    environment.append(item)
                else:
                    environment[index] = item
        if len(arguments) > MAX_RUNTIME_ARGUMENTS or len(environment) > 128:
            raise RecipeOptionError("option choices exceed the runtime bounds")
        tokens = list(self.runtime.entrypoint)
        for argument in arguments:
            tokens.extend(_runtime_argument_tokens(argument))
        try:
            _validate_argv(tokens)
        except ValueError as error:
            raise RecipeOptionError(str(error)) from error
        runtime = self.runtime.model_copy(
            update={"arguments": arguments, "environment": environment}
        )
        return self.model_copy(update={"runtime": runtime})

    @model_validator(mode="after")
    def semantic_rules(self) -> RecipeDefinition:
        self._option_rules()
        refs = [selection.model for selection in self.models]
        if len({(r.kind, r.publisher, r.slug, r.content_sha256) for r in refs}) != len(
            refs
        ):
            raise ValueError("recipe references must be unique")
        settings_kind = self.settings.kind
        has_openai = any(interface.adapter == "openai" for interface in self.interfaces)
        if has_openai != (settings_kind in {"generation", "embedding"}):
            raise ValueError("settings kind must match the serving interface")
        role_names = [role.name for role in self.topology.roles]
        if (
            len(role_names) != len(set(role_names))
            or sum(role.count for role in self.topology.roles)
            != self.topology.node_count
        ):
            raise ValueError("topology roles must be unique and sum to node_count")
        owners = [role for role in self.topology.roles if role.endpoint_owner]
        if len(owners) != 1 or owners[0].count != 1:
            raise ValueError("exactly one single-node role must own the endpoint")
        p = self.topology.parallelism
        if p.tensor * p.pipeline * p.data != self.topology.node_count:
            raise ValueError("parallelism product must equal node_count")
        if set(self.topology.start_order) != set(role_names) or len(
            self.topology.start_order
        ) != len(role_names):
            raise ValueError("topology start order must contain every role once")
        if len({selection.id for selection in self.models}) != len(self.models):
            raise ValueError("model selection IDs must be unique")
        selectors = {
            item.id: item for selection in self.models for item in selection.files
        }
        if len(selectors) != sum(len(selection.files) for selection in self.models):
            raise ValueError("model file selector IDs must be unique")
        for selection in self.models:
            for selector in selection.files:
                if not set(selector.roles) <= set(role_names):
                    raise ValueError(
                        "model file selector roles must match topology role assignments"
                    )
        setting_names = set(self.settings.knobs)
        for name in ("context_tokens", "concurrency", "max_batch_tokens"):
            if getattr(self.settings, name, None) is not None:
                setting_names.add(name)
        if any(
            argument.setting not in setting_names
            for argument in self.runtime.arguments
            if argument.setting is not None
        ):
            raise ValueError("runtime argument references an unknown setting")
        interface_names = [interface.adapter for interface in self.interfaces]
        if len(interface_names) != len(set(interface_names)):
            raise ValueError("interfaces must be unique")
        if self.validation.serving.interface not in interface_names:
            raise ValueError("validation interface is not declared")
        if (
            len(self.validation.serving.checks) == 1
            and self.validation.serving.checks[0].kind == "openai.health"
        ):
            raise ValueError("health alone does not test model serving")
        for check in self.validation.serving.checks:
            if (
                check.kind.endswith(".output")
                and check.kind.removesuffix(".output")
                != self.validation.serving.interface
            ):
                raise ValueError("job serving check does not match interface")
            if (
                check.kind.startswith("openai.")
                and self.validation.serving.interface != "openai"
            ):
                raise ValueError("OpenAI serving check does not match interface")
            if check.kind.endswith(".output"):
                interface = next(
                    interface
                    for interface in self.interfaces
                    if interface.adapter == self.validation.serving.interface
                )
                if not isinstance(interface, RecipeJobInterface):
                    raise ValueError("job serving requires a job interface")
                request = check.request
                if not isinstance(request, RecipeJobServingRequest):
                    raise ValueError("job serving requires a filesystem request")
                output_ids = {slot.id for slot in interface.output.slots}
                if request.output_slot not in output_ids:
                    raise ValueError(
                        "job request output_slot is not declared by the interface"
                    )
                if interface.input is None:
                    if request.input_slots:
                        raise ValueError(
                            "job request input bindings require an interface input"
                        )
                else:
                    declared_input_ids = {
                        slot.id for slot in interface.input.slots or []
                    }
                    if not set(request.input_slots) <= declared_input_ids:
                        raise ValueError(
                            "job request input slot is not declared by the interface"
                        )
        return self

    def _option_rules(self) -> None:
        names = [option.name for option in self.options]
        if len(names) != len(set(names)):
            raise ValueError("recipe option names must be unique")
        touched: dict[tuple[str, str], str] = {}
        for option in self.options:
            keys = {
                *(("arg", a.name) for c in option.choices for a in c.args),
                *(("env", n) for c in option.choices for n in c.env),
            }
            for key in keys:
                if key in touched:
                    raise ValueError(
                        f"two options change the same runtime {key[0]} {key[1]}"
                    )
                touched[key] = option.name
        # Every choice must merge into a valid runtime; the default of every
        # other option stays in place, so one choice at a time is enough.
        for option in self.options:
            for choice in option.choices:
                try:
                    self.with_option_choices({option.name: choice.value})
                except RecipeOptionError as error:
                    raise ValueError(str(error)) from error
