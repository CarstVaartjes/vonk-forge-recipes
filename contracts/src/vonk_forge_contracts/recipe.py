"""Strict public recipe authoring contract.

Only author intent is represented here.  Runtime resolution, image receipts,
engine compatibility, and placement plans remain platform-owned concerns.
"""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    model_validator,
)
from typing_extensions import TypeAliasType


class _RecipeContract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


Identifier = Annotated[StrictStr, Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_-]*$")]
Sha256 = Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]
_SEGMENT = r"(?:[A-Za-z0-9_-][A-Za-z0-9._-]*|\.[A-Za-z0-9_-][A-Za-z0-9._-]*)"
AbsolutePath = Annotated[StrictStr, Field(max_length=256, pattern=rf"^/{_SEGMENT}(?:/{_SEGMENT})*$")]
RelativePath = Annotated[StrictStr, Field(max_length=256, pattern=rf"^{_SEGMENT}(?:/{_SEGMENT})*$")]
Scalar = StrictStr | StrictInt | StrictBool
JsonValue = TypeAliasType("JsonValue", Scalar | None | list["JsonValue"] | dict[StrictStr, "JsonValue"])
ChangeEffect = Literal["none", "restart", "reprepare", "rebuild"]


class RecipeReference(_RecipeContract):
    kind: Literal["model"]
    publisher: StrictStr = Field(min_length=2, max_length=64, pattern=r"^[a-z0-9][a-z0-9-]{1,62}$")
    slug: StrictStr = Field(min_length=2, max_length=64, pattern=r"^[a-z0-9][a-z0-9-]{1,62}$")
    content_sha256: Sha256


class ModelVersionReference(RecipeReference):
    kind: Literal["model"]


class RecipeIdentity(_RecipeContract):
    publisher: StrictStr = Field(min_length=2, max_length=64, pattern=r"^[a-z0-9][a-z0-9-]{1,62}$")
    slug: StrictStr = Field(min_length=2, max_length=64, pattern=r"^[a-z0-9][a-z0-9-]{1,62}$")


class RecipeMetadata(_RecipeContract):
    title: StrictStr = Field(min_length=1, max_length=120)
    description: StrictStr = Field(min_length=1, max_length=4000)
    tags: list[StrictStr] = Field(max_length=20)
    alignment: Literal["standard", "abliterated", "derisked", "other-modified", "unspecified"] | None = None


class RecipeMount(_RecipeContract):
    target: AbsolutePath
    read_only: Literal[True]


class RecipeModelFile(_RecipeContract):
    id: StrictStr = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    file_id: StrictStr = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    roles: list[StrictStr] = Field(min_length=1, max_length=32)
    mount: RecipeMount


class RecipeModelSelection(_RecipeContract):
    id: Identifier
    model: ModelVersionReference
    files: list[RecipeModelFile] = Field(min_length=1, max_length=256)


class BuildContext(_RecipeContract):
    path: RelativePath


class RecipeImage(_RecipeContract):
    repository: StrictStr = Field(min_length=1, max_length=512, pattern=r"^[a-z0-9][a-z0-9._/-]*$")
    digest: Sha256
    platform: Literal["linux/arm64"]


class BuildPatch(_RecipeContract):
    path: RelativePath


class BuildArgument(_RecipeContract):
    name: StrictStr = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    value: Scalar


class BuildNetwork(_RecipeContract):
    mode: Literal["none", "public"]
    hosts: list[StrictStr] = Field(max_length=64)


class RecipeBuildDefinition(_RecipeContract):
    base_image: RecipeImage
    context: BuildContext
    dockerfile: RelativePath
    patches: list[BuildPatch] = Field(max_length=64)
    target: StrictStr | None = Field(default=None, max_length=64)
    arguments: list[BuildArgument] = Field(max_length=64)
    network: BuildNetwork


class RecipeImageExecution(_RecipeContract):
    mode: Literal["image"]
    image: RecipeImage


class RecipeBuildExecution(_RecipeContract):
    mode: Literal["build"]
    build: RecipeBuildDefinition


RecipeExecution = Annotated[RecipeImageExecution | RecipeBuildExecution, Field(discriminator="mode")]


class _RecipeSettings(_RecipeContract):
    knobs: dict[Identifier, RecipeSetting] = Field(default_factory=dict, max_length=64)


class RecipeSetting(_RecipeContract):
    value: Scalar
    change_effect: ChangeEffect


class RecipeGenerationSettings(_RecipeSettings):
    kind: Literal["generation"]
    context_tokens: RecipeIntegerSetting
    concurrency: RecipeIntegerSetting
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


RecipeSettings = Annotated[RecipeGenerationSettings | RecipeEmbeddingSettings | RecipeJobSettings, Field(discriminator="kind")]


class RecipeRuntimeArgument(_RecipeContract):
    name: StrictStr = Field(min_length=1, max_length=64)
    value: Scalar | None = None
    setting: Identifier | None = None

    @model_validator(mode="after")
    def one_source(self) -> RecipeRuntimeArgument:
        if (self.value is None) == (self.setting is None):
            raise ValueError("exactly one of value or setting is required")
        return self


class RecipeRuntimeEnvironment(_RecipeContract):
    name: StrictStr = Field(min_length=1, max_length=128)
    value: Scalar | None = None
    secret: StrictStr | None = None

    @model_validator(mode="after")
    def one_source(self) -> RecipeRuntimeEnvironment:
        if (self.value is None) == (self.secret is None):
            raise ValueError("exactly one of value or secret is required")
        return self


class RecipeReadiness(_RecipeContract):
    strategy: Literal["endpoint-owner", "endpoint-owner-after-all-ranks"]
    path: AbsolutePath
    timeout_seconds: StrictInt = Field(ge=1, le=3600)


class RecipeFailurePolicy(_RecipeContract):
    rank_loss: Literal["not-applicable", "withdraw-endpoint"]
    recovery: Literal["restart-entrypoint", "restart-worker-then-entrypoint"]


Argv = Annotated[list[Annotated[StrictStr, Field(min_length=1, max_length=4096)]], Field(min_length=1, max_length=64)]


class RecipeLifecycle(_RecipeContract):
    pre_start: list[Argv] = Field(max_length=16)
    post_stop: list[Argv] = Field(max_length=16)
    stop_timeout_seconds: StrictInt = Field(ge=1, le=600)
    readiness: RecipeReadiness | None = None
    failure: RecipeFailurePolicy | None = None


class RecipeRuntime(_RecipeContract):
    engine: Identifier
    entrypoint: Argv
    arguments: list[RecipeRuntimeArgument] = Field(max_length=128)
    environment: list[RecipeRuntimeEnvironment] = Field(max_length=128)
    lifecycle: RecipeLifecycle


class RecipeMemoryResources(_RecipeContract):
    kind: Literal["unified", "host", "accelerator"]
    startup_peak_bytes: StrictInt = Field(ge=1)
    steady_state_bytes: StrictInt = Field(ge=1)
    runtime_growth_bytes: StrictInt = Field(ge=0)
    system_reserve_bytes: StrictInt = Field(ge=0)


class RecipeDiskResources(_RecipeContract):
    image_bytes: StrictInt = Field(ge=0)
    artifact_bytes: StrictInt = Field(ge=0)
    staging_bytes: StrictInt = Field(ge=0)
    cache_bytes: StrictInt = Field(ge=0)
    rollback_bytes: StrictInt = Field(ge=0)
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
    world_size: StrictInt = Field(ge=1)
    tensor: StrictInt = Field(ge=1)
    pipeline: StrictInt = Field(ge=1)
    data: StrictInt = Field(ge=1)
    backend: StrictStr = Field(min_length=1, max_length=64)


class RecipeFabric(_RecipeContract):
    connectivity: Literal["none", "connected", "full_mesh", "switch"]
    minimum_bandwidth_mbps: StrictInt = Field(ge=0)


class RecipeTopology(_RecipeContract):
    name: StrictStr = Field(min_length=1, max_length=64)
    mode: Literal["single", "distributed", "tensor_parallel", "pipeline_parallel", "data_parallel", "hybrid", "ray", "mpi"]
    node_count: StrictInt = Field(ge=1)
    roles: list[RecipeTopologyRole] = Field(min_length=1, max_length=32)
    parallelism: RecipeParallelism
    fabric: RecipeFabric
    start_order: list[StrictStr] = Field(min_length=1, max_length=32)
    stop_order: list[StrictStr] = Field(min_length=1, max_length=32)


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
        if self.min_files > self.max_files or self.max_file_bytes > self.max_total_bytes:
            raise ValueError("file slot limits are inconsistent")
        if len(self.media_types) != len(set(self.media_types)) or len(self.extensions) != len(set(self.extensions)):
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
    path: Literal["/inputs"]
    required: StrictBool
    media_types: list[StrictStr] = Field(min_length=1, max_length=16)
    max_bytes: StrictInt = Field(ge=1, le=1073741824)
    slots: list[RecipeInputSlot] | None = Field(default=None, min_length=1, max_length=32)


class RecipeJobOutput(_RecipeContract):
    path: Literal["/outputs"]
    max_total_bytes: StrictInt = Field(ge=1, le=2147483648)
    slots: list[RecipeOutputSlot] = Field(min_length=1, max_length=32)


class RecipeOpenAIInterface(_RecipeContract):
    adapter: Literal["openai"]
    port: StrictInt = Field(ge=1024, le=65535)
    model_aliases: list[Annotated[StrictStr, Field(min_length=1, max_length=120)]] = Field(min_length=1, max_length=16)
    health_path: AbsolutePath


class RecipeJobInterface(_RecipeContract):
    adapter: Literal["image-job", "audio-job", "video-job", "mesh-job", "artifact-job"]
    path: Literal["/outputs"]
    input: RecipeJobInput | None = None
    output: RecipeJobOutput


RecipeInterface = Annotated[RecipeOpenAIInterface | RecipeJobInterface, Field(discriminator="adapter")]


ServingKind = Literal[
    "openai.health", "openai.chat", "openai.vision", "openai.tools", "openai.completion", "openai.embedding",
    "image-job.output", "audio-job.output", "video-job.output", "mesh-job.output", "artifact-job.output",
]
ServingAssertion = Literal[
    "endpoint.healthy", "chat.nonempty", "chat.output-cap", "tools.called", "completion.nonempty", "completion.output-cap",
    "embedding.nonempty", "inference.completed", "artifact.output",
]


class RecipeHttpServingRequest(_RecipeContract):
    transport: Literal["http"]
    method: Literal["GET", "POST"]
    path: AbsolutePath
    body: dict[StrictStr, JsonValue] | None = Field(default=None, min_length=1, max_length=32)

    @model_validator(mode="after")
    def method_body(self) -> RecipeHttpServingRequest:
        if (self.method == "POST") != (self.body is not None):
            raise ValueError("POST requires a body; GET must omit the body")
        return self


class RecipeJobServingRequest(_RecipeContract):
    transport: Literal["job"]
    fixture: RelativePath
    input_path: Literal["/inputs"] | None = None
    input_slots: dict[Identifier, RelativePath] = Field(default_factory=dict, max_length=32)
    output_path: Literal["/outputs"]
    output_slot: Identifier

    @model_validator(mode="after")
    def input_binding(self) -> RecipeJobServingRequest:
        if self.input_path is None and self.input_slots:
            raise ValueError("input slots require input_path /inputs")
        return self


ServingRequest = Annotated[RecipeHttpServingRequest | RecipeJobServingRequest, Field(discriminator="transport")]


class RecipeValidationCheck(_RecipeContract):
    name: Identifier
    kind: ServingKind
    request: ServingRequest
    assertions: list[ServingAssertion] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def executable(self) -> RecipeValidationCheck:
        if len(self.assertions) != len(set(self.assertions)):
            raise ValueError("serving assertions must be unique")
        is_job = self.kind.endswith(".output")
        if is_job != isinstance(self.request, RecipeJobServingRequest):
            raise ValueError("serving request transport must match serving kind")
        if self.kind == "openai.health":
            if not isinstance(self.request, RecipeHttpServingRequest) or self.request.method != "GET" or self.assertions != ["endpoint.healthy"]:
                raise ValueError("health checks require an HTTP GET and endpoint.healthy")
        elif not is_job:
            if not isinstance(self.request, RecipeHttpServingRequest) or self.request.method != "POST":
                raise ValueError("representative OpenAI checks require an HTTP POST")
            body = self.request.body or {}
            if self.kind in {"openai.chat", "openai.vision", "openai.tools"}:
                if self.request.path != "/v1/chat/completions" or not isinstance(body.get("messages"), list) or not body["messages"]:
                    raise ValueError("chat checks require messages at /v1/chat/completions")
                required = "tools.called" if self.kind == "openai.tools" else "chat.nonempty"
                if required not in self.assertions or self.kind == "openai.tools" and not body.get("tools"):
                    raise ValueError("OpenAI check does not exercise its declared behavior")
            elif self.kind == "openai.completion":
                if self.request.path != "/v1/completions" or not body.get("prompt") or "completion.nonempty" not in self.assertions:
                    raise ValueError("completion checks require a prompt and completion.nonempty")
            elif self.kind == "openai.embedding":
                if self.request.path != "/v1/embeddings" or not body.get("input") or "embedding.nonempty" not in self.assertions:
                    raise ValueError("embedding checks require input and embedding.nonempty")
            if any(item.endswith("output-cap") for item in self.assertions) and (type(body.get("max_tokens")) is not int or body["max_tokens"] <= 0):
                raise ValueError("output-cap requires a positive max_tokens request limit")
        else:
            if not self.request.output_slot:
                raise ValueError("job checks require an output slot")
            if "artifact.output" not in self.assertions:
                raise ValueError("job checks require an artifact.output assertion")
        return self


class RecipeServingValidation(_RecipeContract):
    interface: Literal["openai", "image-job", "audio-job", "video-job", "mesh-job", "artifact-job"]
    checks: list[RecipeValidationCheck] = Field(min_length=1, max_length=32)


class RecipeBenchmark(_RecipeContract):
    name: StrictStr = Field(min_length=1, max_length=64)
    framework: StrictStr = Field(min_length=1, max_length=64)
    configuration: dict[StrictStr, Scalar]


class RecipeValidation(_RecipeContract):
    benchmarks: list[RecipeBenchmark] = Field(max_length=32)
    serving: RecipeServingValidation


class RecipeProvenance(_RecipeContract):
    source_kind: Literal["local", "workload_run", "global", "fork"]
    source_reference: StrictStr | None = Field(default=None, max_length=2048)
    attribution: list[StrictStr] = Field(max_length=32)


class RecipeDefinition(_RecipeContract):
    """The sole public recipe authoring contract."""

    schema_version: Literal[2] = 2
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

    @model_validator(mode="after")
    def semantic_rules(self) -> RecipeDefinition:
        refs = [selection.model for selection in self.models]
        if len({(r.kind, r.publisher, r.slug, r.content_sha256) for r in refs}) != len(refs):
            raise ValueError("recipe references must be unique")
        settings_kind = self.settings.kind
        has_openai = any(interface.adapter == "openai" for interface in self.interfaces)
        if has_openai != (settings_kind in {"generation", "embedding"}):
            raise ValueError("settings kind must match the serving interface")
        role_names = [role.name for role in self.topology.roles]
        if len(role_names) != len(set(role_names)) or sum(role.count for role in self.topology.roles) != self.topology.node_count:
            raise ValueError("topology roles must be unique and sum to node_count")
        owners = [role for role in self.topology.roles if role.endpoint_owner]
        if len(owners) != 1 or owners[0].count != 1:
            raise ValueError("exactly one single-node role must own the endpoint")
        p = self.topology.parallelism
        if p.world_size != p.tensor * p.pipeline * p.data or p.world_size != self.topology.node_count:
            raise ValueError("world_size and parallelism product must equal node_count")
        if (self.topology.node_count == 1) != (self.topology.fabric.connectivity == "none"):
            raise ValueError("one-node topology requires no fabric; multi-node topology requires fabric")
        if set(self.topology.start_order) != set(role_names) or len(self.topology.start_order) != len(role_names) or set(self.topology.stop_order) != set(role_names) or len(self.topology.stop_order) != len(role_names):
            raise ValueError("topology orders must contain every role exactly once")
        if len({selection.id for selection in self.models}) != len(self.models):
            raise ValueError("model selection IDs must be unique")
        selectors = {item.id: item for selection in self.models for item in selection.files}
        if len(selectors) != sum(len(selection.files) for selection in self.models):
            raise ValueError("model file selector IDs must be unique")
        for selection in self.models:
            for selector in selection.files:
                if not set(selector.roles) <= set(role_names):
                    raise ValueError("model file selector roles must match topology role assignments")
        setting_names = set(self.settings.knobs)
        for name in ("context_tokens", "concurrency", "max_batch_tokens"):
            if getattr(self.settings, name, None) is not None:
                setting_names.add(name)
        if any(argument.setting not in setting_names for argument in self.runtime.arguments if argument.setting is not None):
            raise ValueError("runtime argument references an unknown setting")
        interface_names = [interface.adapter for interface in self.interfaces]
        if len(interface_names) != len(set(interface_names)):
            raise ValueError("interfaces must be unique")
        if self.validation.serving.interface not in interface_names:
            raise ValueError("validation interface is not declared")
        if len(self.validation.serving.checks) == 1 and self.validation.serving.checks[0].kind == "openai.health":
            raise ValueError("health alone does not test model serving")
        for check in self.validation.serving.checks:
            if check.kind.endswith(".output") and check.kind.removesuffix(".output") != self.validation.serving.interface:
                raise ValueError("job serving check does not match interface")
            if check.kind.startswith("openai.") and self.validation.serving.interface != "openai":
                raise ValueError("OpenAI serving check does not match interface")
        return self
