"""Consumer projections of the library fields used by the sweep.

Extra producer fields are intentionally ignored. A malformed consumed field
invalidates only its row; unavailable assessments still use their defaults.
"""

from pydantic import AliasChoices, AliasPath, BaseModel, ConfigDict, Field
from vonk_forge_contracts.recipe import RecipeValidationCheck


class Projection(BaseModel):
    model_config = ConfigDict(strict=True)


class Interface(Projection):
    adapter: str = ""
    port: int = 0
    model_aliases: list[str] = Field(default_factory=list)


class Disk(Projection):
    artifact_bytes: int = 0


class RoleResources(Projection):
    disk: Disk = Field(default_factory=Disk)


class Role(Projection):
    resources: RoleResources = Field(default_factory=RoleResources)


class Topology(Projection):
    node_count: int = 1
    roles: list[Role] = Field(default_factory=list)


class ModelIdentity(Projection):
    content_sha256: str = Field(
        validation_alias=AliasChoices(
            "content_sha256", AliasPath("model", "content_sha256")
        )
    )


class Serving(Projection):
    checks: list[RecipeValidationCheck] = Field(default_factory=list)


class Validation(Projection):
    serving: Serving = Field(default_factory=Serving)


class RecipeProjection(Projection):
    validation: Validation = Field(default_factory=Validation)
    interfaces: list[Interface] = Field(default_factory=list)
    topology: Topology = Field(default_factory=Topology)
    models: list[ModelIdentity] = Field(default_factory=list)
    runtime_engine: str = Field(
        default="",
        validation_alias=AliasChoices("runtime_engine", AliasPath("runtime", "engine")),
    )
    base_image_digest: str = Field(
        default="",
        validation_alias=AliasChoices(
            "base_image_digest", AliasPath("execution", "build", "base_image", "digest")
        ),
    )


class RecipeIdentity(Projection):
    title: str = ""
    content_sha256: str = ""
    recipe_revision_id: str = ""


class PrebuiltImage(Projection):
    """Signed catalogue image pin, independent of the recipe JSON digest."""

    reference: str = Field(pattern=r"^.+@sha256:[0-9a-f]{64}$")
    build_key: str = Field(pattern=r"^[0-9a-f]{64}$")


class RuntimeImageIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    build_key: str
    image_digest: str


class Resources(Projection):
    memory_bytes: int = 0
    image_bytes: int = 0
    disk_bytes: int = 0


class RecipeRow(Projection):
    selector: str
    # Expand/contract: legacy document shape accepted until every Controller serves
    # projected. Keep both paths until the coordinator removes legacy support.
    projected: RecipeProjection = Field(
        validation_alias=AliasChoices("projected", "document")
    )
    identity: RecipeIdentity = Field(default_factory=RecipeIdentity)
    resources: Resources = Field(default_factory=Resources)
    node_count: int = 1
    usage: list[str] = Field(default_factory=list)
    model_selectors: list[str] = Field(default_factory=list)
    engine: str = ""
    creator: str = ""
    updated_at: str = ""
    prebuilt_image: PrebuiltImage | None = None


class ModelRow(Projection):
    selector: str
    identity: ModelIdentity
    resources: Resources = Field(default_factory=Resources)
