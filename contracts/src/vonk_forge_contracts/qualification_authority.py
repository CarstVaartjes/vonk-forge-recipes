"""Strict contracts for recipe qualification authorities.

The qualification authority binds an ordered catalog to executable batches and
recovery-coverage groups: one representative recipe exercises a failure mode
for every member of its group. Campaign results are a plain log, not receipts.
"""

from __future__ import annotations

import hashlib
import json
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

from ._schema_version import (
    QualificationAuthoritySchemaVersion,
    QualificationCampaignSchemaVersion,
)
from .model import ModelTerritorialRestrictions

Sha256 = Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]
GitSha = Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{40}$")]
RecipeKey = Annotated[
    StrictStr,
    Field(
        min_length=5,
        max_length=127,
        pattern=r"^[a-z0-9][a-z0-9-]{1,62}/[a-z0-9][a-z0-9-]{1,62}$",
    ),
]
Identifier = Annotated[
    StrictStr,
    Field(min_length=1, max_length=256, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$"),
]

RecoveryFailureMode = Literal[
    "single-host-restart",
    "dual-rank-loss-recovery",
    "dual-host-restart",
]
class _QualificationContract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class AuthorityCatalog(_QualificationContract):
    """The immutable catalog release used to derive the authority."""

    __test__ = False

    repository: Annotated[StrictStr, Field(min_length=1, max_length=256)]
    commit: GitSha
    release_tag: Annotated[StrictStr, Field(pattern=r"^v[0-9]+\.[0-9]+\.[0-9]+$")]
    source_commit: GitSha
    catalog_index_sha256: Sha256
    qualification_index_sha256: Sha256
    recipe_count: Annotated[StrictInt, Field(ge=1)]


class AuthorityScope(_QualificationContract):
    """The topology boundary for this physical campaign."""

    __test__ = False

    maximum_node_count: Annotated[StrictInt, Field(ge=1, le=2)]
    recipe_count: Annotated[StrictInt, Field(ge=1)]
    excluded_topology_recipe_keys: list[RecipeKey]

    @model_validator(mode="after")
    def unique_exclusions(self) -> AuthorityScope:
        if len(self.excluded_topology_recipe_keys) != len(
            set(self.excluded_topology_recipe_keys)
        ):
            raise ValueError("excluded topology recipe keys must be unique")
        return self


class AuthorityPackage(_QualificationContract):
    """The exact published recipe package bound by a row."""

    __test__ = False

    path: Annotated[StrictStr, Field(min_length=1, max_length=512)]
    sha256: Sha256
    expected_bytes: Annotated[StrictInt, Field(ge=1, le=2**63 - 1)]
    media_type: Annotated[StrictStr, Field(min_length=3, max_length=128)]


class ModelLicenseReference(_QualificationContract):
    """A Model identity and license fact shown at the operator gate."""

    __test__ = False

    key: RecipeKey
    content_sha256: Sha256
    spdx: Annotated[StrictStr, Field(min_length=1, max_length=128)]
    url: Annotated[StrictStr, Field(min_length=1, max_length=2048)]
    attribution: list[Annotated[StrictStr, Field(min_length=1, max_length=256)]]
    territorial_restrictions: ModelTerritorialRestrictions | None = None


class ReviewGate(_QualificationContract):
    """An explicit per-recipe operator review gate."""

    __test__ = False

    kind: Literal["capacity-review"]
    reason: Annotated[StrictStr, Field(min_length=1, max_length=2048)]


class RecoveryCoverageRef(_QualificationContract):
    """A recipe row's typed reference to one recovery coverage definition."""

    __test__ = False

    coverage_id: Sha256
    failure_mode: RecoveryFailureMode
    representative_recipe: RecipeKey
    role: Literal["representative", "shared-member", "dedicated"]


class RecipeAuthorityRow(_QualificationContract):
    """One recipe's exact catalog, policy and test identity."""

    __test__ = False

    sequence: Annotated[StrictInt, Field(ge=1)]
    key: RecipeKey
    content_sha256: Sha256
    node_count: Annotated[StrictInt, Field(ge=1, le=2)]
    interface: Annotated[StrictStr, Field(min_length=1, max_length=64)]
    recipe_version: Annotated[StrictStr, Field(min_length=1, max_length=64)]
    package: AuthorityPackage
    disposition: Literal["actionable", "capacity-review"]
    model_license_refs: list[ModelLicenseReference]
    qualification_inputs: list[
        Annotated[StrictStr, Field(min_length=1, max_length=128)]
    ]
    smoke_cases: list[Annotated[StrictStr, Field(min_length=1, max_length=128)]]
    review_gates: list[ReviewGate]
    runtime_stack_sha256: Sha256
    topology_sha256: Sha256
    recovery_coverage_refs: list[RecoveryCoverageRef]
    disposition_reason: (
        Annotated[StrictStr, Field(min_length=1, max_length=4096)] | None
    ) = None

    @model_validator(mode="after")
    def consistent_review_gate(self) -> RecipeAuthorityRow:
        kinds = [gate.kind for gate in self.review_gates]
        if len(kinds) != len(set(kinds)):
            raise ValueError("review gate kinds must be unique")
        expected_disposition = (
            "capacity-review" if "capacity-review" in kinds else "actionable"
        )
        if self.disposition != expected_disposition:
            raise ValueError("recipe disposition does not match its review gates")
        return self


class QualificationBatchAssignment(_QualificationContract):
    """One exact recipe and lane assignment in an execution batch."""

    __test__ = False

    recipe: RecipeKey
    lane: Annotated[StrictInt, Field(ge=1, le=2)]
    node_count: Annotated[StrictInt, Field(ge=1, le=2)]


class QualificationBatch(_QualificationContract):
    """One paired single-Spark batch or exclusive topology assignment."""

    __test__ = False

    sequence: Annotated[StrictInt, Field(ge=1)]
    id: Annotated[StrictStr, Field(pattern=r"^batch-[0-9]{3,}$")]
    mode: Literal["paired-single", "single", "exclusive-dual"]
    assignments: list[QualificationBatchAssignment] = Field(min_length=1, max_length=2)

    @model_validator(mode="after")
    def assignments_match_mode(self) -> QualificationBatch:
        lanes = [assignment.lane for assignment in self.assignments]
        recipes = [assignment.recipe for assignment in self.assignments]
        if len(recipes) != len(set(recipes)) or len(lanes) != len(set(lanes)):
            raise ValueError("batch recipe and lane assignments must be unique")
        if self.mode == "paired-single":
            if (
                len(self.assignments) != 2
                or set(lanes) != {1, 2}
                or any(item.node_count != 1 for item in self.assignments)
            ):
                raise ValueError("paired-single batches require two single-node lanes")
        elif self.mode == "single":
            if (
                len(self.assignments) != 1
                or lanes != [1]
                or self.assignments[0].node_count != 1
            ):
                raise ValueError("single batches require one single-node lane")
        elif (
            len(self.assignments) != 1
            or lanes != [1]
            or self.assignments[0].node_count != 2
        ):
            raise ValueError("exclusive-dual batches require one two-node assignment")
        return self


class RecoveryCoverageMember(_QualificationContract):
    """The exact identity of a recipe allowed to use one recovery proof."""

    __test__ = False

    recipe: RecipeKey
    recipe_content_sha256: Sha256
    package_sha256: Sha256
    model_content_sha256s: list[Sha256]
    runtime_stack_sha256: Sha256
    topology_sha256: Sha256

    @model_validator(mode="after")
    def unique_models(self) -> RecoveryCoverageMember:
        if self.model_content_sha256s != sorted(set(self.model_content_sha256s)):
            raise ValueError("recovery Model identities must be unique and sorted")
        return self


class RecoveryCoverageDefinition(_QualificationContract):
    """Normalized recovery-equivalence payload, before its identity is added."""

    __test__ = False

    failure_mode: RecoveryFailureMode
    representative_recipe: RecipeKey
    members: list[RecoveryCoverageMember] = Field(min_length=1)
    shared: StrictBool
    equivalence_rationale: Annotated[StrictStr, Field(min_length=1, max_length=2048)]

    @model_validator(mode="after")
    def valid_coverage_definition(self) -> RecoveryCoverageDefinition:
        keys = [member.recipe for member in self.members]
        if keys != sorted(set(keys)):
            raise ValueError("recovery coverage members must be unique and sorted")
        if self.representative_recipe not in keys:
            raise ValueError("recovery representative must be a coverage member")
        if self.shared != (len(keys) > 1):
            raise ValueError("shared recovery requires multiple exact members")
        if self.failure_mode == "dual-rank-loss-recovery" and self.shared:
            raise ValueError("dual rank-loss recovery remains recipe-specific")
        if (
            self.shared
            and len(
                {
                    (member.runtime_stack_sha256, member.topology_sha256)
                    for member in self.members
                }
            )
            != 1
        ):
            raise ValueError(
                "shared recovery requires identical runtime and topology identities"
            )
        return self


class RecoveryCoverage(RecoveryCoverageDefinition):
    """A reviewed equivalence group for one exact recovery failure mode."""

    __test__ = False

    coverage_id: Sha256

    @model_validator(mode="after")
    def valid_coverage_identity(self) -> RecoveryCoverage:
        if recovery_coverage_id(self) != self.coverage_id:
            raise ValueError("recovery coverage_id does not match its bound identities")
        return self


class QualificationAuthority(_QualificationContract):
    """The current source of recipe qualification order and recovery references."""

    __test__ = False

    schema_version: QualificationAuthoritySchemaVersion
    authority_id: Identifier
    catalog: AuthorityCatalog
    scope: AuthorityScope
    recipes: list[RecipeAuthorityRow] = Field(min_length=1)
    batches: list[QualificationBatch] = Field(min_length=1)
    recovery_coverage: list[RecoveryCoverage] = Field(min_length=1)

    @model_validator(mode="after")
    def coherent_authority(self) -> QualificationAuthority:
        if len(self.recipes) != self.scope.recipe_count:
            raise ValueError("authority recipe count does not match its scope")
        if len(self.recipes) + len(self.scope.excluded_topology_recipe_keys) != (
            self.catalog.recipe_count
        ):
            raise ValueError("authority scope does not account for the full catalog")
        keys = [row.key for row in self.recipes]
        if len(keys) != len(set(keys)) or [
            row.sequence for row in self.recipes
        ] != list(range(1, len(self.recipes) + 1)):
            raise ValueError("authority rows must be unique and sequence-ordered")
        if set(keys) & set(self.scope.excluded_topology_recipe_keys):
            raise ValueError("an in-scope recipe cannot also be excluded")
        rows_by_key = {row.key: row for row in self.recipes}
        if [batch.sequence for batch in self.batches] != list(
            range(1, len(self.batches) + 1)
        ):
            raise ValueError("execution batch sequences must be contiguous")
        if len({batch.id for batch in self.batches}) != len(self.batches):
            raise ValueError("execution batch ids must be unique")
        assigned: list[str] = []
        for batch in self.batches:
            for assignment in batch.assignments:
                row = rows_by_key.get(assignment.recipe)
                if row is None or assignment.node_count != row.node_count:
                    raise ValueError("batch assignment must match one authority row")
                assigned.append(assignment.recipe)
        if sorted(assigned) != sorted(keys):
            raise ValueError("execution batches must cover every in-scope recipe once")

        coverage_by_id = {entry.coverage_id: entry for entry in self.recovery_coverage}
        if len(coverage_by_id) != len(self.recovery_coverage):
            raise ValueError("recovery coverage IDs must be unique")
        expected_modes = {
            1: {"single-host-restart"},
            2: {"dual-rank-loss-recovery", "dual-host-restart"},
        }
        seen_refs: set[tuple[str, RecoveryFailureMode]] = set()
        referenced_members: dict[str, set[str]] = {
            coverage_id: set() for coverage_id in coverage_by_id
        }
        for row in self.recipes:
            refs = {
                (ref.coverage_id, ref.failure_mode): ref
                for ref in row.recovery_coverage_refs
            }
            if len(refs) != len(row.recovery_coverage_refs):
                raise ValueError(f"{row.key} repeats a recovery coverage reference")
            if len(refs) != len(expected_modes[row.node_count]):
                raise ValueError(
                    f"{row.key} must have exactly one reference per recovery mode"
                )
            if {failure_mode for _, failure_mode in refs} != expected_modes[
                row.node_count
            ]:
                raise ValueError(
                    f"{row.key} lacks its required recovery coverage modes"
                )
            for reference in row.recovery_coverage_refs:
                group = coverage_by_id.get(reference.coverage_id)
                if group is None or group.failure_mode != reference.failure_mode:
                    raise ValueError(
                        f"{row.key} references an absent recovery definition"
                    )
                member = next(
                    (member for member in group.members if member.recipe == row.key),
                    None,
                )
                if (
                    member is None
                    or group.representative_recipe != reference.representative_recipe
                ):
                    raise ValueError(
                        f"{row.key} recovery reference is outside its exact scope"
                    )
                model_digests = sorted(
                    {item.content_sha256 for item in row.model_license_refs}
                )
                if (
                    member.recipe_content_sha256 != row.content_sha256
                    or member.package_sha256 != row.package.sha256
                    or member.model_content_sha256s != model_digests
                    or member.runtime_stack_sha256 != row.runtime_stack_sha256
                    or member.topology_sha256 != row.topology_sha256
                ):
                    raise ValueError(f"{row.key} recovery identity is stale")
                role = (
                    "representative"
                    if row.key == group.representative_recipe and group.shared
                    else "shared-member"
                    if group.shared
                    else "dedicated"
                )
                if reference.role != role:
                    raise ValueError(
                        f"{row.key} recovery reference role is inconsistent"
                    )
                seen_refs.add((row.key, reference.failure_mode))
                referenced_members[reference.coverage_id].add(row.key)
        expected_refs = {
            (row.key, mode)
            for row in self.recipes
            for mode in expected_modes[row.node_count]
        }
        if seen_refs != expected_refs:
            raise ValueError("recovery definitions and row references must be exact")
        for group in self.recovery_coverage:
            declared_members = {member.recipe for member in group.members}
            if any(member not in rows_by_key for member in declared_members):
                raise ValueError(
                    "recovery coverage includes a recipe outside the authority"
                )
            if declared_members != referenced_members[group.coverage_id]:
                raise ValueError(
                    "recovery coverage members and row references must be exact"
                )
        return self


class QualificationCampaignOptions(_QualificationContract):
    """Bounded polling and cleanup settings for the runner."""

    __test__ = False

    cleanup: Literal["stop"] = "stop"
    operation_timeout_seconds: Annotated[StrictInt, Field(ge=1, le=86400)] = 86400
    poll_interval_seconds: Annotated[StrictInt, Field(ge=1, le=60)] = 5


class QualificationCampaignManifest(_QualificationContract):
    """The small manifest joining an authority and fixture registry."""

    __test__ = False

    schema_version: QualificationCampaignSchemaVersion
    qualification_authority: Annotated[StrictStr, Field(min_length=1, max_length=512)]
    fixture_manifest: Annotated[StrictStr, Field(min_length=1, max_length=512)]
    options: QualificationCampaignOptions = Field(
        default_factory=QualificationCampaignOptions
    )


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def recovery_coverage_id(
    value: RecoveryCoverage | RecoveryCoverageDefinition | dict[str, object],
) -> str:
    """Hash a coverage definition without its own ID field."""

    if isinstance(value, RecoveryCoverage):
        document = value.model_dump(
            mode="json", exclude={"coverage_id"}, exclude_none=True
        )
    elif isinstance(value, RecoveryCoverageDefinition):
        document = value.model_dump(mode="json", exclude_none=True)
    else:
        document = RecoveryCoverageDefinition.model_validate(value).model_dump(
            mode="json", exclude_none=True
        )
    return _canonical_sha256(document)


def campaign_authority_json_schema() -> dict[str, object]:
    """Return the generated JSON Schema for the paired qualification authority."""

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://api.vonkforge.ai/contracts/qualification-authority-v5.schema.json",
        **QualificationAuthority.model_json_schema(ref_template="#/$defs/{model}"),
    }


def campaign_manifest_json_schema() -> dict[str, object]:
    """Return the generated JSON Schema for the qualification manifest."""

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://api.vonkforge.ai/contracts/qualification-campaign-manifest-v2.schema.json",
        **QualificationCampaignManifest.model_json_schema(
            ref_template="#/$defs/{model}"
        ),
    }


campaign_authority_json_schema.__test__ = False
campaign_manifest_json_schema.__test__ = False
