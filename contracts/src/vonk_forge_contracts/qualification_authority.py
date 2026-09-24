"""Strict contracts for recipe qualification authorities and reusable evidence.

The qualification authority binds an ordered catalog to executable batches and
explicit recovery-coverage identities. Shared recovery is scoped to one exact
failure mode and one reviewed set of recipe, package, model, build and topology
identities. Runtime receipts carry the physical evidence needed to consume that
reference safely.
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
    RecoveryCoverageReceiptSchemaVersion,
)
from .model import ModelTerritorialRestrictions

Sha256 = Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]
GitSha = Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{40}$")]
OciDigest = Annotated[StrictStr, Field(pattern=r"^sha256:[a-f0-9]{64}$")]
RecipeKey = Annotated[
    StrictStr,
    Field(
        min_length=5,
        max_length=127,
        pattern=r"^[a-z0-9][a-z0-9-]{1,62}/[a-z0-9][a-z0-9-]{1,62}$",
    ),
]
NodeId = Annotated[StrictStr, Field(pattern=r"^spk_[0-9a-f]{32}$")]
Identifier = Annotated[
    StrictStr,
    Field(min_length=1, max_length=256, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$"),
]

RecoveryFailureMode = Literal[
    "single-host-restart",
    "dual-rank-loss-recovery",
    "dual-host-restart",
]
RecoveryInvalidationField = Literal[
    "recipe_content_sha256",
    "package_sha256",
    "model_content_sha256s",
    "runtime_stack_sha256",
    "topology_sha256",
    "coverage_membership",
    "runtime_image_digest",
    "platform_build_sha256",
    "agent_build_sha256",
    "target_node_ids",
    "smoke_receipt_sha256",
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
    operator_acceptance_required: StrictBool
    territorial_restrictions: ModelTerritorialRestrictions | None = None


class ReviewGate(_QualificationContract):
    """An explicit per-recipe operator review gate."""

    __test__ = False

    kind: Literal["capacity-review", "operator-acceptance-required"]
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
    disposition: Literal[
        "actionable", "capacity-review", "operator-acceptance-required"
    ]
    operator_acceptance_required: StrictBool
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
        if self.operator_acceptance_required != any(
            reference.operator_acceptance_required
            for reference in self.model_license_refs
        ):
            raise ValueError("operator acceptance must match the exact Model licenses")
        kinds = [gate.kind for gate in self.review_gates]
        if len(kinds) != len(set(kinds)):
            raise ValueError("review gate kinds must be unique")
        if self.operator_acceptance_required != (
            "operator-acceptance-required" in kinds
        ):
            raise ValueError("operator acceptance review gate is inconsistent")
        expected_disposition = (
            "operator-acceptance-required"
            if self.operator_acceptance_required
            else "capacity-review"
            if "capacity-review" in kinds
            else "actionable"
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
    invalidated_by: list[RecoveryInvalidationField]

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
        expected_invalidations = {
            "recipe_content_sha256",
            "package_sha256",
            "model_content_sha256s",
            "runtime_stack_sha256",
            "topology_sha256",
            "coverage_membership",
            "runtime_image_digest",
            "platform_build_sha256",
            "agent_build_sha256",
            "target_node_ids",
            "smoke_receipt_sha256",
        }
        if set(self.invalidated_by) != expected_invalidations:
            raise ValueError("recovery invalidation must bind every evidence identity")
        if len(self.invalidated_by) != len(expected_invalidations):
            raise ValueError("recovery invalidation fields must be unique")
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


class RecoveryNodeEvidence(_QualificationContract):
    """One physical node's restart proof."""

    __test__ = False

    node_id: NodeId
    agent_build_sha256: Sha256
    baseline_boot_id: Annotated[StrictStr, Field(min_length=1, max_length=128)]
    recovered_boot_id: Annotated[StrictStr, Field(min_length=1, max_length=128)]
    baseline_event_id: Identifier
    offline_event_id: Identifier
    recovered_event_id: Identifier

    @model_validator(mode="after")
    def boot_id_changed(self) -> RecoveryNodeEvidence:
        if self.baseline_boot_id == self.recovered_boot_id:
            raise ValueError("recovery receipt must prove a changed boot ID")
        event_ids = [
            self.baseline_event_id,
            self.offline_event_id,
            self.recovered_event_id,
        ]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("host-restart event identities must be unique")
        return self


class RecoveryNodeBuildIdentity(_QualificationContract):
    """The agent build running on a specific physical Spark."""

    __test__ = False

    node_id: NodeId
    agent_build_sha256: Sha256


class RecoveryRankEvidence(_QualificationContract):
    """A dual-Spark rank loss and serving recovery without a host reboot."""

    __test__ = False

    lost_node_id: NodeId
    recovered_node_id: NodeId
    survivor_node_id: NodeId
    node_builds: list[RecoveryNodeBuildIdentity] = Field(min_length=2, max_length=2)
    rank_loss_event_id: Identifier
    route_withdrawal_event_id: Identifier
    rank_recovery_event_id: Identifier
    recovered_smoke_receipt_sha256: Sha256

    @model_validator(mode="after")
    def distinct_rank_proof(self) -> RecoveryRankEvidence:
        if self.lost_node_id != self.recovered_node_id:
            raise ValueError("rank recovery must restore the lost node")
        if self.lost_node_id == self.survivor_node_id:
            raise ValueError("rank loss and survivor must identify distinct nodes")
        node_ids = [node.node_id for node in self.node_builds]
        if len(node_ids) != len(set(node_ids)) or set(node_ids) != {
            self.lost_node_id,
            self.survivor_node_id,
        }:
            raise ValueError("rank recovery must bind both exact node builds")
        event_ids = [
            self.rank_loss_event_id,
            self.route_withdrawal_event_id,
            self.rank_recovery_event_id,
        ]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("rank recovery event identities must be unique")
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
        for row in self.recipes:
            refs = {
                (ref.coverage_id, ref.failure_mode): ref
                for ref in row.recovery_coverage_refs
            }
            if len(refs) != len(row.recovery_coverage_refs):
                raise ValueError(f"{row.key} repeats a recovery coverage reference")
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
        expected_refs = {
            (row.key, mode)
            for row in self.recipes
            for mode in expected_modes[row.node_count]
        }
        if seen_refs != expected_refs:
            raise ValueError("recovery definitions and row references must be exact")
        for group in self.recovery_coverage:
            if any(member.recipe not in rows_by_key for member in group.members):
                raise ValueError(
                    "recovery coverage includes a recipe outside the authority"
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


class RecoveryCoverageReceipt(_QualificationContract):
    """Physical source evidence that may be referenced for shared recovery."""

    __test__ = False

    schema_version: RecoveryCoverageReceiptSchemaVersion
    coverage_id: Sha256
    failure_mode: RecoveryFailureMode
    representative_recipe: RecipeKey
    recipe_content_sha256: Sha256
    package_sha256: Sha256
    model_content_sha256s: list[Sha256]
    runtime_stack_sha256: Sha256
    topology_sha256: Sha256
    runtime_image_digest: OciDigest
    platform_build_sha256: Sha256
    architecture: Literal["linux/arm64"]
    smoke_receipt_sha256: Sha256
    checkpoint_event_ids: list[Identifier] = Field(min_length=1)
    nodes: list[RecoveryNodeEvidence] = Field(default_factory=list, max_length=2)
    rank_recovery: RecoveryRankEvidence | None = None
    passed: Literal[True]

    @model_validator(mode="after")
    def valid_node_proof(self) -> RecoveryCoverageReceipt:
        node_ids = [node.node_id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("recovery receipt node identities must be unique")
        if self.failure_mode == "single-host-restart" and len(self.nodes) != 1:
            raise ValueError("single-host-restart requires one node proof")
        if self.failure_mode == "dual-host-restart" and len(self.nodes) != 2:
            raise ValueError("dual-host-restart requires both node proofs")
        if self.failure_mode in {"single-host-restart", "dual-host-restart"} and (
            self.rank_recovery is not None
        ):
            raise ValueError("host-restart receipts cannot contain rank-loss evidence")
        if self.failure_mode == "dual-rank-loss-recovery" and (
            self.nodes or self.rank_recovery is None
        ):
            raise ValueError(
                "dual rank recovery requires rank evidence and no reboot proof"
            )
        if len(self.checkpoint_event_ids) != len(set(self.checkpoint_event_ids)):
            raise ValueError("recovery checkpoint event identities must be unique")
        if self.failure_mode in {"single-host-restart", "dual-host-restart"}:
            expected_event_ids = {
                event_id
                for node in self.nodes
                for event_id in (
                    node.baseline_event_id,
                    node.offline_event_id,
                    node.recovered_event_id,
                )
            }
        else:
            rank = self.rank_recovery
            if rank is None:
                raise ValueError("dual rank recovery evidence is missing")
            expected_event_ids = {
                rank.rank_loss_event_id,
                rank.route_withdrawal_event_id,
                rank.rank_recovery_event_id,
            }
        if set(self.checkpoint_event_ids) != expected_event_ids:
            raise ValueError(
                "recovery checkpoints must exactly bind the typed source events"
            )
        if self.failure_mode == "dual-rank-loss-recovery":
            rank = self.rank_recovery
            if (
                rank is None
                or rank.recovered_smoke_receipt_sha256 != self.smoke_receipt_sha256
            ):
                raise ValueError("rank recovery smoke identity must match the receipt")
        if self.model_content_sha256s != sorted(set(self.model_content_sha256s)):
            raise ValueError(
                "recovery receipt Model identities must be unique and sorted"
            )
        return self


class RecoveryCoverageUse(_QualificationContract):
    """A durable, digest-addressed use of one representative receipt."""

    __test__ = False

    schema_version: RecoveryCoverageReceiptSchemaVersion
    coverage_id: Sha256
    failure_mode: RecoveryFailureMode
    member_recipe: RecipeKey
    representative_recipe: RecipeKey
    representative_receipt_sha256: Sha256
    member_recipe_content_sha256: Sha256
    member_package_sha256: Sha256
    member_model_content_sha256s: list[Sha256]
    member_runtime_stack_sha256: Sha256
    member_topology_sha256: Sha256
    member_runtime_image_digest: OciDigest
    member_platform_build_sha256: Sha256
    member_nodes: list[RecoveryNodeBuildIdentity] = Field(min_length=1, max_length=2)
    member_smoke_receipt_sha256: Sha256

    @model_validator(mode="after")
    def coherent_reference(self) -> RecoveryCoverageUse:
        if self.member_recipe == self.representative_recipe:
            raise ValueError("a representative cannot consume its own shared receipt")
        node_ids = [node.node_id for node in self.member_nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("recovery use node identities must be unique")
        expected_nodes = 1 if self.failure_mode == "single-host-restart" else 2
        if len(self.member_nodes) != expected_nodes:
            raise ValueError("recovery use must identify every affected Spark build")
        if self.member_model_content_sha256s != sorted(
            set(self.member_model_content_sha256s)
        ):
            raise ValueError("recovery use Model identities must be unique and sorted")
        return self


class RecoveryCoverageReceiptEnvelope(_QualificationContract):
    """A content-addressed source receipt and typed shared-use references."""

    __test__ = False

    receipt: RecoveryCoverageReceipt
    receipt_sha256: Sha256

    @model_validator(mode="after")
    def digest_matches_payload(self) -> RecoveryCoverageReceiptEnvelope:
        if recovery_receipt_sha256(self.receipt) != self.receipt_sha256:
            raise ValueError("representative recovery receipt digest is invalid")
        return self


class RecoveryCoverageConsumption(_QualificationContract):
    """Validate one member's use against the bound group and source receipt."""

    __test__ = False

    coverage: RecoveryCoverage
    representative_receipt: RecoveryCoverageReceiptEnvelope
    use: RecoveryCoverageUse

    @model_validator(mode="after")
    def exact_receipt_and_member(self) -> RecoveryCoverageConsumption:
        coverage = self.coverage
        receipt = self.representative_receipt.receipt
        use = self.use
        if not coverage.shared:
            raise ValueError("a dedicated recovery definition cannot be reused")
        if (
            coverage.coverage_id != receipt.coverage_id
            or coverage.coverage_id != use.coverage_id
            or coverage.failure_mode != receipt.failure_mode
            or coverage.failure_mode != use.failure_mode
            or coverage.representative_recipe != receipt.representative_recipe
            or coverage.representative_recipe != use.representative_recipe
        ):
            raise ValueError("recovery receipt does not match the typed coverage scope")
        representative = next(
            (
                item
                for item in coverage.members
                if item.recipe == receipt.representative_recipe
            ),
            None,
        )
        member = next(
            (item for item in coverage.members if item.recipe == use.member_recipe),
            None,
        )
        if (
            representative is None
            or member is None
            or member.recipe == representative.recipe
        ):
            raise ValueError("recovery use is outside the representative member group")
        if (
            receipt.recipe_content_sha256 != representative.recipe_content_sha256
            or receipt.package_sha256 != representative.package_sha256
            or receipt.model_content_sha256s != representative.model_content_sha256s
            or receipt.runtime_stack_sha256 != representative.runtime_stack_sha256
            or receipt.topology_sha256 != representative.topology_sha256
        ):
            raise ValueError(
                "representative receipt identity differs from its authority"
            )
        if (
            use.member_recipe_content_sha256 != member.recipe_content_sha256
            or use.member_package_sha256 != member.package_sha256
            or use.member_model_content_sha256s != member.model_content_sha256s
            or use.member_runtime_stack_sha256 != member.runtime_stack_sha256
            or use.member_topology_sha256 != member.topology_sha256
        ):
            raise ValueError("member use identity differs from its authority")
        if (
            use.representative_receipt_sha256
            != self.representative_receipt.receipt_sha256
        ):
            raise ValueError(
                "member use does not bind the exact representative receipt"
            )
        if (
            use.member_runtime_image_digest != receipt.runtime_image_digest
            or use.member_platform_build_sha256 != receipt.platform_build_sha256
        ):
            raise ValueError("shared recovery runtime image or platform build changed")
        if receipt.failure_mode in {"single-host-restart", "dual-host-restart"}:
            receipt_node_builds = receipt.nodes
        else:
            rank_recovery = receipt.rank_recovery
            if rank_recovery is None:
                raise ValueError("dual rank recovery evidence is missing")
            receipt_node_builds = rank_recovery.node_builds
        representative_nodes = sorted(
            (node.node_id, node.agent_build_sha256) for node in receipt_node_builds
        )
        member_nodes = sorted(
            (node.node_id, node.agent_build_sha256) for node in use.member_nodes
        )
        if member_nodes != representative_nodes:
            raise ValueError("shared recovery target Spark or agent build changed")
        return self


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


def recovery_receipt_sha256(value: RecoveryCoverageReceipt) -> str:
    """Return the stable digest referenced when another row reuses this receipt."""

    return _canonical_sha256(value.model_dump(mode="json", exclude_none=True))


def campaign_authority_json_schema() -> dict[str, object]:
    """Return the generated JSON Schema for the paired qualification authority."""

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://api.vonkforge.ai/contracts/qualification-authority-v4.schema.json",
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


def recovery_coverage_receipt_json_schema() -> dict[str, object]:
    """Return the schema for a content-addressed representative recovery receipt."""

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://api.vonkforge.ai/contracts/recovery-coverage-receipt-v1.schema.json",
        **RecoveryCoverageReceiptEnvelope.model_json_schema(
            ref_template="#/$defs/{model}"
        ),
    }


def recovery_coverage_use_json_schema() -> dict[str, object]:
    """Return the schema for one durable use of a representative receipt."""

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://api.vonkforge.ai/contracts/recovery-coverage-use-v1.schema.json",
        **RecoveryCoverageUse.model_json_schema(ref_template="#/$defs/{model}"),
    }


def recovery_coverage_consumption_json_schema() -> dict[str, object]:
    """Return the schema for validating a shared use with its source receipt."""

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://api.vonkforge.ai/contracts/recovery-coverage-consumption-v1.schema.json",
        **RecoveryCoverageConsumption.model_json_schema(ref_template="#/$defs/{model}"),
    }


campaign_authority_json_schema.__test__ = False
campaign_manifest_json_schema.__test__ = False
recovery_coverage_receipt_json_schema.__test__ = False
recovery_coverage_use_json_schema.__test__ = False
recovery_coverage_consumption_json_schema.__test__ = False
