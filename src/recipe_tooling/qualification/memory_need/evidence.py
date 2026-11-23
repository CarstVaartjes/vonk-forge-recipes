"""Hardware evidence and measured budgets."""

from __future__ import annotations

from decimal import ROUND_CEILING, Decimal
from pathlib import Path

from recipe_tooling.qualification.documents import recipe_document
from recipe_tooling.qualification.evidence_store import read_evidence
from recipe_tooling.qualification.memory_need import args as mn_args
from recipe_tooling.qualification.memory_need import constants as mn_constants
from recipe_tooling.qualification.memory_need.constants import (
    ADMISSION_MAXIMUM_PEAK_BYTES,
    OS_FLOOR_BYTES,
    TOTAL_BYTES,
    VLLM_DEFAULT_UTILIZATION,
    budget_margin_bytes,
)
from recipe_tooling.qualification.memory_need.geometry import MemoryInputs
from vonk_forge_contracts import (
    PublishedRoleMemory,
    QualificationMeasuredRoleMemory,
    QualificationStatus,
    RecipeDefinition,
    RecipeQualificationEvidence,
    derive_status,
    document_sha256,
)


def _evidence(key: str):
    """The committed hardware evidence for one recipe (tests swap the corpus here)."""
    return read_evidence(mn_constants.ROOT, key)


def unchanged_recipe(
    recipe: RecipeDefinition, document_digest: str, inputs: MemoryInputs
) -> bool:
    """Recognize exact content in the committed snapshot or qualification evidence."""
    if not document_digest:
        return False
    key = f"{recipe.identity.publisher}/{recipe.identity.slug}"
    known = document_digest in inputs.historical_documents.get(key, []) or any(
        record.identity.content_sha256 == document_digest
        for record in _evidence(key).records
    )
    if not known:
        return False
    # Bind the caller's digest AND parsed recipe to the authored bytes. An old
    # digest cannot waive an edited file or an in-memory launch mutation.
    document = recipe_document(recipe.identity.slug, mn_constants.ROOT)
    return (
        document_sha256(document) == document_digest
        and RecipeDefinition.model_validate(document) == recipe
    )


def hardware_proven(
    recipe: RecipeDefinition,
    document_digest: str,
    evidence: RecipeQualificationEvidence | None = None,
) -> bool:
    """Use the qualification reducer, including later failures.

    ``evidence`` is the recipe's evidence when the caller already read it.
    """

    key = f"{recipe.identity.publisher}/{recipe.identity.slug}"
    evidence = evidence if evidence is not None else _evidence(key)
    for record in evidence.records:
        if record.identity.content_sha256 != document_digest:
            continue
        identity = record.identity.model_copy(
            update={"required_sparks": recipe.topology.node_count}
        )
        if (
            derive_status(
                evidence, identity, recipe.topology.node_count, testing_now=False
            ).status
            == QualificationStatus.WORKING
        ):
            return True
    return False


def measured_peak_bytes(used_bytes: int) -> int:
    """A role's peak from a Spark's observed use (OS included): the one conversion."""
    return max(0, used_bytes - OS_FLOOR_BYTES)


def holds_within(declared_bytes: int, used_bytes: int) -> bool:
    """Whether a Spark's observed use (OS included) fits a role's declared peak.

    The one rule for the sweep's memory-envelope note and for measured peaks.
    """
    return measured_peak_bytes(used_bytes) <= declared_bytes


def published_measured_peaks(
    recipe: RecipeDefinition,
    document_digest: str,
    evidence: RecipeQualificationEvidence,
) -> tuple[PublishedRoleMemory, ...]:
    """The measured steady peak per role the catalog carries for exactly this content.

    A role is published only when hardware proved this content (a later recipe
    failure withdraws it) and its row (keyed content, role) either covers every
    launch variant the recipe allows (``keeps_measured_peaks`` for a vLLM budget,
    else a single variant) or measured more than the declaration (raising is
    always safe). A measurement above the admission maximum is carried as
    information and flagged, while ``peak_bytes`` stays the declared peak: the
    Controller admits on the declaration and warns.
    """
    if not hardware_proven(recipe, document_digest, evidence):
        return ()
    rows = {
        row.role: row
        for row in evidence.measured_memory
        if row.content_sha256 == document_digest
    }
    variants = mn_args.memory_variants(recipe)
    published = []
    for role in sorted(recipe.topology.roles, key=lambda item: item.name):
        row = rows.get(role.name)
        if row is None:
            continue
        declared = role.resources.memory.peak_bytes
        measured = measured_peak_bytes(row.steady_bytes)
        covers = (
            all(keeps_measured_peaks(variant, row.budget_bytes) for variant in variants)
            if row.budget_bytes is not None
            else len(variants) == 1
        )
        if not (covers or measured > declared):
            continue
        over = measured > ADMISSION_MAXIMUM_PEAK_BYTES
        published.append(
            PublishedRoleMemory(
                role=role.name,
                peak_bytes=declared if over else measured,
                measured_peak_bytes=measured,
                exceeds_admission_maximum=over,
            )
        )
    return tuple(published)


def load_inputs(root: Path = mn_constants.ROOT) -> MemoryInputs:
    return MemoryInputs.model_validate_json(
        (root / "qualification/memory-inputs.json").read_text()
    )


def vllm_budget_bytes(recipe: RecipeDefinition) -> int | None:
    """vLLM's own utilization budget (gpu-memory-utilization x Spark memory), if pool-less.

    None for an absolute KV pool or another engine. Hardware evidence records it
    for the launch that ran (spark_sweep/memory.py::measurement).
    """
    if recipe.runtime.engine != "vllm" or any(
        mn_args.argument_value(recipe, name) is not None
        for name in ("kv-cache-memory-bytes", "kv-cache-memory")
    ):
        return None
    fraction = mn_args.argument_value(recipe, "gpu-memory-utilization")
    share = Decimal(str(fraction)) if fraction is not None else VLLM_DEFAULT_UTILIZATION
    return int((share * TOTAL_BYTES).to_integral_value(ROUND_CEILING))


def vllm_utilization_envelope(recipe: RecipeDefinition) -> int | None:
    """vLLM's utilization budget plus the margin it holds beside it, if pool-less.

    A speculative drafter is charged inside the budget, whichever mechanism
    loads it (src/recipe_tooling/qualification/vllm_memory_fit.py).
    """
    budget = vllm_budget_bytes(recipe)
    return None if budget is None else budget + budget_margin_bytes(recipe)


def declarable_vllm_envelope(recipe: RecipeDefinition) -> int | None:
    """The vLLM envelope a declared peak is judged and raised against.

    Capped at the offline admission maximum: a peak above what an idle Spark can
    charge is never reserved in full, so no declaration should chase a larger
    envelope. The uncapped envelope stays the admissibility test's input (a
    budget above the maximum is reported there).
    """
    envelope = vllm_utilization_envelope(recipe)
    return None if envelope is None else min(envelope, ADMISSION_MAXIMUM_PEAK_BYTES)


def recipe_envelope(recipe: RecipeDefinition) -> int:
    """The largest declarable vLLM envelope over the recipe's launch variants."""
    return max(
        (
            e
            for v in mn_args.memory_variants(recipe)
            if (e := declarable_vllm_envelope(v))
        ),
        default=0,
    )


def _rows_by_content(
    evidence: RecipeQualificationEvidence,
) -> dict[str, dict[str, QualificationMeasuredRoleMemory]]:
    by_content: dict[str, dict[str, QualificationMeasuredRoleMemory]] = {}
    for row in evidence.measured_memory:
        by_content.setdefault(row.content_sha256, {})[row.role] = row
    return by_content


def measured_budgets(
    recipe: RecipeDefinition, evidence: RecipeQualificationEvidence
) -> dict[str, int]:
    """Budget per content that qualifies for ``measured_budget`` (and for retention).

    A content qualifies when it is still Working (``hardware_proven``) and its rows
    measure every declared role within its declared peak (``holds_within``); its
    budget is the smallest over its rows. The one loop: ``measured_budget`` takes
    the largest, evidence retention keeps the content that gives it.
    """
    budgets: dict[str, int] = {}
    for content, rows in _rows_by_content(evidence).items():
        ran = [row.budget_bytes for row in rows.values()]
        if (
            None not in ran
            and all(
                role.name in rows
                and holds_within(
                    role.resources.memory.peak_bytes, rows[role.name].steady_bytes
                )
                for role in recipe.topology.roles
            )
            and hardware_proven(recipe, content, evidence)
        ):
            budgets[content] = min(b for b in ran if b is not None)
    return budgets


def measured_budget(
    recipe: RecipeDefinition,
    replaced: tuple[RecipeDefinition, str] | None = None,
) -> int | None:
    """The largest vLLM budget under which hardware measured these declared peaks.

    Read from the recipe's evidence table (never git history): a content that is
    still Working (``hardware_proven``) whose rows measure every declared role
    within its declared peak (``holds_within``); a content's budget is the
    smallest over its rows. ``replaced`` is the document (and digest) a refresh
    is replacing, in hand: while its content has no complete rows, proven
    content vouches for peaks no smaller than its own declared ones, under its
    largest budget, unless a row shows a role using more than declared (partial
    rows are "not measured", never a loss of that proof). Once every role of it
    has a row, the rows alone decide, per launch variant. ``None`` when nothing
    qualifies.
    """
    key = f"{recipe.identity.publisher}/{recipe.identity.slug}"
    evidence = _evidence(key)
    by_content = _rows_by_content(evidence)
    budgets = list(measured_budgets(recipe, evidence).values())
    if replaced is not None:
        old, digest = replaced
        before = {r.name: r.resources.memory.peak_bytes for r in old.topology.roles}
        rows = by_content.get(digest, {})
        overrun = any(
            role in before and not holds_within(before[role], row.steady_bytes)
            for role, row in rows.items()
        )
        if (
            not overrun
            and not set(before) <= set(rows)  # complete rows retire the fallback
            and all(
                role.resources.memory.peak_bytes >= before.get(role.name, 2**63)
                for role in recipe.topology.roles
            )
            and hardware_proven(old, digest, evidence)
        ):
            budgets += [
                b for v in mn_args.memory_variants(old) if (b := vllm_budget_bytes(v))
            ]
    return max(budgets, default=None)


def keeps_measured_peaks(launch: RecipeDefinition, measured: int | None) -> bool:
    """Whether one launch variant keeps its declared peaks: they are measured.

    The one rule for the refresh's raise and for the "peak understates" warning:
    hardware measured the peaks (``measured``, from ``measured_budget``) under a
    vLLM budget no smaller than this launch's. Unmeasured estimates heal.
    """
    budget = vllm_budget_bytes(launch)
    return budget is not None and measured is not None and budget <= measured


def raise_peaks_to_envelope(
    recipe: RecipeDefinition, replaced: tuple[RecipeDefinition, str] | None = None
) -> RecipeDefinition | None:
    """The recipe with role peaks raised to its vLLM budget envelope, None if none is.

    A document that moves (``tools/refresh-upstream``) is new content and meets the
    current margins; raising here keeps a refresh from blocking. Peaks are never
    lowered, and a declared load peak keeps its rise above the peak. The envelope
    is capped at the admission maximum. A launch variant whose peaks hardware
    evidence measured under no smaller budget is kept (``keeps_measured_peaks``);
    the peaks rise to the largest capped envelope of the variants that are not,
    including a measured launch whose budget the move raised.
    """
    measured = measured_budget(recipe, replaced)
    envelope = max(
        (
            e
            for v in mn_args.memory_variants(recipe)
            if not keeps_measured_peaks(v, measured)
            and (e := declarable_vllm_envelope(v))
        ),
        default=0,
    )
    if all(r.resources.memory.peak_bytes >= envelope for r in recipe.topology.roles):
        return None
    raised = recipe.model_copy(deep=True)
    for role in raised.topology.roles:
        memory = role.resources.memory
        if memory.peak_bytes >= envelope:
            continue
        if memory.load_peak_bytes is not None:
            memory.load_peak_bytes += envelope - memory.peak_bytes
        memory.peak_bytes = envelope
    return raised
