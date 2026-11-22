"""The retention rule: the evidence the readers consult, nothing else (newest-only)."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

from vonk_forge_contracts import (
    QualificationBlame,
    QualificationEvidenceRecord,
    QualificationOutcome,
    RecipeDefinition,
    RecipeQualificationEvidence,
)

from recipe_tooling.qualification.evidence_store import (
    catalog_recipes,
    read_evidence,
    write_evidence,
)
from recipe_tooling.qualification.memory_need.evidence import measured_budgets

#: The classes `derive_status` and `published_status` read, each by the reducers' own
#: predicate: the newest record of each class, per identity, is all they consult.
EVIDENCE_CLASSES: dict[str, Callable[[QualificationEvidenceRecord], bool]] = {
    "pass": lambda r: r.outcome == QualificationOutcome.PASSED,
    "resolved_pass": lambda r: (
        r.outcome == QualificationOutcome.PASSED and r.platform.resolved
    ),
    "failure": lambda r: r.outcome == QualificationOutcome.FAILED,
    "behavioural": lambda r: r.blame != QualificationBlame.PLATFORM,
    "recipe_blamed": lambda r: r.blame == QualificationBlame.RECIPE,
}


#: The three verdicts of the whole file the status text quotes ("worked before").
FILE_CLASSES = ("pass", "failure", "behavioural")


def _identity_key(record: QualificationEvidenceRecord) -> str:
    return record.identity.model_dump_json()


def _newest_of_classes(
    records: Iterable[QualificationEvidenceRecord],
    names: Iterable[str],
    current: str,
    per_identity: bool,
) -> list[QualificationEvidenceRecord]:
    """The newest record of each class (per identity or file-wide), in `derive_status` order."""
    best: dict[tuple[str, str], tuple[tuple, QualificationEvidenceRecord]] = {}
    for record in records:
        for name in names:
            if not EVIDENCE_CLASSES[name](record):
                continue
            slot = (_identity_key(record) if per_identity else "", name)
            order = (
                record.tested_at,
                record.identity.content_sha256 == current,
                record.outcome == QualificationOutcome.FAILED,
                record.model_dump_json(),
            )
            if slot not in best or order > best[slot][0]:
                best[slot] = (order, record)
    return [record for _, record in best.values()]


def retained(
    evidence: RecipeQualificationEvidence, recipe: RecipeDefinition, current: str
) -> RecipeQualificationEvidence:
    """The evidence the readers consult, nothing else (newest-only).

    Kept: per identity of the current content, its newest record of each class; the
    file's newest pass, failure and behavioural record (any content); the one other
    content `measured_budget` takes its maximum from, with its per-identity newest;
    measured rows only for contents that keep a record. Applying it again, or
    after re-importing history, changes nothing.
    """
    budgets = measured_budgets(recipe, evidence)
    # The best older content: a later failure of the current one must not lose the budget.
    source = max(
        (c for c in budgets if c != current),
        key=lambda c: (budgets[c], c),
        default=None,
    )
    contents = {current} if source is None else {current, source}
    kept = _newest_of_classes(
        (r for r in evidence.records if r.identity.content_sha256 in contents),
        EVIDENCE_CLASSES,
        current,
        True,
    ) + _newest_of_classes(evidence.records, FILE_CLASSES, current, False)
    unique = {r.model_dump_json(): r for r in kept}
    with_records = {r.identity.content_sha256 for r in unique.values()}
    return evidence.model_copy(
        update={
            "records": tuple(
                sorted(
                    unique.values(), key=lambda r: (r.tested_at, r.model_dump_json())
                )
            ),
            "measured_memory": tuple(
                m for m in evidence.measured_memory if m.content_sha256 in with_records
            ),
        }
    )


def retain_evidence(root: Path) -> list[Path]:
    """The one walk that writes the retention: a withdrawn recipe's file goes, the rest shrinks.

    Runs after every merge and refusal of a publisher batch (never inside
    `write_evidence`: a newer record later refused must not have displaced the pass
    it replaced), so each batch re-compacts against the main it merged.
    """
    recipes = catalog_recipes(root)
    base = root / "qualification/evidence"
    changed = []
    for path in sorted(base.rglob("*.json")):
        key = path.relative_to(base).with_suffix("").as_posix()
        if key not in recipes:
            path.unlink()
            changed.append(path)
            continue
        recipe, digest = recipes[key]
        if write_evidence(root, retained(read_evidence(root, key), recipe, digest)):
            changed.append(path)
    return changed
