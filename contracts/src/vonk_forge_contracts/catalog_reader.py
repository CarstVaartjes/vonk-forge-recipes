"""The consumer's one reader of a published ``catalog-index.json``.

An unknown value never breaks the catalog; only integrity edges refuse it.
:func:`entry_outcome` is the one decision, per entry:

=====================  ==========================================================
``REFUSE``             The entry's ``content_sha256`` is not the digest of its
                       document, or its package names another digest. The whole
                       index is refused (:class:`CatalogIndexRefused`).
``SKIP``               The entry is malformed for this release (a missing field,
                       a wrong type, an invalid value, a word it cannot do without)
                       or shares its identity with another entry. Only that entry
                       is left out, reported as a :class:`SkippedEntry`.
``KEEP_UNRECOGNISED``  The entry carries a closed word a newer minor added outside
                       its execution document (a capability, modality, alignment,
                       API endpoint or format, or qualification word; see
                       ``unrecognised``). It is kept with that word as an
                       :class:`UnrecognisedWord`, reported once per word as an
                       :class:`UnrecognisedWordReport`.
``KEEP``               Everything else.
=====================  ==========================================================

Unknown fields are ignored. The envelope is all-or-nothing: an unreadable
envelope, another ``kind`` or ``schema_version`` or another contract major is
refused. Each entry is validated once, here; what the reader returns is typed
and needs no further validation, and its ``content_sha256`` is verified.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Final, Literal, NamedTuple

from pydantic import BaseModel, ConfigDict, Field, StrictStr, ValidationError
from pydantic_core import ErrorDetails

from ._version import __version__
from .canonical import document_sha256
from .catalog_index import CatalogIndex, CatalogModelEntry, CatalogRecipeEntry
from .unrecognised import UnrecognisedWord, read_context

_CONTRACT_MAJOR: Final = int(__version__.split(".", 1)[0])
_SHA256: Final = re.compile(r"[0-9a-f]{64}")
# Pydantic error types that mean "a word this release does not list".
_UNKNOWN_WORD_ERRORS: Final = frozenset({"literal_error", "enum", "union_tag_invalid"})


class CatalogSection(StrEnum):
    MODELS = "catalog_entities"
    RECIPES = "recipes"


class EntryOutcome(StrEnum):
    KEEP = "keep"
    KEEP_UNRECOGNISED = "keep-unrecognised"
    SKIP = "skip"
    REFUSE = "refuse"


class DigestCheck(StrEnum):
    MATCH = "match"
    MISMATCH = "mismatch"
    # No digest or document to compare: the entry is malformed, not tampered.
    UNREADABLE = "unreadable"


class SkipReason(StrEnum):
    MISSING = "missing"
    WRONG_TYPE = "wrong-type"
    INVALID_VALUE = "invalid-value"
    # A word this release does not know where the entry cannot do without it
    # (an execution or API field); a newer platform reads it.
    UNKNOWN_WORD = "unknown-word"
    DUPLICATE = "duplicate"


class RefusalReason(StrEnum):
    ENVELOPE_INVALID = "envelope-invalid"
    # Another ``kind``, ``schema_version`` or contract major.
    SCHEMA_UNSUPPORTED = "schema-unsupported"
    DIGEST_MISMATCH = "digest-mismatch"


class CatalogIndexRefused(ValueError):
    """The index fails an integrity edge; nothing of it may be used."""

    def __init__(self, reason: RefusalReason, detail: str) -> None:
        super().__init__(f"{reason.value}: {detail}")
        self.reason = reason
        self.detail = detail


class _Report(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class SkippedEntry(_Report):
    """One entry left out; the rest of the index applies."""

    kind: Literal["skipped-entry"] = "skipped-entry"
    section: CatalogSection
    # ``publisher/slug`` when the document names it, else its source path or
    # its position in the section.
    entry: StrictStr
    # The dotted location inside the entry; ``entry`` for the entry as a whole.
    field: StrictStr
    reason: SkipReason


class UnrecognisedWordReport(_Report):
    """One word this release does not know, and every kept entry carrying it."""

    kind: Literal["unrecognised-word"] = "unrecognised-word"
    section: CatalogSection
    field: StrictStr
    word: StrictStr
    entries: tuple[StrictStr, ...]


CatalogProblem = Annotated[
    SkippedEntry | UnrecognisedWordReport, Field(discriminator="kind")
]


@dataclass(frozen=True, slots=True)
class EntryRead[E: (CatalogModelEntry, CatalogRecipeEntry)]:
    """One entry's outcome; ``entry`` is ``None`` exactly when it is skipped."""

    outcome: EntryOutcome
    entry: E | None
    problems: tuple[CatalogProblem, ...]


class CatalogRead(NamedTuple):
    """The entries this release reads, and what it left out or did not recognise."""

    index: CatalogIndex
    problems: tuple[CatalogProblem, ...]


def entry_outcome(
    digest: DigestCheck, *, parsed: bool, unrecognised: bool, duplicate: bool
) -> EntryOutcome:
    """The one decision: refuse at an integrity edge, else skip, else keep."""

    match digest:
        case DigestCheck.MISMATCH:
            return EntryOutcome.REFUSE
        case DigestCheck.UNREADABLE:
            return EntryOutcome.SKIP
        case DigestCheck.MATCH if not parsed or duplicate:
            return EntryOutcome.SKIP
        case DigestCheck.MATCH if unrecognised:
            return EntryOutcome.KEEP_UNRECOGNISED
        case DigestCheck.MATCH:
            return EntryOutcome.KEEP


def read_catalog_model_entry(
    raw: object, *, position: int | None = None
) -> EntryRead[CatalogModelEntry]:
    """Read one published Model entry (raises :class:`CatalogIndexRefused`)."""

    return _read_entry(CatalogModelEntry, CatalogSection.MODELS, raw, position)


def read_catalog_recipe_entry(
    raw: object, *, position: int | None = None
) -> EntryRead[CatalogRecipeEntry]:
    """Read one published Recipe entry (raises :class:`CatalogIndexRefused`)."""

    return _read_entry(CatalogRecipeEntry, CatalogSection.RECIPES, raw, position)


def read_catalog(document: Mapping[str, object]) -> CatalogRead:
    """Read a published index: every readable entry, and a report of the rest."""

    if not isinstance(document, Mapping):
        raise CatalogIndexRefused(
            RefusalReason.ENVELOPE_INVALID, "the index is not a JSON object"
        )
    raw_models = document.get(CatalogSection.MODELS.value)
    raw_recipes = document.get(CatalogSection.RECIPES.value)
    if not isinstance(raw_models, list) or not isinstance(raw_recipes, list):
        raise CatalogIndexRefused(
            RefusalReason.ENVELOPE_INVALID, "the index entries are not arrays"
        )
    envelope = _read_envelope(document)
    models, model_problems = _settle(
        CatalogSection.MODELS,
        [
            read_catalog_model_entry(raw, position=position)
            for position, raw in enumerate(raw_models)
        ],
    )
    recipes, recipe_problems = _settle(
        CatalogSection.RECIPES,
        [
            read_catalog_recipe_entry(raw, position=position)
            for position, raw in enumerate(raw_recipes)
        ],
    )
    index = envelope.model_copy(
        update={"catalog_entities": tuple(models), "recipes": tuple(recipes)}
    )
    return CatalogRead(index=index, problems=(*model_problems, *recipe_problems))


def _read_envelope(document: Mapping[str, object]) -> CatalogIndex:
    try:
        envelope = CatalogIndex.model_validate(
            {
                **document,
                CatalogSection.MODELS.value: [],
                CatalogSection.RECIPES.value: [],
            },
            extra="ignore",
        )
    except ValidationError as error:
        fields = {str(item["loc"][0]) for item in error.errors() if item["loc"]}
        if fields & {"kind", "schema_version"}:
            raise CatalogIndexRefused(
                RefusalReason.SCHEMA_UNSUPPORTED,
                f"unsupported index {', '.join(sorted(fields))}",
            ) from error
        raise CatalogIndexRefused(
            RefusalReason.ENVELOPE_INVALID,
            f"invalid index {', '.join(sorted(fields)) or 'envelope'}",
        ) from error
    major = int(envelope.contract_version.split(".", 1)[0])
    if major != _CONTRACT_MAJOR:
        raise CatalogIndexRefused(
            RefusalReason.SCHEMA_UNSUPPORTED,
            f"contract {envelope.contract_version} is not a v{_CONTRACT_MAJOR} release",
        )
    return envelope


def _read_entry[E: (CatalogModelEntry, CatalogRecipeEntry)](
    model: type[E], section: CatalogSection, raw: object, position: int | None
) -> EntryRead[E]:
    digest = _digest_check(section, raw)
    context = read_context()
    entry: E | None = None
    error: ErrorDetails | None = None
    if digest is not DigestCheck.MISMATCH:
        try:
            entry = model.model_validate(raw, extra="ignore", context=context)
        except ValidationError as failure:
            error = failure.errors()[0]
    words = tuple(_unrecognised(entry, ())) if entry is not None else ()
    outcome = entry_outcome(
        digest, parsed=entry is not None, unrecognised=bool(words), duplicate=False
    )
    name = _entry_name(raw, position)
    match outcome:
        case EntryOutcome.REFUSE:
            raise CatalogIndexRefused(
                RefusalReason.DIGEST_MISMATCH,
                f"{section.value} entry {name} does not match its digest",
            )
        case EntryOutcome.SKIP:
            field, reason = _skip_cause(error)
            skipped = SkippedEntry(
                section=section, entry=name, field=field, reason=reason
            )
            return EntryRead(outcome=outcome, entry=None, problems=(skipped,))
        case EntryOutcome.KEEP | EntryOutcome.KEEP_UNRECOGNISED:
            reports = tuple(
                UnrecognisedWordReport(
                    section=section, field=field, word=word, entries=(name,)
                )
                for field, word in dict.fromkeys(words)
            )
            return EntryRead(outcome=outcome, entry=entry, problems=reports)


def _settle[E: (CatalogModelEntry, CatalogRecipeEntry)](
    section: CatalogSection, reads: Sequence[EntryRead[E]]
) -> tuple[list[E], list[CatalogProblem]]:
    """Skip every entry of a duplicated identity; report each word once."""

    counts = Counter(_identity(read.entry) for read in reads if read.entry is not None)
    duplicated = {name for name, count in counts.items() if count > 1}
    kept: list[E] = []
    skipped: list[CatalogProblem] = []
    words: dict[tuple[str, str], list[str]] = {}
    for read in reads:
        if read.entry is None:
            skipped.extend(read.problems)
            continue
        name = _identity(read.entry)
        outcome = entry_outcome(
            DigestCheck.MATCH,
            parsed=True,
            unrecognised=bool(read.problems),
            duplicate=name in duplicated,
        )
        if outcome is EntryOutcome.SKIP:
            skipped.append(
                SkippedEntry(
                    section=section,
                    entry=name,
                    field="document.identity",
                    reason=SkipReason.DUPLICATE,
                )
            )
            continue
        kept.append(read.entry)
        for report in read.problems:
            if isinstance(report, UnrecognisedWordReport):
                words.setdefault((report.field, report.word), []).append(name)
    reports = [
        UnrecognisedWordReport(
            section=section, field=field, word=word, entries=tuple(entries)
        )
        for (field, word), entries in sorted(words.items())
    ]
    return kept, [*skipped, *reports]


def _digest_check(section: CatalogSection, raw: object) -> DigestCheck:
    """Compare the published digests with the document, before any parsing."""

    if not isinstance(raw, Mapping):
        return DigestCheck.UNREADABLE
    document, claimed = raw.get("document"), raw.get("content_sha256")
    if not isinstance(document, Mapping) or not _is_sha256(claimed):
        return DigestCheck.UNREADABLE
    try:
        actual = document_sha256(document)
    except (TypeError, ValueError):
        return DigestCheck.UNREADABLE
    if actual != claimed:
        return DigestCheck.MISMATCH
    package = raw.get("package")
    if section is CatalogSection.RECIPES and isinstance(package, Mapping):
        for in_package, in_entry in (
            ("recipe_content_sha256", "content_sha256"),
            ("source_bundle_sha256", "source_bundle_sha256"),
        ):
            first, second = package.get(in_package), raw.get(in_entry)
            if _is_sha256(first) and _is_sha256(second) and first != second:
                return DigestCheck.MISMATCH
    return DigestCheck.MATCH


def _skip_cause(error: ErrorDetails | None) -> tuple[str, SkipReason]:
    if error is None:  # parsed, but its document cannot be hashed
        return "document", SkipReason.INVALID_VALUE
    field = ".".join(str(part) for part in error["loc"]) or "entry"
    kind = error["type"]
    if kind == "missing":
        return field, SkipReason.MISSING
    if kind in _UNKNOWN_WORD_ERRORS:
        return field, SkipReason.UNKNOWN_WORD
    if kind.endswith("_type") or kind == "is_instance_of":
        return field, SkipReason.WRONG_TYPE
    return field, SkipReason.INVALID_VALUE


def _unrecognised(value: object, path: tuple[str, ...]) -> Iterator[tuple[str, str]]:
    if isinstance(value, UnrecognisedWord):
        yield ".".join(path), str(value)
    elif isinstance(value, BaseModel):
        for name in type(value).model_fields:
            yield from _unrecognised(getattr(value, name), (*path, name))
    elif isinstance(value, list | tuple):
        for item in value:
            yield from _unrecognised(item, path)


def _identity(entry: CatalogModelEntry | CatalogRecipeEntry) -> str:
    identity = entry.document.identity
    return f"{identity.publisher}/{identity.slug}"


def _entry_name(raw: object, position: int | None) -> str:
    if isinstance(raw, Mapping):
        document = raw.get("document")
        identity = document.get("identity") if isinstance(document, Mapping) else None
        if isinstance(identity, Mapping):
            publisher, slug = identity.get("publisher"), identity.get("slug")
            if isinstance(publisher, str) and isinstance(slug, str):
                return f"{publisher}/{slug}"
        source = raw.get("source_path")
        if isinstance(source, str):
            return source
    return f"#{position}" if position is not None else "unidentified"


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None
