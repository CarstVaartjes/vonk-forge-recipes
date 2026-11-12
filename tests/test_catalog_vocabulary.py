"""The published vocabulary is complete, and the index carries it."""

from __future__ import annotations

import pytest
from vonk_forge_contracts import (
    CAPABILITY_TEXT,
    CapabilityName,
    QualificationStatus,
    catalog_vocabulary,
    read_catalog_index,
)
from vonk_forge_contracts.vocabulary import require_complete


def test_a_missing_entry_fails_the_completeness_check() -> None:
    without_one = {k: v for k, v in CAPABILITY_TEXT.items() if k != CapabilityName.OCR}
    with pytest.raises(RuntimeError, match="missing .*ocr"):
        require_complete("capability", without_one, CapabilityName)


def test_the_vocabulary_names_every_capability_and_status_once() -> None:
    vocabulary = catalog_vocabulary()
    assert [c.name for c in vocabulary.capabilities] == [
        m.value for m in CapabilityName
    ]
    assert [s.status for s in vocabulary.statuses] == [
        m.value for m in QualificationStatus
    ]
    known = {category.id for category in vocabulary.categories}
    assert {c.category for c in vocabulary.capabilities} <= known
    assert [s.status for s in vocabulary.statuses if s.verified] == ["Working"]


def test_the_built_index_carries_the_vocabulary(
    built_catalog: tuple[dict, object],
) -> None:
    index = read_catalog_index(built_catalog[0])
    assert index.vocabulary == catalog_vocabulary()
