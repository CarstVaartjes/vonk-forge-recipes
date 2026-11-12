"""The published vocabulary is complete, and the index carries it."""

from __future__ import annotations

import pytest
from vonk_forge_contracts import (
    CapabilityName,
    catalog_vocabulary,
    read_catalog_index,
)
from vonk_forge_contracts.vocabulary import CAPABILITY_TEXT, require_complete


def test_a_missing_entry_fails_the_completeness_check() -> None:
    without_one = {k: v for k, v in CAPABILITY_TEXT.items() if k != CapabilityName.OCR}
    with pytest.raises(RuntimeError, match="missing .*ocr"):
        require_complete("capability", without_one, CapabilityName)


def test_only_working_counts_as_verified() -> None:
    verified = [s.status for s in catalog_vocabulary().statuses if s.verified]
    assert verified == ["Working"]


def test_the_built_index_carries_the_vocabulary(
    built_catalog: tuple[dict, object],
) -> None:
    index = read_catalog_index(built_catalog[0])
    assert index.vocabulary == catalog_vocabulary()
