from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from generated_catalog import GENERATED

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(autouse=True)
def forbid_sweep_evidence_publication(request, monkeypatch):
    """Fake fleet tests must never start the git/GitHub publisher subprocess."""
    if request.node.path.name == "test_qualification_evidence_publisher.py":
        return  # publisher unit tests supply their own fake subprocesses
    from spark_sweep.evidence_publisher import EvidencePublisher

    def forbidden(self, now):
        pytest.fail("sweep tests must disable evidence publication")

    monkeypatch.setattr(EvidencePublisher, "tick", forbidden)


@pytest.fixture(scope="session")
def built_catalog() -> tuple[dict, Path]:
    """The catalog and package directory built once for the whole session.

    One build reads, hashes and gzips the closure of all recipes, which is the
    most expensive thing this suite does. Every test reads the same generated
    outputs; the cases that exercise the archive validators consume the
    produced bytes without writing to the package directory.
    """
    catalog = json.loads((GENERATED / "catalog-index.json").read_text(encoding="utf-8"))
    return catalog, GENERATED / "packages"
