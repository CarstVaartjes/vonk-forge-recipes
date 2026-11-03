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


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "live_evidence: read the committed qualification evidence in memory checks",
    )


@pytest.fixture(autouse=True)
def isolate_memory_evidence(request, monkeypatch, tmp_path_factory):
    """Memory checks see no hardware evidence unless a test opts in.

    Sweep evidence PRs keep proving recipes; a test that asserted a live recipe is
    unproven would fail every evidence batch that proves it.
    """
    if request.node.get_closest_marker("live_evidence"):
        return
    import qualification.memory_need as memory
    from spark_sweep.qualification import read_evidence

    live, empty = memory.ROOT, tmp_path_factory.mktemp("no-evidence")

    def evidence(key):
        # A test that points the module at its own corpus keeps it.
        return read_evidence(empty if memory.ROOT == live else memory.ROOT, key)

    monkeypatch.setattr(memory, "_evidence", evidence)


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
