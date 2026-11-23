from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
from generated_catalog import GENERATED

from recipe_tooling.protocol_wheel import selection
from recipe_tooling.qualification.evidence_store import read_evidence
from recipe_tooling.qualification.memory_need import constants as mn_constants
from recipe_tooling.qualification.memory_need import evidence as mn_evidence
from recipe_tooling.upstream_http import forget_busy
from spark_sweep.evidence_publisher import EvidencePublisher

ROOT = Path(__file__).resolve().parents[1]
# The test import path is `[tool.pytest.ini_options] pythonpath` (the repository, the
# contracts source and src); the selected agent-protocol wheel is added here (the
# adapters overlay the same wheel).
sys.path.insert(0, str(ROOT / selection(ROOT).artifact))


@pytest.fixture(autouse=True)
def no_remembered_rate_limit():
    """A host marked busy by one test must not refuse another test's request."""

    forget_busy()
    yield
    forget_busy()


@pytest.fixture(autouse=True)
def forbid_sweep_evidence_publication(request, monkeypatch):
    """Fake fleet tests must never start the git/GitHub publisher subprocess."""
    if request.node.path.name == "test_qualification_evidence_publisher.py":
        return  # publisher unit tests supply their own fake subprocesses

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

    live, empty = mn_constants.ROOT, tmp_path_factory.mktemp("no-evidence")

    def evidence(key):
        # A test that points the module at its own corpus keeps it.
        return read_evidence(
            empty if live == mn_constants.ROOT else mn_constants.ROOT, key
        )

    monkeypatch.setattr(mn_evidence, "_evidence", evidence)


@pytest.fixture(scope="session")
def built_catalog() -> tuple[dict[str, Any], Path]:
    """The catalog and package directory built once for the whole session.

    One build reads, hashes and gzips the closure of all recipes, which is the
    most expensive thing this suite does. Every test reads the same generated
    outputs; the cases that exercise the archive validators consume the
    produced bytes without writing to the package directory.
    """
    catalog = json.loads((GENERATED / "catalog-index.json").read_text(encoding="utf-8"))
    return catalog, GENERATED / "packages"
