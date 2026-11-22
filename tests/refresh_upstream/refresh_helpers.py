"""Shared fixtures of the refresh_upstream tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from repo_root import ROOT

from recipe_tooling.catalog_model import file_id, roles_for
from recipe_tooling.refresh_upstream import catalog_view as refresh_catalog_view
from recipe_tooling.refresh_upstream import classification as refresh_classification
from recipe_tooling.refresh_upstream import github as refresh_github
from recipe_tooling.refresh_upstream import model as refresh_model
from recipe_tooling.refresh_upstream import publish as refresh_publish

#: The fields a PR from this repository carries (see recipe_tooling.qualification.own_pulls).
OWN = {"isCrossRepository": False, "headRepositoryOwner": {"login": "CarstVaartjes"}}

MODEL = json.loads(
    (
        ROOT / "contracts/src/vonk_forge_contracts/examples/model-definition.json"
    ).read_text()
)
OLD = "0123456789abcdef0123456789abcdef01234567"
NEW = "f" * 40


def comparison(
    status: str = "ahead", files: tuple[str, ...] = (), truncated: bool = False
):
    return refresh_model.Comparison(
        status,
        [],
        [{"filename": name, "status": "modified"} for name in files],
        truncated,
        "u",
    )


def classify(
    cmp,
    patched: set[str] | frozenset[str] = frozenset(),
    problems=(),
    used_changes=None,
):
    return refresh_classification.classify_changes(
        cmp,
        patched=set(patched),
        vendor_problems=list(problems),
        used_changes=used_changes,
    )


PATCH_HOME = {
    "gh": refresh_github,
    "sync_issue": refresh_publish,
    "publish_mechanical": refresh_publish,
}


def files(*paths: str, sha: str = "a" * 64):
    return [
        {
            "path": p,
            "sha256": sha,
            "installed_bytes": 10,
            "roles": roles_for(p),
        }
        for p in paths
    ]


def model_with(paths: tuple[str, ...]):
    document = json.loads(json.dumps(MODEL))
    document["identity"]["slug"] = "demo-" + OLD[:8]
    document["source"]["revision"] = OLD
    document["files"] = [
        {
            "id": file_id(p, "c" * 64),
            "path": p,
            "roles": roles_for(p),
            "sha256": "c" * 64,
            "size_bytes": 10,
        }
        for p in paths
    ]
    return document


def single_recipe_catalog(source_reference: str) -> Any:
    return refresh_catalog_view.Catalog(
        Path(tempfile.gettempdir()),
        recipes={
            "recipe": {
                "identity": {"slug": "recipe"},
                "provenance": {
                    "attribution": ["Example"],
                    "source_reference": source_reference,
                },
                "release": {"version": "1.0.0", "released_at": "2026-01-01"},
                "models": [],
                "execution": {"build": {"context": {"path": "missing"}}},
            }
        },
    )


if __name__ == "__main__":
    unittest.main()


OLD_YAML = """defaults:
  max_num_batched_tokens: 8192
env:
  VLLM_MARLIN_USE_ATOMIC_ADD: 1
command: |
  vllm serve org/model \\
    --trust-remote-code \\
    --max-num-batched-tokens {max_num_batched_tokens}
"""


def failing(
    url: str = "https://github.com/o/r/actions/runs/7/job/9",
) -> list[dict[str, str]]:
    return [
        {
            "name": "Validate producer contracts and packages",
            "conclusion": "FAILURE",
            "detailsUrl": url,
        }
    ]
