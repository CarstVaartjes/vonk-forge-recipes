"""A held base image (policy/base-image-holds.json) stays put: refresh skips it, CI guards it."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from repo_root import ROOT

from recipe_tooling import base_image_holds as holds
from recipe_tooling.refresh_upstream import assessment as refresh_assessment
from recipe_tooling.refresh_upstream import catalog_view as refresh_catalog_view
from recipe_tooling.refresh_upstream import model as refresh_model

REPOSITORY = "docker.io/vllm/vllm-openai"
HELD = "1" * 64
NEWER = "2" * 64
DOCKERFILE = "adapters/x/Dockerfile"


def recipe(digest: str) -> dict[str, Any]:
    return {
        "identity": {"slug": "held"},
        "provenance": {"attribution": ["Example"]},
        "release": {"version": "1.0.0", "released_at": "2026-01-01"},
        "models": [],
        "execution": {
            "build": {
                "base_image": {"repository": REPOSITORY, "digest": digest},
                "context": {"path": "adapters/x"},
                "dockerfile": DOCKERFILE,
            }
        },
    }


def catalog(root: Path, digest: str, from_digest: str | None = None) -> None:
    (root / "recipes").mkdir()
    (root / "recipes/held.json").write_text(json.dumps(recipe(digest)))
    (root / "adapters/x").mkdir(parents=True)
    (root / DOCKERFILE).write_text(
        f"FROM {REPOSITORY}@sha256:{from_digest or digest}\nRUN true\n"
    )
    root.joinpath("policy").mkdir(exist_ok=True)
    (root / holds.HOLDS).write_text(
        json.dumps(
            {
                "held": {
                    "base_image": {"repository": REPOSITORY, "digest": HELD},
                    "version": "v0.27.1",
                    "reason": "newer releases break this recipe",
                    "upstream": "https://github.com/vllm-project/vllm/pull/1",
                }
            }
        )
    )


class CheckTests(unittest.TestCase):
    def problems(self, digest: str, from_digest: str | None = None) -> list[str]:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog(root, digest, from_digest)
            return holds.problems(root)

    def test_the_catalog_keeps_every_held_base_image(self) -> None:
        assert holds.problems(ROOT) == []

    def test_held_digest_passes(self) -> None:
        assert self.problems(HELD) == []

    def test_moving_a_held_recipe_without_its_hold_fails(self) -> None:
        found = self.problems(NEWER)
        assert len(found) == 2  # base_image and the Dockerfile FROM
        assert "update/remove the hold" in found[0]
        assert DOCKERFILE in found[1]

    def test_dockerfile_alone_moving_fails(self) -> None:
        found = self.problems(HELD, from_digest=NEWER)
        assert len(found) == 1
        assert f"sha256:{NEWER}" in found[0]

    def test_hold_needs_reason_and_upstream(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog(root, HELD)
            data = json.loads((root / holds.HOLDS).read_text())
            data["held"]["reason"] = ""
            root.joinpath("policy").mkdir(exist_ok=True)
            (root / holds.HOLDS).write_text(json.dumps(data))
            assert "needs a reason" in holds.problems(root)[0]

    def test_hold_for_a_missing_recipe_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog(root, HELD)
            (root / "recipes/held.json").unlink()
            assert "remove the hold" in holds.problems(root)[0]


class RefreshSkipsHeldTests(unittest.TestCase):
    """A fake refresh whose upstream moved the base image proposes nothing."""

    def assess(self, root: Path, repository: str) -> Any:
        catalog_ = refresh_catalog_view.Catalog(root, recipes={"held": recipe(HELD)})
        target = refresh_model.Target(
            "oci", repository, f"sha256:{NEWER}", "2026-10-16", None, None, "u"
        )
        item = refresh_model.Item("image", "oci", repository, f"sha256:{HELD}", target)
        assessment = refresh_model.Assessment("held", items=[item])
        assessor = refresh_assessment.Assessor(catalog_, object(), None, {})  # pyright: ignore[reportArgumentType]
        with (
            patch(
                "recipe_tooling.refresh_upstream.items.collect_items",
                return_value=assessment,
            ),
            patch.object(refresh_assessment.Assessor, "_image") as move,
        ):
            result = assessor.assess("held")
        return result, move

    def test_held_image_is_not_moved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog(root, HELD)
            result, move = self.assess(root, REPOSITORY)
        move.assert_not_called()
        assert not result.drifted
        assert not result.reasons
        assert not result.edits
        assert "is held" in "".join(result.evidence)

    def test_other_image_still_moves(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog(root, HELD)
            _result, move = self.assess(root, "ghcr.io/creator/other")
        move.assert_called_once()


if __name__ == "__main__":
    unittest.main()
