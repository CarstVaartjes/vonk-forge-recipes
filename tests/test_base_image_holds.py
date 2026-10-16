"""A held base image (base-image-holds.json) stays put: refresh skips it, CI guards it."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from test_refresh_upstream import refresh

ROOT = Path(__file__).resolve().parents[1]
LOADER = importlib.machinery.SourceFileLoader(
    "check_base_image_holds", str(ROOT / "tools/check-base-image-holds")
)
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
assert SPEC is not None
holds = importlib.util.module_from_spec(SPEC)
sys.modules[LOADER.name] = holds
LOADER.exec_module(holds)

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
        self.assertEqual(holds.problems(ROOT), [])

    def test_held_digest_passes(self) -> None:
        self.assertEqual(self.problems(HELD), [])

    def test_moving_a_held_recipe_without_its_hold_fails(self) -> None:
        found = self.problems(NEWER)
        self.assertEqual(len(found), 2)  # base_image and the Dockerfile FROM
        self.assertIn("update/remove the hold", found[0])
        self.assertIn(DOCKERFILE, found[1])

    def test_dockerfile_alone_moving_fails(self) -> None:
        found = self.problems(HELD, from_digest=NEWER)
        self.assertEqual(len(found), 1)
        self.assertIn(f"sha256:{NEWER}", found[0])

    def test_hold_needs_reason_and_upstream(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog(root, HELD)
            data = json.loads((root / holds.HOLDS).read_text())
            data["held"]["reason"] = ""
            (root / holds.HOLDS).write_text(json.dumps(data))
            self.assertIn("needs a reason", holds.problems(root)[0])

    def test_hold_for_a_missing_recipe_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog(root, HELD)
            (root / "recipes/held.json").unlink()
            self.assertIn("remove the hold", holds.problems(root)[0])


class RefreshSkipsHeldTests(unittest.TestCase):
    """A fake refresh whose upstream moved the base image proposes nothing."""

    def assess(self, root: Path, repository: str) -> Any:
        catalog_ = refresh.Catalog(root, recipes={"held": recipe(HELD)})
        target = refresh.Target(
            "oci", repository, f"sha256:{NEWER}", "2026-10-16", None, None, "u"
        )
        item = refresh.Item("image", "oci", repository, f"sha256:{HELD}", target)
        assessment = refresh.Assessment("held", items=[item])
        assessor = refresh.Assessor(catalog_, object(), None, {})
        with (
            patch("refresh_upstream.collect_items", return_value=assessment),
            patch.object(refresh.Assessor, "_image") as move,
        ):
            result = assessor.assess("held")
        return result, move

    def test_held_image_is_not_moved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog(root, HELD)
            result, move = self.assess(root, REPOSITORY)
        move.assert_not_called()
        self.assertFalse(result.drifted)
        self.assertFalse(result.reasons)
        self.assertFalse(result.edits)
        self.assertIn("is held", "".join(result.evidence))

    def test_other_image_still_moves(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog(root, HELD)
            _result, move = self.assess(root, "ghcr.io/creator/other")
        move.assert_called_once()


if __name__ == "__main__":
    unittest.main()
