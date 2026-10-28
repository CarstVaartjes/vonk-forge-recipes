"""A held Model revision (model-revision-holds.json) stays: refresh skips it, CI guards it."""

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
    "check_model_revision_holds", str(ROOT / "tools/check-model-revision-holds")
)
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
assert SPEC is not None
holds = importlib.util.module_from_spec(SPEC)
sys.modules[LOADER.name] = holds
LOADER.exec_module(holds)

REPOSITORY = "poolside/Laguna-S-2.1-NVFP4"
HELD = "0" * 40
DECLINED = "8" * 40
NEWER = "9" * 40


def recipe() -> dict[str, Any]:
    return {
        "identity": {"slug": "held"},
        "provenance": {"attribution": ["Example"]},
        "release": {"version": "1.0.0", "released_at": "2026-01-01"},
        "models": [{"id": "primary", "model": {"slug": "m"}, "files": []}],
        "execution": {"build": {"context": {"path": "adapters/x"}}},
        "topology": {"roles": []},
    }


def catalog(root: Path, revision: str = HELD) -> None:
    (root / "recipes").mkdir()
    (root / "recipes/held.json").write_text(json.dumps(recipe()))
    (root / "models").mkdir()
    (root / "models/m.json").write_text(
        json.dumps(
            {
                "source": {
                    "repository": f"https://huggingface.co/{REPOSITORY}",
                    "revision": revision,
                }
            }
        )
    )
    (root / holds.HOLDS).write_text(
        json.dumps(
            {
                "held": {
                    "model": {"repository": REPOSITORY},
                    "declined": DECLINED,
                    "reason": "the declined revision does not fit one Spark",
                    "upstream": "https://github.com/o/r/pull/1850",
                }
            }
        )
    )


class CheckTests(unittest.TestCase):
    def problems(self, revision: str = HELD, **change: Any) -> list[str]:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog(root, revision)
            if change:
                data = json.loads((root / holds.HOLDS).read_text())
                data["held"].update(change)
                (root / holds.HOLDS).write_text(json.dumps(data))
            return holds.problems(root)

    def test_the_catalog_keeps_every_held_model_revision(self) -> None:
        self.assertEqual(holds.problems(ROOT), [])

    def test_held_revision_passes(self) -> None:
        self.assertEqual(self.problems(), [])

    def test_moving_a_held_recipe_without_its_hold_fails(self) -> None:
        found = self.problems(DECLINED)
        self.assertEqual(len(found), 1)
        self.assertIn("remove the hold", found[0])

    def test_a_newer_revision_passes_so_the_hold_lapses(self) -> None:
        self.assertEqual(self.problems(NEWER), [])

    def test_hold_needs_reason_upstream_and_revision(self) -> None:
        self.assertIn("reason", self.problems(reason="")[0])
        self.assertIn("upstream", self.problems(upstream="#1850")[0])
        self.assertIn("declined", self.problems(declined="826aacdf")[0])
        self.assertIn("extra", self.problems(note="x")[0])

    def test_hold_for_a_missing_recipe_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog(root)
            (root / "recipes/held.json").unlink()
            self.assertIn("remove the hold", holds.problems(root)[0])


class RefreshSkipsDeclinedTests(unittest.TestCase):
    """A fake refresh whose upstream Model moved proposes nothing for the declined one."""

    def assess(self, root: Path, newest: str) -> Any:
        catalog_ = refresh.Catalog(root, recipes={"held": recipe()})
        items = [
            refresh.Item(
                kind,
                "huggingface",
                REPOSITORY,
                HELD,
                refresh.Target(
                    "huggingface", REPOSITORY, newest, "2026-08-25", None, None, "u"
                ),
                "m",
            )
            for kind in ("model", "source")
        ]
        assessment = refresh.Assessment("held", items=items)
        assessor = refresh.Assessor(catalog_, object(), None, {})
        with (
            patch("refresh_upstream.collect_items", return_value=assessment),
            patch.object(refresh.Assessor, "_model") as move,
            patch.object(refresh.Assessor, "_retarget_reference"),
        ):
            result = assessor.assess("held")
        return result, move

    def test_declined_revision_is_not_moved_and_needs_no_review(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog(root)
            result, move = self.assess(root, DECLINED)
        move.assert_not_called()
        self.assertFalse(result.drifted)  # its review issue closes as current
        self.assertFalse(result.reasons)
        self.assertFalse(result.edits)
        self.assertIn("is declined", "".join(result.evidence))

    def test_a_newer_revision_is_assessed_again(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog(root)
            _result, move = self.assess(root, NEWER)
        move.assert_called_once()


if __name__ == "__main__":
    unittest.main()
