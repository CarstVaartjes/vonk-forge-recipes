"""A held Model revision (policy/model-revision-holds.json) stays: refresh skips it, CI guards it."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from repo_root import ROOT

from recipe_tooling import model_revision_holds as holds
from recipe_tooling.refresh_upstream import assessment as refresh_assessment
from recipe_tooling.refresh_upstream import catalog_view as refresh_catalog_view
from recipe_tooling.refresh_upstream import model as refresh_model

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
    root.joinpath("policy").mkdir(exist_ok=True)
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
                root.joinpath("policy").mkdir(exist_ok=True)
                (root / holds.HOLDS).write_text(json.dumps(data))
            return holds.problems(root)

    def test_the_catalog_keeps_every_held_model_revision(self) -> None:
        assert holds.problems(ROOT) == []

    def test_held_revision_passes(self) -> None:
        assert self.problems() == []

    def test_moving_a_held_recipe_without_its_hold_fails(self) -> None:
        found = self.problems(DECLINED)
        assert len(found) == 1
        assert "remove the hold" in found[0]

    def test_a_newer_revision_passes_so_the_hold_lapses(self) -> None:
        assert self.problems(NEWER) == []

    def test_hold_needs_reason_upstream_and_revision(self) -> None:
        assert "reason" in self.problems(reason="")[0]
        assert "upstream" in self.problems(upstream="#1850")[0]
        assert "declined" in self.problems(declined="826aacdf")[0]
        assert "extra" in self.problems(note="x")[0]

    def test_hold_for_a_missing_recipe_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog(root)
            (root / "recipes/held.json").unlink()
            assert "remove the hold" in holds.problems(root)[0]


class RefreshSkipsDeclinedTests(unittest.TestCase):
    """A fake refresh whose upstream Model moved proposes nothing for the declined one."""

    def assess(self, root: Path, newest: str) -> Any:
        catalog_ = refresh_catalog_view.Catalog(root, recipes={"held": recipe()})
        items = [
            refresh_model.Item(
                kind,
                "huggingface",
                REPOSITORY,
                HELD,
                refresh_model.Target(
                    "huggingface", REPOSITORY, newest, "2026-08-25", None, None, "u"
                ),
                "m",
            )
            for kind in ("model", "source")
        ]
        assessment = refresh_model.Assessment("held", items=items)
        assessor = refresh_assessment.Assessor(catalog_, object(), None, {})  # pyright: ignore[reportArgumentType]
        with (
            patch(
                "recipe_tooling.refresh_upstream.items.collect_items",
                return_value=assessment,
            ),
            patch.object(refresh_assessment.Assessor, "_model") as move,
            patch.object(refresh_assessment.Assessor, "_retarget_reference"),
        ):
            result = assessor.assess("held")
        return result, move

    def test_declined_revision_is_not_moved_and_needs_no_review(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog(root)
            result, move = self.assess(root, DECLINED)
        move.assert_not_called()
        assert not result.drifted  # its review issue closes as current
        assert not result.reasons
        assert not result.edits
        assert "is declined" in "".join(result.evidence)

    def test_a_newer_revision_is_assessed_again(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog(root)
            _result, move = self.assess(root, NEWER)
        move.assert_called_once()


if __name__ == "__main__":
    unittest.main()
