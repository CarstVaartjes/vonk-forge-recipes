from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar

ROOT = Path(__file__).resolve().parents[1]
LOADER = importlib.machinery.SourceFileLoader(
    "refresh_upstream", str(ROOT / "tools/refresh-upstream")
)
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
assert SPEC is not None
refresh = importlib.util.module_from_spec(SPEC)
sys.modules[LOADER.name] = refresh
LOADER.exec_module(refresh)

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
    return refresh.Comparison(
        status,
        [],
        [{"filename": name, "status": "modified"} for name in files],
        truncated,
        "u",
    )


def classify(cmp, patched: set[str] | frozenset[str] = frozenset(), problems=()):
    return refresh.classify_changes(
        cmp, patched=set(patched), vendor_problems=list(problems)
    )


class PinningRuleTests(unittest.TestCase):
    def test_newest_plain_version_tag_wins(self) -> None:
        tags = [
            {"name": n} for n in ("v1.9.0", "v1.10.0", "v1.10.0-rc1", "nightly", "v0.3")
        ]
        self.assertEqual(refresh.pick_version_tag(tags)["name"], "v1.10.0")

    def test_no_version_tag_means_follow_the_head(self) -> None:
        self.assertIsNone(refresh.pick_version_tag([{"name": "latest"}, {"name": "x"}]))


class ClassificationTests(unittest.TestCase):
    def test_docs_and_code_only_is_mechanical(self) -> None:
        cmp = comparison(
            files=("src/model.py", "docs/guide.md", ".github/ci.yml", "logo.png")
        )
        self.assertEqual(classify(cmp), [])

    def test_launch_script_config_readme_and_patches_are_not(self) -> None:
        for name in (
            "run.sh",
            "Dockerfile",
            "serve.py",
            "configs/a.yaml",
            "README.md",
            "x.patch",
        ):
            with self.subTest(name=name):
                self.assertTrue(classify(comparison(files=(name,))))

    def test_downgrade_divergence_and_truncation_are_not(self) -> None:
        for cmp in (
            comparison("behind"),
            comparison("diverged"),
            comparison(truncated=True),
        ):
            self.assertTrue(classify(cmp))

    def test_patch_target_change_and_vendor_problems_are_not(self) -> None:
        self.assertTrue(classify(comparison(files=("lib/a.py",)), patched={"lib/a.py"}))
        self.assertTrue(classify(comparison(), problems=["x differs"]))
        self.assertEqual(
            classify(comparison(files=("lib/b.py",)), patched={"lib/a.py"}), []
        )

    def test_renamed_away_blocking_file_counts(self) -> None:
        cmp = refresh.Comparison(
            "ahead",
            [],
            [
                {
                    "filename": "old.txt",
                    "previous_filename": "run.sh",
                    "status": "renamed",
                }
            ],
            False,
            "u",
        )
        self.assertTrue(classify(cmp))

    def test_patch_targets_are_read_from_diff_headers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "a.patch").write_text("--- a/lib/x.py\n+++ b/lib/x.py\n@@\n")
            self.assertEqual(refresh.patch_targets(Path(tmp)), {"lib/x.py"})


def files(*paths: str, sha: str = "a" * 64):
    return [
        {
            "path": p,
            "sha256": sha,
            "installed_bytes": 10,
            "roles": refresh.catalog._roles(p),
        }
        for p in paths
    ]


def model_with(paths: tuple[str, ...]):
    document = json.loads(json.dumps(MODEL))
    document["identity"]["slug"] = "demo-" + OLD[:8]
    document["source"]["revision"] = OLD
    document["files"] = [
        {
            "id": refresh.file_id(p, "c" * 64),
            "path": p,
            "roles": refresh.catalog._roles(p),
            "sha256": "c" * 64,
            "size_bytes": 10,
        }
        for p in paths
    ]
    return document


class ModelTests(unittest.TestCase):
    def test_same_paths_with_new_digests_is_compatible(self) -> None:
        old = model_with(("model.safetensors", "config.json"))
        self.assertEqual(
            refresh.model_shape_problems(
                old, files("model.safetensors", "config.json", "NOTES.md")
            ),
            [],
        )

    def test_removed_and_new_weight_files_are_not(self) -> None:
        old = model_with(("model.safetensors", "config.json"))
        self.assertTrue(refresh.model_shape_problems(old, files("model.safetensors")))
        self.assertTrue(
            refresh.model_shape_problems(
                old, files("model.safetensors", "config.json", "model-2.safetensors")
            )
        )

    def test_refreshed_model_renames_and_remaps_ids(self) -> None:
        old = model_with(("model.safetensors",))
        target = refresh.Target(
            "huggingface", "o/r", NEW, "2026-09-30", None, None, "u"
        )
        new = refresh.refreshed_model(old, files("model.safetensors"), target)
        self.assertEqual(new["identity"]["slug"], "demo-ffffffff")
        self.assertEqual(new["source"]["revision"], NEW)
        self.assertEqual(
            new["files"][0]["id"], refresh.file_id("model.safetensors", "a" * 64)
        )


class ReleaseTests(unittest.TestCase):
    recipe: ClassVar[dict[str, object]] = {
        "release": {"version": "1.0.0", "released_at": "2026-06-01"}
    }

    def item(self, version=None, date="2026-09-01", kind="source"):
        target = refresh.Target("github", "o/r", NEW, date, None, version, "u")
        return refresh.Item(kind, "github", "o/r", OLD, target)

    def test_upstream_version_and_date_when_published(self) -> None:
        self.assertEqual(
            refresh.released(self.recipe, [self.item("2.4.1")]), ("2.4.1", "2026-09-01")
        )

    def test_own_version_is_bumped_when_upstream_has_none(self) -> None:
        self.assertEqual(
            refresh.released(self.recipe, [self.item()]), ("1.0.1", "2026-09-01")
        )

    def test_invalid_upstream_spelling_falls_back_to_own_version(self) -> None:
        self.assertEqual(
            refresh.released(self.recipe, [self.item("1.6 beta")])[0], "1.0.1"
        )


if __name__ == "__main__":
    unittest.main()
