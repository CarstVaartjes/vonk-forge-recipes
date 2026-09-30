from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

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

    def test_provider_gate_change_creates_review_without_retargeting_model(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model = model_with(("model.safetensors",))
            model["identity"]["slug"] = "gated-model"
            model["source"] = {
                "repository": "https://huggingface.co/example/model",
                "revision": OLD,
            }
            model["requires_token"] = False
            recipe = {
                "identity": {"slug": "recipe"},
                "models": [{"model": {"slug": "gated-model"}}],
                "execution": {"build": {"context": {"path": "missing-context"}}},
            }
            catalog = refresh.Catalog(
                Path(temporary),
                recipes={"recipe": recipe},
                models={"gated-model": model},
            )
            target = refresh.Target(
                "huggingface",
                "example/model",
                OLD,
                "2026-09-30",
                None,
                None,
                "https://huggingface.co/example/model/tree/" + OLD,
                access_status="gated",
            )

            result = refresh.collect_items(
                catalog, "recipe", object(), lambda _provider, _repo: target
            )

            self.assertFalse(result.items)
            self.assertTrue(result.drifted)
            self.assertIn("provider access changed to gated", result.reasons[0])

    def test_restricted_token_required_or_unspecified_model_is_unverified(self) -> None:
        for requires_token in (True, None):
            with (
                self.subTest(requires_token=requires_token),
                tempfile.TemporaryDirectory() as temporary,
            ):
                model = model_with(("model.safetensors",))
                model["identity"]["slug"] = "restricted-model"
                model["source"] = {
                    "repository": "https://huggingface.co/example/model",
                    "revision": OLD,
                }
                if requires_token is not None:
                    model["requires_token"] = requires_token
                else:
                    model.pop("requires_token", None)
                recipe = {
                    "identity": {"slug": "recipe"},
                    "models": [{"model": {"slug": "restricted-model"}}],
                    "execution": {"build": {"context": {"path": "missing-context"}}},
                }
                catalog = refresh.Catalog(
                    Path(temporary),
                    recipes={"recipe": recipe},
                    models={"restricted-model": model},
                )

                def restricted(_provider: str, _repo: str):
                    raise refresh.AccessRestricted("anonymous metadata denied")

                result = refresh.collect_items(catalog, "recipe", object(), restricted)

                self.assertFalse(result.items)
                self.assertFalse(result.reasons)
                self.assertFalse(result.drifted)
                self.assertTrue(result.unreachable)
                self.assertIn("no credentials were requested", result.unreachable[0])

    def test_unknown_anonymous_access_status_is_unverified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model = model_with(("model.safetensors",))
            model["identity"]["slug"] = "unknown-access-model"
            model["source"] = {
                "repository": "https://huggingface.co/example/model",
                "revision": OLD,
            }
            model["requires_token"] = False
            recipe = {
                "identity": {"slug": "recipe"},
                "models": [{"model": {"slug": "unknown-access-model"}}],
                "execution": {"build": {"context": {"path": "missing-context"}}},
            }
            catalog = refresh.Catalog(
                Path(temporary),
                recipes={"recipe": recipe},
                models={"unknown-access-model": model},
            )
            target = refresh.Target(
                "huggingface",
                "example/model",
                OLD,
                "2026-09-30",
                None,
                None,
                "https://huggingface.co/example/model/tree/" + OLD,
                access_status="unknown",
            )

            result = refresh.collect_items(
                catalog, "recipe", object(), lambda _provider, _repo: target
            )

            self.assertFalse(result.items)
            self.assertFalse(result.reasons)
            self.assertFalse(result.drifted)
            self.assertTrue(result.unreachable)
            self.assertIn("access status is unknown", result.unreachable[0])

    def test_huggingface_resolver_records_explicit_access_status(self) -> None:
        class FakeHttp:
            def json(self, _url, _provider, *, anonymous=False):
                self.anonymous = anonymous
                return {"sha": OLD, "gated": "auto", "private": False}

        http = FakeHttp()
        target = refresh.resolve_huggingface(http, "example/model")
        self.assertEqual(target.access_status, "gated")
        self.assertTrue(http.anonymous)


class EmbeddedSourceReviewTests(unittest.TestCase):
    def test_unrelated_default_head_advance_does_not_block_safe_refresh(
        self,
    ) -> None:
        class FakeProvider:
            def json(self, url: str, _provider: str):
                if url.endswith("/repos/example/dependency"):
                    return {"default_branch": "main"}
                if url.endswith("/commits/main"):
                    return {
                        "sha": NEW,
                        "html_url": "https://github.com/example/dependency/commit/new",
                        "commit": {"committer": {"date": "2026-09-30T00:00:00Z"}},
                    }
                raise AssertionError(url)

        embedded = {
            "coverage": {},
            "recipes": [
                {
                    "recipe_id": "recipe",
                    "inputs": [
                        {
                            "kind": "download-source",
                            "repository": "example/dependency",
                            "revision": OLD,
                        }
                    ],
                }
            ],
        }
        refresh.drift.observe_embedded_heads(embedded, FakeProvider())
        catalog = refresh.Catalog(
            Path(tempfile.gettempdir()),
            recipes={
                "recipe": {
                    "identity": {"slug": "recipe"},
                    "provenance": {
                        "attribution": ["Example"],
                        "source_reference": f"https://github.com/example/recipe/tree/{OLD}",
                    },
                    "release": {
                        "version": "1.0.0",
                        "released_at": "2026-01-01",
                    },
                    "models": [],
                    "execution": {"build": {"context": {"path": "missing"}}},
                }
            },
        )
        assessor = refresh.Assessor(catalog, object(), None, embedded)
        target = refresh.Target(
            "github",
            "example/recipe",
            NEW,
            "2026-09-30",
            "v2.0.0",
            "2.0.0",
            "https://github.com/example/recipe/commit/new",
        )
        assessor.target = lambda _provider, _repo: target
        with patch(
            "refresh_upstream.compare_commits",
            return_value=refresh.Comparison(
                "ahead", [], [], False, "https://github.com/example/recipe/compare"
            ),
        ):
            result = assessor.assess("recipe")

        self.assertEqual(
            embedded["recipes"][0]["inputs"][0]["candidate_status"], "advanced"
        )
        self.assertFalse(result.reasons)
        self.assertTrue(result.mechanical)
        self.assertIn(
            "review evidence, not a configured upgrade channel", result.evidence[0]
        )
        self.assertIn(NEW, result.edits["recipes/recipe.json"].decode())


class NeverDowngradeTests(unittest.TestCase):
    def collect(self, status: str):
        catalog = refresh.Catalog(
            Path(tempfile.gettempdir()),
            recipes={
                "recipe": {
                    "identity": {"slug": "recipe"},
                    "provenance": {
                        "attribution": ["Example"],
                        "source_reference": f"https://github.com/example/recipe/tree/{OLD}",
                    },
                    "release": {"version": "1.0.0", "released_at": "2026-01-01"},
                    "models": [],
                    "execution": {"build": {"context": {"path": "missing"}}},
                }
            },
        )
        target = refresh.Target(
            "github", "example/recipe", NEW, "2026-09-30", "v2.0.0", "2.0.0", "u"
        )
        with patch(
            "refresh_upstream.compare_commits",
            return_value=refresh.Comparison(status, [], [], False, "u"),
        ):
            return refresh.collect_items(
                catalog, "recipe", object(), lambda _p, _r: target
            )

    def test_pin_ahead_of_release_is_current(self) -> None:
        result = self.collect("behind")
        self.assertFalse(result.drifted)
        self.assertFalse(result.reasons)

    def test_release_ahead_of_pin_is_refreshed(self) -> None:
        result = self.collect("ahead")
        self.assertEqual(len(result.items), 1)
        self.assertEqual(result.items[0].comparison.status, "ahead")
        self.assertEqual(classify(result.items[0].comparison), [])

    def test_diverged_history_needs_review(self) -> None:
        result = self.collect("diverged")
        self.assertEqual(len(result.items), 1)
        self.assertTrue(classify(result.items[0].comparison))


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


class OpenPullRequestTests(unittest.TestCase):
    footprint = (
        {"recipes/a.json", "models/m-1.json"},
        ["adapters/llm/a/"],
    )

    def test_foreign_pr_touching_recipe_model_or_adapter_blocks(self) -> None:
        for name in ("recipes/a.json", "models/m-1.json", "adapters/llm/a/Dockerfile"):
            with self.subTest(name=name):
                self.assertEqual(
                    refresh.blocking_prs(self.footprint, {7: {name, "README.md"}}), [7]
                )

    def test_unrelated_pr_does_not_block(self) -> None:
        opens = {7: {"recipes/b.json", "adapters/llm/ab/Dockerfile", "README.md"}}
        self.assertEqual(refresh.blocking_prs(self.footprint, opens), [])

    def test_footprint_covers_recipe_models_and_context(self) -> None:
        catalog = refresh.Catalog(ROOT)
        catalog.recipes["a"] = {
            "models": [{"model": {"slug": "m-1"}}],
            "execution": {"build": {"context": {"path": "adapters/llm/a"}}},
        }
        self.assertEqual(refresh.recipe_footprint(catalog, "a"), self.footprint)


if __name__ == "__main__":
    unittest.main()
