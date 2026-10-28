from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import json
import os
import re
import sys
import tempfile
import types
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import patch

from tools._catalog_model import roles_for

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


def classify(
    cmp,
    patched: set[str] | frozenset[str] = frozenset(),
    problems=(),
    used_changes=None,
):
    return refresh.classify_changes(
        cmp,
        patched=set(patched),
        vendor_problems=list(problems),
        used_changes=used_changes,
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
            "id": refresh.file_id(p, "c" * 64),
            "path": p,
            "roles": roles_for(p),
            "sha256": "c" * 64,
            "size_bytes": 10,
        }
        for p in paths
    ]
    return document


class ModelTests(unittest.TestCase):
    def test_same_paths_with_new_weight_digests_is_compatible(self) -> None:
        old = model_with(("model.safetensors", "README.md"))
        self.assertEqual(
            refresh.model_shape_problems(
                old, files("model.safetensors", "README.md", "NOTES.md")
            ),
            [],
        )

    def test_changed_configuration_tokenizer_or_runtime_file_is_not(self) -> None:
        for path in ("config.json", "tokenizer.json", "dspark.py"):
            with self.subTest(path=path):
                old = model_with(("model.safetensors", path))
                self.assertTrue(
                    refresh.model_shape_problems(old, files("model.safetensors", path))
                )

    def test_removed_and_new_weight_files_are_not(self) -> None:
        old = model_with(("model.safetensors", "config.json"))
        self.assertTrue(refresh.model_shape_problems(old, files("model.safetensors")))
        self.assertTrue(
            refresh.model_shape_problems(
                old, files("model.safetensors", "config.json", "model-2.safetensors")
            )
        )

    def curated(self, old_paths, before, after):
        old = model_with(old_paths)
        item = refresh.Item(
            "model",
            "huggingface",
            "o/r",
            OLD,
            refresh.Target("huggingface", "o/r", NEW, "2026-09-30", None, None, "u"),
        )
        assessor = refresh.Assessor(refresh.Catalog(Path(".")), object(), None, {})
        with patch.object(refresh.Assessor, "_inventory", return_value=files(*before)):
            kept = assessor._curated(old, item, files(*after))
        return old, kept

    def test_files_the_document_left_out_are_not_new(self) -> None:
        # #620: LTX-2.3's Model catalogues one checkpoint of eight; a README-only
        # revision is not "six new weights files".
        repository = ("a.safetensors", "b.safetensors", "README.md")
        old, kept = self.curated(("a.safetensors",), repository, repository)
        self.assertEqual([f["path"] for f in kept], ["a.safetensors"])
        self.assertEqual(refresh.model_shape_problems(old, kept), [])

    def test_files_added_upstream_are_still_new(self) -> None:
        old, kept = self.curated(
            ("a.safetensors",),
            ("a.safetensors", "b.safetensors"),
            ("a.safetensors", "b.safetensors", "c.safetensors"),
        )
        self.assertEqual([f["path"] for f in kept], ["a.safetensors", "c.safetensors"])
        self.assertIn(
            "new weights file", " ".join(refresh.model_shape_problems(old, kept))
        )

    def test_a_recipe_is_moved_only_by_files_it_mounts(self) -> None:
        # #530: the EXL3 recipe mounts config and two shards of the official
        # checkpoint; a new chat template and reference code concern other recipes.
        old = model_with(("config.json", "model.safetensors", "inference/model.py"))
        recipe = {
            "models": [
                {
                    "model": {"slug": old["identity"]["slug"]},
                    "files": [{"file_id": old["files"][0]["id"]}],
                }
            ]
        }
        self.assertEqual(refresh.mounted_paths(recipe, old), {"config.json"})
        new = files("config.json", "model.safetensors", "inference/model.py")
        for entry in new:
            entry["sha256"] = "c" * 64 if entry["path"] == "config.json" else "e" * 64
        new += files("chat_template.jinja")
        self.assertTrue(refresh.mounted_unchanged(old, new, {"config.json"}))
        new[0]["sha256"] = "f" * 64
        self.assertFalse(refresh.mounted_unchanged(old, new, {"config.json"}))
        everything = [{"file_id": f["id"]} for f in old["files"]]
        recipe["models"][0]["files"] = everything
        self.assertIsNone(refresh.mounted_paths(recipe, old))

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

    def test_refreshed_model_follows_the_source_between_split_and_whole(self) -> None:
        old = model_with(("model.safetensors",))
        target = refresh.Target(
            "huggingface", "o/r", NEW, "2026-09-30", None, None, "u"
        )
        split = files("model.safetensors")
        split[0]["installed_bytes"] = 20
        split[0]["parts"] = [
            {
                "path": f"model.safetensors.part0{i}",
                "sha256": "b" * 64,
                "size_bytes": 10,
            }
            for i in range(2)
        ]
        moved = refresh.refreshed_model(old, split, target)
        self.assertEqual(len(moved["files"][0]["parts"]), 2)
        back = refresh.refreshed_model(moved, files("model.safetensors"), target)
        self.assertNotIn("parts", back["files"][0])

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
        # The recipe uses the whole repository and code changed, so the move is a
        # refresh (a move that changes nothing that runs is not one; see
        # NothingRunsTests and test_published_image_follow).
        with (
            patch(
                "refresh_upstream.compare_commits",
                return_value=refresh.Comparison(
                    "ahead",
                    [],
                    [{"filename": "src/model.py", "status": "modified"}],
                    False,
                    "https://github.com/example/recipe/compare",
                ),
            ),
            patch.object(refresh.Assessor, "_used_changes", return_value=None),
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


def single_recipe_catalog(source_reference: str) -> object:
    return refresh.Catalog(
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


class NothingRunsTests(unittest.TestCase):
    """A move that changes nothing a recipe runs is not a refresh (#1085, #848, #1282)."""

    TARGET = refresh.Target(
        "github", "example/recipe", NEW, "2026-09-30", None, None, "u"
    )

    def test_docs_benchmarks_tests_and_licences_do_not_run(self) -> None:
        for name in (
            "README.md",
            "VALIDATION_2026-10-08.md",
            "bench/requirements-sdk.txt",
            "bench/results/2026-10-08/run/result.json",
            "benchmarks/x.sh",
            "tests/test_a.py",
            "docs/guide.rst",
            "LICENSE",
            "NOTICE.txt",
            ".gitignore",
            "assets/chart.svg",
        ):
            with self.subTest(name=name):
                self.assertFalse(refresh.runs(name))
        for name in ("src/model.py", "requirements.txt", "run.sh", "Makefile", "a.c"):
            with self.subTest(name=name):
                self.assertTrue(refresh.runs(name))

    def test_only_complete_docs_comparisons_change_nothing(self) -> None:
        docs = comparison(files=("README.md", "bench/results/a.json", "LICENSE"))
        self.assertTrue(refresh.nothing_runs(docs, frozenset()))
        self.assertFalse(refresh.nothing_runs(docs, {"README.md"}))  # adapter copies it
        self.assertFalse(
            refresh.nothing_runs(comparison(files=("README.md", "src/a.py")), set())
        )
        self.assertFalse(
            refresh.nothing_runs(
                comparison(files=("README.md",), truncated=True), set()
            )
        )

    def assess(self, comparison_, *, used=None, report=None):
        catalog = single_recipe_catalog(f"https://github.com/example/recipe/tree/{OLD}")
        assessor = refresh.Assessor(catalog, object(), None, {})
        assessor.target = lambda _provider, _repo: self.TARGET
        patches = [
            patch("refresh_upstream.compare_commits", return_value=comparison_),
            patch.object(refresh.Assessor, "_used_changes", return_value=used),
        ]
        if report is not None:
            pin = refresh.AdapterPin("example/recipe", OLD, Path("missing"))
            patches += [
                patch("refresh_upstream.adapter_pin", return_value=pin),
                patch("refresh_upstream.patch_targets", return_value=set()),
                patch.object(
                    refresh.vendored,
                    "fetch_tree",
                    return_value=refresh.vendored.Tree({"README.md": "b" * 40}, True),
                ),
                patch.object(refresh.vendored, "check_adapter", return_value=report),
                patch.object(refresh.Assessor, "_retarget_adapter"),
                patch.object(refresh.Assessor, "_follow_vendored", return_value=None),
            ]
        for active in patches:
            active.start()
            self.addCleanup(active.stop)
        return assessor.assess("recipe")

    def test_whole_repository_move_of_docs_and_benchmarks_is_current(self) -> None:
        result = self.assess(
            comparison(files=("README.md", "bench/results/a.json", "AGENTS.md"))
        )
        self.assertFalse(result.drifted)
        self.assertFalse(result.reasons)

    def test_whole_repository_code_move_is_still_a_refresh(self) -> None:
        result = self.assess(comparison(files=("README.md", "src/engine.py")))
        self.assertTrue(result.mechanical)

    def test_stale_vendored_readme_copy_alone_is_current(self) -> None:
        message = "adapters/x: upstream-README.md: differs from upstream README.md at f"
        report = refresh.vendored.Report("adapters/x")
        report.problems.append(message)
        report.stale.append((message, "upstream-README.md", "README.md"))
        result = self.assess(comparison(files=("README.md",)), used=[], report=report)
        self.assertFalse(result.drifted)

    def test_huggingface_source_reference_alone_is_current(self) -> None:
        catalog = single_recipe_catalog(
            f"https://huggingface.co/example/model/tree/{OLD}"
        )
        assessor = refresh.Assessor(catalog, object(), None, {})
        assessor.target = lambda _provider, _repo: refresh.Target(
            "huggingface", "example/model", NEW, "2026-09-30", None, None, "u"
        )
        with patch("refresh_upstream.follow_huggingface_branch", return_value=None):
            result = assessor.assess("recipe")
        self.assertFalse(result.drifted)
        self.assertFalse(result.edits)


class MergeGateTests(unittest.TestCase):
    main_head = "a" * 40

    def test_unchanged_prepared_pr_is_armed_after_publication_recovers(self) -> None:
        assessment = refresh.Assessment("recipe", edits={"recipe.json": b"new pin\n"})
        merges: list[tuple[str, ...]] = []
        published = False
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.dict(
                os.environ,
                {
                    "GIT_CONFIG_GLOBAL": os.devnull,
                    "GIT_CONFIG_NOSYSTEM": "1",
                    "GIT_CONFIG_COUNT": "0",
                    "GIT_AUTHOR_NAME": "Refresh test",
                    "GIT_AUTHOR_EMAIL": "refresh@example.test",
                    "GIT_COMMITTER_NAME": "Refresh test",
                    "GIT_COMMITTER_EMAIL": "refresh@example.test",
                },
            ),
            patch("refresh_upstream.verify", return_value=None),
        ):
            root = Path(temporary)
            refresh.git("init", "-q", "--initial-branch=main", cwd=root)
            refresh.git("remote", "add", "origin", str(root), cwd=root)
            (root / "recipe.json").write_bytes(b"old pin\n")
            refresh.git("add", "recipe.json", cwd=root)
            refresh.git("commit", "-qm", "Base", cwd=root)
            main = refresh.git("rev-parse", "HEAD", cwd=root)
            refresh.git("update-ref", "refs/remotes/origin/main", main, cwd=root)
            refresh.git("switch", "-qc", "refresh/recipe", cwd=root)
            refresh.apply_edits(root, assessment.edits)
            refresh.git("commit", "-qam", "Prepared refresh", cwd=root)
            prepared = refresh.git("rev-parse", "HEAD", cwd=root)

            def github(*arguments: str) -> str:
                if arguments[:3] == ("pr", "list", "--head"):
                    return json.dumps([{"number": 421, "state": "OPEN", "body": ""}])
                if arguments[:2] == ("pr", "list"):
                    return "[]"
                if arguments[0] == "api":
                    return main
                if arguments[:2] == ("run", "list"):
                    return json.dumps(
                        [
                            {
                                "headSha": main,
                                "status": "completed",
                                "conclusion": "success",
                            }
                        ]
                        if published
                        else []
                    )
                if arguments[:2] == ("pr", "merge"):
                    merges.append(arguments)
                    return ""
                raise AssertionError(f"Unexpected GitHub mutation: {arguments}")

            with patch("refresh_upstream.gh", side_effect=github):
                waiting = refresh.Summary()
                refresh.publish_mechanical(root, assessment, waiting)
                self.assertEqual(waiting.armed, [])
                self.assertEqual(merges, [])
                published = True
                resumed = refresh.Summary()
                refresh.publish_mechanical(root, assessment, resumed)

            self.assertEqual(resumed.armed, [421])
            self.assertEqual(merges, [("pr", "merge", "421", "--auto", "--squash")])
            self.assertEqual(
                refresh.git("rev-parse", "refs/heads/refresh/recipe", cwd=root),
                prepared,
            )

    def test_existing_armed_pr_does_not_block_a_disjoint_mechanical_merge(self) -> None:
        summary = refresh.Summary()
        summary.healthy = True
        pulls = [
            {
                "number": 419,
                "autoMergeRequest": {"enabledAt": "now"},
                "files": [{"path": "recipes/a.json"}],
            },
            {"number": 421, "files": [{"path": "recipes/b.json"}]},
        ]
        with (
            patch("refresh_upstream.gh_json", return_value=pulls),
            patch("refresh_upstream.gh") as gh,
        ):
            allowed = refresh.arm_mechanical_pr(421, "recipe-b", summary)

        self.assertTrue(allowed)
        self.assertEqual(summary.armed, [421])
        gh.assert_called_once_with("pr", "merge", "421", "--auto", "--squash")

    def test_shared_model_file_defers_the_second_pr(self) -> None:
        summary = refresh.Summary()
        summary.healthy = True
        pulls = [
            {
                "number": 419,
                "autoMergeRequest": {"enabledAt": "now"},
                "files": [{"path": "models/shared.json"}],
            },
            {
                "number": 421,
                "files": [{"path": "recipes/b.json"}, {"path": "models/shared.json"}],
            },
        ]
        with (
            patch("refresh_upstream.gh_json", return_value=pulls),
            patch("refresh_upstream.gh") as gh,
        ):
            allowed = refresh.arm_mechanical_pr(421, "recipe-b", summary)

        self.assertFalse(allowed)
        self.assertEqual(summary.armed, [])
        self.assertIn("shares files with armed PR #419", summary.deferred[0])
        gh.assert_not_called()

    def test_file_overlap_counts_prs_armed_earlier_in_the_run(self) -> None:
        pulls = [
            {"number": 1, "files": [{"path": "models/m.json"}]},
            {"number": 2, "files": [{"path": "models/m.json"}]},
            {"number": 3, "files": [{"path": "models/m.json"}]},
        ]
        self.assertEqual(refresh.file_overlaps(2, pulls, armed=[1]), [1])
        self.assertEqual(refresh.file_overlaps(2, pulls), [])

    def test_pending_publication_keeps_prepared_pr_unarmed(self) -> None:
        summary = refresh.Summary()
        with (
            patch("refresh_upstream.gh_json", side_effect=[[]]),
            patch("refresh_upstream.gh", return_value=self.main_head) as gh,
        ):
            allowed = refresh.arm_mechanical_pr(421, "recipe", summary)

        self.assertFalse(allowed)
        self.assertEqual(summary.armed, [])
        self.assertIn("did not succeed", summary.deferred[0])
        self.assertFalse(any("merge" in call.args for call in gh.call_args_list))

    def test_successful_publication_arms_every_green_pr_in_one_run(self) -> None:
        summary = refresh.Summary()
        runs = [
            {
                "headSha": self.main_head,
                "status": "completed",
                "conclusion": "success",
            }
        ]
        pulls = [
            {"number": n, "files": [{"path": f"recipes/r{n}.json"}]}
            for n in (421, 422, 423)
        ]
        with (
            patch("refresh_upstream.gh_json", side_effect=[runs, pulls, pulls, pulls]),
            patch("refresh_upstream.gh", return_value=self.main_head) as gh,
        ):
            results = [
                refresh.arm_mechanical_pr(n, f"recipe-{n}", summary)
                for n in (421, 422, 423)
            ]

        self.assertEqual(results, [True, True, True])
        self.assertEqual(summary.armed, [421, 422, 423])
        merges = [call for call in gh.call_args_list if "merge" in call.args]
        self.assertEqual(len(merges), 3)

    def test_inflight_publication_does_not_block_but_failure_does(self) -> None:
        done = {"status": "completed", "conclusion": "success"}
        inflight = {"status": "in_progress", "conclusion": ""}
        cancelled = {"status": "completed", "conclusion": "cancelled"}
        failed = {"status": "completed", "conclusion": "failure"}
        self.assertTrue(refresh.publication_healthy([inflight, cancelled, done]))
        self.assertFalse(refresh.publication_healthy([inflight, failed, done]))
        self.assertFalse(refresh.publication_healthy([]))

    def test_open_refresh_prs_are_armed_up_front(self) -> None:
        summary = refresh.Summary()
        summary.healthy = True
        listing = [
            {"number": 1, "headRefName": "refresh/a", "autoMergeRequest": None},
            {"number": 2, "headRefName": "refresh/b", "autoMergeRequest": {"x": 1}},
            {"number": 3, "headRefName": "codex/c", "autoMergeRequest": None},
        ]
        calls = []

        def gh_json(*arguments: str):
            calls.append(arguments)
            if "autoMergeRequest,files" in arguments[-1]:
                return [{"number": 1, "files": []}]
            return listing

        with (
            patch("refresh_upstream.gh_json", side_effect=gh_json),
            patch("refresh_upstream.gh") as gh,
        ):
            refresh.arm_open_mechanical_prs(summary)

        self.assertEqual(summary.armed, [1])
        gh.assert_called_once_with("pr", "merge", "1", "--auto", "--squash")

    def test_prepare_bound_counts_pushed_prs_only(self) -> None:
        self.assertEqual(refresh.MAX_MECHANICAL_PER_RUN, 6)
        summary = refresh.Summary()
        self.assertEqual(summary.prepared, 0)

    def test_chain_guard_detects_a_burst_not_a_single_recent_run(self) -> None:
        now = datetime(2026, 10, 2, 12, 0, tzinfo=UTC)

        def run(minutes_ago: int, **extra: object) -> dict[str, object]:
            when = now - timedelta(minutes=minutes_ago)
            return {
                "databaseId": 7,
                "status": "completed",
                "conclusion": "success",
                "updatedAt": when.isoformat(),
            } | extra

        burst = [run(m) for m in (1, 5, 9, 14)]
        self.assertTrue(refresh.refresh_is_spinning(burst, 99, now))
        # One run that just finished (the one ahead in the queue) does not skip us.
        self.assertFalse(refresh.refresh_is_spinning([run(1)], 99, now))
        self.assertFalse(
            refresh.refresh_is_spinning([run(m) for m in (40, 50, 60, 70)], 99, now)
        )
        # Active, cancelled and the current run never count.
        mixed = [
            run(1, status="in_progress"),
            run(2, conclusion="cancelled"),
            run(3, databaseId=99),
            run(4),
        ]
        self.assertFalse(refresh.refresh_is_spinning(mixed + burst[:2], 99, now))

    def test_chain_guard_command(self) -> None:
        def decide(event: str, conclusion: str, runs: list[dict[str, object]]) -> str:
            with tempfile.NamedTemporaryFile("w+", suffix=".out") as output:
                arguments = refresh.argparse.Namespace(
                    event=event, run_id=99, upstream_conclusion=conclusion
                )
                with (
                    patch("refresh_upstream.gh_json", return_value=runs),
                    patch.dict(os.environ, {"GITHUB_OUTPUT": output.name}),
                ):
                    refresh.command_guard(arguments)
                return Path(output.name).read_text().strip()

        recent = [
            {
                "databaseId": 7 + n,
                "status": "completed",
                "conclusion": "success",
                "updatedAt": datetime.now(UTC).isoformat(),
            }
            for n in range(4)
        ]
        self.assertEqual(decide("schedule", "", recent), "run=true")
        self.assertEqual(decide("workflow_dispatch", "", recent), "run=true")
        self.assertEqual(decide("workflow_run", "success", []), "run=true")
        self.assertEqual(decide("workflow_run", "failure", []), "run=false")
        self.assertEqual(decide("workflow_run", "success", recent), "run=false")


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
        with patch("refresh_upstream.pin_is_newer_by_date", return_value=False):
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


class ReadmeAndScopeTests(unittest.TestCase):
    def test_readme_is_not_a_blocking_file(self) -> None:
        self.assertFalse(refresh.is_blocking("README.md"))
        self.assertFalse(refresh.is_blocking("models/readme.rst"))
        self.assertTrue(refresh.is_blocking("run.sh"))

    def test_ui_mate_unrelated_upstream_changes_are_mechanical(self) -> None:
        # #491: 3 commits, 300+ files (truncated), none of them used by the recipe.
        cmp = refresh.Comparison(
            "ahead",
            [],
            [
                {"filename": "README.md", "status": "modified"},
                {
                    "filename": "osworker_bench/CUA-Gym-Hub/deploy-all.sh",
                    "status": "added",
                },
            ],
            True,
            "u",
        )
        self.assertEqual(classify(cmp, used_changes=[]), [])
        # Without a verified used set the same change is judged file by file.
        self.assertTrue(classify(cmp))

    def test_used_file_change_is_a_reason(self) -> None:
        self.assertTrue(
            classify(comparison(), used_changes=["agents/ui_mate_agent.py"])
        )

    def test_diverged_is_not_a_reason_once_used_files_are_verified(self) -> None:
        self.assertEqual(classify(comparison("diverged"), used_changes=[]), [])
        self.assertTrue(classify(comparison("diverged")))
        self.assertTrue(
            classify(comparison("diverged"), used_changes=[], problems=["x differs"])
        )

    def test_licence_removal_is_informational(self) -> None:
        cmp = refresh.Comparison(
            "ahead", [], [{"filename": "LICENSE", "status": "removed"}], False, "u"
        )
        item = refresh.Item(
            "source",
            "github",
            "o/r",
            OLD,
            refresh.Target("github", "o/r", NEW, "2026-09-30", None, None, "u"),
        )
        self.assertEqual(classify(cmp, used_changes=[]), [])
        self.assertIn("never gate", refresh.github_evidence(item, cmp, set()))


class AdapterUsageTests(unittest.TestCase):
    def usage(self, files: dict[str, str]):
        with tempfile.TemporaryDirectory() as tmp:
            for name, text in files.items():
                (Path(tmp) / name).write_text(text)
            return refresh.adapter_usage(Path(tmp), "o/Repo", OLD)

    def test_wrapper_without_references_uses_nothing(self) -> None:
        self.assertEqual(
            self.usage({"Dockerfile": "FROM x\nCOPY a /a\n"}), (set(), False)
        )

    def test_explicit_urls_name_files(self) -> None:
        text = (
            "curl https://raw.githubusercontent.com/o/repo/main/cfg/a.yaml\n"
            "# https://github.com/o/Repo/blob/v1/docs/b.md\n"
        )
        self.assertEqual(
            self.usage({"setup.sh": text}), ({"cfg/a.yaml", "docs/b.md"}, False)
        )

    def test_clone_or_commit_named_archive_uses_the_whole_repository(self) -> None:
        clone = "RUN git clone https://github.com/o/Repo /src\n"
        self.assertTrue(self.usage({"Dockerfile": clone})[1])
        self.assertTrue(self.usage({f"tensorfold-{OLD}.tar.gz": ""})[1])
        codeload = f"ADD https://codeload.github.com/o/Repo/tar.gz/{OLD} /src.tgz\n"
        self.assertTrue(self.usage({"Dockerfile": codeload})[1])


class DivergedReleaseTests(unittest.TestCase):
    class FakeHttp:
        def __init__(self, pin: str, release: str) -> None:
            self.dates = {OLD: pin, NEW: release}

        def github(self, path: str):
            return {
                "commit": {"committer": {"date": self.dates[path.rsplit("/", 1)[1]]}}
            }

    def test_pin_on_main_newer_than_release_branch_commit(self) -> None:
        # #485: pin b9ce4b69 on main (1.3.0rc13) against release 1.2.1 on a release branch.
        http = self.FakeHttp("2026-09-01T00:00:00Z", "2026-04-20T00:00:00Z")
        self.assertTrue(refresh.pin_is_newer_by_date(http, "o/r", OLD, NEW))

    def test_release_newer_than_pin_is_not_current(self) -> None:
        http = self.FakeHttp("2026-04-01T00:00:00Z", "2026-04-20T00:00:00Z")
        self.assertFalse(refresh.pin_is_newer_by_date(http, "o/r", OLD, NEW))

    def test_unreadable_dates_are_not_current(self) -> None:
        self.assertFalse(refresh.pin_is_newer_by_date(object(), "o/r", OLD, NEW))

    def test_image_wrapper_diverged_release_behind_pin_is_current(self) -> None:
        collector = NeverDowngradeTests()
        with patch("refresh_upstream.pin_is_newer_by_date", return_value=True):
            self.assertFalse(collector.collect("diverged").drifted)
        with patch("refresh_upstream.pin_is_newer_by_date", return_value=False):
            self.assertEqual(len(collector.collect("diverged").items), 1)


class ModelScopeTests(unittest.TestCase):
    def old_model(self, paths: tuple[str, ...]):
        document = model_with(paths)
        for entry in document["files"]:  # ids as older catalogs wrote them
            entry["id"] = f"{entry['path'].split('.')[0].lower()}-{'c' * 12}"
        return document

    def test_old_format_ids_do_not_block(self) -> None:
        # #476: only README.md changed; every id predates the path-digest suffix.
        old = self.old_model(
            (".gitattributes", "README.md", "config.json", "model.safetensors")
        )
        new = files(".gitattributes", "config.json", "model.safetensors", "README.md")
        for entry in new:
            entry["sha256"] = "c" * 64 if entry["path"] != "README.md" else "d" * 64
        self.assertEqual(refresh.model_shape_problems(old, new), [])

    def test_same_fixture_with_config_digest_change_needs_review(self) -> None:
        old = self.old_model(("README.md", "config.json", "model.safetensors"))
        new = files("README.md", "config.json", "model.safetensors")
        for entry in new:
            entry["sha256"] = "d" * 64 if entry["path"] == "config.json" else "c" * 64
        problems = refresh.model_shape_problems(old, new)
        self.assertEqual(len(problems), 1)
        self.assertIn("config.json", problems[0])

    def test_removed_licence_and_readme_are_informational(self) -> None:
        old = self.old_model(("LICENSE", "README.md", "model.safetensors"))
        new = files("model.safetensors")
        for entry in new:
            entry["sha256"] = "c" * 64
        self.assertEqual(refresh.model_shape_problems(old, new), [])
        target = refresh.Target(
            "huggingface", "o/r", NEW, "2026-09-30", None, None, "u"
        )
        refreshed = refresh.refreshed_model(old, new, target)
        self.assertEqual([f["path"] for f in refreshed["files"]], ["model.safetensors"])
        item = refresh.Item("model", "huggingface", "o/r", OLD, target)
        self.assertIn("never gate", refresh.model_evidence(item, old, new))

    def test_restructured_repository_needs_review(self) -> None:
        # #473: turboderp moved its files; config.json and shards are gone, new weights.
        old = self.old_model(
            ("config.json", "model-00001-of-00002.safetensors", "tokenizer.json")
        )
        new = files("cal_trace.safetensors", "measurement.json")
        joined = " ".join(refresh.model_shape_problems(old, new))
        self.assertIn("config.json", joined)
        self.assertIn("model-00001-of-00002.safetensors", joined)
        self.assertIn("new weights file", joined)

    def test_tensorfold_whole_repo_adapter_keeps_judging_every_file(self) -> None:
        cmp = comparison(files=("src/tensorfold/cuda/server.py", "pyproject.toml"))
        reasons = classify(cmp)  # used_changes=None: the adapter uses the whole repo
        self.assertIn("server.py", " ".join(reasons))

    def test_file_id_matches_a_fresh_catalog(self) -> None:
        tree = [
            {
                "type": "file",
                "path": "sub/model-1.safetensors",
                "size": 5,
                "lfs": {"oid": "e" * 64},
            }
        ]
        (entry,) = refresh.catalog.inventory("o/r", NEW, tree)
        self.assertEqual(entry["id"], refresh.file_id(entry["path"], "e" * 64))


OLD_YAML = """defaults:
  max_num_batched_tokens: 8192
env:
  VLLM_MARLIN_USE_ATOMIC_ADD: 1
command: |
  vllm serve org/model \\
    --trust-remote-code \\
    --max-num-batched-tokens {max_num_batched_tokens}
"""


class VendoredCreatorMoveTests(unittest.TestCase):
    """A vendored file the recipe's own creator changed follows when nothing it
    references in it is gone (the 15 eugr recipes of refresh run 38469862398)."""

    def assess(self, new_yaml: str, *, dockerfile: str = "", ours=(), extra=None):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        adapter = root / "adapters/x"
        (adapter / "recipes").mkdir(parents=True)
        (adapter / "recipes/r.yaml").write_text(OLD_YAML)
        (adapter / "vllm-wrapper.py").write_text("print('wrapper')\n")
        (adapter / "Dockerfile").write_text(
            "FROM example/engine@sha256:" + "a" * 64 + "\nENTRYPOINT []\n" + dockerfile
        )
        for name, text in (extra or {}).items():
            (adapter / name).write_text(text)
        (adapter / "vendored-upstream.json").write_text(
            json.dumps(
                {
                    "ours": ["vllm-wrapper.py", *ours],
                    "source": {"repo": "example/recipe", "commit": OLD},
                }
            )
        )
        recipe = {
            "identity": {"slug": "recipe"},
            "provenance": {
                "attribution": ["Example"],
                "source_reference": f"https://github.com/example/recipe/tree/{OLD}",
            },
            "release": {"version": "1.0.0", "released_at": "2026-01-01"},
            "models": [],
            "execution": {"build": {"context": {"path": "adapters/x"}}},
            "runtime": {
                "arguments": [
                    {"name": "trust-remote-code", "setting": None, "value": True},
                    {"name": "served-model-name", "setting": None, "value": "m"},
                ],
                "environment": [{"name": "VLLM_MARLIN_USE_ATOMIC_ADD", "value": "1"}],
                "entrypoint": ["/opt/vonk/bin/vllm", "serve", "/models"],
            },
        }
        catalog = refresh.Catalog(root, recipes={"recipe": recipe})
        assessor = refresh.Assessor(catalog, object(), None, {})
        assessor.target = lambda _provider, _repo: refresh.Target(
            "github", "example/recipe", NEW, "2026-10-13", None, None, "u"
        )
        data = new_yaml.encode()
        tree = refresh.vendored.Tree(
            {"recipes/r.yaml": refresh.vendored.git_blob_sha(data)}, True
        )
        for active in (
            patch(
                "refresh_upstream.compare_commits",
                return_value=comparison(files=("recipes/r.yaml",)),
            ),
            patch.object(refresh.vendored, "fetch_tree", return_value=tree),
            patch("refresh_upstream.upstream_blob", return_value=data),
        ):
            active.start()
            self.addCleanup(active.stop)
        return assessor.assess("recipe")

    def test_verbatim_file_with_its_flags_intact_is_mechanical(self) -> None:
        new_yaml = OLD_YAML.replace("8192", "4096")
        result = self.assess(new_yaml)
        self.assertTrue(result.mechanical, result.reasons)
        self.assertEqual(result.edits["adapters/x/recipes/r.yaml"], new_yaml.encode())
        self.assertIn(NEW, result.edits["adapters/x/vendored-upstream.json"].decode())
        body = refresh.pr_body(result)
        self.assertIn("-  max_num_batched_tokens: 8192", body)
        self.assertIn("+  max_num_batched_tokens: 4096", body)
        for checked in ("`--trust-remote-code`", "`VLLM_MARLIN_USE_ATOMIC_ADD`"):
            self.assertIn(checked, body)
        # A flag the creator's file never stated is not checked against it.
        self.assertNotIn("`--served-model-name`", body)

    def test_a_patched_or_edited_file_stays_a_review(self) -> None:
        new_yaml = OLD_YAML.replace("8192", "4096")
        cases = {
            "sed": {"dockerfile": "RUN sed -i 's/a/b/' \\\n    /app/recipes/r.yaml\n"},
            "patch": {
                "ours": ("local.patch",),
                "extra": {
                    "local.patch": "--- a/recipes/r.yaml\n+++ b/recipes/r.yaml\n"
                },
            },
        }
        for name, case in cases.items():
            with self.subTest(name=name):
                result = self.assess(new_yaml, **case)
                self.assertFalse(result.mechanical)
                self.assertIn("not vendored verbatim", " ".join(result.reasons))
                self.assertNotIn("adapters/x/recipes/r.yaml", result.edits)

    def test_running_the_file_is_not_editing_it(self) -> None:
        result = self.assess(
            OLD_YAML.replace("8192", "4096"),
            dockerfile="RUN cd /app && bash ./recipes/r.yaml 2>/dev/null\n",
        )
        self.assertTrue(result.mechanical, result.reasons)

    def test_a_referenced_flag_gone_upstream_stays_a_review_naming_it(self) -> None:
        result = self.assess(OLD_YAML.replace("    --trust-remote-code \\\n", ""))
        self.assertFalse(result.mechanical)
        reasons = " ".join(result.reasons)
        self.assertIn("`--trust-remote-code`", reasons)
        self.assertIn("no longer has", reasons)


class ModelReExportTests(unittest.TestCase):
    """A re-export of the same Model repository is its newest revision (#566)."""

    def shards(self, count: int) -> tuple[str, ...]:
        return tuple(
            f"model-{i:05d}-of-{count:05d}.safetensors" for i in range(1, count + 1)
        )

    def move(self, mounted: tuple[str, ...] | None):
        old = model_with(("config.json", "tokenizer.json", *self.shards(36)))
        old["source"]["repository"] = "https://huggingface.co/o/r"
        new = files("config.json", "HYBRID.json", *self.shards(41), sha="e" * 64)
        new += files("tokenizer.json")
        selected = [f for f in old["files"] if mounted is None or f["path"] in mounted]
        mount = {"target": "/models"}
        recipe = {
            "identity": {"slug": "recipe"},
            "models": [
                {
                    "id": "primary",
                    "model": {"slug": old["identity"]["slug"], "content_sha256": "x"},
                    "files": [
                        {
                            "id": "primary-" + f["id"][:40],
                            "file_id": f["id"],
                            "mount": mount,
                            "roles": ["entrypoint"],
                        }
                        for f in selected
                    ],
                }
            ],
            "topology": {
                "roles": [{"resources": {"disk": {"artifact_bytes": 1_000_000}}}]
            },
        }
        catalog = refresh.Catalog(
            Path("."),
            recipes={"recipe": recipe},
            models={old["identity"]["slug"]: old},
        )
        assessor = refresh.Assessor(catalog, object(), None, {})
        item = refresh.Item(
            "model",
            "huggingface",
            "o/r",
            OLD,
            refresh.Target("huggingface", "o/r", NEW, "2026-10-06", None, None, "u"),
            old["identity"]["slug"],
        )
        inventories = {OLD: files(*(f["path"] for f in old["files"])), NEW: new}
        result = refresh.Assessment("recipe", items=[item])
        new_recipe = json.loads(json.dumps(recipe))
        edits: dict[str, bytes | None] = {}
        with patch.object(
            refresh.Assessor,
            "_inventory",
            side_effect=lambda _repo, revision: inventories[revision],
        ):
            assessor._model(item, recipe, new_recipe, result, edits)
        return result, new_recipe, edits

    def test_resharded_re_export_moves_a_recipe_mounting_everything(self) -> None:
        result, new_recipe, edits = self.move(None)
        self.assertEqual(result.reasons, [])
        self.assertFalse(result.items[0].noop)
        written = edits[f"models/demo-{NEW[:8]}.json"]
        assert written is not None
        moved = json.loads(written)
        paths = {f["path"] for f in moved["files"]}
        self.assertIn("HYBRID.json", paths)
        self.assertIn("model-00041-of-00041.safetensors", paths)
        self.assertNotIn("model-00036-of-00036.safetensors", paths)
        mounted = {s["file_id"] for s in new_recipe["models"][0]["files"]}
        self.assertEqual(mounted, {f["id"] for f in moved["files"]})

    def test_a_named_mounted_file_gone_stays_a_review_naming_it(self) -> None:
        result, _, edits = self.move(
            ("config.json", "model-00036-of-00036.safetensors")
        )
        self.assertIn(
            "file this recipe mounts is gone upstream: "
            "`model-00036-of-00036.safetensors`",
            result.reasons[0],
        )
        self.assertFalse(edits)

    def test_a_kit_declared_revision_keeps_its_file_set(self) -> None:
        old = model_with(("config.json", *self.shards(2)))
        new = files("config.json", *self.shards(3))
        self.assertTrue(refresh.model_problems(old, new, None, declared=True))
        self.assertEqual(refresh.model_problems(old, new, None, declared=False), [])


class ArchiveLabelTests(unittest.TestCase):
    OLD_DIGEST = "1" * 64
    NEW_DIGEST = "2" * 64
    OTHER = "3" * 64

    def item(self):
        target = refresh.Target("github", "o/r", NEW, "2026-09-30", None, None, "u")
        return refresh.Item("source", "github", "o/r", OLD, target)

    def digest(self, _repo: str, commit: str) -> str:
        return self.OLD_DIGEST if commit == OLD else self.NEW_DIGEST

    def test_archive_label_of_the_pin_moves_with_it(self) -> None:
        text = (
            f'LABEL a.archive-sha256="{self.OLD_DIGEST}" \\\n'
            f'      b.mia-source-archive-sha256="{self.OTHER}"\n'
        )
        with patch("refresh_upstream.archive_sha256", self.digest):
            moved, changed = refresh.retarget_archive_labels(text, self.item())
        self.assertTrue(changed)
        self.assertIn(f'a.archive-sha256="{self.NEW_DIGEST}"', moved)
        self.assertIn(f'mia-source-archive-sha256="{self.OTHER}"', moved)

    def test_the_download_check_moves_with_the_label(self) -> None:
        # The LTX adapters state the digest twice; moving only the label left the
        # sha256sum check on the old archive and the build would fail.
        text = (
            f'LABEL a.archive-sha256="{self.OLD_DIGEST}"\n'
            f"RUN curl -o /tmp/s.tar.gz https://codeload.github.com/o/r/tar.gz/{OLD} \\\n"
            f"    && printf '%s  %s\\n' {self.OLD_DIGEST} /tmp/s.tar.gz | sha256sum -c\n"
            f"    && printf '%s  %s\\n' {self.OTHER} /tmp/wheel.whl | sha256sum -c\n"
        )
        with patch("refresh_upstream.archive_sha256", self.digest):
            moved, changed = refresh.retarget_archive_labels(text, self.item())
        self.assertTrue(changed)
        self.assertNotIn(self.OLD_DIGEST, moved)
        self.assertEqual(moved.count(self.NEW_DIGEST), 2)
        self.assertIn(self.OTHER, moved)

    def test_unrelated_archive_labels_are_left_alone(self) -> None:
        text = f'LABEL b.archive-sha256="{self.OTHER}"\n'
        with patch("refresh_upstream.archive_sha256", self.digest):
            self.assertEqual(
                refresh.retarget_archive_labels(text, self.item()), (text, False)
            )

    def test_fetch_failure_is_a_review_reason_not_a_stale_label(self) -> None:
        def broken(_repo: str, _commit: str) -> str:
            raise RuntimeError("offline")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / "adapter"
            directory.mkdir()
            (directory / "Dockerfile").write_text(
                f'LABEL x.archive-sha256="{self.OLD_DIGEST}" y="{OLD}"\n'
            )
            assessor = refresh.Assessor(
                refresh.Catalog(root), object(), None, {"recipes": []}
            )
            pin = refresh.AdapterPin("o/r", OLD, directory)
            result = refresh.Assessment("recipe")
            edits: dict = {}
            with patch("refresh_upstream.archive_sha256", broken):
                assessor._retarget_adapter(pin, self.item(), edits, result)
            self.assertTrue(result.reasons)
            with patch("refresh_upstream.archive_sha256", self.digest):
                result = refresh.Assessment("recipe")
                assessor._retarget_adapter(pin, self.item(), edits, result)
            new = edits["adapter/Dockerfile"].decode()
            self.assertIn(self.NEW_DIGEST, new)
            self.assertIn(NEW, new)
            self.assertFalse(result.reasons)


class EnvelopeTests(unittest.TestCase):
    def retarget(
        self, artifact_bytes: int, old_size: int, new_size: int, prefix: str = "sel-"
    ):
        old = model_with(("model.safetensors",))
        selection = prefix + old["files"][0]["id"]
        old["files"][0]["size_bytes"] = old_size
        fresh = json.loads(json.dumps(old))
        fresh["identity"]["slug"] = "demo-" + NEW[:8]
        fresh["files"][0]["sha256"] = "d" * 64
        fresh["files"][0]["size_bytes"] = new_size
        fresh["files"][0]["id"] = refresh.file_id("model.safetensors", "d" * 64)
        recipe = {
            "models": [
                {
                    "id": "primary",
                    "model": {"slug": old["identity"]["slug"], "content_sha256": "x"},
                    "files": [
                        {
                            "id": selection,
                            "file_id": old["files"][0]["id"],
                        }
                    ],
                }
            ],
            "topology": {
                "roles": [{"resources": {"disk": {"artifact_bytes": artifact_bytes}}}]
            },
        }
        result = refresh.Assessment("recipe")
        assessor = refresh.Assessor(
            refresh.Catalog(Path(".")), object(), None, {"recipes": []}
        )
        assessor._retarget_model(recipe, recipe, old, fresh, result)
        self.selected = recipe["models"][0]["files"][0]
        return recipe["topology"]["roles"][0]["resources"]["disk"], result

    def test_a_recipe_mounting_every_file_mounts_the_new_ones(self) -> None:
        # #531: the official checkpoint gained chat_template.jinja; an SGLang recipe
        # that mounts the whole Model must mount it too, and a dropped README leaves.
        old = model_with(("model.safetensors", "README.md"))
        fresh = json.loads(json.dumps(old))
        fresh["identity"]["slug"] = "demo-" + NEW[:8]
        fresh["files"] = [
            dict(old["files"][0]),
            {
                "id": refresh.file_id("chat_template.jinja", "e" * 64),
                "path": "chat_template.jinja",
                "roles": ["tokenizer"],
                "sha256": "e" * 64,
                "size_bytes": 7,
            },
        ]
        mount = {"target": "/models"}
        recipe = {
            "models": [
                {
                    "id": "primary",
                    "model": {"slug": old["identity"]["slug"], "content_sha256": "x"},
                    "files": [
                        {
                            "id": "primary-" + f["id"],
                            "file_id": f["id"],
                            "mount": mount,
                            "roles": ["entrypoint"],
                        }
                        for f in old["files"]
                    ],
                }
            ],
            "topology": {
                "roles": [{"resources": {"disk": {"artifact_bytes": 1_000_000}}}]
            },
        }
        result = refresh.Assessment("recipe")
        assessor = refresh.Assessor(
            refresh.Catalog(Path(".")), object(), None, {"recipes": []}
        )
        assessor._retarget_model(recipe, recipe, old, fresh, result)
        self.assertFalse(result.reasons)
        selected = recipe["models"][0]["files"]
        self.assertEqual(
            sorted(s["file_id"] for s in selected),
            sorted(f["id"] for f in fresh["files"]),
        )
        self.assertTrue(all(s["mount"] == mount for s in selected))
        disk = recipe["topology"]["roles"][0]["resources"]["disk"]
        self.assertEqual(disk["artifact_bytes"], 1_000_000 - 10 + 7)

    def test_selected_file_id_follows_the_new_file_within_the_contract_bound(
        self,
    ) -> None:
        # #1290: a long model-reference prefix plus the new file id overflowed 64.
        prefix = "dependency-nemotron-3-5-lightning-30b-a3b-nvfp4-dspark-"
        _disk, result = self.retarget(10_000_000, 10_000_000, 10_000_000, prefix)
        self.assertFalse(result.reasons)
        new_id = refresh.file_id("model.safetensors", "d" * 64)
        self.assertEqual(self.selected["file_id"], new_id)
        self.assertLessEqual(len(self.selected["id"]), 64)
        self.assertTrue(self.selected["id"].startswith(prefix + new_id[:5]))
        _disk, _result = self.retarget(10_000_000, 10_000_000, 10_000_000)
        self.assertEqual(self.selected["id"], "sel-" + new_id)

    def test_small_size_change_shifts_a_derived_envelope(self) -> None:
        disk, result = self.retarget(10_000_000, 10_000_000, 9_999_600)
        self.assertEqual(disk["artifact_bytes"], 9_999_600)
        self.assertFalse(result.reasons)

    def test_small_size_change_keeps_a_declared_allowance(self) -> None:
        disk, result = self.retarget(10_500_000, 10_000_000, 9_999_600)
        self.assertEqual(disk["artifact_bytes"], 10_499_600)
        self.assertFalse(result.reasons)

    def test_material_size_change_needs_review(self) -> None:
        _disk, result = self.retarget(10_000_000, 10_000_000, 10_200_000)
        self.assertTrue(result.reasons)
        self.assertIn("more than 1%", result.reasons[0])


class BranchFollowingTests(unittest.TestCase):
    """A pin on a non-default branch is compared with that branch, not the default."""

    BRANCH_HEAD = "b" * 40
    MAIN_HEAD = "a" * 40

    def hf_catalog(self, root: str):
        model = model_with(("model.safetensors",))
        model["identity"]["slug"] = "branch-model"
        model["source"] = {
            "repository": "https://huggingface.co/example/model",
            "revision": OLD,
        }
        recipe = {
            "identity": {"slug": "recipe"},
            "models": [{"model": {"slug": "branch-model"}}],
            "execution": {"build": {"context": {"path": "missing-context"}}},
        }
        return refresh.Catalog(
            Path(root), recipes={"recipe": recipe}, models={"branch-model": model}
        )

    def hf_collect(self, branches, histories):
        class FakeHttp:
            def json(self, url, _provider, *, anonymous=False):
                if url.endswith("/refs"):
                    return {"branches": branches}
                name = url.split("/commits/")[1].split("?")[0]
                return histories[name]

        main = refresh.Target(
            "huggingface",
            "example/model",
            self.MAIN_HEAD,
            "2026-09-30",
            None,
            None,
            "u",
            access_status="public",
        )
        with tempfile.TemporaryDirectory() as root:
            return refresh.collect_items(
                self.hf_catalog(root), "recipe", FakeHttp(), lambda _p, _r: main
            )

    def test_huggingface_pin_at_the_head_of_a_per_bpw_branch_is_current(self) -> None:
        result = self.hf_collect(
            [
                {"name": "main", "targetCommit": self.MAIN_HEAD},
                {"name": "3.00bpw", "targetCommit": OLD},
            ],
            {},
        )
        self.assertFalse(result.items)
        self.assertFalse(result.drifted)

    def test_huggingface_branch_that_moved_ahead_is_the_target(self) -> None:
        result = self.hf_collect(
            [
                {"name": "main", "targetCommit": self.MAIN_HEAD},
                {"name": "3.00bpw", "targetCommit": self.BRANCH_HEAD},
                {"name": "2.00bpw", "targetCommit": "c" * 40},
            ],
            {
                "main": [{"id": self.MAIN_HEAD}],
                "3.00bpw": [
                    {"id": self.BRANCH_HEAD, "date": "2026-10-01T00:00:00.000Z"},
                    {"id": OLD},
                ],
                "2.00bpw": [{"id": "c" * 40}],
            },
        )
        self.assertEqual(len(result.items), 1)
        self.assertEqual(result.items[0].target.sha, self.BRANCH_HEAD)
        self.assertEqual(result.items[0].target.branch, "3.00bpw")

    def test_huggingface_pin_in_the_default_history_follows_the_default(self) -> None:
        result = self.hf_collect(
            [
                {"name": "main", "targetCommit": self.MAIN_HEAD},
                {"name": "other", "targetCommit": "c" * 40},
            ],
            {"main": [{"id": self.MAIN_HEAD}, {"id": OLD}]},
        )
        self.assertEqual(len(result.items), 1)
        self.assertEqual(result.items[0].target.sha, self.MAIN_HEAD)
        self.assertIsNone(result.items[0].target.branch)

    def github_collect(self, *, heads, branches, statuses):
        class FakeHttp:
            def github(self, path):
                if path == "repos/example/recipe":
                    return {"default_branch": "main"}
                if path.endswith("/branches-where-head"):
                    return [{"name": n} for n in heads]
                if path.startswith("repos/example/recipe/branches"):
                    return [{"name": n, "commit": {"sha": h}} for n, h in branches]
                if "/compare/" in path:
                    head = path.split("...")[1].split("?")[0]
                    return {"status": statuses[head], "ahead_by": 2}
                if path.endswith("/tags?per_page=100"):
                    return []
                sha = path.rsplit("/", 1)[1]
                return {
                    "sha": MainSha.get(sha, sha),
                    "html_url": "u/" + sha,
                    "commit": {"committer": {"date": "2026-10-01T00:00:00Z"}},
                }

        MainSha: dict[str, str] = {"main": self.MAIN_HEAD}
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
        main = refresh.Target(
            "github", "example/recipe", self.MAIN_HEAD, "2026-09-30", None, None, "u"
        )
        with (
            patch(
                "refresh_upstream.compare_commits",
                side_effect=lambda _h, _r, _old, new: refresh.Comparison(
                    "diverged" if new == self.MAIN_HEAD else "ahead", [], [], False, "u"
                ),
            ),
            patch("refresh_upstream.pin_is_newer_by_date", return_value=False),
        ):
            return refresh.collect_items(
                catalog, "recipe", FakeHttp(), lambda _p, _r: main
            )

    def test_github_pin_at_the_head_of_a_feature_branch_is_current(self) -> None:
        result = self.github_collect(
            heads=["feature"],
            branches=[],
            statuses={self.MAIN_HEAD: "diverged"},
        )
        self.assertFalse(result.items)
        self.assertFalse(result.drifted)

    def test_github_feature_branch_that_moved_ahead_is_the_target(self) -> None:
        result = self.github_collect(
            heads=[],
            branches=[("main", self.MAIN_HEAD), ("feature", self.BRANCH_HEAD)],
            statuses={self.MAIN_HEAD: "diverged", self.BRANCH_HEAD: "ahead"},
        )
        self.assertEqual(len(result.items), 1)
        self.assertEqual(result.items[0].target.sha, self.BRANCH_HEAD)
        self.assertEqual(result.items[0].target.branch, "feature")

    def test_github_pin_in_the_default_history_is_not_a_branch(self) -> None:
        result = self.github_collect(
            heads=[],
            branches=[("feature", self.BRANCH_HEAD)],
            statuses={self.MAIN_HEAD: "ahead", self.BRANCH_HEAD: "ahead"},
        )
        self.assertEqual(len(result.items), 1)
        self.assertIsNone(result.items[0].target.branch)


class ReleaseLineTests(unittest.TestCase):
    """An adapter that declares ``source.line`` follows the newest release in that line."""

    REPO = "example/engine"
    V066 = "6" * 40
    V109 = "9" * 40
    BETA = "b" * 40
    RELEASES: ClassVar[list[dict[str, object]]] = [
        {"tag_name": "v1.0.0-beta.5", "prerelease": True, "draft": False},
        {"tag_name": "v1.0.9", "prerelease": False, "draft": False},
        {"tag_name": "v0.60.1", "prerelease": False, "draft": False},
        {"tag_name": "v0.6.6", "prerelease": False, "draft": False, "body": "0.6.6"},
        {"tag_name": "v0.6.0", "prerelease": False, "draft": False},
    ]
    SHAS: ClassVar[dict[str, str]] = {
        "v1.0.9": V109,
        "v1.0.0-beta.5": BETA,
        "v0.6.6": V066,
        "v0.6.0": OLD,
        "v0.60.1": "a" * 40,
    }

    def collect(self, pin: str, line: str | None = "0.6"):
        releases, shas, repo = self.RELEASES, self.SHAS, self.REPO

        class FakeHttp:
            def github(self, path: str):
                if path == f"repos/{repo}/releases/latest":
                    return releases[1]
                if path == f"repos/{repo}/releases?per_page=100":
                    return releases
                ref = path.rsplit("/", 1)[1]
                return {
                    "sha": shas.get(ref, ref),
                    "html_url": "u/" + ref,
                    "commit": {"committer": {"date": "2026-10-06T00:00:00Z"}},
                }

        http = FakeHttp()
        with tempfile.TemporaryDirectory() as tmp:
            adapter = Path(tmp) / "adapter"
            adapter.mkdir()
            (adapter / "Dockerfile").write_text("FROM scratch\n")
            source = {"repo": repo, "commit": pin} | ({"line": line} if line else {})
            (adapter / "vendored-upstream.json").write_text(
                json.dumps({"ours": [], "source": source})
            )
            catalog = refresh.Catalog(
                Path(tmp),
                recipes={
                    "recipe": {
                        "identity": {"slug": "recipe"},
                        "provenance": {
                            "attribution": ["Example"],
                            "source_reference": f"https://github.com/{repo}/tree/{pin}",
                        },
                        "release": {"version": "0.6.0", "released_at": "2026-09-30"},
                        "models": [],
                        "execution": {"build": {"context": {"path": "adapter"}}},
                    }
                },
            )
            with patch(
                "refresh_upstream.compare_commits",
                side_effect=lambda _h, _r, _old, _new: comparison("ahead"),
            ):
                return refresh.collect_items(
                    catalog,
                    "recipe",
                    http,
                    lambda _p, r: refresh.resolve_github(http, r),
                )

    def test_newest_release_inside_the_line_is_the_target(self) -> None:
        result = self.collect(OLD)
        self.assertEqual([i.target.sha for i in result.items], [self.V066])
        self.assertEqual(result.items[0].target.version, "0.6.6")
        self.assertEqual(len(result.notes), 1)
        self.assertIn("example/engine 1.0", result.notes[0])

    def test_pin_at_the_newest_of_its_line_is_current(self) -> None:
        result = self.collect(self.V066)
        self.assertFalse(result.items)
        self.assertFalse(result.drifted)
        self.assertEqual(len(result.notes), 1)  # reported, never a move

    def test_without_a_line_the_newest_release_wins(self) -> None:
        result = self.collect(OLD, line=None)
        self.assertEqual([i.target.sha for i in result.items], [self.V109])
        self.assertFalse(result.notes)

    def test_newer_line_is_reported_once_and_stays_stable(self) -> None:
        result = self.collect(OLD)
        first = refresh.issue_body(result)
        self.assertEqual(first.count("Newer release line available"), 1)
        self.assertEqual(refresh.issue_body(result, first), first)
        self.assertNotIn("Upstream moved", refresh.issue_body(result, first))
        later = refresh.Target(
            "github", self.REPO, "c" * 40, "", "v1.0.10", "1.0.10", "u"
        )
        newest = refresh.Target(
            "github", self.REPO, self.V109, "", "v1.0.9", "1.0.9", "u"
        )
        self.assertEqual(
            refresh.newer_line_note(self.REPO, "0.6", later),
            refresh.newer_line_note(self.REPO, "0.6", newest),
        )

    def test_a_line_is_a_version_prefix_not_a_string_prefix(self) -> None:
        self.assertFalse(refresh.in_line((0, 60, 1), refresh.line_parts("0.6")))
        self.assertTrue(refresh.in_line((0, 6, 6), refresh.line_parts("0.6")))
        with self.assertRaises(RuntimeError):
            refresh.line_parts("0.6.x")


class AppCommandBoundaryTests(unittest.TestCase):
    def test_renewed_app_token_reaches_both_gh_and_git(self) -> None:
        from unittest.mock import Mock

        auth = Mock()
        auth.environment.side_effect = [
            {"GH_TOKEN": "first-app-token"},
            {"GH_TOKEN": "renewed-app-token"},
        ]
        completed = __import__("subprocess").CompletedProcess([], 0, "", "")
        with (
            patch.object(refresh, "APP_AUTH", auth),
            patch.dict(refresh.os.environ, {"GH_TOKEN": "actions-token"}),
            patch.object(refresh.subprocess, "run", return_value=completed) as run,
        ):
            refresh.gh("pr", "list")
            refresh.git("push", "origin", cwd=ROOT)
        self.assertEqual(
            run.call_args_list[0].kwargs["env"]["GH_TOKEN"], "first-app-token"
        )
        self.assertEqual(
            run.call_args_list[1].kwargs["env"]["GH_TOKEN"], "renewed-app-token"
        )

    def test_missing_app_key_fails_before_recipe_work(self) -> None:
        with (
            patch.object(sys, "argv", ["refresh-upstream", "--github-app"]),
            patch.dict(refresh.os.environ, {}, clear=True),
            patch.object(refresh, "command_run") as command,
            patch.object(refresh, "APP_AUTH", None),
            self.assertRaisesRegex(RuntimeError, "not configured"),
        ):
            refresh.main()
        command.assert_not_called()


class AppKeyLifetimeTests(unittest.TestCase):
    def test_signing_key_is_removed_before_recipe_subprocesses_and_auth_is_closed(
        self,
    ) -> None:
        from unittest.mock import Mock

        auth = Mock()

        def command(arguments):
            self.assertNotIn("REFRESH_APP_PRIVATE_KEY", refresh.os.environ)
            return 0

        with (
            patch.object(sys, "argv", ["refresh-upstream", "--github-app"]),
            patch.dict(
                refresh.os.environ,
                {
                    "REFRESH_APP_CLIENT_ID": "client",
                    "REFRESH_APP_PRIVATE_KEY": "private-signing-key",
                    "GITHUB_REPOSITORY": "owner/repo",
                },
            ),
            patch.object(refresh, "RefreshAppAuth", return_value=auth),
            patch.object(refresh, "command_run", side_effect=command),
            patch.object(refresh, "APP_AUTH", None),
        ):
            self.assertEqual(refresh.main(), 0)
            auth.close.assert_called_once()
            self.assertIsNone(refresh.APP_AUTH)


class AppVerificationTests(unittest.TestCase):
    def test_live_verification_does_not_start_recipe_work(self) -> None:
        from unittest.mock import Mock

        auth = Mock()
        with (
            patch.object(
                sys, "argv", ["refresh-upstream", "--github-app", "--verify-app"]
            ),
            patch.dict(
                refresh.os.environ,
                {
                    "REFRESH_APP_CLIENT_ID": "client",
                    "REFRESH_APP_PRIVATE_KEY": "key",
                    "GITHUB_REPOSITORY": "owner/repo",
                },
            ),
            patch.object(refresh, "RefreshAppAuth", return_value=auth),
            patch.object(refresh, "APP_AUTH", None),
            patch.object(
                refresh,
                "gh_json",
                return_value={"repositories": [{"full_name": "owner/repo"}]},
            ),
            patch.object(refresh, "command_run") as command,
        ):
            self.assertEqual(refresh.main(), 0)
            command.assert_not_called()
            auth.close.assert_called_once()

    def test_broader_token_scope_is_refused(self) -> None:
        from unittest.mock import Mock

        auth = Mock()
        with (
            patch.object(
                sys, "argv", ["refresh-upstream", "--github-app", "--verify-app"]
            ),
            patch.dict(
                refresh.os.environ,
                {
                    "REFRESH_APP_CLIENT_ID": "client",
                    "REFRESH_APP_PRIVATE_KEY": "key",
                    "GITHUB_REPOSITORY": "owner/repo",
                },
            ),
            patch.object(refresh, "RefreshAppAuth", return_value=auth),
            patch.object(refresh, "APP_AUTH", None),
            patch.object(
                refresh,
                "gh_json",
                return_value={
                    "repositories": [
                        {"full_name": "owner/repo"},
                        {"full_name": "owner/other"},
                    ]
                },
            ),
            self.assertRaisesRegex(RuntimeError, "scope differs"),
        ):
            refresh.main()
        auth.close.assert_called_once()


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


class SelfHealingTests(unittest.TestCase):
    def test_check_state(self) -> None:
        self.assertEqual(refresh.check_state(None), "none")
        self.assertEqual(refresh.check_state(failing()), "failed")
        self.assertEqual(
            refresh.check_state(
                [{"conclusion": "CANCELLED"}, {"conclusion": "SUCCESS"}]
            ),
            "cancelled",
        )
        self.assertEqual(
            refresh.check_state([{"status": "IN_PROGRESS", "conclusion": ""}]),
            "pending",
        )
        self.assertEqual(
            refresh.check_state([{"conclusion": "SUCCESS"}, {"conclusion": "SKIPPED"}]),
            "passed",
        )

    def test_verify_runs_every_data_check_of_validate_yml(self) -> None:
        workflow = (ROOT / ".github/workflows/validate.yml").read_text()
        ci = set(re.findall(r"python3 (tools/[\w-]+)", workflow))
        # run elsewhere or needing a base ref / a built index, not data the refresh edits
        skipped = {
            "tools/build-catalog-index",
            "tools/build-family-aware-coverage",
            "tools/check-recipe-topology",
            "tools/check-image-provenance",
            "tools/generate-contract-examples",
            "tools/generate-contract-schemas",
            "tools/build-release-bundle",
        }
        covered = {part for _, cmd in refresh.verify_steps() for part in cmd}
        self.assertEqual(sorted(t for t in ci - skipped if t not in covered), [])
        self.assertIn("check", dict(refresh.verify_steps())["kit-declared pins"])

    def republish(
        self, rollup, *, pin: bytes = b"new pin\n", main_moves: bool = False
    ) -> tuple[list, str, str, Any, bool]:
        """A refresh PR prepared earlier; this run assesses ``pin`` against main."""
        assessment = refresh.Assessment("recipe", edits={"recipe.json": pin})
        calls: list[tuple[str, ...]] = []
        env = {
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@example.test",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@example.test",
        }
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, env):
            root = Path(temporary)
            refresh.git("init", "-q", "--initial-branch=main", cwd=root)
            refresh.git("remote", "add", "origin", str(root), cwd=root)
            (root / "recipe.json").write_bytes(b"old pin\n")
            refresh.git("add", "recipe.json", cwd=root)
            refresh.git("commit", "-qm", "Base", cwd=root)
            refresh.git("switch", "-qc", "refresh/recipe", cwd=root)
            refresh.apply_edits(root, {"recipe.json": b"new pin\n"})
            refresh.git("commit", "-qam", "Prepared refresh", cwd=root)
            before = refresh.git("rev-parse", "HEAD", cwd=root)
            refresh.git("switch", "-q", "main", cwd=root)
            if main_moves:  # another recipe merged meanwhile
                (root / "other.json").write_bytes(b"other\n")
                refresh.git("add", "other.json", cwd=root)
                refresh.git("commit", "-qm", "Other change", cwd=root)
            main = refresh.git("rev-parse", "HEAD", cwd=root)
            refresh.git("update-ref", "refs/remotes/origin/main", main, cwd=root)

            def github(*arguments: str, check: bool = True) -> str:
                calls.append(arguments)
                if arguments[:3] == ("pr", "list", "--head"):
                    return json.dumps(
                        [
                            {
                                "number": 9,
                                "state": "OPEN",
                                "body": "",
                                "statusCheckRollup": rollup,
                            }
                        ]
                    )
                if arguments[:2] == ("run", "view"):
                    return "line one\nBOOM: kit lock is stale\n"
                if arguments[:2] == ("pr", "list"):
                    return "[]"
                if arguments[:2] == ("run", "list"):
                    return "[]"
                return main

            with (
                patch("refresh_upstream.gh", side_effect=github),
                patch("refresh_upstream.verify", return_value=None),
            ):
                refresh.publish_mechanical(root, assessment, refresh.Summary())
            after = refresh.git("rev-parse", "refs/heads/refresh/recipe", cwd=root)
            forward = (
                refresh.run(
                    ["git", "merge-base", "--is-ancestor", before, after],
                    cwd=root,
                    check=False,
                ).returncode
                == 0
            )
            if after != before:
                self.assertEqual(
                    refresh.git("show", f"{after}:recipe.json", cwd=root),
                    pin.decode().strip(),
                )
                self.assertEqual(
                    refresh.git("diff", "--name-only", main, after, cwd=root),
                    "recipe.json",
                )
        return calls, before, after, assessment, forward

    def test_failed_checks_on_the_carried_refresh_are_reported_not_pushed(
        self,
    ) -> None:
        calls, before, after, assessment, _ = self.republish(failing())
        self.assertEqual(before, after)
        self.assertFalse(
            [c for c in calls if c[:2] in {("pr", "edit"), ("pr", "merge")}]
        )
        self.assertIn("not pushed again", assessment.reasons[0])
        self.assertIn("BOOM: kit lock is stale", "".join(assessment.evidence))

    def test_cancelled_checks_are_reported_not_pushed(self) -> None:
        _, before, after, assessment, _ = self.republish([{"conclusion": "CANCELLED"}])
        self.assertEqual(before, after)
        self.assertIn("cancelled", assessment.reasons[0])

    def test_passing_unchanged_pr_is_left_alone(self) -> None:
        calls, before, after, _, _ = self.republish([{"conclusion": "SUCCESS"}])
        self.assertEqual(before, after)
        self.assertFalse([c for c in calls if c[:2] == ("pr", "edit")])

    def test_main_moving_elsewhere_pushes_nothing(self) -> None:
        _, before, after, assessment, _ = self.republish(
            [{"conclusion": "SUCCESS"}], main_moves=True
        )
        self.assertEqual(before, after)
        self.assertEqual(assessment.reasons, [])

    def test_a_newer_refresh_moves_the_branch_forward_without_force(self) -> None:
        for rollup in ([{"conclusion": "SUCCESS"}], failing()):
            with self.subTest(rollup=rollup):
                calls, before, after, assessment, forward = self.republish(
                    rollup, pin=b"newer pin\n", main_moves=True
                )
                self.assertNotEqual(before, after)
                self.assertTrue(forward, "the push must be a fast-forward")
                self.assertEqual(assessment.reasons, [])
                self.assertTrue([c for c in calls if c[:2] == ("pr", "edit")])

    def test_a_refreshed_proven_recipe_rewrites_its_guard_entry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / refresh.PROVEN_BYTES).parent.mkdir(parents=True)
            (root / refresh.PROVEN_BYTES).write_text(
                json.dumps({"vonk-forge/a": "0" * 64, "vonk-forge/b": "1" * 64})
            )
            edits: dict[str, bytes | None] = {"recipes/a.json": b"moved\n"}
            refresh.follow_proven_bytes(root, edits)
            guard = json.loads(edits[refresh.PROVEN_BYTES] or b"")
            self.assertEqual(
                guard["vonk-forge/a"], hashlib.sha256(b"moved\n").hexdigest()
            )
            self.assertEqual(guard["vonk-forge/b"], "1" * 64)
            unproven: dict[str, bytes | None] = {"recipes/c.json": b"moved\n"}
            refresh.follow_proven_bytes(root, unproven)
            self.assertNotIn(refresh.PROVEN_BYTES, unproven)

    def test_failure_evidence_has_the_log_tail(self) -> None:
        with patch("refresh_upstream.gh", return_value="x\nBOOM: stale\n"):
            text = refresh.failed_check_evidence(failing())
        self.assertIn("BOOM: stale", text)
        self.assertIn("actions/runs/7", text)


class SharedContextTests(unittest.TestCase):
    def catalog(self):
        def recipe(path: str) -> dict:
            return {"execution": {"build": {"context": {"path": path}}}}

        return types.SimpleNamespace(
            recipes={
                "b": recipe("adapters/x/"),
                "a": recipe("adapters/x"),
                "c": recipe("adapters/y"),
            }
        )

    def test_recipes_sharing_a_context_are_grouped_sorted(self) -> None:
        self.assertEqual(
            refresh.sharing_groups(self.catalog()), {"a": ["a", "b"], "b": ["a", "b"]}
        )

    def assessment(self, recipe_id: str, **kwargs):
        item = refresh.Item(
            "source",
            "github",
            "o/r",
            OLD,
            refresh.Target("github", "o/r", NEW, "d", None, None, "u"),
        )
        return refresh.Assessment(recipe_id, items=[item], **kwargs)

    def test_group_is_one_assessment_with_all_edits(self) -> None:
        assessed = {
            "a": self.assessment("a", edits={"recipes/a.json": b"1", "lock": b"L"}),
            "b": self.assessment("b", edits={"recipes/b.json": b"2", "lock": b"L"}),
        }
        combined = refresh.combine_group(["a", "b"], assessed)
        assert not isinstance(combined, str)
        self.assertEqual(combined.recipe_id, "a")
        self.assertEqual(
            set(combined.edits), {"recipes/a.json", "recipes/b.json", "lock"}
        )
        self.assertIn("a, b", refresh.pr_body(combined))

    def test_group_waits_when_a_sibling_is_not_mechanical(self) -> None:
        assessed = {
            "a": self.assessment("a", edits={"lock": b"L"}),
            "b": self.assessment("b", reasons=["needs review"]),
        }
        self.assertIsInstance(refresh.combine_group(["a", "b"], assessed), str)
        self.assertIsInstance(
            refresh.combine_group(["a", "b"], {"a": assessed["a"]}), str
        )

    def test_a_current_sibling_does_not_hold_the_group_back(self) -> None:
        # Since 2026-10-19 the Nemotron Lightning DSpark Model moved for three recipes
        # whose contexts they share with current siblings; every run deferred them.
        current = refresh.Assessment("a")
        assessed = {
            "a": current,
            "b": self.assessment("b", edits={"recipes/b.json": b"2"}),
            "c": self.assessment("c", edits={"recipes/c.json": b"3"}),
        }
        combined = refresh.combine_group(["a", "b", "c"], assessed)
        assert not isinstance(combined, str), combined
        self.assertEqual(combined.recipe_id, "a")
        self.assertEqual(set(combined.edits), {"recipes/b.json", "recipes/c.json"})
        self.assertEqual(combined.label, "b, c")
        self.assertIsInstance(refresh.combine_group(["a"], {"a": current}), str)

    def test_group_rejects_conflicting_shared_edits(self) -> None:
        assessed = {
            "a": self.assessment("a", edits={"lock": b"L1"}),
            "b": self.assessment("b", edits={"lock": b"L2"}),
        }
        self.assertIsInstance(refresh.combine_group(["a", "b"], assessed), str)


class ReviewIssueLifecycleTests(unittest.TestCase):
    """One issue per recipe for life, showing the current assessment (#1282, #1283)."""

    TARGET = refresh.Target("github", "o/r", NEW, "2026-09-30", None, None, "u")

    def assessment(self, recipe_id: str, *, item=False, reason="", unreachable=""):
        result = refresh.Assessment(recipe_id)
        if item:
            result.items.append(
                refresh.Item("source", "github", "o/r", OLD, self.TARGET)
            )
        if reason:
            result.reasons.append(reason)
        if unreachable:
            result.unreachable.append(unreachable)
        return result

    def run_gh(self, issues: dict[str, list[dict]], call, *arguments):
        calls: list[tuple[str, ...]] = []

        def gh_json(*args: str):
            return issues[args[args.index("--state") + 1]]

        def gh(*args: str, check: bool = True):
            calls.append(args)
            return ""

        with (
            patch("refresh_upstream.gh_json", side_effect=gh_json),
            patch("refresh_upstream.gh", side_effect=gh),
        ):
            call(*arguments)
        return calls

    def test_only_current_gone_or_handled_recipes_close(self) -> None:
        names = ["current", "deferred", "handled", "waiting", "gone"]
        issues = {
            "open": [
                {"number": n, "title": refresh.issue_title(name)}
                for n, name in enumerate(names, start=1)
            ]
        }
        assessed = {
            "current": self.assessment("current"),
            "deferred": self.assessment("deferred", item=True),
            "handled": self.assessment("handled", item=True),
            "waiting": self.assessment("waiting", unreachable="x not found"),
        }
        summary = refresh.Summary()
        calls = self.run_gh(
            issues, refresh.close_current_issues, assessed, {"handled"}, summary
        )
        closed = [c[2] for c in calls if c[:2] == ("issue", "close")]
        self.assertEqual(closed, ["1", "3", "5"])

    def test_an_older_duplicate_issue_closes_in_favour_of_the_newest(self) -> None:
        # #1563/#1714: two open issues for one recipe; only the newest is kept current.
        title = refresh.issue_title("drifted")
        issues = {
            "open": [
                {"number": 1563, "title": title},
                {"number": 1714, "title": title},
            ]
        }
        assessed = {"drifted": self.assessment("drifted", item=True, reason="x")}
        calls = self.run_gh(
            issues, refresh.close_current_issues, assessed, set(), refresh.Summary()
        )
        closes = [c for c in calls if c[:2] == ("issue", "close")]
        self.assertEqual([c[2] for c in closes], ["1563"])
        self.assertIn("#1714", closes[0][-1])

    def test_a_recipe_that_needs_review_again_reopens_its_issue(self) -> None:
        title = refresh.issue_title("recipe")
        issues = {
            "open": [],
            "closed": [
                {"number": 7, "title": title, "body": "old"},
                {"number": 9, "title": title, "body": "older"},
            ],
        }
        result = self.assessment("recipe", item=True, reason="run.sh changed")
        calls = self.run_gh(issues, refresh.sync_issue, result, refresh.Summary())
        self.assertEqual(calls[0], ("issue", "reopen", "9"))
        self.assertEqual(calls[1][:3], ("issue", "edit", "9"))
        self.assertFalse([c for c in calls if c[:2] == ("issue", "create")])

    def test_first_review_creates_one_issue(self) -> None:
        result = self.assessment("recipe", item=True, reason="run.sh changed")
        calls = self.run_gh(
            {"open": [], "closed": []}, refresh.sync_issue, result, refresh.Summary()
        )
        self.assertEqual([c[:2] for c in calls], [("issue", "create")])

    def test_a_waiting_recipe_updates_only_an_open_issue(self) -> None:
        result = self.assessment("recipe", unreachable="commit not on GitHub")
        title = refresh.issue_title("recipe")
        none = self.run_gh(
            {"open": [], "closed": [{"number": 3, "title": title, "body": ""}]},
            refresh.sync_issue,
            result,
            refresh.Summary(),
            False,
        )
        self.assertEqual(none, [])
        calls = self.run_gh(
            {"open": [{"number": 4, "title": title, "body": "stale reason"}]},
            refresh.sync_issue,
            result,
            refresh.Summary(),
            False,
        )
        self.assertEqual(calls[0][:3], ("issue", "edit", "4"))
        body = calls[0][4]
        self.assertIn("waits for an upstream", body)
        self.assertIn("## Why it waits", body)
        self.assertIn("- commit not on GitHub", body)


class StalePullRequestTests(unittest.TestCase):
    def close(self, assessed, groups=None, published=None, issues=()):
        pulls = [
            {"number": 1, "headRefName": "refresh/a", "body": ""},
            {"number": 2, "headRefName": "refresh/b", "body": ""},
        ]
        closed: list[tuple[str, ...]] = []

        def github(*arguments: str, check: bool = True) -> str:
            if arguments[0] == "pr" and arguments[1] == "close":
                closed.append(arguments)
            return ""

        def github_json(*arguments: str):
            return pulls if arguments[0] == "pr" else list(issues)

        with (
            patch("refresh_upstream.gh", side_effect=github),
            patch("refresh_upstream.gh_json", side_effect=github_json),
        ):
            refresh.close_stale_refresh_prs(
                assessed, refresh.Summary(), groups, published
            )
        return {c[2]: c[-1] for c in closed}

    def item(self):
        return refresh.Item(
            "source",
            "github",
            "o/r",
            OLD,
            refresh.Target("github", "o/r", NEW, "d", None, None, "u"),
        )

    def test_current_gone_and_review_prs_are_closed_with_a_reason(self) -> None:
        assessed = {
            "a": refresh.Assessment("a"),
            "b": refresh.Assessment("b", items=[self.item()], reasons=["why"]),
        }
        closed = self.close(
            assessed, issues=[{"number": 77, "title": refresh.issue_title("b")}]
        )
        self.assertIn("current again", closed["1"])
        self.assertIn("#77", closed["2"])
        self.assertIn("no longer exists", self.close({})["1"])

    def test_mechanical_published_and_unreachable_prs_stay(self) -> None:
        mech = refresh.Assessment("a", items=[self.item()])
        down = refresh.Assessment("b", items=[self.item()], unreachable=["x"])
        self.assertEqual(self.close({"a": mech, "b": down}), {})
        review = refresh.Assessment("a", items=[self.item()], reasons=["ci"])
        self.assertEqual(self.close({"a": review, "b": down}, published={"a"}), {})

    def test_a_sibling_pr_of_a_shared_context_is_closed(self) -> None:
        mech = {
            "a": refresh.Assessment("a", items=[self.item()]),
            "b": refresh.Assessment("b", items=[self.item()]),
        }
        closed = self.close(mech, groups={"a": ["a", "b"], "b": ["a", "b"]})
        self.assertEqual(list(closed), ["2"])
        self.assertIn("shares its build context", closed["2"])
