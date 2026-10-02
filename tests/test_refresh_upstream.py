from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
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
        summary.receipt = True
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
        summary.receipt = True
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
        self.assertIn("publish.yml receipt", summary.deferred[0])
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

    def test_prepare_bound_counts_pushed_prs_only(self) -> None:
        self.assertEqual(refresh.MAX_MECHANICAL_PER_RUN, 6)
        summary = refresh.Summary()
        self.assertEqual(summary.prepared, 0)

    def test_chain_guard_skips_failed_publication_and_recent_runs(self) -> None:
        now = datetime(2026, 10, 2, 12, 0, tzinfo=UTC)

        def run(minutes_ago: int, **extra: object) -> dict[str, object]:
            when = now - timedelta(minutes=minutes_ago)
            return {
                "databaseId": 7,
                "status": "completed",
                "conclusion": "success",
                "updatedAt": when.isoformat().replace("+00:00", "Z"),
            } | extra

        self.assertTrue(refresh.recent_refresh_ran([run(3)], 99, now, 10))
        self.assertFalse(refresh.recent_refresh_ran([run(30)], 99, now, 10))
        # Active or queued runs never block a chained run (the concurrency group queues).
        self.assertFalse(
            refresh.recent_refresh_ran([run(1, status="in_progress")], 99, now, 10)
        )
        # The current run never blocks itself.
        self.assertFalse(
            refresh.recent_refresh_ran([run(1, databaseId=99)], 99, now, 10)
        )
        self.assertFalse(
            refresh.recent_refresh_ran([run(1, conclusion="cancelled")], 99, now, 10)
        )

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
                "databaseId": 7,
                "status": "completed",
                "conclusion": "success",
                "updatedAt": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            }
        ]
        self.assertEqual(decide("schedule", "", recent), "run=true")
        self.assertEqual(decide("workflow_dispatch", "", recent), "run=true")
        self.assertEqual(decide("workflow_run", "success", []), "run=true")
        self.assertEqual(decide("workflow_run", "failure", []), "run=false")
        self.assertEqual(decide("workflow_run", "success", recent), "run=false")

    def test_only_behind_refresh_branches_are_updated(self) -> None:
        pulls = [
            {"number": 1, "headRefName": "refresh/a", "mergeStateStatus": "BEHIND"},
            {"number": 2, "headRefName": "refresh/b", "mergeStateStatus": "CLEAN"},
            {"number": 3, "headRefName": "codex/x", "mergeStateStatus": "BEHIND"},
        ]
        self.assertEqual(
            [p["number"] for p in refresh.stale_mechanical_prs(pulls)], [1]
        )
        summary = refresh.Summary()
        with (
            patch("refresh_upstream.gh_json", return_value=pulls),
            patch("refresh_upstream.gh", return_value="") as gh,
        ):
            refresh.update_stale_prs(summary)
        gh.assert_called_once_with("pr", "update-branch", "1", check=False)


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
    def retarget(self, artifact_bytes: int, old_size: int, new_size: int):
        old = model_with(("model.safetensors",))
        old["files"][0]["size_bytes"] = old_size
        fresh = json.loads(json.dumps(old))
        fresh["identity"]["slug"] = "demo-" + NEW[:8]
        fresh["files"][0]["sha256"] = "d" * 64
        fresh["files"][0]["size_bytes"] = new_size
        fresh["files"][0]["id"] = refresh.file_id("model.safetensors", "d" * 64)
        recipe = {
            "models": [
                {
                    "model": {"slug": old["identity"]["slug"], "content_sha256": "x"},
                    "files": [
                        {
                            "id": "sel-" + old["files"][0]["id"],
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
        return recipe["topology"]["roles"][0]["resources"]["disk"], result

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
