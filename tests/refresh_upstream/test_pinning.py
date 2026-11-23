"""Refresh-upstream tests: pinning."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

import pytest
from refresh_helpers import NEW, OLD, classify, comparison, model_with

from recipe_tooling.refresh_upstream import catalog_view as refresh_catalog_view
from recipe_tooling.refresh_upstream import classification as refresh_classification
from recipe_tooling.refresh_upstream import items as refresh_items
from recipe_tooling.refresh_upstream import model as refresh_model
from recipe_tooling.refresh_upstream import pinning as refresh_pinning
from recipe_tooling.refresh_upstream import render as refresh_render


class PinningRuleTests(unittest.TestCase):
    def test_newest_plain_version_tag_wins(self) -> None:
        tags = [
            {"name": n} for n in ("v1.9.0", "v1.10.0", "v1.10.0-rc1", "nightly", "v0.3")
        ]
        assert refresh_pinning.pick_version_tag(tags)["name"] == "v1.10.0"  # pyright: ignore[reportOptionalSubscript]

    def test_no_version_tag_means_follow_the_head(self) -> None:
        assert (
            refresh_pinning.pick_version_tag([{"name": "latest"}, {"name": "x"}])
            is None
        )


class NeverDowngradeTests(unittest.TestCase):
    def collect(self, status: str):
        catalog = refresh_catalog_view.Catalog(
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
        target = refresh_model.Target(
            "github", "example/recipe", NEW, "2026-09-30", "v2.0.0", "2.0.0", "u"
        )
        with patch(
            "recipe_tooling.refresh_upstream.items.compare_commits",
            return_value=refresh_model.Comparison(status, [], [], False, "u"),
        ):
            return refresh_items.collect_items(
                catalog,
                "recipe",
                object(),  # pyright: ignore[reportArgumentType]
                lambda _p, _r: target,
            )

    def test_pin_ahead_of_release_is_current(self) -> None:
        result = self.collect("behind")
        assert not result.drifted
        assert not result.reasons

    def test_release_ahead_of_pin_is_refreshed(self) -> None:
        result = self.collect("ahead")
        assert len(result.items) == 1
        assert result.items[0].comparison.status == "ahead"  # pyright: ignore[reportOptionalMemberAccess]
        assert classify(result.items[0].comparison) == []

    def test_diverged_history_needs_review(self) -> None:
        with patch(
            "recipe_tooling.refresh_upstream.items.pin_is_newer_by_date",
            return_value=False,
        ):
            result = self.collect("diverged")
        assert len(result.items) == 1
        assert classify(result.items[0].comparison)


class ReleaseTests(unittest.TestCase):
    recipe: ClassVar[dict[str, object]] = {
        "release": {"version": "1.0.0", "released_at": "2026-06-01"}
    }

    def item(self, version=None, date="2026-09-01", kind="source"):
        target = refresh_model.Target("github", "o/r", NEW, date, None, version, "u")
        return refresh_model.Item(kind, "github", "o/r", OLD, target)

    def test_upstream_version_and_date_when_published(self) -> None:
        assert refresh_classification.released(self.recipe, [self.item("2.4.1")]) == (
            "2.4.1",
            "2026-09-01",
        )

    def test_own_version_is_bumped_when_upstream_has_none(self) -> None:
        assert refresh_classification.released(self.recipe, [self.item()]) == (
            "1.0.1",
            "2026-09-01",
        )

    def test_invalid_upstream_spelling_falls_back_to_own_version(self) -> None:
        assert (
            refresh_classification.released(self.recipe, [self.item("1.6 beta")])[0]
            == "1.0.1"
        )


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
        assert refresh_items.pin_is_newer_by_date(http, "o/r", OLD, NEW)  # pyright: ignore[reportArgumentType]

    def test_release_newer_than_pin_is_not_current(self) -> None:
        http = self.FakeHttp("2026-04-01T00:00:00Z", "2026-04-20T00:00:00Z")
        assert not refresh_items.pin_is_newer_by_date(http, "o/r", OLD, NEW)  # pyright: ignore[reportArgumentType]

    def test_unreadable_dates_are_not_current(self) -> None:
        assert not refresh_items.pin_is_newer_by_date(object(), "o/r", OLD, NEW)  # pyright: ignore[reportArgumentType]

    def test_image_wrapper_diverged_release_behind_pin_is_current(self) -> None:
        collector = NeverDowngradeTests()
        with patch(
            "recipe_tooling.refresh_upstream.items.pin_is_newer_by_date",
            return_value=True,
        ):
            assert not collector.collect("diverged").drifted
        with patch(
            "recipe_tooling.refresh_upstream.items.pin_is_newer_by_date",
            return_value=False,
        ):
            assert len(collector.collect("diverged").items) == 1


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
        return refresh_catalog_view.Catalog(
            Path(root), recipes={"recipe": recipe}, models={"branch-model": model}
        )

    def hf_collect(self, branches, histories):
        class FakeHttp:
            def json(self, url, _provider, *, anonymous=False):
                if url.endswith("/refs"):
                    return {"branches": branches}
                name = url.split("/commits/")[1].split("?")[0]
                return histories[name]

        main = refresh_model.Target(
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
            return refresh_items.collect_items(
                self.hf_catalog(root),
                "recipe",
                FakeHttp(),  # pyright: ignore[reportArgumentType]
                lambda _p, _r: main,
            )

    def test_huggingface_pin_at_the_head_of_a_per_bpw_branch_is_current(self) -> None:
        result = self.hf_collect(
            [
                {"name": "main", "targetCommit": self.MAIN_HEAD},
                {"name": "3.00bpw", "targetCommit": OLD},
            ],
            {},
        )
        assert not result.items
        assert not result.drifted

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
        assert len(result.items) == 1
        assert result.items[0].target.sha == self.BRANCH_HEAD
        assert result.items[0].target.branch == "3.00bpw"

    def test_huggingface_pin_in_the_default_history_follows_the_default(self) -> None:
        result = self.hf_collect(
            [
                {"name": "main", "targetCommit": self.MAIN_HEAD},
                {"name": "other", "targetCommit": "c" * 40},
            ],
            {"main": [{"id": self.MAIN_HEAD}, {"id": OLD}]},
        )
        assert len(result.items) == 1
        assert result.items[0].target.sha == self.MAIN_HEAD
        assert result.items[0].target.branch is None

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
        catalog = refresh_catalog_view.Catalog(
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
        main = refresh_model.Target(
            "github", "example/recipe", self.MAIN_HEAD, "2026-09-30", None, None, "u"
        )
        with (
            patch(
                "recipe_tooling.refresh_upstream.items.compare_commits",
                side_effect=lambda _h, _r, _old, new: refresh_model.Comparison(
                    "diverged" if new == self.MAIN_HEAD else "ahead", [], [], False, "u"
                ),
            ),
            patch(
                "recipe_tooling.refresh_upstream.items.pin_is_newer_by_date",
                return_value=False,
            ),
        ):
            return refresh_items.collect_items(
                catalog,
                "recipe",
                FakeHttp(),  # pyright: ignore[reportArgumentType]
                lambda _p, _r: main,
            )

    def test_github_pin_at_the_head_of_a_feature_branch_is_current(self) -> None:
        result = self.github_collect(
            heads=["feature"],
            branches=[],
            statuses={self.MAIN_HEAD: "diverged"},
        )
        assert not result.items
        assert not result.drifted

    def test_github_feature_branch_that_moved_ahead_is_the_target(self) -> None:
        result = self.github_collect(
            heads=[],
            branches=[("main", self.MAIN_HEAD), ("feature", self.BRANCH_HEAD)],
            statuses={self.MAIN_HEAD: "diverged", self.BRANCH_HEAD: "ahead"},
        )
        assert len(result.items) == 1
        assert result.items[0].target.sha == self.BRANCH_HEAD
        assert result.items[0].target.branch == "feature"

    def test_github_pin_in_the_default_history_is_not_a_branch(self) -> None:
        result = self.github_collect(
            heads=[],
            branches=[("feature", self.BRANCH_HEAD)],
            statuses={self.MAIN_HEAD: "ahead", self.BRANCH_HEAD: "ahead"},
        )
        assert len(result.items) == 1
        assert result.items[0].target.branch is None


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
            catalog = refresh_catalog_view.Catalog(
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
                "recipe_tooling.refresh_upstream.items.compare_commits",
                side_effect=lambda _h, _r, _old, _new: comparison("ahead"),
            ):
                return refresh_items.collect_items(
                    catalog,
                    "recipe",
                    http,  # pyright: ignore[reportArgumentType]
                    lambda _p, r: refresh_pinning.resolve_github(http, r),  # pyright: ignore[reportArgumentType]
                )

    def test_newest_release_inside_the_line_is_the_target(self) -> None:
        result = self.collect(OLD)
        assert [i.target.sha for i in result.items] == [self.V066]
        assert result.items[0].target.version == "0.6.6"
        assert len(result.notes) == 1
        assert "example/engine 1.0" in result.notes[0]

    def test_pin_at_the_newest_of_its_line_is_current(self) -> None:
        result = self.collect(self.V066)
        assert not result.items
        assert not result.drifted
        assert len(result.notes) == 1  # reported, never a move

    def test_without_a_line_the_newest_release_wins(self) -> None:
        result = self.collect(OLD, line=None)
        assert [i.target.sha for i in result.items] == [self.V109]
        assert not result.notes

    def test_newer_line_is_reported_once_and_stays_stable(self) -> None:
        result = self.collect(OLD)
        first = refresh_render.issue_body(result)
        assert first.count("Newer release line available") == 1
        assert refresh_render.issue_body(result, first) == first
        assert "Upstream moved" not in refresh_render.issue_body(result, first)
        later = refresh_model.Target(
            "github", self.REPO, "c" * 40, "", "v1.0.10", "1.0.10", "u"
        )
        newest = refresh_model.Target(
            "github", self.REPO, self.V109, "", "v1.0.9", "1.0.9", "u"
        )
        assert refresh_pinning.newer_line_note(
            self.REPO, "0.6", later
        ) == refresh_pinning.newer_line_note(self.REPO, "0.6", newest)

    def test_a_line_is_a_version_prefix_not_a_string_prefix(self) -> None:
        assert not refresh_pinning.in_line(
            (0, 60, 1), refresh_pinning.line_parts("0.6")
        )
        assert refresh_pinning.in_line((0, 6, 6), refresh_pinning.line_parts("0.6"))
        with pytest.raises(RuntimeError):
            refresh_pinning.line_parts("0.6.x")
