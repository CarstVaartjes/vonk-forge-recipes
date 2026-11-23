"""Refresh-upstream tests: publish."""

from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from refresh_helpers import NEW, OLD, OWN
from repo_root import ROOT

from recipe_tooling.refresh_upstream import catalog_view as refresh_catalog_view
from recipe_tooling.refresh_upstream import model as refresh_model
from recipe_tooling.refresh_upstream import publish as refresh_publish
from recipe_tooling.refresh_upstream import render as refresh_render


class OpenPullRequestTests(unittest.TestCase):
    footprint = (
        {"recipes/a.json", "models/m-1.json", "display/models/m-1.json"},
        ["adapters/llm/a/"],
    )

    def test_foreign_pr_touching_recipe_model_or_adapter_blocks(self) -> None:
        for name in (
            "recipes/a.json",
            "models/m-1.json",
            "display/models/m-1.json",
            "adapters/llm/a/Dockerfile",
        ):
            with self.subTest(name=name):
                assert refresh_publish.blocking_prs(
                    self.footprint, {7: {name, "README.md"}}
                ) == [7]

    def test_unrelated_pr_does_not_block(self) -> None:
        opens = {7: {"recipes/b.json", "adapters/llm/ab/Dockerfile", "README.md"}}
        assert refresh_publish.blocking_prs(self.footprint, opens) == []

    def test_footprint_covers_recipe_models_and_context(self) -> None:
        catalog = refresh_catalog_view.Catalog(ROOT)
        catalog.recipes["a"] = {
            "models": [{"model": {"slug": "m-1"}}],
            "execution": {"build": {"context": {"path": "adapters/llm/a"}}},
        }
        assert refresh_publish.recipe_footprint(catalog, "a") == self.footprint


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
        assert (
            refresh_publish.sharing_groups(self.catalog())  # pyright: ignore[reportArgumentType]
            == {"a": ["a", "b"], "b": ["a", "b"]}
        )

    def assessment(self, recipe_id: str, **kwargs):
        item = refresh_model.Item(
            "source",
            "github",
            "o/r",
            OLD,
            refresh_model.Target("github", "o/r", NEW, "d", None, None, "u"),
        )
        return refresh_model.Assessment(recipe_id, items=[item], **kwargs)

    def test_group_is_one_assessment_with_all_edits(self) -> None:
        assessed = {
            "a": self.assessment("a", edits={"recipes/a.json": b"1", "lock": b"L"}),
            "b": self.assessment("b", edits={"recipes/b.json": b"2", "lock": b"L"}),
        }
        combined = refresh_publish.combine_group(["a", "b"], assessed)
        assert not isinstance(combined, str)
        assert combined.recipe_id == "a"
        assert set(combined.edits) == {"recipes/a.json", "recipes/b.json", "lock"}
        assert "a, b" in refresh_render.pr_body(combined)

    def test_group_waits_when_a_sibling_is_not_mechanical(self) -> None:
        assessed = {
            "a": self.assessment("a", edits={"lock": b"L"}),
            "b": self.assessment("b", reasons=["needs review"]),
        }
        assert isinstance(refresh_publish.combine_group(["a", "b"], assessed), str)
        assert isinstance(
            refresh_publish.combine_group(["a", "b"], {"a": assessed["a"]}), str
        )

    def test_a_current_sibling_does_not_hold_the_group_back(self) -> None:
        # Since 2026-10-19 the Nemotron Lightning DSpark Model moved for three recipes
        # whose contexts they share with current siblings; every run deferred them.
        current = refresh_model.Assessment("a")
        assessed = {
            "a": current,
            "b": self.assessment("b", edits={"recipes/b.json": b"2"}),
            "c": self.assessment("c", edits={"recipes/c.json": b"3"}),
        }
        combined = refresh_publish.combine_group(["a", "b", "c"], assessed)
        assert not isinstance(combined, str), combined
        assert combined.recipe_id == "a"
        assert set(combined.edits) == {"recipes/b.json", "recipes/c.json"}
        assert combined.label == "b, c"
        assert isinstance(refresh_publish.combine_group(["a"], {"a": current}), str)

    def test_a_group_pr_survives_a_run_that_did_not_republish_it(self) -> None:
        # Lead "a" is current; "b" still moves but this run deferred it (cap, open PR).
        pulls = [{"number": 5, "headRefName": "refresh/a", "body": "", **OWN}]
        assessed = {
            "a": refresh_model.Assessment("a"),
            "b": self.assessment("b", edits={"recipes/b.json": b"2"}),
        }
        calls: list[tuple[str, ...]] = []
        with (
            patch(
                "recipe_tooling.refresh_upstream.github.gh_json",
                side_effect=[pulls, [], []],
            ),
            patch(
                "recipe_tooling.refresh_upstream.github.gh",
                side_effect=lambda *a, **k: calls.append(a),
            ),
        ):
            refresh_publish.close_stale_refresh_prs(
                assessed,
                refresh_publish.Summary(),
                {"a": ["a", "b"], "b": ["a", "b"]},
                set(),
            )
        assert not [c for c in calls if c[:2] == ("pr", "close")]

    def test_a_group_issue_keeps_the_failed_check_evidence(self) -> None:
        # A current sibling with evidence of its own must not shift the slice that
        # carries publishing's "Failed checks" section into the lead's issue.
        current = refresh_model.Assessment("a", evidence=["a: newest is shown"])
        moving = self.assessment("b", edits={"recipes/b.json": b"2"})
        moving.evidence.append("b: model moved")
        combined = refresh_publish.combine_group(
            ["a", "b"], {"a": current, "b": moving}
        )
        assert not isinstance(combined, str), combined

        def publish(_root, assessment, _summary):
            assessment.reasons.append("refresh PR #5 checks failed")
            assessment.evidence.append("### Failed checks of PR #5")
            return 5

        with patch(
            "recipe_tooling.refresh_upstream.publish.publish_mechanical",
            side_effect=publish,
        ):
            number = refresh_publish.publish_group(
                Path("."), combined, moving, refresh_publish.Summary()
            )
        assert number == 5
        assert "### Failed checks of PR #5" in moving.evidence
        assert "refresh PR #5 checks failed" in moving.reasons

    def test_group_rejects_conflicting_shared_edits(self) -> None:
        assessed = {
            "a": self.assessment("a", edits={"lock": b"L1"}),
            "b": self.assessment("b", edits={"lock": b"L2"}),
        }
        assert isinstance(refresh_publish.combine_group(["a", "b"], assessed), str)


class ReviewIssueLifecycleTests(unittest.TestCase):
    """One issue per recipe for life, showing the current assessment (#1282, #1283)."""

    TARGET = refresh_model.Target("github", "o/r", NEW, "2026-09-30", None, None, "u")

    def assessment(self, recipe_id: str, *, item=False, reason="", unreachable=""):
        result = refresh_model.Assessment(recipe_id)
        if item:
            result.items.append(
                refresh_model.Item("source", "github", "o/r", OLD, self.TARGET)
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
            patch(
                "recipe_tooling.refresh_upstream.github.gh_json", side_effect=gh_json
            ),
            patch("recipe_tooling.refresh_upstream.github.gh", side_effect=gh),
        ):
            call(*arguments)
        return calls

    def test_only_current_gone_or_handled_recipes_close(self) -> None:
        names = ["current", "deferred", "handled", "waiting", "gone"]
        issues = {
            "open": [
                {"number": n, "title": refresh_render.issue_title(name)}
                for n, name in enumerate(names, start=1)
            ]
        }
        assessed = {
            "current": self.assessment("current"),
            "deferred": self.assessment("deferred", item=True),
            "handled": self.assessment("handled", item=True),
            "waiting": self.assessment("waiting", unreachable="x not found"),
        }
        summary = refresh_publish.Summary()
        calls = self.run_gh(
            issues, refresh_publish.close_current_issues, assessed, {"handled"}, summary
        )
        closed = [c[2] for c in calls if c[:2] == ("issue", "close")]
        assert closed == ["1", "3", "5"]

    def test_an_older_duplicate_issue_closes_in_favour_of_the_newest(self) -> None:
        # #1563/#1714: two open issues for one recipe; only the newest is kept current.
        title = refresh_render.issue_title("drifted")
        issues = {
            "open": [
                {"number": 1563, "title": title},
                {"number": 1714, "title": title},
            ]
        }
        assessed = {"drifted": self.assessment("drifted", item=True, reason="x")}
        calls = self.run_gh(
            issues,
            refresh_publish.close_current_issues,
            assessed,
            set(),
            refresh_publish.Summary(),
        )
        closes = [c for c in calls if c[:2] == ("issue", "close")]
        assert [c[2] for c in closes] == ["1563"]
        assert "#1714" in closes[0][-1]

    def test_a_recipe_that_needs_review_again_reopens_its_issue(self) -> None:
        title = refresh_render.issue_title("recipe")
        issues = {
            "open": [],
            "closed": [
                {"number": 7, "title": title, "body": "old"},
                {"number": 9, "title": title, "body": "older"},
            ],
        }
        result = self.assessment("recipe", item=True, reason="run.sh changed")
        calls = self.run_gh(
            issues, refresh_publish.sync_issue, result, refresh_publish.Summary()
        )
        assert calls[0] == ("issue", "reopen", "9")
        assert calls[1][:3] == ("issue", "edit", "9")
        assert not [c for c in calls if c[:2] == ("issue", "create")]

    def test_first_review_creates_one_issue(self) -> None:
        result = self.assessment("recipe", item=True, reason="run.sh changed")
        calls = self.run_gh(
            {"open": [], "closed": []},
            refresh_publish.sync_issue,
            result,
            refresh_publish.Summary(),
        )
        assert [c[:2] for c in calls] == [("issue", "create")]

    def test_a_waiting_recipe_updates_only_an_open_issue(self) -> None:
        result = self.assessment("recipe", unreachable="commit not on GitHub")
        title = refresh_render.issue_title("recipe")
        none = self.run_gh(
            {"open": [], "closed": [{"number": 3, "title": title, "body": ""}]},
            refresh_publish.sync_issue,
            result,
            refresh_publish.Summary(),
            False,
        )
        assert none == []
        calls = self.run_gh(
            {"open": [{"number": 4, "title": title, "body": "stale reason"}]},
            refresh_publish.sync_issue,
            result,
            refresh_publish.Summary(),
            False,
        )
        assert calls[0][:3] == ("issue", "edit", "4")
        body = calls[0][4]
        assert "waits for an upstream" in body
        assert "## Why it waits" in body
        assert "- commit not on GitHub" in body

    def test_memory_estimate_report_keeps_one_issue_current(self) -> None:
        title = refresh_model.MEMORY_ESTIMATE_TITLE
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.md"
            report.write_text("- recipe: peak understates need 1\n")
            created = self.run_gh(
                {"open": [], "closed": []},
                refresh_publish.command_memory_estimate_issue,
                report,
            )
            reopened = self.run_gh(
                {"open": [], "closed": [{"number": 5, "title": title, "body": ""}]},
                refresh_publish.command_memory_estimate_issue,
                report,
            )
            report.write_text("")
            closed = self.run_gh(
                {"open": [{"number": 5, "title": title, "body": "old"}]},
                refresh_publish.command_memory_estimate_issue,
                report,
            )
        assert [c[:2] for c in created if c[0] == "issue"] == [("issue", "create")]
        assert refresh_model.LABEL_MEMORY_ESTIMATE in created[-1]
        assert [c[:3] for c in reopened if c[0] == "issue"] == [
            ("issue", "reopen", "5"),
            ("issue", "edit", "5"),
        ]
        assert [c[:3] for c in closed if c[0] == "issue"] == [("issue", "close", "5")]


class StalePullRequestTests(unittest.TestCase):
    def close(self, assessed, groups=None, published=None, issues=()):
        pulls = [
            {"number": 1, "headRefName": "refresh/a", "body": "", **OWN},
            {"number": 2, "headRefName": "refresh/b", "body": "", **OWN},
        ]
        closed: list[tuple[str, ...]] = []

        def github(*arguments: str, check: bool = True) -> str:
            if arguments[0] == "pr" and arguments[1] == "close":
                closed.append(arguments)
            return ""

        def github_json(*arguments: str):
            return pulls if arguments[0] == "pr" else list(issues)

        with (
            patch("recipe_tooling.refresh_upstream.github.gh", side_effect=github),
            patch(
                "recipe_tooling.refresh_upstream.github.gh_json",
                side_effect=github_json,
            ),
        ):
            refresh_publish.close_stale_refresh_prs(
                assessed, refresh_publish.Summary(), groups, published
            )
        return {c[2]: c[-1] for c in closed}

    def item(self):
        return refresh_model.Item(
            "source",
            "github",
            "o/r",
            OLD,
            refresh_model.Target("github", "o/r", NEW, "d", None, None, "u"),
        )

    def test_current_gone_and_review_prs_are_closed_with_a_reason(self) -> None:
        assessed = {
            "a": refresh_model.Assessment("a"),
            "b": refresh_model.Assessment("b", items=[self.item()], reasons=["why"]),
        }
        closed = self.close(
            assessed, issues=[{"number": 77, "title": refresh_render.issue_title("b")}]
        )
        assert "current again" in closed["1"]
        assert "#77" in closed["2"]
        assert "no longer exists" in self.close({})["1"]

    def test_mechanical_published_and_unreachable_prs_stay(self) -> None:
        mech = refresh_model.Assessment("a", items=[self.item()])
        down = refresh_model.Assessment("b", items=[self.item()], unreachable=["x"])
        assert self.close({"a": mech, "b": down}) == {}
        review = refresh_model.Assessment("a", items=[self.item()], reasons=["ci"])
        assert self.close({"a": review, "b": down}, published={"a"}) == {}

    def test_a_sibling_pr_of_a_shared_context_is_closed(self) -> None:
        mech = {
            "a": refresh_model.Assessment("a", items=[self.item()]),
            "b": refresh_model.Assessment("b", items=[self.item()]),
        }
        closed = self.close(mech, groups={"a": ["a", "b"], "b": ["a", "b"]})
        assert list(closed) == ["2"]
        assert "shares its build context" in closed["2"]
