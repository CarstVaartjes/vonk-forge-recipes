"""Refresh-upstream tests: cli."""

from __future__ import annotations

import argparse
import contextlib
import os
import subprocess
import sys
import types
import unittest
from typing import Any
from unittest.mock import Mock, patch

import pytest
from refresh_helpers import NEW, OLD, OWN, PATCH_HOME
from repo_root import ROOT

from recipe_tooling.refresh_upstream import apply as refresh_apply
from recipe_tooling.refresh_upstream import assessment as refresh_assessment
from recipe_tooling.refresh_upstream import catalog_view as refresh_catalog_view
from recipe_tooling.refresh_upstream import cli as refresh_cli
from recipe_tooling.refresh_upstream import github as refresh
from recipe_tooling.refresh_upstream import github as refresh_github
from recipe_tooling.refresh_upstream import model as refresh_model
from recipe_tooling.refresh_upstream import publish as refresh_publish
from recipe_tooling.upstream_drift import embedded as drift_embedded
from recipe_tooling.upstream_drift import heads as drift_heads
from recipe_tooling.upstream_drift import inventory as drift_inventory


class AppCommandBoundaryTests(unittest.TestCase):
    def test_renewed_app_token_reaches_both_gh_and_git(self) -> None:

        auth = Mock()
        auth.environment.side_effect = [
            {"GH_TOKEN": "first-app-token"},
            {"GH_TOKEN": "renewed-app-token"},
        ]
        completed = __import__("subprocess").CompletedProcess([], 0, "", "")
        with (
            patch.object(refresh_apply, "APP_AUTH", auth),
            patch.dict(os.environ, {"GH_TOKEN": "actions-token"}),
            patch.object(subprocess, "run", return_value=completed) as run,
        ):
            refresh_github.gh("pr", "list")
            refresh.git("push", "origin", cwd=ROOT)
        assert run.call_args_list[0].kwargs["env"]["GH_TOKEN"] == "first-app-token"
        assert run.call_args_list[1].kwargs["env"]["GH_TOKEN"] == "renewed-app-token"

    def test_missing_app_key_fails_before_recipe_work(self) -> None:
        with (
            patch.object(sys, "argv", ["refresh-upstream", "--github-app"]),
            patch.dict(os.environ, {}, clear=True),
            patch.object(refresh_cli, "command_run") as command,
            patch.object(refresh_apply, "APP_AUTH", None),
            pytest.raises(RuntimeError, match="not configured"),
        ):
            refresh_cli.main()
        command.assert_not_called()


class PerRecipeFaultTests(unittest.TestCase):
    """One recipe's unexpected error is that recipe's failure; the others still run."""

    def drifted(self, recipe_id: str, **kwargs):
        item = refresh_model.Item(
            "source",
            "github",
            "o/r",
            OLD,
            refresh_model.Target("github", "o/r", NEW, "d", None, None, "u"),
        )
        return refresh_model.Assessment(recipe_id, items=[item], **kwargs)

    def run_refresh(self, assessments, *, assess_fault=None, **patches):
        """Run command_run over the assessments; return (exit code, summary).

        GitHub is never reached: ``gh`` and ``sync_issue`` are patched unless a test
        passes its own; ``self.synced`` lists the recipes whose issue was synced.
        """
        written: list[Any] = []
        self.synced: list[str] = []
        patches = {
            "gh": lambda *_arguments, **_keywords: "",
            "sync_issue": lambda result, *_rest, **_keywords: self.synced.append(
                result.recipe_id
            ),
            **patches,
        }

        class FakeAssessor:
            def __init__(self, *_arguments) -> None:
                pass

            def assess(self, recipe_id: str):
                if recipe_id == assess_fault:
                    raise ValueError("Expecting value: line 1 column 1 (char 0)")
                return assessments[recipe_id]

        def read_optional(source, action, *_arguments, **_keywords):
            if source == "catalog assessment":
                return action()
            return None

        catalog = types.SimpleNamespace(recipes=dict.fromkeys(assessments))
        arguments = argparse.Namespace(cache_dir=None, dry_run=False)
        with (
            contextlib.ExitStack() as stack,
            patch.object(refresh_github, "git", return_value="head"),
            patch.object(refresh_cli, "read_optional", side_effect=read_optional),
            patch.object(refresh_catalog_view.Catalog, "load", return_value=catalog),
            patch.object(drift_inventory, "load_manifest", return_value=({}, {})),
            patch.object(drift_inventory, "discover_watches"),
            patch.object(
                drift_embedded, "embedded_inventory", return_value={"coverage": {}}
            ),
            patch.object(drift_heads, "observe_embedded_heads"),
            patch.object(refresh_assessment, "Assessor", FakeAssessor),
            patch.object(refresh_github, "ensure_labels"),
            patch.object(refresh_publish, "foreign_open_prs", return_value={}),
            patch.object(refresh_publish, "sharing_groups", return_value={}),
            patch.object(refresh_publish, "recipe_footprint", return_value=(set(), [])),
            patch.object(refresh_publish, "close_current_issues"),
            patch.object(refresh_publish, "close_stale_refresh_prs"),
            patch.object(
                refresh_publish,
                "write_summary",
                side_effect=lambda summary, *_rest: written.append(summary),
            ),
            patch("traceback.print_exc"),
        ):
            for name, fn in patches.items():
                stack.enter_context(
                    patch.object(PATCH_HOME[name], name, side_effect=fn)
                )
            code = refresh_cli.command_run(arguments)
        (summary,) = written
        return code, summary

    def test_an_error_in_a_recipe_step_does_not_starve_the_next(self) -> None:
        published: list[str] = []

        def publish(_root, assessment, _summary):
            if assessment.recipe_id == "a":
                raise ValueError("Expecting value: line 1 column 1 (char 0)")
            published.append(assessment.recipe_id)
            return 7

        code, summary = self.run_refresh(
            {"a": self.drifted("a"), "b": self.drifted("b")},
            publish_mechanical=publish,
        )
        assert code == 1
        assert published == ["b"]
        assert len(summary.failures) == 1
        assert summary.failures[0].startswith("a: ValueError: Expecting value")

    def test_an_over_limit_recipe_keeps_its_open_refresh_pr(self) -> None:
        synced: list[tuple[str, bool, list[str]]] = []
        calls: list[tuple[str, ...]] = []

        def sync(assessment, _summary, *, create=True):
            synced.append((assessment.recipe_id, create, list(assessment.reasons)))

        assessments = {"a": self.drifted("a")}
        with patch.object(refresh_model, "MAX_MECHANICAL_PER_RUN", 0):
            _, summary = self.run_refresh(assessments, sync_issue=sync)

        assert [(r, c) for r, c, _ in synced] == [("a", False)]
        assert "over this run's limit" in synced[0][2][0]
        assert assessments["a"].prepared  # the result itself is untouched
        pulls = [{"number": 5, "headRefName": "refresh/a", "body": "", **OWN}]
        with (
            patch.object(refresh_github, "gh_json", side_effect=[pulls, [], []]),
            patch.object(
                refresh_github, "gh", side_effect=lambda *a, **k: calls.append(a)
            ),
        ):
            refresh_publish.close_stale_refresh_prs(assessments, summary)
        assert not [c for c in calls if c[:2] == ("pr", "close")]

    def test_an_error_syncing_an_unreachable_recipe_does_not_starve_the_next(
        self,
    ) -> None:
        published: list[str] = []

        def sync(assessment, _summary, **_keywords):
            raise ValueError("Expecting value: line 1 column 1 (char 0)")

        def publish(_root, assessment, _summary):
            published.append(assessment.recipe_id)
            return 7

        code, summary = self.run_refresh(
            {
                "a": refresh_model.Assessment("a", unreachable=["x not found"]),
                "b": self.drifted("b"),
            },
            sync_issue=sync,
            publish_mechanical=publish,
        )
        assert code == 1
        assert published == ["b"]
        assert summary.failures[0].startswith("a: ValueError:")

    def test_an_error_assessing_one_recipe_does_not_stop_the_others(self) -> None:
        published: list[str] = []

        def publish(_root, assessment, _summary):
            published.append(assessment.recipe_id)
            return 7

        code, summary = self.run_refresh(
            {"a": self.drifted("a"), "b": self.drifted("b")},
            assess_fault="a",
            publish_mechanical=publish,
        )
        assert code == 1
        assert published == ["b"]
        assert self.synced == []  # the failed recipe's issue is untouched
        assert summary.unreachable == []
        assert summary.failures[0].startswith("a: ValueError: Expecting value")


class AppKeyLifetimeTests(unittest.TestCase):
    def test_signing_key_is_removed_before_recipe_subprocesses_and_auth_is_closed(
        self,
    ) -> None:

        auth = Mock()

        def command(arguments):
            assert "REFRESH_APP_PRIVATE_KEY" not in os.environ
            return 0

        with (
            patch.object(sys, "argv", ["refresh-upstream", "--github-app"]),
            patch.dict(
                os.environ,
                {
                    "REFRESH_APP_CLIENT_ID": "client",
                    "REFRESH_APP_PRIVATE_KEY": "private-signing-key",
                    "GITHUB_REPOSITORY": "owner/repo",
                },
            ),
            patch.object(refresh_cli, "RefreshAppAuth", return_value=auth),
            patch.object(refresh_cli, "command_run", side_effect=command),
            patch.object(refresh_cli, "materialize_or_exit"),
            patch.object(refresh_apply, "APP_AUTH", None),
        ):
            assert refresh_cli.main() == 0
            auth.close.assert_called_once()
            assert refresh_apply.APP_AUTH is None


class AppVerificationTests(unittest.TestCase):
    def test_live_verification_does_not_start_recipe_work(self) -> None:

        auth = Mock()
        with (
            patch.object(
                sys, "argv", ["refresh-upstream", "--github-app", "--verify-app"]
            ),
            patch.dict(
                os.environ,
                {
                    "REFRESH_APP_CLIENT_ID": "client",
                    "REFRESH_APP_PRIVATE_KEY": "key",
                    "GITHUB_REPOSITORY": "owner/repo",
                },
            ),
            patch.object(refresh_cli, "RefreshAppAuth", return_value=auth),
            patch.object(refresh_apply, "APP_AUTH", None),
            patch.object(
                refresh_github,
                "gh_json",
                return_value={"repositories": [{"full_name": "owner/repo"}]},
            ),
            patch.object(refresh_cli, "command_run") as command,
        ):
            assert refresh_cli.main() == 0
            command.assert_not_called()
            auth.close.assert_called_once()

    def test_broader_token_scope_is_refused(self) -> None:

        auth = Mock()
        with (
            patch.object(
                sys, "argv", ["refresh-upstream", "--github-app", "--verify-app"]
            ),
            patch.dict(
                os.environ,
                {
                    "REFRESH_APP_CLIENT_ID": "client",
                    "REFRESH_APP_PRIVATE_KEY": "key",
                    "GITHUB_REPOSITORY": "owner/repo",
                },
            ),
            patch.object(refresh_cli, "RefreshAppAuth", return_value=auth),
            patch.object(refresh_apply, "APP_AUTH", None),
            patch.object(
                refresh_github,
                "gh_json",
                return_value={
                    "repositories": [
                        {"full_name": "owner/repo"},
                        {"full_name": "owner/other"},
                    ]
                },
            ),
            pytest.raises(RuntimeError, match="scope differs"),
        ):
            refresh_cli.main()
        auth.close.assert_called_once()
