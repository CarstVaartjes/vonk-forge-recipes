"""Refresh-upstream tests: apply."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import patch

from refresh_helpers import OWN, failing
from repo_root import ROOT

from recipe_tooling.refresh_upstream import apply as refresh_apply
from recipe_tooling.refresh_upstream import github as refresh
from recipe_tooling.refresh_upstream import model as refresh_model
from recipe_tooling.refresh_upstream import publish as refresh_publish


def _branch_tip(root: Path, branch: str) -> str:
    """The commit a local branch points at, read from the ref files (loose, then packed)."""
    loose = root / ".git/refs/heads" / branch
    if loose.is_file():
        return loose.read_text().strip()
    for line in (root / ".git/packed-refs").read_text().splitlines():
        if line.endswith(f" refs/heads/{branch}"):
            return line.split()[0]
    raise AssertionError(f"no branch {branch}")


class MergeGateTests(unittest.TestCase):
    main_head = "a" * 40

    def test_unchanged_prepared_pr_is_armed_after_publication_recovers(self) -> None:
        assessment = refresh_model.Assessment(
            "recipe", edits={"recipe.json": b"new pin\n"}
        )
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
            patch("recipe_tooling.refresh_upstream.apply.verify", return_value=None),
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
            refresh_apply.apply_edits(root, assessment.edits)
            refresh.git("commit", "-qam", "Prepared refresh", cwd=root)
            prepared = refresh.git("rev-parse", "HEAD", cwd=root)

            def github(*arguments: str) -> str:
                if arguments[:3] == ("pr", "list", "--head"):
                    return json.dumps(
                        [{"number": 421, "state": "OPEN", "body": "", **OWN}]
                    )
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

            with patch("recipe_tooling.refresh_upstream.github.gh", side_effect=github):
                waiting = refresh_publish.Summary()
                refresh_publish.publish_mechanical(root, assessment, waiting)
                assert waiting.armed == []
                assert merges == []
                published = True
                resumed = refresh_publish.Summary()
                refresh_publish.publish_mechanical(root, assessment, resumed)

            assert resumed.armed == [421]
            assert merges == [("pr", "merge", "421", "--auto", "--squash")]
            self.assertEqual(
                refresh.git("rev-parse", "refs/heads/refresh/recipe", cwd=root),
                prepared,
            )

    def test_leftover_edits_of_a_previous_recipe_do_not_block_the_next(self) -> None:
        # Run 41704192720: an "already current" recipe returned with its edits staged,
        # and every later `git switch --detach origin/main` aborted on them.
        assessment = refresh_model.Assessment(
            "recipe", edits={"recipe.json": b"new pin\n"}
        )
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
            patch("recipe_tooling.refresh_upstream.apply.verify", return_value=None),
            patch(
                "recipe_tooling.refresh_upstream.github.gh",
                side_effect=lambda *arguments: (
                    "https://github.test/pull/7"
                    if arguments[:2] == ("pr", "create")
                    else "[]"
                ),
            ),
        ):
            root = Path(temporary)
            refresh.git("init", "-q", "--initial-branch=main", cwd=root)
            refresh.git("remote", "add", "origin", str(root), cwd=root)
            (root / "recipe.json").write_bytes(b"old pin\n")
            refresh.git("add", "recipe.json", cwd=root)
            refresh.git("commit", "-qm", "Base", cwd=root)
            (root / "recipe.json").write_bytes(b"newer main\n")
            (root / "added.json").write_bytes(b"main adds\n")
            refresh.git("add", "added.json", cwd=root)
            refresh.git("commit", "-qam", "Main moves", cwd=root)
            main = refresh.git("rev-parse", "HEAD", cwd=root)
            refresh.git("update-ref", "refs/remotes/origin/main", main, cwd=root)
            refresh.git("switch", "-q", "--detach", "HEAD~1", cwd=root)
            (root / "recipe.json").write_bytes(b"left behind\n")
            refresh.git("add", "-A", cwd=root)
            # An edit that raised before `git add -A` leaves a file untracked at a
            # path the moved main tracks.
            (root / "added.json").write_bytes(b"half applied\n")

            number = refresh_publish.publish_mechanical(
                root, assessment, refresh_publish.Summary()
            )

            self.assertEqual(refresh.git("rev-parse", "HEAD", cwd=root), main)
            assert (root / "added.json").read_bytes() == b"main adds\n"
            self.assertEqual(
                refresh.git("diff", "--name-only", "origin/main", cwd=root),
                "recipe.json",
            )
            assert number == 7

    def test_a_policy_or_contract_change_on_main_defers_a_stale_run(self) -> None:
        assessment = refresh_model.Assessment(
            "recipe", edits={"recipe.json": b"new pin\n"}
        )
        for moved in ("policy/rules.json", "contracts/src/model.py"):
            with self.subTest(moved=moved):

                def git(*arguments: str, moved: str = moved, **_: object) -> str:
                    # the entries of main's changed set under the given pathspecs
                    if arguments[0] == "diff":
                        specs = arguments[arguments.index("--") + 1 :]
                        return "\n".join(p for p in [moved] if p.startswith(specs))
                    return ""

                summary = refresh_publish.Summary()
                summary.started = "a" * 40
                with (
                    patch(
                        "recipe_tooling.refresh_upstream.github.gh", return_value="[]"
                    ),
                    patch(
                        "recipe_tooling.refresh_upstream.github.git", side_effect=git
                    ),
                    patch(
                        "recipe_tooling.refresh_upstream.apply.verify",
                        return_value=None,
                    ),
                ):
                    number = refresh_publish.publish_mechanical(
                        Path(), assessment, summary
                    )

                assert number is None
                assert f"main changed {moved}" in summary.deferred[0]

    def test_arming_after_a_push_sees_the_pushed_files(self) -> None:
        assessment = refresh_model.Assessment(
            "recipe-b", edits={"recipes/b.json": b"new pin\n"}
        )
        summary = refresh_publish.Summary()
        summary.healthy = True
        summary.armed = [419]
        armed = {
            "number": 419,
            **OWN,
            "autoMergeRequest": {"enabledAt": "now"},
            "files": [{"path": "recipes/a.json"}],
        }
        # The listing before the push does not show the overlap; the push added it.
        summary.open_pulls = [
            armed,
            {"number": 421, **OWN, "files": [{"path": "recipes/b.json"}]},
        ]
        after = [
            armed,
            {"number": 421, **OWN, "files": [{"path": "recipes/a.json"}]},
        ]

        def listing(*arguments: str) -> list[dict[str, Any]]:
            if "--head" in arguments:
                return [{"number": 421, "state": "OPEN", "body": "", **OWN}]
            return after

        calls: list[tuple[str, ...]] = []
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch(
                "recipe_tooling.refresh_upstream.github.gh_json", side_effect=listing
            ),
            patch(
                "recipe_tooling.refresh_upstream.github.gh",
                side_effect=lambda *a, **k: calls.append(a) or "",
            ),
            patch("recipe_tooling.refresh_upstream.github.git", return_value=""),
            patch("recipe_tooling.refresh_upstream.apply.verify", return_value=None),
        ):
            refresh_publish.publish_mechanical(Path(temporary), assessment, summary)

        assert not [c for c in calls if c[:2] == ("pr", "merge")]
        assert "#419" in summary.deferred[0]

    def test_open_pulls_are_listed_once_per_run(self) -> None:
        summary = refresh_publish.Summary()
        summary.healthy = True
        pulls = [
            {"number": n, **OWN, "files": [{"path": f"recipes/r{n}.json"}]}
            for n in (421, 422, 423)
        ]
        with (
            patch(
                "recipe_tooling.refresh_upstream.github.gh_json", return_value=pulls
            ) as listing,
            patch("recipe_tooling.refresh_upstream.github.gh"),
        ):
            for n in (421, 422, 423):
                assert refresh_publish.arm_mechanical_pr(n, f"recipe-{n}", summary)

        assert listing.call_count == 1

    def test_existing_armed_pr_does_not_block_a_disjoint_mechanical_merge(self) -> None:
        summary = refresh_publish.Summary()
        summary.healthy = True
        pulls = [
            {
                "number": 419,
                **OWN,
                "autoMergeRequest": {"enabledAt": "now"},
                "files": [{"path": "recipes/a.json"}],
            },
            {"number": 421, **OWN, "files": [{"path": "recipes/b.json"}]},
        ]
        with (
            patch("recipe_tooling.refresh_upstream.github.gh_json", return_value=pulls),
            patch("recipe_tooling.refresh_upstream.github.gh") as gh,
        ):
            allowed = refresh_publish.arm_mechanical_pr(421, "recipe-b", summary)

        assert allowed
        assert summary.armed == [421]
        gh.assert_called_once_with("pr", "merge", "421", "--auto", "--squash")

    def test_shared_model_file_defers_the_second_pr(self) -> None:
        summary = refresh_publish.Summary()
        summary.healthy = True
        pulls = [
            {
                "number": 419,
                **OWN,
                "autoMergeRequest": {"enabledAt": "now"},
                "files": [{"path": "models/shared.json"}],
            },
            {
                "number": 421,
                **OWN,
                "files": [{"path": "recipes/b.json"}, {"path": "models/shared.json"}],
            },
        ]
        with (
            patch("recipe_tooling.refresh_upstream.github.gh_json", return_value=pulls),
            patch("recipe_tooling.refresh_upstream.github.gh") as gh,
        ):
            allowed = refresh_publish.arm_mechanical_pr(421, "recipe-b", summary)

        assert not allowed
        assert summary.armed == []
        assert "shares files with armed PR #419" in summary.deferred[0]
        gh.assert_not_called()

    def test_file_overlap_counts_prs_armed_earlier_in_the_run(self) -> None:
        pulls = [
            {"number": 1, **OWN, "files": [{"path": "models/m.json"}]},
            {"number": 2, **OWN, "files": [{"path": "models/m.json"}]},
            {"number": 3, **OWN, "files": [{"path": "models/m.json"}]},
        ]
        assert refresh_publish.file_overlaps(2, pulls, armed=[1]) == [1]
        assert refresh_publish.file_overlaps(2, pulls) == []

    def test_failed_checks_free_the_file_lane(self) -> None:
        def pulls(rollup: list[dict[str, str]]) -> list[dict[str, object]]:
            return [
                {
                    "number": 1,
                    **OWN,
                    "autoMergeRequest": {"mergeMethod": "SQUASH"},
                    "statusCheckRollup": rollup,
                    "files": [{"path": "models/m.json"}],
                },
                {"number": 2, **OWN, "files": [{"path": "models/m.json"}]},
            ]

        assert (
            refresh_publish.file_overlaps(2, pulls([{"conclusion": "FAILURE"}])) == []
        )
        assert (
            refresh_publish.file_overlaps(2, pulls([{"conclusion": "CANCELLED"}])) == []
        )
        assert refresh_publish.file_overlaps(2, pulls([{"conclusion": "SUCCESS"}])) == [
            1
        ]
        assert refresh_publish.file_overlaps(2, pulls([{"status": "IN_PROGRESS"}])) == [
            1
        ]

    def test_pending_publication_keeps_prepared_pr_unarmed(self) -> None:
        summary = refresh_publish.Summary()
        with (
            patch("recipe_tooling.refresh_upstream.github.gh_json", side_effect=[[]]),
            patch(
                "recipe_tooling.refresh_upstream.github.gh", return_value=self.main_head
            ) as gh,
        ):
            allowed = refresh_publish.arm_mechanical_pr(421, "recipe", summary)

        assert not allowed
        assert summary.armed == []
        assert "did not succeed" in summary.deferred[0]
        assert not any("merge" in call.args for call in gh.call_args_list)

    def test_successful_publication_arms_every_green_pr_in_one_run(self) -> None:
        summary = refresh_publish.Summary()
        runs = [
            {
                "headSha": self.main_head,
                "status": "completed",
                "conclusion": "success",
            }
        ]
        pulls = [
            {"number": n, **OWN, "files": [{"path": f"recipes/r{n}.json"}]}
            for n in (421, 422, 423)
        ]
        with (
            patch(
                "recipe_tooling.refresh_upstream.github.gh_json",
                side_effect=[runs, pulls, pulls, pulls],
            ),
            patch(
                "recipe_tooling.refresh_upstream.github.gh", return_value=self.main_head
            ) as gh,
        ):
            results = [
                refresh_publish.arm_mechanical_pr(n, f"recipe-{n}", summary)
                for n in (421, 422, 423)
            ]

        assert results == [True, True, True]
        assert summary.armed == [421, 422, 423]
        merges = [call for call in gh.call_args_list if "merge" in call.args]
        assert len(merges) == 3

    def test_inflight_publication_does_not_block_but_failure_does(self) -> None:
        done = {"status": "completed", "conclusion": "success"}
        inflight = {"status": "in_progress", "conclusion": ""}
        cancelled = {"status": "completed", "conclusion": "cancelled"}
        failed = {"status": "completed", "conclusion": "failure"}
        assert refresh_publish.publication_healthy([inflight, cancelled, done])
        assert not refresh_publish.publication_healthy([inflight, failed, done])
        assert not refresh_publish.publication_healthy([])

    def test_open_refresh_prs_are_armed_up_front(self) -> None:
        summary = refresh_publish.Summary()
        summary.healthy = True
        listing = [
            {"number": 1, "headRefName": "refresh/a", "autoMergeRequest": None, **OWN},
            {
                "number": 2,
                "headRefName": "refresh/b",
                "autoMergeRequest": {"x": 1},
                **OWN,
            },
            {"number": 3, "headRefName": "codex/c", "autoMergeRequest": None, **OWN},
        ]
        calls = []

        def gh_json(*arguments: str):
            calls.append(arguments)
            if "autoMergeRequest,files" in arguments[-1]:
                return [{"number": 1, "files": [], **OWN}]
            return listing

        with (
            patch(
                "recipe_tooling.refresh_upstream.github.gh_json", side_effect=gh_json
            ),
            patch("recipe_tooling.refresh_upstream.github.gh") as gh,
        ):
            refresh_publish.arm_open_mechanical_prs(summary)

        assert summary.armed == [1]
        gh.assert_called_once_with("pr", "merge", "1", "--auto", "--squash")

    def test_prepare_bound_counts_pushed_prs_only(self) -> None:
        assert refresh_model.MAX_MECHANICAL_PER_RUN == 6
        summary = refresh_publish.Summary()
        assert summary.prepared == 0

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
        assert refresh_publish.refresh_is_spinning(burst, 99, now)
        # One run that just finished (the one ahead in the queue) does not skip us.
        assert not refresh_publish.refresh_is_spinning([run(1)], 99, now)
        assert not refresh_publish.refresh_is_spinning(
            [run(m) for m in (40, 50, 60, 70)], 99, now
        )
        # Active, cancelled and the current run never count.
        mixed = [
            run(1, status="in_progress"),
            run(2, conclusion="cancelled"),
            run(3, databaseId=99),
            run(4),
        ]
        assert not refresh_publish.refresh_is_spinning(mixed + burst[:2], 99, now)

    def test_chain_guard_command(self) -> None:
        def decide(event: str, conclusion: str, runs: list[dict[str, object]]) -> str:
            with tempfile.NamedTemporaryFile("w+", suffix=".out") as output:
                arguments = argparse.Namespace(
                    event=event, run_id=99, upstream_conclusion=conclusion
                )
                with (
                    patch(
                        "recipe_tooling.refresh_upstream.github.gh_json",
                        return_value=runs,
                    ),
                    patch.dict(os.environ, {"GITHUB_OUTPUT": output.name}),
                ):
                    refresh_publish.command_guard(arguments)
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
        assert decide("schedule", "", recent) == "run=true"
        assert decide("workflow_dispatch", "", recent) == "run=true"
        assert decide("workflow_run", "success", []) == "run=true"
        assert decide("workflow_run", "failure", []) == "run=false"
        assert decide("workflow_run", "success", recent) == "run=false"


class SelfHealingTests(unittest.TestCase):
    def test_check_state(self) -> None:
        assert refresh_publish.check_state(None) == "none"
        assert refresh_publish.check_state(failing()) == "failed"
        assert (
            refresh_publish.check_state(
                [{"conclusion": "CANCELLED"}, {"conclusion": "SUCCESS"}]
            )
            == "cancelled"
        )
        assert (
            refresh_publish.check_state([{"status": "IN_PROGRESS", "conclusion": ""}])
            == "pending"
        )
        assert (
            refresh_publish.check_state(
                [{"conclusion": "SUCCESS"}, {"conclusion": "SKIPPED"}]
            )
            == "passed"
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
            "tools/added-lines",
            "tools/check-test-git",
            "tools/check-final-stage-ldd",
            "tools/check-serial-reads",
            "tools/check-nccl-ib-hca",
            "tools/generate-contract-examples",
            "tools/generate-contract-schemas",
            "tools/build-release-bundle",
            "tools/prebuilt-images",  # reads the platform pin, not refresh data
            "tools/check-added-file-size",  # checks a PR diff, not refresh data
            "tools/contract-version",  # publication stamps its version, no refresh data
        }
        covered = {part for _, cmd in refresh_apply.verify_steps() for part in cmd}
        assert sorted(t for t in ci - skipped if t not in covered) == []
        assert "check" in dict(refresh_apply.verify_steps())["kit-declared pins"]

    def republish(
        self, rollup, *, pin: bytes = b"new pin\n", main_change: str | None = None
    ) -> tuple[list[Any], str, str, Any, bool]:
        """A refresh PR prepared earlier; this run assesses ``pin`` against main."""
        assessment = refresh_model.Assessment("recipe", edits={"recipe.json": pin})
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
            refresh_apply.apply_edits(root, {"recipe.json": b"new pin\n"})
            refresh.git("commit", "-qam", "Prepared refresh", cwd=root)
            before = refresh.git("rev-parse", "HEAD", cwd=root)
            refresh.git("switch", "-q", "main", cwd=root)
            if main_change:  # another PR merged meanwhile
                (root / main_change).parent.mkdir(parents=True, exist_ok=True)
                (root / main_change).write_bytes(b"changed on main\n")
                refresh.git("add", main_change, cwd=root)
                refresh.git("commit", "-qm", f"Change {main_change}", cwd=root)
            main = refresh.git("rev-parse", "HEAD", cwd=root)
            refresh.git("update-ref", "refs/remotes/origin/main", main, cwd=root)

            def github(*arguments: str, check: bool = True) -> str:
                calls.append(arguments)
                if arguments[:3] == ("pr", "list", "--head"):
                    return json.dumps(
                        [
                            {
                                "number": 9,
                                **OWN,
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
                patch("recipe_tooling.refresh_upstream.github.gh", side_effect=github),
                patch(
                    "recipe_tooling.refresh_upstream.apply.verify", return_value=None
                ),
            ):
                refresh_publish.publish_mechanical(
                    root, assessment, refresh_publish.Summary()
                )
            after = _branch_tip(root, "refresh/recipe")
            forward = (
                refresh_apply.run(
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
                self.assertEqual(  # forward: the old head and main are its parents
                    refresh.git("rev-parse", f"{after}^1", f"{after}^2", cwd=root),
                    f"{before}\n{main}",
                )
        return calls, before, after, assessment, forward

    def test_failed_checks_on_the_carried_refresh_are_reported_not_pushed(
        self,
    ) -> None:
        calls, before, after, assessment, _ = self.republish(failing())
        assert before == after
        assert not [c for c in calls if c[:2] in {("pr", "edit"), ("pr", "merge")}]
        assert "#9" in assessment.reasons[0]
        assert "BOOM: kit lock is stale" in "".join(assessment.evidence)

    def test_cancelled_checks_are_reported_not_pushed(self) -> None:
        _, before, after, assessment, _ = self.republish([{"conclusion": "CANCELLED"}])
        assert before == after
        assert "cancelled" in assessment.reasons[0]

    def test_passing_unchanged_pr_is_left_alone(self) -> None:
        calls, before, after, _, _ = self.republish([{"conclusion": "SUCCESS"}])
        assert before == after
        assert not [c for c in calls if c[:2] == ("pr", "edit")]

    def test_main_moving_elsewhere_pushes_nothing(self) -> None:
        _, before, after, assessment, _ = self.republish(
            [{"conclusion": "SUCCESS"}], main_change="other.json"
        )
        assert before == after
        assert assessment.reasons == []

    def test_a_newer_refresh_moves_the_branch_forward_without_force(self) -> None:
        for rollup in ([{"conclusion": "SUCCESS"}], failing()):
            with self.subTest(rollup=rollup):
                calls, before, after, assessment, forward = self.republish(
                    rollup, pin=b"newer pin\n", main_change="other.json"
                )
                assert before != after
                assert forward, "the push must be a fast-forward"
                assert assessment.reasons == []
                assert [c for c in calls if c[:2] == ("pr", "edit")]

    def test_main_touching_the_refreshed_file_moves_the_branch_forward(self) -> None:
        _, before, after, assessment, forward = self.republish(
            [{"conclusion": "SUCCESS"}], main_change="recipe.json"
        )
        assert before != after
        assert forward
        assert assessment.reasons == []

    def test_a_failed_pr_moves_forward_once_main_changes_its_checks(self) -> None:
        for rollup in (failing(), [{"conclusion": "CANCELLED"}]):
            with self.subTest(rollup=rollup):
                _, before, after, assessment, forward = self.republish(
                    rollup, main_change="tests/test_fixed.py"
                )
                assert before != after
                assert forward
                assert assessment.reasons == []

    def test_refresh_branches_are_never_force_pushed_or_updated_in_place(
        self,
    ) -> None:
        source = "\n".join(
            path.read_text()
            for path in sorted(
                (ROOT / "src/recipe_tooling/refresh_upstream").glob("*.py")
            )
        )
        pushes = re.findall(r'"push",[^)]*\)', source)
        assert pushes
        assert not [p for p in pushes if "force" in p], pushes
        assert '"update-branch"' not in source

    def test_a_refreshed_proven_recipe_rewrites_its_guard_entry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / refresh_model.PROVEN_BYTES).parent.mkdir(parents=True)
            (root / refresh_model.PROVEN_BYTES).write_text(
                json.dumps({"vonk-forge/a": "0" * 64, "vonk-forge/b": "1" * 64})
            )
            edits: dict[str, bytes | None] = {"recipes/a.json": b"moved\n"}
            refresh_publish.follow_proven_bytes(root, edits)
            guard = json.loads(edits[refresh_model.PROVEN_BYTES] or b"")
            assert guard["vonk-forge/a"] == hashlib.sha256(b"moved\n").hexdigest()
            assert guard["vonk-forge/b"] == "1" * 64
            unproven: dict[str, bytes | None] = {"recipes/c.json": b"moved\n"}
            refresh_publish.follow_proven_bytes(root, unproven)
            assert refresh_model.PROVEN_BYTES not in unproven

    def test_failure_evidence_has_the_log_tail(self) -> None:
        with patch(
            "recipe_tooling.refresh_upstream.github.gh", return_value="x\nBOOM: stale\n"
        ):
            text = refresh_publish.failed_check_evidence(failing())
        assert "BOOM: stale" in text
        assert "actions/runs/7" in text
