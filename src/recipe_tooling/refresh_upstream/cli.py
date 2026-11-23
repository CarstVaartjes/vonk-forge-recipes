"""Keep recipes on the pinning rule; hand the cases a script cannot vouch for to review.

Pinning rule: when an upstream publishes releases or version tags, pin the newest
published version (by its commit); otherwise follow the latest commit. Model weights
on Hugging Face follow the newest revision. Recipes are refreshed in place (same id).

``run`` (the hourly workflow) looks at every recipe, its adapter pin and its Models:

* current            -> nothing; an open review issue is closed;
* mechanical refresh -> one ``refresh/<recipe-id>`` branch and pull request;
* anything else      -> the recipe is left alone and one issue per recipe records why.

A refresh is mechanical only when everything below holds (see ``classify_changes``):
the new commit is a descendant of the pin; every vendored file is upstream's at the
new commit (a creator's changed file is carried over when the adapter vendors it verbatim
and every flag, env var and entrypoint the recipe references in it is still there); no
patch target, launch script, configuration, Dockerfile or README changed; every Model file
the recipe mounts by name still exists (a re-export of the same repository is the newest
revision); and the whole test suite still passes.
"""

from __future__ import annotations

import argparse
import os
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from recipe_tooling.adapter_sources import materialize_or_exit
from recipe_tooling.qualification.tolerant import InputReason, read_optional
from recipe_tooling.refresh_auth import RefreshAppAuth
from recipe_tooling.refresh_upstream import apply as refresh_apply
from recipe_tooling.refresh_upstream import assessment as refresh_assessment
from recipe_tooling.refresh_upstream import catalog_view as refresh_catalog_view
from recipe_tooling.refresh_upstream import github as refresh_github
from recipe_tooling.refresh_upstream import http as refresh_http
from recipe_tooling.refresh_upstream import model as refresh_model
from recipe_tooling.refresh_upstream import publish as refresh_publish
from recipe_tooling.upstream_drift import embedded as drift_embedded
from recipe_tooling.upstream_drift import heads as drift_heads
from recipe_tooling.upstream_drift import inventory as drift_inventory


class AnonymousHeadClient:
    """Adapt the refresh HTTP cache for public-only embedded Git observations."""

    def __init__(self, http: refresh_http.Http) -> None:
        self.http = http

    def json(self, url: str, provider: str) -> Any:
        return self.http.json(url, provider, anonymous=True)


def record_failure(
    summary: refresh_publish.Summary, recipe_id: str, error: Exception
) -> None:
    """Record one recipe's failure; the full traceback goes to stderr for debugging."""
    traceback.print_exc()
    summary.failures.append(f"{recipe_id}: {type(error).__name__}: {str(error)[:300]}")


def command_run(arguments: argparse.Namespace) -> int:
    root = refresh_model.ROOT
    cache_dir = arguments.cache_dir
    summary = refresh_publish.Summary(
        started=refresh_github.git("rev-parse", "HEAD", cwd=root)
    )
    if not arguments.dry_run:
        for source, action, reason in (
            (
                "publication receipt",
                refresh_publish.publish_if_stale,
                InputReason.PUBLICATION_UNAVAILABLE,
            ),
            (
                "arm open PRs",
                lambda: refresh_publish.arm_open_mechanical_prs(summary),
                InputReason.HOUSEKEEPING_UNAVAILABLE,
            ),
        ):
            read_optional(
                source,
                action,
                lambda diagnostic: summary.failures.append(
                    diagnostic.model_dump_json()
                ),
                reason,
            )

    http = refresh_http.Http(
        github_token=os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN"),
        hf_token=os.environ.get("HF_TOKEN"),
        cache_path=None if cache_dir is None else cache_dir / "etags.json",
    )
    if cache_dir is not None:
        os.environ.setdefault("VONK_UPSTREAM_CACHE", str(cache_dir / "github-trees"))

    failed: set[str] = set()  # recipes whose assessment or processing raised

    def assess_catalog():
        catalog_ = refresh_catalog_view.Catalog.load(root)
        defaults, overrides = drift_inventory.load_manifest(
            root / "policy/upstream-watch.json"
        )
        watches = drift_inventory.discover_watches(root, defaults, overrides)
        embedded = drift_embedded.embedded_inventory(root, watches)
        drift_heads.observe_embedded_heads(embedded, AnonymousHeadClient(http))
        assessor = refresh_assessment.Assessor(catalog_, http, cache_dir, embedded)

        def assess_one(recipe_id: str) -> refresh_model.Assessment:
            try:
                return assessor.assess(recipe_id)
            except Exception as error:  # noqa: BLE001 - one recipe's fault must not starve the rest
                record_failure(summary, recipe_id, error)
                failed.add(recipe_id)
                # skipped this run: the loop leaves its issue and PRs untouched and
                # the close functions skip an unreachable assessment
                return refresh_model.Assessment(
                    recipe_id, unreachable=["assessment failed"]
                )

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(assess_one, sorted(catalog_.recipes)))
        return catalog_, results, embedded

    assessment = read_optional(
        "catalog assessment",
        assess_catalog,
        lambda diagnostic: summary.failures.append(diagnostic.model_dump_json()),
        InputReason.ASSESSMENT_UNAVAILABLE,
    )
    if assessment is None:
        refresh_publish.write_summary(
            summary, http, os.environ.get("GITHUB_STEP_SUMMARY"), {"coverage": {}}
        )
        return 1
    catalog_, results, embedded = assessment
    assessed = {r.recipe_id: r for r in results}
    read_optional(
        "upstream HTTP cache",
        http.save,
        lambda diagnostic: summary.failures.append(diagnostic.model_dump_json()),
    )
    if arguments.dry_run:
        for result in results:
            if result.drifted or result.unreachable:
                state = (
                    "unreachable"
                    if result.unreachable
                    else "mechanical"
                    if result.mechanical
                    else "review-pr"  # prepared, never armed
                    if result.for_review
                    else "review"
                )
                print(
                    state,
                    result.recipe_id,
                    "; ".join(result.unreachable + result.reasons + result.held)[:300],
                )
        return 0
    refresh_github.ensure_labels()
    summary.current = sum(1 for r in results if not r.drifted and not r.unreachable)
    review_ids: set[str] = set()
    open_prs = refresh_publish.foreign_open_prs()
    groups = refresh_publish.sharing_groups(catalog_)
    grouped: set[str] = set()
    published: set[str] = set()  # refresh PRs regenerated by this run
    handled: set[str] = set()  # recipes (group members too) an open refresh PR covers
    prepared_prs: dict[str, int | None] = {}  # group member -> the group's PR
    for result in results:
        if result.recipe_id in failed:  # its assessment raised; already recorded
            continue
        try:
            if result.drifted and (
                blockers := refresh_publish.blocking_prs(
                    refresh_publish.recipe_footprint(catalog_, result.recipe_id),
                    open_prs,
                )
            ):
                review_ids.add(result.recipe_id)  # leave any existing issue alone
                summary.deferred.append(
                    f"{result.recipe_id}: skipped: open PR "
                    + ", ".join(f"#{n}" for n in blockers)
                )
                continue
            if result.unreachable:
                summary.unreachable.append(
                    f"{result.recipe_id}: {'; '.join(result.unreachable)}"
                )
                refresh_publish.sync_issue(result, summary, create=False)
                continue
            summary.newer_lines += [f"{result.recipe_id}: {n}" for n in result.notes]
            if not result.drifted:
                continue
            if result.prepared:
                if summary.prepared >= refresh_model.MAX_MECHANICAL_PER_RUN:
                    summary.deferred.append(
                        f"{result.recipe_id}: mechanical, over this run's limit"
                    )
                    continue
                members = groups.get(result.recipe_id)
                if members is None:
                    if (
                        refresh_publish.publish_mechanical(root, result, summary)
                        is not None
                    ):
                        published.add(result.recipe_id)
                        handled.add(result.recipe_id)
                elif result.recipe_id in grouped:
                    if result.held or result.group_outcome:  # its own issue says why
                        if result.held:
                            result.pull_request = prepared_prs.get(result.recipe_id)
                        review_ids.add(result.recipe_id)
                        refresh_publish.sync_issue(result, summary)
                    continue
                else:
                    grouped.update(members)
                    combined = refresh_publish.combine_group(members, assessed)
                    held = [
                        m
                        for m in members
                        if refresh_publish.blocking_prs(
                            refresh_publish.recipe_footprint(catalog_, m), open_prs
                        )
                    ]
                    if held and not isinstance(combined, str):
                        combined = f"{', '.join(held)} has an open pull request"
                    if isinstance(combined, str):
                        review_ids.add(result.recipe_id)
                        refresh_publish.explain_members(members, assessed, [combined])
                        summary.deferred.append(
                            f"{result.recipe_id}: refreshed together with "
                            f"{', '.join(m for m in members if m != result.recipe_id)} "
                            f"or not at all: {combined}"
                        )
                    else:
                        number = refresh_publish.publish_group(
                            root, combined, result, summary
                        )
                        prepared_prs.update(dict.fromkeys(members, number))
                        if number is not None:
                            published.add(combined.recipe_id)
                            handled.update(members)
                            refresh_publish.supersede_member_prs(
                                members, combined.recipe_id, number, summary
                            )
                        else:
                            refresh_publish.explain_members(
                                members,
                                assessed,
                                combined.reasons
                                or ["its group pull request was not published"],
                            )
            if result.reasons or result.held or result.group_outcome:
                review_ids.add(result.recipe_id)
                refresh_publish.sync_issue(result, summary)
        except Exception as error:  # noqa: BLE001 - one recipe's fault must not starve the rest
            failed.add(result.recipe_id)
            record_failure(summary, result.recipe_id, error)
    for source, action in (
        (
            "close current issues",
            lambda: refresh_publish.close_current_issues(
                assessed, handled - review_ids - failed, summary
            ),
        ),
        (
            "close stale refresh PRs",
            lambda: refresh_publish.close_stale_refresh_prs(
                assessed, summary, groups, published
            ),
        ),
    ):
        read_optional(
            source,
            action,
            lambda diagnostic: summary.failures.append(diagnostic.model_dump_json()),
            InputReason.HOUSEKEEPING_UNAVAILABLE,
        )
    refresh_publish.write_summary(
        summary, http, os.environ.get("GITHUB_STEP_SUMMARY"), embedded
    )
    return 1 if summary.failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir", type=Path, help="persist ETags and upstream trees here"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="classify and print; change nothing"
    )
    parser.add_argument(
        "--github-app",
        action="store_true",
        help="require renewable repository-scoped App credentials for git/gh",
    )
    parser.add_argument(
        "--verify-app",
        action="store_true",
        help="verify App authentication and exact repository scope; change no recipes",
    )
    parser.add_argument(
        "--guard",
        action="store_true",
        help="only decide whether a chained refresh should proceed (needs gh, no App)",
    )
    parser.add_argument(
        "--memory-estimate-issue",
        type=Path,
        metavar="REPORT",
        help="only keep the one memory-estimate issue current from this report (needs gh, no App)",
    )
    parser.add_argument("--event", default="schedule")
    parser.add_argument("--run-id", type=int, default=0)
    parser.add_argument("--upstream-conclusion", default="success")
    arguments = parser.parse_args()
    if arguments.guard:
        return refresh_publish.command_guard(arguments)
    if arguments.memory_estimate_issue:
        return refresh_publish.command_memory_estimate_issue(
            arguments.memory_estimate_issue
        )
    if arguments.verify_app and not arguments.github_app:
        parser.error("--verify-app requires --github-app")
    if arguments.github_app:
        refresh_apply.APP_AUTH = RefreshAppAuth(
            os.environ.get("REFRESH_APP_CLIENT_ID", ""),
            os.environ.pop("REFRESH_APP_PRIVATE_KEY", ""),
            os.environ.get("GITHUB_REPOSITORY", ""),
        )
        refresh_apply.APP_AUTH.environment()  # Fail before upstream work when setup is incomplete.
    try:
        if arguments.verify_app:
            installation = refresh_github.gh_json("api", "installation/repositories")
            repositories = [repo["full_name"] for repo in installation["repositories"]]
            expected = os.environ["GITHUB_REPOSITORY"]
            if repositories != [expected]:
                raise RuntimeError(
                    f"Refresh App token scope differs from {expected}: {repositories}"
                )
            print(
                f"Refresh App authentication verified; token restricted to {expected}."
            )
            return 0
        # Assessments compare the adapters' files, declared vendor files included.
        materialize_or_exit(refresh_model.ROOT)
        return command_run(arguments)
    finally:
        if refresh_apply.APP_AUTH is not None:
            refresh_apply.APP_AUTH.close()
            refresh_apply.APP_AUTH = None
