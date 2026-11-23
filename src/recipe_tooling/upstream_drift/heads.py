"""Observing the heads of the embedded references."""

from __future__ import annotations

from collections import Counter

from recipe_tooling.upstream_drift.model import RemoteNotFound
from recipe_tooling.upstream_drift.observe import (
    ProviderClient,
    _github_commit,
    _required_string,
)


def observe_embedded_heads(
    inventory: dict[str, object], client: ProviderClient
) -> None:
    """Attach advertised GitHub default-branch candidates without retargeting pins."""
    head_client = (
        ProviderClient(opener=client.opener)
        if isinstance(client, ProviderClient)
        else client
    )
    records = inventory.get("recipes", [])
    if not isinstance(records, list):
        return
    memo: dict[str, tuple[str | None, str | None, str | None, str | None]] = {}
    candidate_counts: Counter[str] = Counter()
    for recipe in records:
        if not isinstance(recipe, dict) or not isinstance(recipe.get("inputs"), list):
            continue
        for item in recipe["inputs"]:
            if not isinstance(item, dict) or not isinstance(
                item.get("repository"), str
            ):
                continue
            repository = item["repository"]
            if repository not in memo:
                try:
                    repo = head_client.json(
                        f"https://api.github.com/repos/{repository}", "github"
                    )
                    branch = _required_string(repo, "default_branch", "repository")
                    revision, observed_at, url = _github_commit(
                        head_client, repository, branch
                    )
                    memo[repository] = (revision, observed_at, url, None)
                except (RemoteNotFound, RuntimeError, KeyError, TypeError) as error:
                    memo[repository] = (None, None, None, str(error)[:256])
            observed, observed_at, url, error_text = memo[repository]
            item["candidate_channel"] = "advertised-default-branch"
            item["candidate_revision"] = observed
            item["candidate_observed_at"] = observed_at
            item["candidate_url"] = url
            pinned = item.get("revision")
            if error_text:
                item["candidate_status"] = "unknown"
                item["candidate_reason"] = (
                    "default-branch observation failed; retained the embedded pin: "
                    + error_text
                )
            elif observed is None:
                item["candidate_status"] = "unknown"
                item["candidate_reason"] = "provider returned no advertised head"
            elif not isinstance(pinned, str):
                item["candidate_status"] = "no-immutable-pin"
                item["candidate_reason"] = (
                    "default-branch head is review evidence; the Dockerfile source has no recognized immutable commit"
                )
            elif observed == pinned:
                item["candidate_status"] = "same"
                item["candidate_reason"] = (
                    "advertised default branch is at the embedded commit"
                )
            else:
                item["candidate_status"] = "advanced"
                item["candidate_reason"] = (
                    "default branch advanced; candidate is review evidence only. Retain specialized behavior until compatibility is established; no embedded pin was changed"
                )
            candidate_counts[str(item["candidate_status"])] += 1
    coverage = inventory.get("coverage")
    if isinstance(coverage, dict):
        coverage["counts_by_candidate_status"] = dict(sorted(candidate_counts.items()))
