"""The image workflow builds each missing image once and the index pins it.

A Controller prefers the digest the signed index pins for a recipe, so the
selection must not rebuild an image that exists, must build a key shared by
several recipes once, and the index must pin exactly the digest the registry
serves for that recipe's key.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import runpy
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
LOADER = importlib.machinery.SourceFileLoader(
    "prebuilt_images", str(ROOT / "tools/prebuilt-images")
)
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
assert SPEC is not None
tool = importlib.util.module_from_spec(SPEC)
sys.modules[LOADER.name] = tool
LOADER.exec_module(tool)
INDEX = runpy.run_path(str(ROOT / "tools/build-catalog-index"))

OWNER = "CarstVaartjes"
SHARED, OWN = "a" * 64, "b" * 64
DIGEST = "sha256:" + "d" * 64


def _plan() -> dict[str, Any]:
    def recipe(slug: str, context: str) -> dict[str, Any]:
        return {
            "slug": slug,
            "version": "1.0",
            "content_sha256": ({"other": "e", "single": "f"}.get(slug, slug[0])) * 64,
            "source_path": f"recipes/{slug}.json",
            "context": context,
        }

    return {
        "images": [
            {
                "build_key": SHARED,
                "recipes": [
                    recipe("single", "adapters/x/shared"),
                    recipe("dual", "adapters/x/shared"),
                ],
            },
            {"build_key": OWN, "recipes": [recipe("other", "adapters/y/own")]},
        ]
    }


def _registry(tags: dict[tuple[str, str], str]):
    def lookup(owner: str, name: str, tag: str) -> str | None:
        assert owner == OWNER
        return tags.get((name, tag))

    return lookup


def test_a_shared_missing_key_is_built_once_for_every_recipe_that_needs_it() -> None:
    selection = tool.select(
        _plan(), owner=OWNER, candidates={"single", "dual"}, lookup=_registry({})
    )
    assert [item["build_key"] for item in selection["builds"]] == [SHARED]
    assert [recipe["slug"] for recipe in selection["builds"][0]["recipes"]] == [
        "single",
        "dual",
    ]
    assert selection["builds"][0]["recipes"][0]["repository"] == (
        "ghcr.io/carstvaartjes/vonk-forge-recipe-single"
    )
    assert selection["copies"] == []


def test_an_existing_key_is_copied_not_rebuilt_and_present_images_are_skipped() -> None:
    registry = _registry(
        {
            ("vonk-forge-recipe-single", f"build-{SHARED}"): DIGEST,
            ("vonk-forge-recipe-other", f"build-{OWN}"): DIGEST,
        }
    )
    selection = tool.select(_plan(), owner=OWNER, candidates=None, lookup=registry)
    assert selection["builds"] == []
    assert selection["copies"] == [
        {
            "from": f"ghcr.io/carstvaartjes/vonk-forge-recipe-single@{DIGEST}",
            "to": f"ghcr.io/carstvaartjes/vonk-forge-recipe-dual:build-{SHARED}",
            "version_tag": "ghcr.io/carstvaartjes/vonk-forge-recipe-dual:1.0",
        }
    ]


def test_pr_candidate_is_promoted_by_digest_without_rebuilding() -> None:
    registry = _registry({("vonk-forge-recipe-single", f"candidate-{SHARED}"): DIGEST})
    selection = tool.select(_plan(), owner=OWNER, candidates=None, lookup=registry)
    assert [row["build_key"] for row in selection["builds"]] == [OWN]
    assert {row["from"] for row in selection["copies"]} == {
        f"{tool.repository(OWNER, 'single')}@{DIGEST}"
    }
    assert {row["to"] for row in selection["copies"]} == {
        f"{tool.repository(OWNER, slug)}:build-{SHARED}" for slug in ("single", "dual")
    }


def test_pr_selection_uses_build_identity_and_ignores_missing_historical_images():
    base = _plan()
    head = _plan()
    # Evidence or model/recipe metadata changes do not change executable keys.
    head["images"][0]["recipes"][0]["content_sha256"] = "9" * 64
    candidates = tool.changed_candidates(head, base)
    assert candidates == set()
    assert tool.select(
        head, owner=OWNER, candidates=candidates, lookup=_registry({})
    ) == {"builds": [], "copies": [], "deferred": [], "deferred_slugs": []}
    # A shared context change schedules every affected recipe, one ARM64 build.
    head["images"][0]["build_key"] = "c" * 64
    candidates = tool.changed_candidates(head, base)
    assert candidates == {"single", "dual"}
    selection = tool.select(
        head, owner=OWNER, candidates=candidates, lookup=_registry({})
    )
    assert [row["build_key"] for row in selection["builds"]] == ["c" * 64]


def test_new_recipe_using_an_unpublished_shared_key_still_requires_proof():
    base = _plan()
    head = _plan()
    head["images"][1]["recipes"][0]["slug"] = "new-recipe"
    assert tool.changed_candidates(head, base) == {"new-recipe"}


def test_pr_cannot_silently_omit_a_recipe_the_image_planner_refused(
    tmp_path, monkeypatch
):
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "images": [],
                "skipped": [{"slug": "broken", "reason": "bad build inputs"}],
            }
        )
    )
    output = tmp_path / "selection.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prebuilt-images",
            "select",
            "--base-plan",
            str(plan),
            "--plan",
            str(plan),
            "--owner",
            OWNER,
            "--output",
            str(output),
        ],
    )
    with pytest.raises(SystemExit, match="PR image planner refused.*broken"):
        tool.main()
    assert not output.exists()


@pytest.mark.parametrize("present", [False, True])
def test_publish_builds_only_the_changed_key_without_a_pr_candidate(
    tmp_path, monkeypatch, present
):
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(_plan()))
    # One changed key always has PR proof; inject a missing candidate for
    # the other key (cancelled/expired PR build or a merge-result key change).
    tags = {(tool.package("single"), f"candidate-{SHARED}"): DIGEST}
    if present:
        tags[(tool.package("other"), f"candidate-{OWN}")] = DIGEST
    select = tool.select
    monkeypatch.setattr(
        tool, "select", lambda plan, **kw: select(plan, **kw, lookup=_registry(tags))
    )
    output = tmp_path / "selection.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prebuilt-images",
            "select",
            "--plan",
            str(plan_path),
            "--owner",
            OWNER,
            "--output",
            str(output),
        ],
    )
    assert tool.main() == 0
    selection = json.loads(output.read_text())
    assert [row["build_key"] for row in selection["builds"]] == (
        [] if present else [OWN]
    )
    assert len(selection["copies"]) == (3 if present else 2)
    assert all(f"@{DIGEST}" in row["from"] for row in selection["copies"])
    assert selection["deferred"] == []


def test_index_pins_each_recipes_own_digest_and_omits_missing_images() -> None:
    registry = _registry({("vonk-forge-recipe-other", f"build-{OWN}"): DIGEST})
    images, missing = tool.resolve(_plan(), owner=OWNER, lookup=registry)
    assert images == {
        "e" * 64: {
            "reference": f"ghcr.io/carstvaartjes/vonk-forge-recipe-other@{DIGEST}",
            "build_key": OWN,
        }
    }
    assert missing == ["dual", "single"]

    # The index carries the entry beside exactly the revision it belongs to.
    rows = [{"content_sha256": "e" * 64}, {"content_sha256": "f" * 64}]
    INDEX["pin_prebuilt_images"](rows, images)
    assert rows == [
        {"content_sha256": "e" * 64, "prebuilt_image": images["e" * 64]},
        {"content_sha256": "f" * 64},
    ]


OLD_PIN = "1" * 40


def test_the_platform_pin_is_read_from_the_publication_workflow() -> None:
    assert tool.platform_ref(f"x:\n  platform_ref: {OLD_PIN}\n") == OLD_PIN
    assert tool.platform_ref("platform_ref: not-a-commit\n") is None
    workflow = (ROOT / ".github/workflows/publish.yml").read_text(encoding="utf-8")
    assert tool.platform_ref(workflow) is not None


def test_explicit_selection_limits_backfills_only() -> None:
    assert tool.candidate_slugs([]) is None
    assert tool.candidate_slugs(["all"]) is None
    assert tool.candidate_slugs(["other"]) == {"other"}


def test_new_contract_publication_reconciles_both_unresolved_tensorfold_keys(
    tmp_path, monkeypatch
) -> None:
    # Run 39667637833: 2.5.0 has no publication tag. Both images were missing,
    # but select produced zero builds because the default baseline was empty.
    keys = {
        "deepseek-v4-1-flash-1017-mia-tensorfold-dual": "05af95ff114943f8d37c3401cbd0ce5febbf5b1c7eb9c3f6256e1bbd664d42f8",
        "deepseek-v4-1-flash-1017-exl3-2-9bpw-tonyd2wild-tensorfold-dual": "cfd0ced0fee052a50f415790bd090b701c49a3ac14ecfc16604f585660012eeb",
    }
    plan = {
        "images": [
            {"build_key": key, "recipes": [{"slug": slug}]}
            for slug, key in keys.items()
        ]
    }
    plan_path, selection_path = tmp_path / "plan.json", tmp_path / "selection.json"
    plan_path.write_text(json.dumps(plan))
    select = tool.select
    monkeypatch.setattr(
        tool,
        "select",
        lambda plan, **kwargs: select(plan, **kwargs, lookup=_registry({})),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prebuilt-images",
            "select",
            "--plan",
            str(plan_path),
            "--owner",
            OWNER,
            "--output",
            str(selection_path),
        ],
    )
    assert tool.main() == 0
    selection = json.loads(selection_path.read_text())
    assert {row["build_key"] for row in selection["builds"]} == set(keys.values())
    # On the next publication only registry evidence suppresses builds, even
    # when no recipe file changed or the release tag advanced past this run.
    registry = _registry(
        {(tool.package(slug), f"build-{key}"): DIGEST for slug, key in keys.items()}
    )
    assert select(plan, owner=OWNER, candidates=None, lookup=registry)["builds"] == []


def test_a_recipe_declared_unbuildable_is_not_retried_unless_asked_for() -> None:
    excluded = {"dual": "compiles for hours on the hosted runner"}

    everything = tool.without_excluded(_plan(), None, excluded, requested=["all"])
    assert everything == {"single", "other"}
    changed = tool.without_excluded(_plan(), {"dual", "other"}, excluded, requested=[])
    assert changed == {"other"}
    # Naming the recipe is an explicit request to try it again.
    assert tool.without_excluded(_plan(), {"dual"}, excluded, requested=["dual"]) == {
        "dual"
    }


def test_every_recipe_without_an_image_says_why() -> None:
    plan = _plan()
    plan["skipped"] = [{"slug": "refused", "reason": "dockerfile.heredoc_forbidden"}]
    reasons = tool.missing_reasons(
        plan, ["dual", "single"], {"dual": "compiles for hours"}
    )

    assert set(reasons) == {"refused", "dual", "single"}
    assert "dockerfile.heredoc_forbidden" in reasons["refused"]
    assert "compiles for hours" in reasons["dual"]
    assert "build failed" in reasons["single"]


def test_declared_exclusions_name_existing_recipes_and_give_a_reason() -> None:
    for slug, reason in tool.read_exclusions(ROOT / tool.EXCLUSIONS).items():
        assert (ROOT / "recipes" / f"{slug}.json").is_file(), slug
        assert reason.strip(), slug


def test_matrix_overflow_continues_after_publication_and_failed_first_batch():
    plan = {
        "images": [
            {
                "build_key": str(i),
                "recipes": [
                    {
                        "slug": f"r{i:04}",
                        "content_sha256": str(i),
                        "source_path": f"recipes/r{i}.json",
                        "context": f"adapters/r{i}",
                    }
                ],
            }
            for i in range(520)
        ]
    }
    first = tool.select(plan, owner=OWNER, candidates=None, lookup=_registry({}))
    assert len(first["builds"]) == tool.MAX_BUILDS
    assert first.get("deferred_slugs") is not None, (
        "overflow needs an automatic continuation payload"
    )
    assert len(first["deferred_slugs"]) == 264
    remaining = tool.candidate_slugs(first["deferred_slugs"])
    # Even if the release tag advances and every first-batch build fails, the
    # workflow explicitly carries the tail; registry evidence suppresses success.
    second = tool.select(plan, owner=OWNER, candidates=remaining, lookup=_registry({}))
    third = tool.select(
        plan,
        owner=OWNER,
        candidates=set(second["deferred_slugs"]),
        lookup=_registry({}),
    )
    assert len(second["builds"]) == tool.MAX_BUILDS and len(third["builds"]) == 8
    covered = {
        row["build_key"] for batch in (first, second, third) for row in batch["builds"]
    }
    assert len(covered) == 520 and not third["deferred"]
    workflow = (ROOT / ".github/workflows/recipe-images.yml").read_text()
    assert (
        "schedule:" in workflow
        and "github.event_name == 'schedule' && 'all'" in workflow
    )
    assert "needs.plan.outputs.deferred_slugs" in workflow
    assert (
        "gh workflow run recipe-images.yml" in workflow
        and '-f slugs="$DEFERRED"' in workflow
    )
    assert "needs.build.result == 'success'" not in workflow
