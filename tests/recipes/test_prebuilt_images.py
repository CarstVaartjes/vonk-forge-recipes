"""The image workflow builds each missing image once and the index pins it.

A Controller prefers the digest the signed index pins for a recipe, so the
selection must not rebuild an image that exists, must build a key shared by
several recipes once, and the index must pin exactly the digest the registry
serves for that recipe's key.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

import pytest
from repo_root import ROOT

from recipe_tooling import build_plan, catalog_index, prebuilt_images
from recipe_tooling.upstream_http import UpstreamBusy

tool = prebuilt_images
INDEX = vars(catalog_index)

OWNER = "CarstVaartjes"
SHARED, OWN = "a" * 64, "b" * 64
DIGEST = "sha256:" + "d" * 64


def _plan() -> dict[str, Any]:
    def recipe(slug: str, context: str) -> dict[str, Any]:
        return {
            "slug": slug,
            "version": "1.0",
            "content_sha256": ({"other": "e", "single": "f"}.get(slug, slug[0])) * 64,
            "source_bundle_sha256": "5" * 64,
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
            ("vonk-forge-recipe-single", tool.build_tag(SHARED)): DIGEST,
            ("vonk-forge-recipe-other", tool.build_tag(OWN)): DIGEST,
        }
    )
    selection = tool.select(_plan(), owner=OWNER, candidates=None, lookup=registry)
    assert selection["builds"] == []
    assert selection["copies"] == [
        {
            "from": f"ghcr.io/carstvaartjes/vonk-forge-recipe-single@{DIGEST}",
            "to": f"ghcr.io/carstvaartjes/vonk-forge-recipe-dual:{tool.build_tag(SHARED)}",
            "version_tag": "ghcr.io/carstvaartjes/vonk-forge-recipe-dual:1.0",
        }
    ]


def test_pr_candidate_is_promoted_by_digest_without_rebuilding() -> None:
    registry = _registry(
        {("vonk-forge-recipe-single", tool.candidate_tag(SHARED)): DIGEST}
    )
    selection = tool.select(_plan(), owner=OWNER, candidates=None, lookup=registry)
    assert [row["build_key"] for row in selection["builds"]] == [OWN]
    assert {row["from"] for row in selection["copies"]} == {
        f"{tool.repository(OWNER, 'single')}@{DIGEST}"
    }
    assert {row["to"] for row in selection["copies"]} == {
        f"{tool.repository(OWNER, slug)}:{tool.build_tag(SHARED)}"
        for slug in ("single", "dual")
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
    ) == {
        "builds": [],
        "copies": [],
        "deferred": [],
        "deferred_slugs": [],
        "adopted": [],
        "kept": [],
        "busy": [],
    }
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
    tags = {(tool.package("single"), tool.candidate_tag(SHARED)): DIGEST}
    if present:
        tags[(tool.package("other"), tool.candidate_tag(OWN))] = DIGEST
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


def _busy_for_shared(owner: str, name: str, tag: str) -> str | None:
    if tag.endswith(SHARED):
        raise UpstreamBusy("ghcr.io is rate limited")
    return None


def test_a_busy_image_is_skipped_and_the_other_is_still_built() -> None:
    selection = tool.select(
        _plan(), owner=OWNER, candidates=None, lookup=_busy_for_shared
    )
    assert [row["build_key"] for row in selection["builds"]] == [OWN]
    assert selection["busy"] == [SHARED]


def test_a_pull_request_with_a_busy_image_refuses_instead_of_skipping_its_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(_plan()))
    base_path = tmp_path / "base.json"
    base_path.write_text(json.dumps({"images": []}))  # every key changed
    real = tool.select
    monkeypatch.setattr(
        tool, "select", lambda plan, **kw: real(plan, **kw, lookup=_busy_for_shared)
    )

    def run(*extra: str) -> int:
        monkeypatch.setattr(
            sys,
            "argv",
            ["prebuilt-images", "select", "--plan", str(plan_path), "--owner", OWNER,
             "--output", str(tmp_path / "out.json"), *extra],
        )  # fmt: skip
        return tool.main()

    assert run() == 0  # publish and scheduled runs keep the skip
    assert json.loads((tmp_path / "out.json").read_text())["busy"] == [SHARED]
    with pytest.raises(SystemExit, match="registry busy"):
        run("--base-plan", str(base_path))


def test_index_pins_each_recipes_own_digest_and_omits_missing_images() -> None:
    registry = _registry({("vonk-forge-recipe-other", tool.build_tag(OWN)): DIGEST})
    images, missing, _carried = tool.resolve(_plan(), owner=OWNER, lookup=registry)
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


def _pin(
    slug: str, digest: str, key: str, bundle: str = "5" * 64
) -> dict[str, str | None]:
    return {
        "reference": f"ghcr.io/carstvaartjes/vonk-forge-recipe-{slug}@{digest}",
        "build_key": key,
        "source_bundle_sha256": bundle,
    }


def _image(pin: dict[str, str | None]) -> dict[str, str | None]:
    return {"reference": pin["reference"], "build_key": pin["build_key"]}


def test_a_busy_lookup_keeps_the_pin_of_unchanged_input_whatever_its_key() -> None:
    def busy(owner: str, name: str, tag: str) -> str | None:
        raise OSError("429 too many requests")

    previous = "sha256:" + "1" * 64
    published = {
        "e" * 64: _pin("other", previous, "9" * 64),  # same input, a legacy key
        "f" * 64: _pin("single", previous, OWN, "6" * 64),  # build context changed
    }
    images, missing, carried = tool.resolve(
        _plan(), owner=OWNER, lookup=busy, published=published
    )
    assert images == {"e" * 64: _image(published["e" * 64])}
    assert carried == ["other"]
    assert missing == ["dual", "single"]


def test_a_successful_lookup_pins_the_new_digest_over_the_published_one() -> None:
    published = {"e" * 64: _pin("other", "sha256:" + "1" * 64, OWN)}
    registry = _registry({("vonk-forge-recipe-other", tool.build_tag(OWN)): DIGEST})
    images, _missing, carried = tool.resolve(
        _plan(), owner=OWNER, lookup=registry, published=published
    )
    assert images["e" * 64] == _image(_pin("other", DIGEST, OWN))
    assert carried == []


def test_the_published_pins_are_read_from_the_release_bundle(tmp_path: Path) -> None:
    pin = _image(_pin("other", DIGEST, OWN))
    index = {
        "recipes": [
            {
                "content_sha256": "e" * 64,
                "source_bundle_sha256": "5" * 64,
                "prebuilt_image": pin,
            },
            {"content_sha256": "f" * 64},
        ]
    }
    source = tmp_path / "catalog-index.json"
    source.write_text(json.dumps(index))

    def release(arguments: list[str]) -> None:
        with tarfile.open(arguments[arguments.index("--output") + 1], "w") as archive:
            archive.add(source, arcname="catalog-index.json")

    def unavailable(arguments: list[str]) -> None:
        raise subprocess.CalledProcessError(1, arguments)

    assert tool.published_pins("o/r", gh=release) == {
        "e" * 64: {**pin, "source_bundle_sha256": "5" * 64}
    }
    assert tool.published_pins("o/r", gh=unavailable) is None

    def not_a_tar(arguments: list[str]) -> None:
        Path(arguments[arguments.index("--output") + 1]).write_bytes(b"not a tar")

    assert tool.published_pins("o/r", gh=not_a_tar) is None

    def odd_index(arguments: list[str]) -> None:
        source.write_text(json.dumps({"recipes": ["x"]}))
        release(arguments)

    assert tool.published_pins("o/r", gh=odd_index) is None


def test_a_pin_naming_a_foreign_package_is_not_carried() -> None:
    def none_found(owner: str, name: str, tag: str) -> str | None:
        return None

    foreign = {
        "reference": "evil.example/x/y@sha256:" + "1" * 64,
        "build_key": OWN,
        "source_bundle_sha256": "5" * 64,
    }
    images, missing, carried = tool.resolve(
        _plan(), owner=OWNER, lookup=none_found, published={"e" * 64: foreign}
    )
    assert images == {}
    assert carried == []
    assert "other" in missing


OLD_PIN = "1" * 40


def test_the_platform_pin_is_read_from_the_publication_workflow() -> None:
    assert tool.platform_ref(f"x:\n  platform_ref: {OLD_PIN}\n") == OLD_PIN
    assert tool.platform_ref("platform_ref: not-a-commit\n") is None
    workflow = (ROOT / ".github/workflows/publish.yml").read_text(encoding="utf-8")
    assert tool.platform_ref(workflow) is not None


def test_every_platform_pin_equals_publish_workflows() -> None:
    """`uses:` takes no expression, so every literal must equal the owner's."""

    workflows = ROOT / ".github/workflows"
    owner = tool.platform_ref((workflows / "publish.yml").read_text("utf-8"))
    assert owner is not None
    pins = {
        sha
        for workflow in workflows.glob("*.yml")
        for found in re.findall(
            r"vonk-forge/\.github/workflows/[\w-]+\.yml@(\S+)|platform_ref:\s*(\S+)",
            workflow.read_text("utf-8"),
        )
        for sha in found
        if sha
    }
    assert pins == {owner}


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
        {
            (tool.package(slug), tool.build_tag(key)): DIGEST
            for slug, key in keys.items()
        }
    )
    assert select(plan, owner=OWNER, candidates=None, lookup=registry)["builds"] == []


def test_every_recipe_without_an_image_says_why() -> None:
    plan = _plan()
    plan["skipped"] = [{"slug": "refused", "reason": "dockerfile.heredoc_forbidden"}]
    reasons = tool.missing_reasons(plan, ["dual", "single"])

    assert set(reasons) == {"refused", "dual", "single"}
    assert "dockerfile.heredoc_forbidden" in reasons["refused"]
    assert "build failed" in reasons["single"]


@pytest.mark.parametrize(("late", "attempts"), [(False, 3), (True, 1)])
def test_a_build_retry_starts_only_if_it_fits_the_job_timeout(
    tmp_path: Path, late: bool, attempts: int
) -> None:
    # Run the build job's own retry function with a failing builder. Early, a
    # transient failure retries; a failure past the retry budget fails the job
    # at once instead of starting an attempt the job timeout would kill.
    workflow = (ROOT / ".github/workflows/recipe-images.yml").read_text()
    build = workflow.split("\n  build:\n", 1)[1].split("\n  resolve:\n", 1)[0]
    # timeout-minutes cannot read env, so the retry deadline repeats it: both
    # copies must name the same job timeout, less a margin for the push.
    timeout = re.search(r"^    timeout-minutes: (\d+)$", build, re.MULTILINE)
    budget = re.search(
        r"deadline=\$\(\(JOB_START \+ \((\d+) - (\d+)\) \* 60\)\)", build
    )
    assert timeout is not None
    assert budget is not None
    assert int(budget.group(1)) == int(timeout.group(1))
    margin = int(budget.group(2))
    assert 0 < margin < int(timeout.group(1))
    elapsed_minutes = int(timeout.group(1)) - margin + 5 if late else 0
    lines = [line.strip() for line in build.splitlines()]
    start = next(i for i, line in enumerate(lines) if line.startswith("deadline="))
    end = lines.index("}", start)
    script = "\n".join(
        [
            "sleep() { :; }",
            *lines[start : end + 1],
            f"build_with_retry sh -c 'echo x >> {tmp_path}/calls; exit 1'",
        ]
    )
    now = int(time.time())
    result = subprocess.run(
        ["bash", "-c", script],
        env={**os.environ, "JOB_START": str(now - elapsed_minutes * 60)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1, result.stderr
    assert (tmp_path / "calls").read_text().count("x") == attempts


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
    assert len(second["builds"]) == tool.MAX_BUILDS
    assert len(third["builds"]) == 8
    covered = {
        row["build_key"] for batch in (first, second, third) for row in batch["builds"]
    }
    assert len(covered) == 520
    assert not third["deferred"]
    workflow = (ROOT / ".github/workflows/recipe-images.yml").read_text()
    assert "schedule:" in workflow
    assert "github.event_name == 'schedule' && 'all'" in workflow
    assert "needs.plan.outputs.deferred_slugs" in workflow
    assert "gh workflow run recipe-images.yml" in workflow
    assert '-f slugs="$DEFERRED"' in workflow
    assert "needs.build.result == 'success'" not in workflow


# D3: the recipe-owned key v2. The first run after it adopts every legacy image of
# unchanged input by copying its digest (no build), and a restart anywhere heals.
K1 = {"single": "7" * 64, "dual": "8" * 64, "other": "9" * 64}
LEGACY = {slug: "sha256:" + digit * 64 for slug, digit in zip(K1, "123", strict=True)}


class Registry:
    """A fake GHCR; ``copy`` applies a selection's copies as the copy job does."""

    def __init__(self, busy: bool = False, failing: frozenset[str] = frozenset()):
        self.tags: dict[tuple[str, str], str] = {}
        self.busy = busy
        self.failing = failing

    def lookup(self, owner: str, name: str, tag: str) -> str | None:
        assert owner == OWNER
        if self.busy:
            raise UpstreamBusy("ghcr.io is rate limited")
        return self.tags.get((name, tag))

    def copy(self, selection: dict[str, Any], limit: int | None = None) -> None:
        for done, copy in enumerate(selection["copies"]):
            if done == limit:
                return  # the copy job was killed here
            name, tag = copy["to"].rsplit("/", 1)[1].split(":")
            if name not in self.failing:  # a 5xx leaves the tag absent
                self.tags[(name, tag)] = copy["from"].split("@", 1)[1]


def _published() -> dict[str, dict[str, str | None]]:
    """The release before the migration: each recipe pins its own adapted image."""
    return {
        recipe["content_sha256"]: _pin(
            recipe["slug"], LEGACY[recipe["slug"]], K1[recipe["slug"]]
        )
        for image in _plan()["images"]
        for recipe in image["recipes"]
    }


def _run(registry: Registry, **kwargs: Any) -> dict[str, Any]:
    return tool.select(
        _plan(),
        owner=OWNER,
        candidates=kwargs.pop("candidates", None),
        lookup=registry.lookup,
        published=kwargs.pop("published", _published()),
        **kwargs,
    )


def _pins(registry: Registry, published: Any = None) -> dict[str, Any]:
    images, missing, _carried = tool.resolve(
        _plan(),
        owner=OWNER,
        lookup=registry.lookup,
        published=_published() if published is None else published,
    )
    assert missing == []
    return images


def _digests(images: dict[str, Any]) -> dict[str, str]:
    return {content: pin["reference"].split("@")[1] for content, pin in images.items()}


@pytest.mark.parametrize("requested", [[], ["all"]])  # a main push; the nightly run
def test_the_first_run_after_the_key_change_adopts_and_builds_nothing(requested):
    registry = Registry()
    selection = _run(registry, candidates=tool.candidate_slugs(requested))
    assert selection["builds"] == []
    assert selection["adopted"] == ["dual", "other", "single"]
    # Each recipe keeps its own digest, even where two now share one key.
    assert {(row["from"], row["to"]) for row in selection["copies"]} == {
        (
            f"{tool.repository(OWNER, slug)}@{LEGACY[slug]}",
            f"{tool.repository(OWNER, slug)}:{tool.build_tag(key)}",
        )
        for key, slugs in ((SHARED, ("single", "dual")), (OWN, ("other",)))
        for slug in slugs
    }
    registry.copy(selection)
    images = _pins(registry)
    assert {pin["build_key"] for pin in images.values()} == {SHARED, OWN}
    assert _digests(images) == _digests(_published())
    after = _run(registry, candidates=tool.candidate_slugs(requested))
    assert (after["builds"], after["copies"]) == ([], [])


def test_a_copy_job_killed_midway_heals_on_the_next_run_without_a_build():
    registry = Registry()
    registry.copy(_run(registry), limit=1)
    # A publish between the two runs pins every recipe, digests unchanged.
    assert _digests(_pins(registry)) == _digests(_published())
    second = _run(registry)
    assert second["builds"] == []
    assert len(second["copies"]) == 2
    # The recipe left over adopts its own image, never its key-sibling's.
    assert all(
        row["from"].split("@")[0] == row["to"].split(":")[0] for row in second["copies"]
    )
    registry.copy(second)
    assert _digests(_pins(registry)) == _digests(_published())


def test_a_failed_copy_carries_the_legacy_pin_by_content_until_the_next_run():
    registry = Registry(failing=frozenset({tool.package("other")}))
    registry.copy(_run(registry))
    images = _pins(registry)
    assert images["e" * 64] == _image(_published()["e" * 64])  # {X, K1}
    registry.failing = frozenset()
    retry = _run(registry)
    assert (retry["builds"], retry["adopted"]) == ([], ["other"])


def test_a_busy_registry_plans_nothing_and_publishes_the_previous_pins():
    registry = Registry(busy=True)
    selection = _run(registry)
    assert (selection["builds"], selection["copies"]) == ([], [])
    assert sorted(selection["busy"]) == sorted([SHARED, OWN])
    assert _pins(registry) == {
        content: _image(pin) for content, pin in _published().items()
    }


def test_adoption_wins_over_a_candidate_of_the_same_input():
    registry = Registry()
    candidate = "sha256:" + "4" * 64
    registry.tags[(tool.package("other"), tool.candidate_tag(OWN))] = candidate
    selection = _run(registry, candidates={"other"})
    assert selection["builds"] == []
    assert [row["from"] for row in selection["copies"]] == [
        f"{tool.repository(OWNER, 'other')}@{LEGACY['other']}"
    ]
    registry.copy(selection)
    assert (
        registry.tags[(tool.package("other"), tool.build_tag(OWN))] == (LEGACY["other"])
    )


def test_a_publish_killed_before_its_upload_republishes_the_same_digests():
    registry = Registry()
    registry.copy(_run(registry))
    # The old release (K1 pins) stayed live; the rerun plans nothing new.
    rerun = _run(registry)
    assert (rerun["builds"], rerun["copies"]) == ([], [])
    assert _digests(_pins(registry)) == _digests(_published())


@pytest.mark.parametrize("change", ["source-bundle", "v2-built"])
def test_changed_input_is_built_not_adopted(change: str):
    registry = Registry()
    published = _published()
    if change == "source-bundle":
        # The content digest names the document only: a context edit is new input.
        published["e" * 64]["source_bundle_sha256"] = "6" * 64
    else:
        # A v2-built pin whose key moved (the build policy changed) is rebuilt.
        registry.tags[(tool.package("other"), tool.build_tag(K1["other"]))] = DIGEST
    selection = _run(registry, candidates={"other"}, published=published)
    assert selection["adopted"] == []
    assert [row["build_key"] for row in selection["builds"]] == [OWN]


@pytest.mark.parametrize(
    ("labels", "rebuilt"),
    [
        ({tool.LEGACY_LABEL: "vonk.runtime-contract.vllm.v1"}, True),
        ({"org.opencontainers.image.source": "https://github.com/o/r"}, False),
        (None, False),  # unreadable labels are no evidence: keep the image
    ],
)
def test_rebuild_adopted_replaces_only_an_adopted_image(labels, rebuilt):
    registry = Registry()
    registry.tags[(tool.package("other"), tool.build_tag(OWN))] = LEGACY["other"]

    def read(reference: str) -> dict[str, str]:
        assert reference == f"{tool.repository(OWNER, 'other')}@{LEGACY['other']}"
        if labels is None:
            raise subprocess.CalledProcessError(1, ["skopeo"])
        return labels

    selection = _run(registry, candidates={"other"}, rebuild={"other"}, labels=read)
    assert [row["build_key"] for row in selection["builds"]] == (
        [OWN] if rebuilt else []
    )
    assert selection["kept"] == ([] if rebuilt else ["other"])
    assert selection["copies"] == []


def test_rebuild_adopted_needs_named_recipes(tmp_path, monkeypatch):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(_plan()))
    monkeypatch.setattr(
        sys,
        "argv",
        ["prebuilt-images", "select", "--plan", str(plan), "--owner", OWNER,
         "--slugs", "all", "--rebuild-adopted", "--output", str(tmp_path / "s.json")],
    )  # fmt: skip
    with pytest.raises(SystemExit, match="needs the recipes named"):
        tool.main()


def _library(root: Path, dockerfile: bytes = b"FROM scratch\n") -> Path:
    """A built library of one recipe: its index row and its package."""
    context = "adapters/x/one"
    files = {f"{context}/Dockerfile": dockerfile, f"{context}/run.sh": b"exec x\n"}
    manifest = {
        "build_inputs": [{"kind": "oci-image", "reference": "r@sha256:" + "0" * 64}]
    }
    (root / "packages").mkdir(parents=True)
    with tarfile.open(root / "packages/one.tar.gz", "w:gz") as archive:
        for name, payload in {
            "manifest.json": json.dumps(manifest).encode(),
            **files,
        }.items():
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    document = {
        "identity": {"publisher": "vonk-forge", "slug": "one"},
        "release": {"version": "1.0"},
        "execution": {
            "build": {
                "base_image": {"repository": "r", "digest": "0" * 64},
                "context": {"path": context},
                "dockerfile": f"{context}/Dockerfile",
            }
        },
    }
    index = {
        "source_commit": "c" * 40,
        "recipes": [
            {
                "document": document,
                "content_sha256": "a" * 64,
                # The catalog derives it from the context bytes.
                "source_bundle_sha256": hashlib.sha256(dockerfile).hexdigest(),
                "source_path": "recipes/one.json",
                "package": {"path": "packages/one.tar.gz"},
            }
        ],
    }
    (root / "catalog-index.json").write_text(json.dumps(index))
    return root


def test_a_pull_request_with_unchanged_recipe_inputs_plans_no_build(tmp_path):
    policy = json.loads(build_plan.POLICY.read_text())
    base = build_plan.write_library_image_plan(
        _library(tmp_path / "base"), tmp_path / "base-out", policy
    )
    head = build_plan.write_library_image_plan(
        _library(tmp_path / "head"), tmp_path / "head-out", policy
    )
    assert base["skipped"] == []
    assert tool.changed_candidates(head, base) == set()
    [image] = head["images"]
    context = tmp_path / "head-out" / image["build_key"] / "context"
    assert sorted(path.name for path in context.iterdir()) == ["Dockerfile", "run.sh"]
    assert "context/Dockerfile" in image["recipe_build_arguments"]
    changed = build_plan.write_library_image_plan(
        _library(tmp_path / "changed", b"FROM scratch\nRUN true\n"),
        tmp_path / "changed-out",
        policy,
    )
    assert tool.changed_candidates(changed, base) == {"one"}


@pytest.mark.parametrize("command", ["select", "resolve"])
def test_an_unreadable_release_plans_and_pins_nothing(tmp_path, monkeypatch, command):
    # Legacy tags only: without the published pins every recipe would look
    # missing and be rebuilt (select) or unpinned (resolve).
    monkeypatch.setattr(tool, "published_pins", lambda repository: None)
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(_plan()))
    output = tmp_path / "out.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["prebuilt-images", command, "--plan", str(plan), "--owner", OWNER,
         "--repository", "o/r", "--output", str(output)],
    )  # fmt: skip
    with pytest.raises(SystemExit, match="not readable"):
        tool.main()
    assert not output.exists()
