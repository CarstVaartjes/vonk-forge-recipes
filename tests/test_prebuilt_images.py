"""The image workflow builds each missing image once and the index pins it.

A Controller prefers the digest the signed index pins for a recipe, so the
selection must not rebuild an image that exists, must build a key shared by
several recipes once, and the index must pin exactly the digest the registry
serves for that recipe's key.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import runpy
import subprocess
import sys
from pathlib import Path

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


def _plan() -> dict[str, object]:
    def recipe(slug: str, context: str) -> dict[str, object]:
        return {
            "slug": slug,
            "version": "1.0",
            "content_sha256": slug[0] * 64,
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


def test_changed_paths_select_recipes_by_document_or_build_context() -> None:
    plan = _plan()
    assert tool.changed_slugs(plan, ["adapters/x/shared/Dockerfile"]) == {
        "single",
        "dual",
    }
    assert tool.changed_slugs(plan, ["recipes/other.json", "README.md"]) == {"other"}
    assert tool.changed_slugs(plan, ["adapters/x/shared-older/Dockerfile"]) == set()


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


def test_index_pins_each_recipes_own_digest_and_omits_missing_images() -> None:
    registry = _registry({("vonk-forge-recipe-other", f"build-{OWN}"): DIGEST})
    images, missing = tool.resolve(_plan(), owner=OWNER, lookup=registry)
    assert images == {
        "o" * 64: {
            "reference": f"ghcr.io/carstvaartjes/vonk-forge-recipe-other@{DIGEST}",
            "build_key": OWN,
        }
    }
    assert missing == ["dual", "single"]

    # The index carries the entry beside exactly the revision it belongs to.
    rows = [{"content_sha256": "o" * 64}, {"content_sha256": "s" * 64}]
    INDEX["pin_prebuilt_images"](rows, images)
    assert rows == [
        {"content_sha256": "o" * 64, "prebuilt_image": images["o" * 64]},
        {"content_sha256": "s" * 64},
    ]


OLD_PIN, NEW_PIN = "1" * 40, "2" * 40


def _publication_repo(tmp_path: Path, monkeypatch) -> str:
    """A repo whose last publication pinned OLD_PIN; returns that commit."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    workflow = tmp_path / ".github/workflows/publish.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text(f"with:\n  platform_ref: {OLD_PIN}\n", encoding="utf-8")
    git("init", "-q")
    git("add", "-A")
    git("commit", "-q", "-m", "published")
    baseline = git("rev-parse", "HEAD")
    monkeypatch.chdir(tmp_path)
    return baseline


def _repin(tmp_path: Path, pin: str) -> None:
    workflow = tmp_path / ".github/workflows/publish.yml"
    workflow.write_text(f"with:\n  platform_ref: {pin}\n", encoding="utf-8")
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qam", "next"],
        cwd=tmp_path,
        check=True,
    )


def test_the_platform_pin_is_read_from_the_publication_workflow() -> None:
    assert tool.platform_ref(f"x:\n  platform_ref: {OLD_PIN}\n") == OLD_PIN
    assert tool.platform_ref("platform_ref: not-a-commit\n") is None
    workflow = (ROOT / ".github/workflows/publish.yml").read_text(encoding="utf-8")
    assert tool.platform_ref(workflow) is not None


def test_a_moved_platform_pin_considers_every_recipe(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    baseline = _publication_repo(tmp_path, monkeypatch)
    workflow = Path(".github/workflows/publish.yml")

    # Same pin, nothing changed: nothing is a candidate.
    unchanged = tool.candidate_slugs(_plan(), [], baseline=baseline, workflow=workflow)
    assert unchanged == set()

    # A moved pin can move the keys of recipes nobody touched.
    _repin(tmp_path, NEW_PIN)
    moved = tool.candidate_slugs(_plan(), [], baseline=baseline, workflow=workflow)
    assert moved is None
    assert "platform ref moved" in capsys.readouterr().out

    # Only keys whose image is missing are built, so unchanged images stay.
    registry = _registry({("vonk-forge-recipe-other", f"build-{OWN}"): DIGEST})
    selection = tool.select(_plan(), owner=OWNER, candidates=moved, lookup=registry)
    assert [item["build_key"] for item in selection["builds"]] == [SHARED]


def test_explicit_requests_and_a_missing_baseline_ignore_the_pin(
    tmp_path: Path, monkeypatch
) -> None:
    baseline = _publication_repo(tmp_path, monkeypatch)
    workflow = Path(".github/workflows/publish.yml")
    _repin(tmp_path, NEW_PIN)
    assert tool.candidate_slugs(
        _plan(), ["other"], baseline=baseline, workflow=workflow
    ) == {"other"}
    assert (
        tool.candidate_slugs(_plan(), ["all"], baseline=baseline, workflow=workflow)
        is None
    )
    assert tool.candidate_slugs(_plan(), [], baseline="", workflow=workflow) == set()


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
