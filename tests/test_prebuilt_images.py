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
