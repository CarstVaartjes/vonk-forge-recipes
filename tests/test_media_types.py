"""Reject bad catalog lists before publication; canonicalize outbound bindings."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError
from vonk_forge_contracts import RecipeDefinition

from qualification.definitions_loader import load_definitions
from qualification.media_types import (
    CanonicalMediaTypes,
    canonical_media_types,
    require_canonical_recipe_media,
)
from spark_sweep.definitions import Definitions
from spark_sweep.smoke_job import _binding

ROOT = Path(__file__).resolve().parents[1]
MOSS = "vonk-forge/moss-vl-realtime-11b-pytorch-single"
MOSS_MEDIA = ["application/x-ndjson", "video/mp4"]


@pytest.mark.parametrize("media", ["Video/MP4", "video/mp4; codecs=h264", "video/*"])
def test_shared_producer_rejects_unsupported_media_spellings(media: str) -> None:
    with pytest.raises(ValueError, match="lowercase and parameter-free"):
        canonical_media_types([media])


def test_moss_binding_canonicalizes_the_actual_failed_list(tmp_path: Path) -> None:
    definitions = Definitions.load()
    spec = definitions.artifact(MOSS)
    assert spec is not None
    # Inject the original definition, including a duplicate from merged slots.
    limits = {
        **spec["output_limits"],
        "allowed_media_types": [*reversed(MOSS_MEDIA), "video/mp4"],
    }
    binding = _binding({**spec, "output_limits": limits}, definitions, tmp_path)
    emitted = json.loads(binding.read_text())["create"]["output_limits"][
        "allowed_media_types"
    ]
    assert emitted == MOSS_MEDIA
    assert limits["allowed_media_types"] == [*reversed(MOSS_MEDIA), "video/mp4"]


@pytest.mark.parametrize(
    "media",
    [
        list(reversed(MOSS_MEDIA)),
        ["video/mp4", "video/mp4"],
        ["Video/MP4"],
        ["video/mp4; codecs=h264"],
        ["video/*"],
        [],
    ],
)
@pytest.mark.parametrize("case_override", [False, True])
def test_catalog_build_rejects_noncanonical_qualification_lists(
    tmp_path: Path, media: list[str], case_override: bool
) -> None:
    (tmp_path / "recipes").mkdir()
    (tmp_path / "shared.json").write_text("{}")
    path = ROOT / "qualification/recipes/moss-vl-realtime-11b-pytorch-single.json"
    entry = json.loads(path.read_text())
    spec = entry["recipes"]
    target = spec["cases"][0] if case_override else spec
    target["output_limits"] = {**spec["output_limits"], "allowed_media_types": media}
    (tmp_path / "recipes" / path.name).write_text(json.dumps(entry))
    with pytest.raises(ValueError, match="output_limits.allowed_media_types"):
        load_definitions(tmp_path)


@pytest.mark.parametrize(
    "media",
    [
        list(reversed(MOSS_MEDIA)),
        ["video/mp4", "video/mp4"],
        ["Video/MP4"],
        ["video/mp4; codecs=h264"],
    ],
)
def test_catalog_packaging_rejects_noncanonical_slot_lists(media: list[str]) -> None:
    document = json.loads(
        (ROOT / "recipes/moss-vl-realtime-11b-pytorch-single.json").read_text()
    )
    document["interfaces"][0]["output"]["slots"][0]["media_types"] = media
    with pytest.raises(ValidationError):
        runpy.run_path(str(ROOT / "tools/build-catalog-index"))["recipe_package"](
            document
        )


def test_all_catalog_recipes_and_qualification_cases_use_canonical_media() -> None:
    for path in (ROOT / "recipes").glob("*.json"):
        require_canonical_recipe_media(
            RecipeDefinition.model_validate_json(path.read_bytes())
        )
    definitions = load_definitions()
    adapter = TypeAdapter(CanonicalMediaTypes)
    for spec in definitions["recipes"].values():
        for case in [spec, *spec.get("cases", [])]:
            if "output_limits" in case:
                media = case["output_limits"]["allowed_media_types"]
                assert adapter.validate_python(media) == canonical_media_types(media)
    assert (
        definitions["recipes"][MOSS]["output_limits"]["allowed_media_types"]
        == MOSS_MEDIA
    )
