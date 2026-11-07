"""The deploy-order guard: a recipe may not use what the pinned platform cannot read."""

from __future__ import annotations

import json
import runpy
from pathlib import Path
from typing import Any, get_args

import pytest
from vonk_forge_contracts import CONTRACT_VERSION, RecipeApi, RecipeDefinition
from vonk_forge_contracts.introduced import (
    API_MODE_SINCE,
    SERVING_KIND_SINCE,
    parse_version,
    unreadable_features,
)
from vonk_forge_contracts.recipe import ServingKind

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "contracts/src/vonk_forge_contracts/examples"
GUARD = runpy.run_path(str(ROOT / "tools/check-platform-contracts"))


def _lock(version: str) -> str:
    return (
        'source = "https://github.com/CarstVaartjes/vonk-forge-recipes.git"\n'
        'subdirectory = "contracts"\n'
        f'package = "vonk-forge-public-contracts=={version}"\n'
    )


def _recipe() -> RecipeDefinition:
    text = (EXAMPLES / "recipe-audio-serving.json").read_text("utf-8")
    return RecipeDefinition.model_validate(json.loads(text))


def _api(name: str) -> RecipeApi:
    text = (EXAMPLES / f"recipe-api-{name}.json").read_text("utf-8")
    return RecipeApi.model_validate(json.loads(text))


def test_every_serving_kind_and_api_mode_states_when_it_arrived() -> None:
    assert set(SERVING_KIND_SINCE) == set(get_args(ServingKind))
    blocks = get_args(RecipeApi.model_fields["api"].annotation)
    modes = {get_args(block.model_fields["mode"].annotation)[0] for block in blocks}
    assert set(API_MODE_SINCE) == modes
    for since in [*SERVING_KIND_SINCE.values(), *API_MODE_SINCE.values()]:
        assert parse_version(since) <= parse_version(CONTRACT_VERSION)


def test_the_audio_kinds_arrived_in_4_4_0_and_the_modes_in_4_3_0() -> None:
    assert SERVING_KIND_SINCE["openai.transcription"] == "4.4.0"
    assert SERVING_KIND_SINCE["openai.speech"] == "4.4.0"
    assert API_MODE_SINCE["audio_transcription"] == "4.3.0"
    assert API_MODE_SINCE["audio_speech"] == "4.3.0"


def test_a_4_4_0_kind_recipe_fails_at_platform_4_1_0_and_passes_at_4_4_0() -> None:
    document = _recipe()
    assert unreadable_features(document, None, "4.1.0") == [
        "serving kind openai.speech needs contracts 4.4.0",
        "serving kind openai.transcription needs contracts 4.4.0",
    ]
    assert unreadable_features(document, None, "4.4.0") == []
    assert unreadable_features(document, None, "5.0.0") == []


def test_an_api_mode_needs_its_own_minimum() -> None:
    api = _api("speech")
    assert unreadable_features(_plain_recipe(), api, "4.2.9") == [
        "api mode audio_speech needs contracts 4.3.0"
    ]
    assert unreadable_features(_plain_recipe(), api, "4.3.0") == []


def _plain_recipe() -> RecipeDefinition:
    data: dict[str, Any] = json.loads(
        (EXAMPLES / "recipe-audio-serving.json").read_text("utf-8")
    )
    checks = data["validation"]["serving"]["checks"]
    data["validation"]["serving"]["checks"] = [
        c for c in checks if not c["kind"].endswith(("transcription", "speech"))
    ]
    return RecipeDefinition.model_validate(data)


def test_the_platform_version_is_read_from_the_lock() -> None:
    assert GUARD["platform_contracts_version"](_lock("4.1.0")) == "4.1.0"
    for bad in ("", _lock("4.1.0") * 2, _lock("4.1")):
        with pytest.raises(ValueError):
            GUARD["platform_contracts_version"](bad)


def test_the_library_is_readable_by_the_pinned_platform() -> None:
    # Against the contracts version this checkout publishes the library must pass.
    slugs = sorted(p.stem for p in (ROOT / "recipes").glob("*.json"))
    assert GUARD["problems"](ROOT, slugs, CONTRACT_VERSION) == []


def test_the_guard_names_the_recipe_and_both_versions(tmp_path: Path) -> None:
    for folder in ("recipes", "display"):
        (tmp_path / folder).mkdir()
    (tmp_path / "recipes/x.json").write_text(
        (EXAMPLES / "recipe-audio-serving.json").read_text("utf-8"), encoding="utf-8"
    )
    old = GUARD["problems"](tmp_path, ["x"], "4.1.0")
    assert old == [
        "x: serving kind openai.speech needs contracts 4.4.0, the pinned platform reads 4.1.0",
        "x: serving kind openai.transcription needs contracts 4.4.0, the pinned platform reads 4.1.0",
    ]
    assert GUARD["problems"](tmp_path, ["x"], "4.4.0") == []
