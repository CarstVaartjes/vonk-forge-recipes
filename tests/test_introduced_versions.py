"""The deploy-order guard: a recipe may not use what the pinned platform cannot read."""

from __future__ import annotations

import json
import runpy
import shutil
from pathlib import Path
from typing import get_args

import pytest
from documents import model_display
from vonk_forge_contracts import CapabilityName, RecipeDefinition, read_recipe
from vonk_forge_contracts.recipe import ServingKind

from qualification import introduced
from qualification.introduced import (
    CAPABILITY_SINCE,
    SERVING_KIND_SINCE,
    parse_version,
    unreadable_features,
)

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "contracts/src/vonk_forge_contracts/examples"
GUARD = runpy.run_path(str(ROOT / "tools/check-platform-contracts"))


def _lock(version: str) -> str:
    return (
        'source = "https://github.com/CarstVaartjes/vonk-forge-recipes.git"\n'
        'subdirectory = "contracts"\n'
        f'package = "vonk-forge-public-contracts=={version}"\n'
    )


def _audio_recipe() -> RecipeDefinition:
    return read_recipe(
        json.loads((EXAMPLES / "recipe-audio-serving.json").read_text("utf-8"))
    )


def _without_audio_kinds(recipe: RecipeDefinition) -> RecipeDefinition:
    serving = recipe.validation.serving
    checks = [
        c for c in serving.checks if not c.kind.endswith(("transcription", "speech"))
    ]
    kept = serving.model_copy(update={"checks": checks})
    return recipe.model_copy(
        update={"validation": recipe.validation.model_copy(update={"serving": kept})}
    )


def test_every_serving_kind_and_capability_states_when_it_arrived() -> None:
    assert set(SERVING_KIND_SINCE) == set(get_args(ServingKind))
    assert set(CAPABILITY_SINCE) == set(CapabilityName)
    for since in [*SERVING_KIND_SINCE.values(), *CAPABILITY_SINCE.values()]:
        parse_version(since)


def test_the_audio_kinds_arrived_in_4_4_0() -> None:
    assert SERVING_KIND_SINCE["openai.transcription"] == "4.4.0"
    assert SERVING_KIND_SINCE["openai.speech"] == "4.4.0"


def test_a_4_4_0_kind_recipe_fails_at_platform_4_1_0_and_passes_at_4_4_0() -> None:
    recipe = _audio_recipe()
    assert unreadable_features(recipe, [], "4.1.0") == [
        "serving kind openai.speech needs contracts 4.4.0",
        "serving kind openai.transcription needs contracts 4.4.0",
    ]
    assert unreadable_features(recipe, [], "4.4.0") == []
    assert unreadable_features(recipe, [], "5.0.0") == []
    assert unreadable_features(_without_audio_kinds(recipe), [], "4.1.0") == []


def test_a_capability_newer_than_the_platform_is_refused(monkeypatch) -> None:
    recipe = _without_audio_kinds(_audio_recipe())
    display = _any_display()
    name = display.capabilities[0]
    assert unreadable_features(recipe, [display], "4.1.0") == []
    monkeypatch.setattr(
        introduced, "CAPABILITY_SINCE", {**CAPABILITY_SINCE, name: "4.9.0"}
    )
    assert unreadable_features(recipe, [display], "4.8.0") == [
        f"capability {name.value} needs contracts 4.9.0"
    ]
    assert unreadable_features(recipe, [display], "4.9.0") == []


def _any_display():
    slug = min(p.stem for p in (ROOT / "display/models").glob("*.json"))
    return model_display(slug, ROOT)


def test_the_platform_version_is_read_from_the_lock() -> None:
    assert GUARD["platform_contracts_version"](_lock("4.1.0")) == "4.1.0"
    for bad in ("", _lock("4.1.0") * 2, _lock("4.1")):
        with pytest.raises(ValueError):
            GUARD["platform_contracts_version"](bad)


def test_the_guard_names_the_recipe_and_both_versions(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / "recipes").mkdir()
    shutil.copy(EXAMPLES / "recipe-audio-serving.json", tmp_path / "recipes/x.json")
    monkeypatch.setitem(
        GUARD["problems"].__globals__,
        "model_display",
        lambda slug, root: _any_display(),
    )
    old = GUARD["problems"](tmp_path, ["x"], "4.1.0")
    assert [m.split(",")[0] for m in old] == [
        "x: serving kind openai.speech needs contracts 4.4.0",
        "x: serving kind openai.transcription needs contracts 4.4.0",
    ]
    assert all(m.endswith("the pinned platform reads 4.1.0") for m in old)
    assert GUARD["problems"](tmp_path, ["x"], "4.4.0") == []


def test_models_no_recipe_selects_are_gated_too(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "display/models").mkdir(parents=True)
    display = _any_display()
    name = display.capabilities[0]
    monkeypatch.setitem(
        GUARD["model_problems"].__globals__, "model_display", lambda slug, root: display
    )
    (tmp_path / "display/models/orphan.json").touch()
    monkeypatch.setattr(
        introduced, "CAPABILITY_SINCE", {**CAPABILITY_SINCE, name: "4.9.0"}
    )
    assert GUARD["model_problems"](tmp_path, "4.8.0") == [
        f"model orphan: capability {name.value} needs contracts 4.9.0, "
        "the pinned platform reads 4.8.0"
    ]
    assert GUARD["model_problems"](tmp_path, "4.9.0") == []
