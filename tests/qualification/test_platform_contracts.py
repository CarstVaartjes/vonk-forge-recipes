"""The channel gate: a recipe may not use a discriminator the platform does not execute."""

from __future__ import annotations

import json
import runpy
import shutil
from pathlib import Path
from typing import Any

from repo_root import ROOT

from vonk_forge_contracts import CONTRACT_MAJOR, RecipeApi, read_recipe

EXAMPLES = ROOT / "contracts/src/vonk_forge_contracts/examples"
GUARD = runpy.run_path(str(ROOT / "tools/check-platform-contracts"))


def _features(tmp_path: Path, **override: Any) -> Path:
    document = {
        "contracts_version": "5.7.0",
        "contract_major": CONTRACT_MAJOR,
        "serving_kinds": ["openai.speech", "openai.transcription", "openai.vision"],
        "effort_kinds": ["levels", "text"],
        "api_modes": ["chat"],
        "adapters": ["openai"],
        "source_providers": ["huggingface"],
        **override,
    }
    path = tmp_path / "platform-features.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _library(tmp_path: Path) -> Path:
    """One audio recipe, its Model, and a Model display with a capability no platform lists."""

    root = tmp_path / "library"
    for directory in ("recipes", "models", "display/models"):
        (root / directory).mkdir(parents=True)
    shutil.copy(EXAMPLES / "recipe-audio-serving.json", root / "recipes/x.json")
    shutil.copy(
        EXAMPLES / "model-definition.json", root / "models/synthetic-tiny-fp16.json"
    )
    display = json.loads((EXAMPLES / "model-display.json").read_text("utf-8"))
    display["capabilities"] = ["zz-new-capability"]
    (root / "display/models/synthetic-tiny-fp16.json").write_text(
        json.dumps(display), encoding="utf-8"
    )
    return root


def test_a_serving_kind_the_platform_does_not_execute_fails(tmp_path: Path) -> None:
    root = _library(tmp_path)
    without_speech = _features(
        tmp_path, serving_kinds=["openai.transcription", "openai.vision"]
    )
    argv = ["--library-root", str(root), "--platform-features"]
    assert GUARD["main"]([*argv, str(without_speech)]) == 1


def test_the_stable_report_warns_on_words_and_fails_only_on_the_major(
    tmp_path: Path, capsys
) -> None:
    root = _library(tmp_path)
    argv = ["--library-root", str(root), "--report-words", "--platform-features"]
    without_speech = _features(
        tmp_path, serving_kinds=["openai.transcription", "openai.vision"]
    )
    assert GUARD["main"]([*argv, str(without_speech)]) == 0
    assert "openai.speech" in capsys.readouterr().out
    older = _features(tmp_path, contract_major=CONTRACT_MAJOR - 1)
    assert GUARD["main"]([*argv, str(older)]) == 1


def test_an_unlisted_capability_is_not_gated(tmp_path: Path) -> None:
    root = _library(tmp_path)
    full = _features(tmp_path)
    assert (
        GUARD["main"](["--library-root", str(root), "--platform-features", str(full)])
        == 0
    )


def test_api_modes_and_effort_kinds_are_executed_words() -> None:
    recipe = read_recipe(
        json.loads((EXAMPLES / "recipe-options.json").read_text("utf-8"))
    )
    document = json.loads((EXAMPLES / "recipe-api-options.json").read_text("utf-8"))
    for choice in document["option_choices"]:
        choice["api"]["reasoning"] = {
            "kind": "always-on",
            "effort": {
                "kind": "text",
                "via": "top_level",
                "key": "reasoning_effort",
                "default": None,
            },
        }
    used = GUARD["words_used"](
        recipe, ["github-lfs"], RecipeApi.model_validate(document)
    )
    assert {
        ("api_modes", "chat"),
        ("effort_kinds", "text"),
        ("source_providers", "github-lfs"),
    } <= used


def test_a_contracts_major_ahead_of_the_platform_is_refused_one_way_only() -> None:
    major_problem = GUARD["major_problem"]
    assert major_problem(6, 5)
    assert major_problem(5, 5) is None
    assert major_problem(5, 6) is None
