"""A TTS recipe's smoke path never renders ``$VOICE`` without its voice.

The live sweep crash-looped on a TTS recipe: ``Sweep._chat_facts`` rendered the cases only
to ask "does this recipe have any?", and rendering ``$VOICE`` without the api document's
voice raised ``CaseRenderError``, which the sweep counted as a smoke-tool crash and retried
(reloading the 64 GB model each time).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from sweep_fakes import FakeModel, FakeRecipe, Gateway, make_sweep

from qualification.documents import recipe_api
from spark_sweep import run
from spark_sweep.definitions import Definitions
from spark_sweep.smoke import SmokeResult

ROOT = Path(__file__).resolve().parents[1]
SLUG = "qwen3-tts-12hz-1-7b-customvoice-vllm-omni-single"


@pytest.fixture
def gateway():
    gateway = Gateway()
    yield gateway
    gateway.stop()


def _tts_definitions(recipe: FakeRecipe, *, with_api: bool) -> Definitions:
    shared = json.loads((ROOT / "qualification" / "shared.json").read_text("utf-8"))
    definitions = Definitions(
        {
            "fixtures": shared["fixtures"],
            "service_case_templates": shared["service_case_templates"],
            "service_recipes": {
                recipe.key: {"alias": recipe.slug, "smoke_cases": ["TTS_SPEECH"]}
            },
            "recipes": {},
        },
        ROOT / "qualification",
    )
    if with_api:
        definitions._recipe_apis[SLUG] = recipe_api(SLUG, ROOT)
    return definitions


def _drive(sweep: Any, clock: Any, done: Any) -> None:
    for _ in range(4000):
        sweep.tick()
        if done():
            return
        clock.sleep(sweep.cfg.poll_seconds)
    raise AssertionError("the sweep never reached the expected state")


def _codes(tmp_path: Path) -> list[str]:
    path = tmp_path / "results.jsonl"
    rows = path.read_text().splitlines() if path.exists() else []
    return [json.loads(line).get("code", "") for line in rows]


def test_a_tts_recipe_with_voices_passes_the_chat_facts_check(
    tmp_path: Path, gateway: Gateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe = FakeRecipe(SLUG)
    sweep, _, clock = make_sweep(tmp_path, [recipe], [FakeModel("m1")], gateway=gateway)
    sweep.defs = _tts_definitions(recipe, with_api=True)
    smoked: list[Any] = []

    def stub(*args: Any) -> SmokeResult:
        smoked.append(args[5])  # the facts the sweep resolved
        return SmokeResult(ok=True, kind="service")

    monkeypatch.setattr(run, "smoke_service", stub)
    _drive(sweep, clock, lambda: smoked)

    assert smoked == [None]  # reviewed cases carry their own request: no fallback facts
    assert "smoke.crash" not in _codes(tmp_path)
    assert "smoke-tool" not in sweep.state.data["infra"]


def test_a_tts_recipe_without_voices_is_one_typed_failure_not_a_retry_loop(
    tmp_path: Path, gateway: Gateway
) -> None:
    recipe = FakeRecipe(SLUG)
    sweep, _, clock = make_sweep(tmp_path, [recipe], [FakeModel("m1")], gateway=gateway)
    sweep.defs = _tts_definitions(recipe, with_api=False)

    _drive(sweep, clock, lambda: "case.cases" in _codes(tmp_path))
    for _ in range(300):  # a retry loop would keep going: run on and count
        sweep.tick()
        clock.sleep(sweep.cfg.poll_seconds)

    codes = _codes(tmp_path)
    assert codes.count("case.cases") == 1
    assert "smoke.crash" not in codes
    assert "smoke-tool" not in sweep.state.data["infra"]
