"""Audio serving checks: kinds, the multipart request, and the staged fixture."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from vonk_forge_contracts import RecipeDefinition
from vonk_forge_contracts.recipe import RecipeMultipartServingRequest
from vonk_forge_contracts.resolver import (
    ContractResolutionError,
    validate_recipe_package_paths,
)

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "contracts/src/vonk_forge_contracts/examples"
FIXTURE = "qualification/fixtures/jfk-inaugural-10s.wav"


def _example() -> dict[str, object]:
    return json.loads((EXAMPLES / "recipe-audio-serving.json").read_text("utf-8"))


def _checks(document: dict[str, object]) -> list[dict[str, object]]:
    return document["validation"]["serving"]["checks"]  # type: ignore[index]


def test_the_audio_example_carries_both_kinds_and_stages_one_fixture() -> None:
    recipe = RecipeDefinition.model_validate(_example())
    kinds = [check.kind for check in recipe.validation.serving.checks]
    assert kinds[-2:] == ["openai.transcription", "openai.speech"]
    transcribe = recipe.validation.serving.checks[-2]
    assert isinstance(transcribe.request, RecipeMultipartServingRequest)
    assert transcribe.fixture_paths() == [FIXTURE]
    assert recipe.validation.serving.checks[-1].fixture_paths() == []


def test_a_multipart_fixture_is_part_of_the_package_closure() -> None:
    recipe = RecipeDefinition.model_validate(_example())
    build = recipe.execution.build
    base = [build.context.path, build.dockerfile]
    validate_recipe_package_paths(recipe, [*base, FIXTURE])
    with pytest.raises(ContractResolutionError, match=FIXTURE):
        validate_recipe_package_paths(recipe, base)


@pytest.mark.parametrize(
    "change",
    [
        lambda check: check["request"]["file"].update(media_type="audio/mpeg"),
        lambda check: check["request"]["file"].update(field="audio"),
        lambda check: check["request"].update(path="/v1/audio/translations"),
        lambda check: check["request"].update(method="GET"),
        lambda check: check["request"].update(fields={}),
        lambda check: check["request"].update(fields={"language": "en"}),
        lambda check: check["request"]["fields"].update(file="x"),
        lambda check: check["request"].update(extra=1),
        lambda check: check["request"]["file"].update(fixture="/etc/passwd"),
    ],
)
def test_a_transcription_request_outside_the_shape_is_refused(change) -> None:  # type: ignore[no-untyped-def]
    document = _example()
    change(_checks(document)[-2])
    with pytest.raises(ValidationError):
        RecipeDefinition.model_validate(document)


@pytest.mark.parametrize(
    "body",
    [
        {"model": "m", "voice": "alloy", "response_format": "wav"},
        {"model": "m", "input": "x", "response_format": "wav"},
        {"model": "m", "input": "x", "voice": "alloy"},
        {"model": "m", "input": "x", "voice": "alloy", "response_format": "mp3"},
        {"model": "m", "input": "x", "voice": 7, "response_format": "wav"},
        {"model": "m", "input": ["x"], "voice": "alloy", "response_format": "wav"},
    ],
)
def test_a_speech_check_needs_input_voice_and_wav(body: dict[str, str]) -> None:
    document = _example()
    _checks(document)[-1]["request"]["body"] = body  # type: ignore[index]
    with pytest.raises(ValidationError, match="speech checks require"):
        RecipeDefinition.model_validate(document)


def test_a_speech_check_stays_at_its_own_path() -> None:
    document = _example()
    _checks(document)[-1]["request"]["path"] = "/v1/chat/completions"  # type: ignore[index]
    with pytest.raises(ValidationError, match="speech checks require"):
        RecipeDefinition.model_validate(document)


def test_the_raw_jfk_fixture_is_the_recorded_public_domain_clip() -> None:
    shared = json.loads((ROOT / "qualification/shared.json").read_text("utf-8"))
    entry = shared["fixtures"]["service-speech-jfk-wav"]
    raw = (ROOT / FIXTURE).read_bytes()
    assert len(raw) == entry["size_bytes"]
    assert hashlib.sha256(raw).hexdigest() == entry["sha256"]
    assert raw[:4] == b"RIFF" and raw[8:12] == b"WAVE"
    assert entry["provenance"]["license_spdx"] == "PD-USGov"
    assert "commons.wikimedia.org" in entry["provenance"]["source_url"]
