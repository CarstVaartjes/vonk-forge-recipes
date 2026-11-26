"""Shared service cases: functional assertions gate, quality assertions inform."""

from __future__ import annotations

import json
from datetime import timedelta
from typing import Any

import pytest
from documents import recipe as read_recipe
from repo_root import ROOT

from recipe_tooling.qualification.case_assertions import (
    invalid_case_assertions,
    require_valid_case_assertions,
)
from recipe_tooling.qualification.definitions_loader import load_definitions
from spark_sweep.smoke import check_assertions, check_quality
from vonk_forge_contracts import (
    QualificationEvidenceRecord,
    QualificationOutcome,
    QualificationRuntimeImage,
    QualificationStatus,
    QualificationUnverifiedReason,
    RecipeQualificationEvidence,
    published_status,
)
from vonk_forge_contracts.canonical import document_sha256
from vonk_forge_contracts.recipe import RecipeJobSettings

TEMPLATES: dict[str, Any] = json.loads(
    (ROOT / "qualification" / "shared.json").read_text(encoding="utf-8")
)["service_case_templates"]


def _chat(content: Any, **message: Any) -> dict[str, Any]:
    return {
        "model": "alias",
        "choices": [
            {
                "message": {"content": content, **message},
                "finish_reason": "stop",
            }
        ],
    }


def _tool_call(name: str, arguments: str) -> dict[str, Any]:
    return {
        "model": "alias",
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {"function": {"name": name, "arguments": arguments}}
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }


def _functional(case: str, response: dict[str, Any]) -> None:
    template = json.loads(
        json.dumps(TEMPLATES[case]["assertions"]).replace("$ALIAS", "alias")
    )
    check_assertions(response, json.dumps(response), template)


def _quality(case: str, response: dict[str, Any]) -> list[dict[str, Any]]:
    return check_quality(
        case, response, json.dumps(response), TEMPLATES[case]["quality_assertions"]
    )


# (case, a wrong but well-formed reply, the right reply)
ANSWERS = [
    ("A391", _chat("305"), _chat("391")),
    ("A323", _chat("324"), _chat("323")),
    ("V_RED", _chat("blue"), _chat("red.")),
    ("V7", _chat("1"), _chat("7")),
    (
        "T_REPORT",
        _tool_call("report_temperature", '{"city": "Rotterdam"}'),
        _tool_call("report_temperature", '{"city": "Amsterdam"}'),
    ),
    (
        "T_PRODUCT",
        _tool_call("record_product", '{"left": 17, "right": 19, "product": 320}'),
        _tool_call("record_product", '{"product": 323, "right": 19, "left": 17}'),
    ),
]


@pytest.mark.parametrize(("case", "wrong", "right"), ANSWERS)
def test_a_wrong_answer_passes_the_functional_check_and_misses_quality(
    case: str, wrong: dict[str, Any], right: dict[str, Any]
) -> None:
    _functional(case, wrong)
    [miss] = [note for note in _quality(case, wrong) if not note["ok"]]
    assert miss["case"] == case
    assert miss["got"]
    _functional(case, right)
    hits = _quality(case, right)
    assert hits
    assert all(hit["ok"] and hit["expected"] for hit in hits)


def test_a_correct_answer_in_prose_runs_with_a_format_note() -> None:
    """phi-4-multimodal-instruct (sglang) answered A323 in a sentence: it runs."""
    phi4 = _chat("The product of 17 multiplied by 19 is 323.")
    _functional("A323", phi4)
    notes = _quality("A323", phi4)
    assert [(n["expected"], n["ok"]) for n in notes] == [
        ("323", True),
        ("only the integer in content", False),
    ]


# (case, a correct answer in prose, the answer, the format note)
PROSE = [
    ("A391", "The answer is 391.", "391", "only the integer in content"),
    ("NEM_A391_OFF", "17 x 23 = 391", "391", "only the integer in content"),
    ("NEM_REASON", "So the result is 391.", "391", "only the integer in content"),
    ("V7", "The digit is 7.", "7", "only the digit in content"),
    ("V_RED", "The color is red.", "red", "only one word in content"),
    ("LFM_OCR", "The text says 7", "7", "only one character in content"),
]


@pytest.mark.parametrize(("case", "content", "answer", "format_note"), PROSE)
def test_an_answer_in_the_wrong_format_runs_with_a_format_note(
    case: str, content: str, answer: str, format_note: str
) -> None:
    """The format of the answer belongs to the creator: a quality note, not a failure."""
    reasoning = {"NEM_REASON": "thinking", "NEM_A391_OFF": None}
    message = {"reasoning_content": reasoning[case]} if case in reasoning else {}
    reply = _chat(content, **message)
    if case == "LFM_OCR":
        reply["usage"] = {"completion_tokens": 5}
    _functional(case, reply)
    notes = _quality(case, reply)
    assert [(n["expected"], n["ok"]) for n in notes] == [
        (answer, True),
        (format_note, False),
    ]


@pytest.mark.parametrize(
    ("case", "broken"),
    [
        ("A391", _chat("")),
        ("A391", _chat(None)),
        ("A391", _chat("391", reasoning_content="<think>x</think>")),
        ("A323", _chat("323 <|im_end|>")),
        ("A323", _chat("")),
        ("A323", _chat(None)),
        ("A323", _chat("323", reasoning_content="<think>x</think>")),
        ("NEM_A391_OFF", _chat("")),
        ("NEM_A391_OFF", _chat("391", reasoning_content="x")),
        ("NEM_REASON", _chat("", reasoning_content="x")),
        ("NEM_REASON", _chat("391")),
        ("V_RED", _chat("")),
        ("V_RED", _chat(None)),
        ("V7", _chat("")),
        ("V7", _chat(None)),
        ("LFM_OCR", _chat("")),
        ("LFM_OCR", _chat("7 <|im_end|>")),
        ("T_REPORT", _tool_call("report_temperature", "{}")),
        ("T_REPORT", _tool_call("other_tool", '{"city": "Amsterdam"}')),
        ("T_REPORT", _tool_call("report_temperature", "not json")),
        ("T_PRODUCT", _tool_call("record_product", '{"left": 17}')),
        (
            "T_PRODUCT",
            _tool_call("record_product", '{"left": "a", "right": 1, "product": 2}'),
        ),
    ],
)
def test_a_malformed_reply_fails_the_functional_check(
    case: str, broken: dict[str, Any]
) -> None:
    with pytest.raises(Exception, match="failed|contains|missing"):
        _functional(case, broken)


def test_every_shared_case_is_valid_and_quality_names_its_expectation() -> None:
    require_valid_case_assertions(load_definitions())
    assert any(t.get("quality_assertions") for t in TEMPLATES.values())


@pytest.mark.parametrize(
    ("template", "problem"),
    [
        ({"assertions": []}, "needs functional assertions"),
        (
            {
                "assertions": [{"kind": "path.nonempty", "path": "x"}],
                "quality_assertions": [
                    {"kind": "path.equals", "path": "x", "value": 1}
                ],
            },
            "must say what it expected",
        ),
        (
            {
                "assertions": [{"kind": "path.nonempty", "path": "x"}],
                "quality_assertions": [
                    {"kind": "raw.not-contains", "values": [], "expected": "x"}
                ],
            },
            "must name the text",
        ),
        (
            {
                "assertions": [{"kind": "path.nonempty", "path": "x", "expected": "x"}],
            },
            "unknown fields",
        ),
        (
            {
                "assertions": [{"kind": "path.regex", "path": "x", "value": "("}],
            },
            "does not compile",
        ),
        (
            {"assertions": [{"kind": "path.shrug", "path": "x"}]},
            "unsupported kind",
        ),
    ],
)
def test_the_split_is_enforced(template: dict[str, Any], problem: str) -> None:
    problems = invalid_case_assertions({"service_case_templates": {"X": template}})
    assert any(problem in p for p in problems)
    with pytest.raises(ValueError):
        require_valid_case_assertions({"service_case_templates": {"X": template}})


@pytest.mark.parametrize("case", ["A391_UNPARSED", "A323_UNPARSED"])
def test_unparsed_thinking_in_content_is_valid_and_only_a_quality_note(
    case: str,
) -> None:
    """A recipe without a reasoning parser serves the model's raw output."""
    raw = _chat('<think>User asks: "What is 17 mult')
    _functional(case, raw)
    misses = [item for item in _quality(case, raw) if not item["ok"]]
    assert {item["expected"] for item in misses} >= {
        "only the integer in content",
        "no reasoning tags in content",
    }
    answered = _chat("391" if case.startswith("A391") else "323")
    _functional(case, answered)
    assert all(item["ok"] for item in _quality(case, answered))


@pytest.mark.parametrize("case", ["A391_UNPARSED", "A323_UNPARSED"])
@pytest.mark.parametrize("broken", [_chat(""), _chat(None), _chat("3 <|channel>")])
def test_unparsed_cases_still_fail_an_empty_or_leaking_reply(
    case: str, broken: dict[str, Any]
) -> None:
    with pytest.raises(Exception, match="failed|contains|missing"):
        _functional(case, broken)


# These recipes have hardware passes, but no job knobs for the expensive
# defaults. Adding them would change the proven Recipe identity. Hunyuan3D-Omni's
# knobs choose its control, not a cheaper smoke, so it stays here.
PROVEN_FIXED_MEDIA = {
    "hunyuan-video-foley-xl-pytorch-single",
    "hunyuan-video-foley-xxl-pytorch-single",
    "hunyuan3d-omni-pytorch-single",
    "ltx-2-19b-dev-bf16-diffusers-single",
    "ltx-2-19b-distilled-fp8-diffusers-single",
    "ltx-2-3-22b-distilled-1-1-diffusers-single",
    "minimax-music3-drowzeys-tensorfold-single",
    "pixal3d-pytorch-single",
    "step1x-3d-geometry-pytorch-single",
    "step1x-3d-label-geometry-pytorch-single",
    "step1x-3d-texture-pytorch-single",
    "triposg-pytorch-single",
    "wan-2-2-i2v-14b-comfyui-single",
    "wan-2-2-t2v-14b-comfyui-single",
    "wan-2-2-ti2v-5b-comfyui-single",
}
MINIMAL_MEDIA_PARAMETERS = {
    "mova-360p-diffusers-single": {"num-frames": 1, "num-inference-steps": 2},
    "mova-720p-diffusers-single": {"num-frames": 1, "num-inference-steps": 2},
    "hunyuan-video-15-distilled-diffusers-single": {
        "num-frames": 1,
        "num-inference-steps": 2,
    },
    "hunyuan-video-15-t2v-diffusers-single": {
        "num-frames": 1,
        "num-inference-steps": 2,
    },
    "hunyuan-video-15-i2v-step-distilled-diffusers-single": {"num-frames": 1},
    "ltx-2-19b-dev-fp4-pytorch-single": {"num-frames": 1, "num-inference-steps": 2},
    "fasth3-8step-v2-drowzeys-tensorfold-single": {"num-frames": 124},
    "minimax-h3-keys-heretic-drowzeys-comfyui-single": {"num-frames": 124},
    "stable-audio-3-small-sfx-pytorch-single": {"duration-seconds": 2},
    # the 1 second smoke video caps the duration at its own length
    "mmaudio-large-44k-v2-pytorch-single": {"duration-seconds": 1.0},
    "minimax-h3-diffusers-single": {"num-frames": 124, "num-inference-steps": 4},
    "minimax-h3-fl2va-diffusers-single": {"num-frames": 124, "num-inference-steps": 4},
    "minimax-h3-fl2va-knapcio-vllm-omni-dual": {"num-inference-steps": 2},
    "minimax-h3-ref2va-knapcio-vllm-omni-dual": {
        "duration": 16,
        "num-inference-steps": 2,
    },
    "wan-dancer-14b-pytorch-single": {
        "style": "street",
        "num-inference-steps-global": 2,
        "num-inference-steps-local": 2,
    },
    "wan-dancer-14b-disk-offload-pytorch-single": {
        "style": "street",
        "num-inference-steps-global": 2,
        "num-inference-steps-local": 2,
    },
}
# These unproven recipes share their entire context with a hardware pass.
# Keep the single adapter unchanged rather than fork it for smaller smokes.
SHARED_PROVEN_MEDIA = {
    "ltx-2-19b-distilled-diffusers-single": "ltx-2-19b-distilled-fp8-diffusers-single",
    "trellis-2-4b-pytorch-single": "pixal3d-pytorch-single",
}
# The audio length follows the smoke fixture's single short sentence, so the
# smoke is minimal by its input rather than by a generation control.
MINIMAL_BY_FIXTURE_MEDIA = {"qwen3-tts-12hz-1-7b-pytorch-single"}


def test_every_media_smoke_has_minimal_controls_or_reviewed_identity_boundary():
    """New media cases cannot silently inherit a full-quality generation."""
    seen: set[str] = set()
    for path in sorted((ROOT / "qualification/recipes").glob("*.json")):
        definition = json.loads(path.read_text())
        spec = definition.get("recipes", {})
        if spec.get("interface") not in {"video-job", "audio-job", "mesh-job"}:
            continue
        slug = path.stem
        seen.add(slug)
        if slug in MINIMAL_MEDIA_PARAMETERS:
            assert spec["parameters"] == MINIMAL_MEDIA_PARAMETERS[slug]
            recipe = read_recipe(slug)
            bindings = {a.setting for a in recipe.runtime.arguments}
            assert set(spec["parameters"]) <= bindings
            assert isinstance(recipe.settings, RecipeJobSettings)
            knobs = recipe.settings.knobs
            assert set(spec["parameters"]) <= knobs.keys()
            for name, value in spec["parameters"].items():
                knob = knobs[name]
                if knob.allowed_values is not None:
                    assert value in knob.allowed_values, name
                if knob.minimum is not None and knob.maximum is not None:
                    assert knob.minimum <= value <= knob.maximum, name
                    if knob.step is not None:
                        assert (value - knob.minimum) % knob.step == 0, name
            if "num-frames" in spec["parameters"]:
                video = next(
                    a for a in spec["assertions"] if a["kind"] == "video-metadata"
                )
                assert video["frame_count"] == spec["parameters"]["num-frames"]
                duration = video["frame_count"] / video["fps"]
                assert (
                    video["minimum_duration_seconds"]
                    <= duration
                    <= video["maximum_duration_seconds"]
                )
        elif slug in MINIMAL_BY_FIXTURE_MEDIA:
            assert not spec.get("parameters")
            # The inputs are the reviewed fixtures (their bytes are sha256-pinned
            # in qualification/shared.json), not a size or duration bound.
            assert {item["fixture"] for item in spec["inputs"]} == {
                "tts-text",
                "tts-voice",
            }
            assert [
                {item["fixture"] for item in case["inputs"]}
                for case in spec.get("cases", [])
            ] == [{"tts-text", "tts-reference-audio", "tts-reference-text"}]
        elif slug in SHARED_PROVEN_MEDIA:
            assert SHARED_PROVEN_MEDIA[slug] in PROVEN_FIXED_MEDIA
            recipe = json.loads((ROOT / "recipes" / path.name).read_text())
            proven = json.loads(
                (ROOT / "recipes" / f"{SHARED_PROVEN_MEDIA[slug]}.json").read_text()
            )
            assert recipe["execution"]["build"] == proven["execution"]["build"]
            assert not spec.get("parameters")
            assert not recipe["settings"]["knobs"]
        elif slug == "skintokens-pytorch-single":
            # One tiny rigged mesh, with no iterative generation controls.
            assert spec["inputs"] == [
                {"slot": "mesh", "fixture": "skintokens-rigged-figure"}
            ]
        else:
            assert slug in PROVEN_FIXED_MEDIA, (
                f"review minimal smoke controls for {slug}"
            )
    assert seen == (
        PROVEN_FIXED_MEDIA
        | MINIMAL_MEDIA_PARAMETERS.keys()
        | MINIMAL_BY_FIXTURE_MEDIA
        | SHARED_PROVEN_MEDIA.keys()
        | {"skintokens-pytorch-single"}
    )


def test_qualification_parameters_do_not_enter_recipe_content_identity(built_catalog):

    catalog, _ = built_catalog
    for entry in catalog["recipes"]:
        if (
            entry["document"]["identity"]["slug"]
            in MINIMAL_MEDIA_PARAMETERS.keys()
            | SHARED_PROVEN_MEDIA.keys()
            | PROVEN_FIXED_MEDIA
        ):
            document = json.loads(
                (
                    ROOT / "recipes" / f"{entry['document']['identity']['slug']}.json"
                ).read_text()
            )
            assert entry["content_sha256"] == document_sha256(document)


def test_protocol_refresh_preserves_prior_pass_without_certifying_new_context() -> None:

    slug = "ltx-2-19b-distilled-fp8-diffusers-single"
    # Freeze the actual historical input as a literal: retention drops old records.
    passed = QualificationEvidenceRecord.model_validate(
        {
            "identity": {
                "content_sha256": "5e875bed66400c52901e70c6d0480f308a71766de3e73055fe8d733edd9b3de2",
                "model_digests": [
                    "7bcc3f9e353351a39e3167859c2d9d4a44d71c0a27dd11420c44b7032cdcba5a",
                    "87499d7f112bb61c3736f19f9066b6e808ce3ea0708ded89a236ba432b2bf665",
                ],
                "required_sparks": 1,
                "models_resolved": True,
            },
            "outcome": "passed",
            "cause_code": "qualification.hardware_pass",
            "cause_summary": "Hardware checks passed",
            "evidence_refs": [
                "232803cbe0cc81f2810c6326e59b5681adc6c423027303feb45d0ad33976973f"
            ],
            "tested_at": "2026-10-21T02:47:59.039345Z",
            "platform": {
                "controller_source_sha": "a9583d00aa3b1bf57945692feba3a4a0f0be646c",
                "hardware_class": "GB10",
                "spark_count": 1,
                "topology": "single",
            },
            "quality_notes": [
                "Carried from the identical document before the display split"
            ],
        }
    )
    assert passed.outcome == QualificationOutcome.PASSED
    history = RecipeQualificationEvidence(
        recipe_key=f"vonk-forge/{slug}", records=(passed,)
    )
    assert passed.identity.runtime_image is None
    # This prospective image is a synthetic fixture, never inferred from the
    # reviewed source-bundle hash or written into public hardware evidence.
    current = passed.identity.model_copy(
        update={
            "runtime_image": QualificationRuntimeImage(
                build_key="f" * 64,
                image_digest="sha256:" + "e" * 64,
            )
        }
    )
    view = published_status(history, current)
    assert view.last_pass == passed
    assert view.status == QualificationStatus.NOT_YET_VERIFIED
    assert view.reason == QualificationUnverifiedReason.INCOMPLETE_PROVENANCE
    # Fresh, complete evidence clears the uncertainty without a manual waiver.
    platform = passed.platform.model_copy(update={"agent_version": "0.0.1"})
    assert platform.resolved
    measured = passed.model_copy(
        update={
            "identity": current,
            "platform": platform,
            "tested_at": passed.tested_at + timedelta(seconds=1),
        }
    )
    updated = history.model_copy(update={"records": (*history.records, measured)})
    recovered = published_status(updated, current)
    assert recovered.status == QualificationStatus.WORKING
    assert recovered.last_pass == measured
    assert history.records == (passed,)
