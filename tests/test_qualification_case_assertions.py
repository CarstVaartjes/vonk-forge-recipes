"""Shared service cases: functional assertions gate, quality assertions inform."""

from __future__ import annotations

import hashlib
import json
import runpy
import shutil
from pathlib import Path
from typing import Any

import pytest

from qualification.case_assertions import (
    invalid_case_assertions,
    require_valid_case_assertions,
)
from qualification.definitions_loader import load_definitions
from spark_sweep.smoke import check_assertions, check_quality

ROOT = Path(__file__).resolve().parents[1]
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
    assert miss["case"] == case and miss["got"]
    _functional(case, right)
    hits = _quality(case, right)
    assert hits and all(hit["ok"] and hit["expected"] for hit in hits)


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


# These recipes have hardware passes, but no job knobs/request controls for the
# expensive defaults. Adding them would change the proven Recipe identity.
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
}
# These unproven recipes share their entire context with a hardware pass.
# Keep the single adapter unchanged rather than fork it for smaller smokes.
SHARED_PROVEN_MEDIA = {
    "ltx-2-19b-distilled-diffusers-single": "ltx-2-19b-distilled-fp8-diffusers-single",
    "trellis-2-4b-pytorch-single": "pixal3d-pytorch-single",
}
REQUEST_CONTROLLED_MEDIA = {
    "minimax-h3-diffusers-single",
    "minimax-h3-fl2va-diffusers-single",
    "wan-dancer-14b-pytorch-single",
    "wan-dancer-14b-disk-offload-pytorch-single",
}


def test_every_media_smoke_has_minimal_controls_or_reviewed_identity_boundary():
    """New media cases cannot silently inherit a full-quality generation."""
    definitions = load_definitions()
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
            recipe = json.loads((ROOT / "recipes" / path.name).read_text())
            bindings = {a.get("setting") for a in recipe["runtime"]["arguments"]}
            assert set(spec["parameters"]) <= bindings
            assert set(spec["parameters"]) <= recipe["settings"]["knobs"].keys()
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
        elif slug in REQUEST_CONTROLLED_MEDIA:
            fixtures = definitions["fixtures"]
            for case in [spec, *spec.get("cases", [])]:
                control = next(
                    i for i in case["inputs"] if i["slot"] in {"request", "controls"}
                )
                request = json.loads(
                    (
                        ROOT / "qualification" / fixtures[control["fixture"]]["path"]
                    ).read_text()
                )
                if slug.startswith("minimax"):
                    assert request["profile"] == "smoke-only"
                    assert request["num_frames"] == 120
                    video = next(
                        a for a in spec["assertions"] if a["kind"] == "video-metadata"
                    )
                    assert video["frame_count"] == request["num_frames"]
                    duration = request["num_frames"] / video["fps"]
                    assert (
                        video["minimum_audio_duration_seconds"]
                        <= duration
                        <= video["maximum_audio_duration_seconds"]
                    )
                else:
                    assert request["num_inference_steps_global"] == 2
                    assert request["num_inference_steps_local"] == 2
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
        | REQUEST_CONTROLLED_MEDIA
        | SHARED_PROVEN_MEDIA.keys()
        | {"skintokens-pytorch-single"}
    )


def test_qualification_parameters_do_not_enter_recipe_content_identity(built_catalog):
    from vonk_forge_contracts.canonical import document_sha256

    catalog, _ = built_catalog
    for entry in catalog["recipes"]:
        if (
            entry["document"]["identity"]["slug"]
            in MINIMAL_MEDIA_PARAMETERS.keys()
            | REQUEST_CONTROLLED_MEDIA
            | SHARED_PROVEN_MEDIA.keys()
            | PROVEN_FIXED_MEDIA
        ):
            document = json.loads(
                (
                    ROOT / "recipes" / f"{entry['document']['identity']['slug']}.json"
                ).read_text()
            )
            assert entry["content_sha256"] == document_sha256(document)


# Reviewed source identities, not hardware verdicts. Pixal3D's context matches
# its 2026-10-20 pass. LTX intentionally adopts the reviewed protocol artifact;
# its 2026-10-21 pass remains historical evidence, not proof of these new bytes.
# Recipe-only protection misses changes in a shared image build context.
REVIEWED_SHARED_IDENTITIES = {
    "ltx-2-19b-distilled-fp8-diffusers-single": (
        "2bdcd7561497a20a66cd02cde2db09e5db89a866ca73c4842ed65e12e2a864c3",
        "8baa9284cbeb16f52a453735c2abece195bed47934e8f7ba9bc03bc9e4593229",
    ),
    "pixal3d-pytorch-single": (
        "fbcc650e981638662e48920c0f867ffc7dd69302fbe8749545c26159ab824b6e",
        "7d171fe64ef2da27ed5bc9d31da69a3c2ec8dd54139961bba320afd7cc81354a",
    ),
}


def _assert_reviewed_shared_identity(root: Path, slug: str) -> None:
    recipe_path = root / "recipes" / f"{slug}.json"
    recipe = json.loads(recipe_path.read_text())
    expected_recipe, expected_source = REVIEWED_SHARED_IDENTITIES[slug]
    assert hashlib.sha256(recipe_path.read_bytes()).hexdigest() == expected_recipe, (
        f"{slug}: reviewed Recipe bytes changed"
    )
    bundle = runpy.run_path(str(ROOT / "tools/build-catalog-index"))["source_bundle"]
    context = root / recipe["execution"]["build"]["context"]["path"]
    assert bundle(context)[2] == expected_source, (
        f"{slug}: reviewed image context bytes changed"
    )


@pytest.mark.parametrize("slug", REVIEWED_SHARED_IDENTITIES)
def test_reviewed_shared_recipe_and_image_bytes(slug: str) -> None:
    _assert_reviewed_shared_identity(ROOT, slug)


@pytest.mark.parametrize("slug", REVIEWED_SHARED_IDENTITIES)
@pytest.mark.parametrize("mutate_context", [False, True])
def test_reviewed_identity_guard_detects_shared_context_or_document_change(
    tmp_path: Path, slug: str, mutate_context: bool
) -> None:
    recipe_path = Path("recipes") / f"{slug}.json"
    recipe = json.loads((ROOT / recipe_path).read_text())
    context = Path(recipe["execution"]["build"]["context"]["path"])
    (tmp_path / "recipes").mkdir()
    shutil.copy2(ROOT / recipe_path, tmp_path / recipe_path)
    shutil.copytree(ROOT / context, tmp_path / context)
    _assert_reviewed_shared_identity(tmp_path, slug)
    # Even an unused helper or whitespace changes the image's source identity.
    target = (
        tmp_path / context / "unused-helper.py"
        if mutate_context
        else tmp_path / recipe_path
    )
    with target.open("a") as stream:
        stream.write("\n")
    with pytest.raises(AssertionError, match="reviewed .* bytes changed"):
        _assert_reviewed_shared_identity(tmp_path, slug)


def test_protocol_refresh_preserves_prior_pass_without_certifying_new_context() -> None:
    from datetime import UTC, datetime, timedelta

    from vonk_forge_contracts import (
        QualificationOutcome,
        QualificationRuntimeImage,
        QualificationStatus,
        QualificationUnverifiedReason,
        published_status,
    )

    from spark_sweep.qualification import read_evidence

    slug = "ltx-2-19b-distilled-fp8-diffusers-single"
    historical_context = (
        "32d29663893be2ac98eeb7ef032d3e0a2fd68a61aa3c613c6b517aea5fb8e62e"
    )
    assert REVIEWED_SHARED_IDENTITIES[slug][1] != historical_context
    proof = read_evidence(ROOT, f"vonk-forge/{slug}")
    # Freeze the actual historical input: future evidence publication must not
    # turn this boundary regression into a constraint on the live catalog.
    passed = next(
        record
        for record in proof.records
        if record.outcome == QualificationOutcome.PASSED
        and record.tested_at == datetime(2026, 10, 21, 2, 47, 59, 39345, tzinfo=UTC)
    )
    history = proof.model_copy(update={"records": (passed,)})
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
