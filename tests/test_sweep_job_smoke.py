"""A job recipe's smoke: readiness by a held run, one artifact job, its output assertions."""

from __future__ import annotations

import io
import json
import struct
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from spark_sweep import verdict
from spark_sweep.catalog import Fleet, Presence, parse_recipe, serving_run
from spark_sweep.definitions import Definitions
from spark_sweep.job_assertions import EVALUATED, NOT_EVALUATED, Output, evaluate
from spark_sweep.smoke import AssertionFailed
from spark_sweep.smoke_job import smoke_job
from spark_sweep.vonkctl import Vonkctl

ROOT = Path(__file__).resolve().parents[1]
OCR = "vonk-forge/hunyuanocr-1-5-vllm-dflash-single"
DEFINITIONS = Definitions.load()


def _recipe(key: str = OCR, adapter: str = "artifact-job") -> Any:
    row = {
        "selector": key,
        "document": {"interfaces": [{"adapter": adapter, "model_aliases": ["ocr"]}]},
    }
    recipe = parse_recipe(row)
    assert recipe is not None and recipe.is_job
    return recipe


def _archive(**manifest_overrides: Any) -> bytes:
    spec = DEFINITIONS.artifact(OCR)
    assert spec is not None
    (assertion,) = [a for a in spec["assertions"] if a["kind"] == "document-archive"]
    manifest = {
        **assertion["manifest_equals"],
        "sampling": assertion["sampling_equals"],
        "documents": [
            {
                "characters": 9,
                "early_stopped_tail_repetition": False,
                "input": assertion["input_name"],
                "output": assertion["output_name"],
            }
        ],
        **manifest_overrides,
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        archive.writestr(assertion["output_name"], "Digit: 7\n")
    return buffer.getvalue()


class Controller:
    """A scripted ``vonkctl recipe job``: each step answers from its queue, in order."""

    def __init__(
        self, output: bytes = b"", **answers: Sequence[tuple[int, Any]]
    ) -> None:
        self.output = output
        self.answers = {step: list(queue) for step, queue in answers.items()}
        self.calls: list[list[str]] = []
        self.binding: dict[str, Any] = {}

    def __call__(self, argv: Sequence[str], timeout: float) -> tuple[int, str, str]:
        args = list(argv[3:])  # vonkctl --json --no-input ...
        assert args[:2] == ["recipe", "job"]
        self.calls.append(args[2:])
        step = args[2]
        if step == "create":
            self.binding = json.loads(Path(args[args.index("--file") + 1]).read_text())
        if step == "download":
            path = Path(args[args.index("--output") + 1]) / "output.zip"
            path.write_bytes(self.output)
            document: Any = {"files": [{"name": "output.zip", "path": str(path)}]}
            return 0, json.dumps(document), ""
        queue = self.answers[step]
        code, document = queue.pop(0) if len(queue) > 1 else queue[0]
        return code, json.dumps(document), ""

    @property
    def steps(self) -> list[str]:
        return [call[0] for call in self.calls]


JOB = {"id": "11111111-1111-4111-8111-111111111111", "run_id": "run-1"}
OUTPUT = [{"name": "output.zip", "media_type": "application/zip"}]


def _job(state: str | None, **fields: Any) -> tuple[int, dict[str, Any]]:
    return 0, {**JOB, "state": state, **fields}


def _smoke(controller: Controller, *, clock: list[float] | None = None) -> Any:
    clock = clock if clock is not None else [0.0]

    def sleep(seconds: float) -> None:
        clock[0] += seconds

    return smoke_job(
        _recipe(),
        DEFINITIONS,
        Vonkctl("vonkctl", runner=controller),
        "run-1",
        "lane-1",
        sleep=sleep,
        now=lambda: clock[0],
    )


def test_a_job_is_created_submitted_followed_downloaded_and_its_outputs_hold() -> None:
    controller = Controller(
        _archive(),
        create=[_job(None, preparation="ready")],
        submit=[_job("queued")],
        detail=[
            _job("running"),
            _job(
                "succeeded",
                output_files=OUTPUT,
                result_evidence={"elapsed_milliseconds": 5},
            ),
        ],
    )
    result = _smoke(controller)
    assert result.ok, result.failure
    assert controller.steps == ["create", "submit", "detail", "detail", "download"]
    create, submit = controller.calls[0], controller.calls[1]
    assert create[create.index("--run") + 1] == "run-1"
    assert submit[1] == JOB["id"]
    # The binding names the definition's fixture as the input, under its own file name.
    assert controller.binding["create"]["interface"] == "artifact-job"
    assert controller.binding["create"]["inputs"] == [
        {"slot": "document", "name": "digit7.png", "media_type": "image/png"}
    ]
    assert list(controller.binding["input_paths"]) == ["digit7.png"]
    (case,) = result.cases
    assert case["job_id"] == JOB["id"] and case["elapsed_ms"] == 5
    assert "not_evaluated" not in case


def test_a_smoked_again_job_reconnects_with_the_same_request_keys() -> None:
    def keys() -> list[str]:
        controller = Controller(
            _archive(),
            create=[_job("succeeded", output_files=OUTPUT)],
        )
        assert _smoke(controller).ok
        assert controller.steps == [
            "create",
            "download",
        ]  # already submitted: no resubmit
        create = controller.calls[0]
        return [create[create.index("--request-key") + 1]]

    assert keys() == keys()


def test_a_draft_left_by_an_interrupted_create_is_uploaded_before_submit() -> None:
    controller = Controller(
        _archive(),
        create=[_job(None, preparation="draft")],
        upload=[_job(None, preparation="ready")],
        submit=[_job("succeeded", output_files=OUTPUT)],
    )
    assert _smoke(controller).ok
    assert controller.steps == ["create", "upload", "submit", "download"]


def test_an_output_that_misses_its_assertion_fails_the_recipe() -> None:
    controller = Controller(
        _archive(model="someone/else"),
        create=[_job(None, preparation="ready")],
        submit=[_job("succeeded", output_files=OUTPUT)],
    )
    result = _smoke(controller)
    assert result.failure is not None
    assert result.failure.klass == "smoke-assertion"
    assert result.failure.evidence["assertion"] == "document-archive"
    assert verdict.disposition(result.failure) is verdict.Disposition.FAIL


@pytest.mark.parametrize(
    ("ended", "outcome"),
    [
        (_job("failed", status_reason="exit 1"), verdict.Disposition.FAIL),
        (
            _job("failed", result_evidence={"failure_kind": "agent-lease-expired"}),
            verdict.Disposition.OBSERVE,
        ),
        (
            _job("failed", result_evidence={"active_scope_may_remain": True}),
            verdict.Disposition.OBSERVE,
        ),
        (_job("cancelled"), verdict.Disposition.OBSERVE),
        (_job("needs-operator"), verdict.Disposition.OBSERVE),
    ],
)
def test_a_job_that_ends_unsuccessfully_is_the_recipes_only_without_a_platform_cause(
    ended: tuple[int, dict[str, Any]], outcome: verdict.Disposition
) -> None:
    controller = Controller(create=[_job(None, preparation="ready")], submit=[ended])
    result = _smoke(controller)
    assert result.failure is not None
    assert verdict.disposition(result.failure) is outcome
    assert result.failure.evidence["job_id"] == JOB["id"]
    assert "download" not in controller.steps


def test_a_controller_refusal_is_a_wait_never_a_verdict() -> None:
    refusal = {
        "error_type": "control_api",
        "code": "controller.unavailable",
        "detail": "down",
    }
    controller = Controller(create=[(2, refusal)])
    result = _smoke(controller)
    assert result.failure is not None
    assert verdict.disposition(result.failure) is verdict.Disposition.WAIT


def test_a_transient_refusal_while_following_keeps_following() -> None:
    refusal = {"error_type": "control_api", "code": "controller.transport_timeout"}
    controller = Controller(
        _archive(),
        create=[_job(None, preparation="ready")],
        submit=[_job("queued")],
        detail=[(2, refusal), _job("succeeded", output_files=OUTPUT)],
    )
    assert _smoke(controller).ok


def test_a_job_that_never_ends_within_its_own_timeout_is_observed_again() -> None:
    controller = Controller(
        create=[_job(None, preparation="ready")],
        submit=[_job("queued")],
        detail=[_job("running")],
    )
    result = _smoke(controller)
    assert result.failure is not None and result.failure.code == "job.not_ended"
    assert verdict.disposition(result.failure) is verdict.Disposition.OBSERVE


def test_a_job_recipe_without_a_definition_is_recipe_data() -> None:
    result = smoke_job(
        _recipe("vonk-forge/no-such-job"),
        DEFINITIONS,
        Vonkctl("vonkctl", runner=Controller()),
        "run-1",
        "lane-1",
    )
    assert result.failure is not None and result.failure.klass == "recipe-data"


# -- assertions ------------------------------------------------------------------------


def _png(width: int, height: int) -> bytes:
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + header + b"\0" * 4


def test_structural_and_image_assertions() -> None:
    outputs = [Output("a.png", "image/png", _png(64, 32))]
    assertions = [
        {"kind": "file-count", "exact": 1},
        {"kind": "file-names", "exact": ["a.png"]},
        {"kind": "media-type", "allowed": ["image/png"]},
        {"kind": "media-type-counts", "counts": {"image/png": 1}},
        {"kind": "minimum-bytes", "value": 22},
        {"kind": "format", "format": "png"},
        {
            "kind": "image-metadata",
            "media_type": "image/png",
            "width": 64,
            "height": 32,
        },
        {"kind": "glb-structure", "minimum_meshes": 1},
    ]
    assert evaluate(assertions, outputs) == ["glb-structure"]
    with pytest.raises(AssertionFailed) as missed:
        evaluate(
            [{"kind": "image-metadata", "media_type": "image/png", "width": 65}],
            outputs,
        )
    assert missed.value.kind == "image-metadata"
    with pytest.raises(AssertionFailed):
        evaluate([{"kind": "format", "format": "mp4"}], outputs)


def test_every_assertion_kind_the_definitions_use_is_decided_or_named_not_evaluated() -> (
    None
):
    """A new kind in a definition must be added to one of the two sets, never silently passed."""
    assert not EVALUATED & NOT_EVALUATED
    used = {
        assertion["kind"]
        for spec in DEFINITIONS.document["recipes"].values()
        for assertion in [*spec.get("assertions", [])]
        + [a for case in spec.get("cases", []) for a in case.get("assertions", [])]
    }
    assert used <= EVALUATED | NOT_EVALUATED, used - EVALUATED - NOT_EVALUATED


# -- readiness -------------------------------------------------------------------------


def _fleet(route_state: str, healthy: bool = True) -> Fleet:
    return Fleet(
        (), (Presence("spk_a", "ocr", "run-1", healthy, route_state, "running"),)
    )


def test_a_job_run_serves_while_held_with_its_route_withdrawn() -> None:
    assert serving_run(_fleet("withdrawn"), "ocr", ["spk_a"], 1, job=True) == "run-1"
    # Before vonk-forge #1494 a held job run reads unhealthy: not ready, never a guess.
    assert (
        serving_run(_fleet("withdrawn", healthy=False), "ocr", ["spk_a"], 1, job=True)
        is None
    )
    assert serving_run(_fleet("published"), "ocr", ["spk_a"], 1, job=True) is None
    # A service run still needs its published route.
    assert serving_run(_fleet("withdrawn"), "ocr", ["spk_a"], 1) is None
    assert serving_run(_fleet("published"), "ocr", ["spk_a"], 1) == "run-1"
