"""A job recipe's smoke: readiness by a held run, one artifact job, its output assertions."""

from __future__ import annotations

import io
import json
import struct
import warnings
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from vonk_forge_contracts.qualification_status import QualificationGlbProfile

from spark_sweep import verdict
from spark_sweep.catalog import Fleet, Presence, parse_recipe, serving_run
from spark_sweep.definitions import Definitions
from spark_sweep.job_assertions import EVALUATED, NOT_EVALUATED, Output, evaluate
from spark_sweep.smoke import AssertionFailed
from spark_sweep.smoke_job import job_budget_seconds, smoke_job
from spark_sweep.vonkctl import Vonkctl

ROOT = Path(__file__).resolve().parents[1]
# An OCR artifact job with a document-archive output: the definition the withdrawn
# HunyuanOCR recipe carried, kept as test data so the job smoke stays exercised.
OCR = "vonk-forge/ocr-document-job"
_CATALOG = Definitions.load()
DEFINITIONS = Definitions(
    {
        **_CATALOG.document,
        "recipes": {
            **_CATALOG.document["recipes"],
            OCR: json.loads(
                (ROOT / "tests/fixtures/sweep-job-document-archive.json").read_text(
                    encoding="utf-8"
                )
            ),
        },
    },
    _CATALOG.root,
)


def _recipe(key: str = OCR, adapter: str = "artifact-job") -> Any:
    row = {
        "selector": key,
        "projected": {"interfaces": [{"adapter": adapter, "model_aliases": ["ocr"]}]},
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
        self,
        output: bytes = b"",
        evidence: tuple[int, Any] | None = None,
        **answers: Sequence[tuple[int, Any]],
    ) -> None:
        self.output = output
        # ``vonkctl fleet evidence``: (exit code, bundle written to --output or error).
        self.evidence = evidence
        self.answers = {step: list(queue) for step, queue in answers.items()}
        self.calls: list[list[str]] = []
        self.binding: dict[str, Any] = {}

    def __call__(self, argv: Sequence[str], timeout: float) -> tuple[int, str, str]:
        args = list(argv[3:])  # vonkctl --json --no-input ...
        if args[:2] == ["fleet", "evidence"] and self.evidence is not None:
            self.calls.append(args)
            code, answer = self.evidence
            if code != 0:
                return code, json.dumps(answer), ""
            Path(args[args.index("--output") + 1]).write_text(json.dumps(answer))
            document = {"operation_id": args[2], "attempt": 1, "output": "x"}
            return 0, json.dumps(document), ""
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


OPERATION = "22222222-2222-4222-8222-222222222222"


def _bundle(stderr: str, exit_code: int | None = None) -> dict[str, Any]:
    """A ``vonkctl fleet evidence`` bundle (schema 2): the exit code is typed."""
    tail = {"truncated": False, "dropped_bytes": 0, "dropped_lines": 0}
    return {
        "schema_version": 2,
        "context": {"operation_id": OPERATION, "attempt": 1},
        "error_code": "recipe_job_run_failed",
        "summary": "job adapter exited unsuccessfully",
        "exit_code": exit_code,
        "diagnostics": {
            "stdout": {"text": "", **tail},
            "stderr": {"text": stderr, **tail},
            # Text that looks like an exit code is never read as one.
            "preflight": [{"name": "exit_code", "value": "99"}],
        },
    }


def test_a_failed_job_carries_its_exit_code_and_stderr_tail() -> None:
    stderr = "x" * 5000 + "\n/opt/vonk/bin/comfyui-job: 1: not found\n"
    controller = Controller(
        evidence=(0, _bundle(stderr, exit_code=127)),
        create=[_job(None, preparation="ready")],
        submit=[_job("failed", operation_id=OPERATION, status_reason="exit")],
    )
    result = _smoke(controller)
    assert result.failure is not None
    assert verdict.disposition(result.failure) is verdict.Disposition.FAIL
    proof = result.failure.evidence
    assert proof["exit_code"] == 127
    assert proof["stderr_tail"].endswith("comfyui-job: 1: not found\n")
    assert len(proof["stderr_tail"].encode()) <= 2048
    assert proof["evidence_ref"] == {"operation_id": OPERATION, "attempt": 1}
    assert "evidence_unavailable" not in proof
    assert controller.calls[-1][:3] == ["fleet", "evidence", OPERATION]


@pytest.mark.parametrize("status", [404, 500])
def test_an_evidence_read_that_fails_never_changes_the_verdict(status: int) -> None:
    error = {
        "error_type": "control_api",
        "code": "controller.not_found" if status == 404 else "controller.error",
        "status": status,
    }
    ended = _job("failed", operation_id=OPERATION, status_reason="exit")
    controller = Controller(
        evidence=(1, error),
        create=[_job(None, preparation="ready")],
        submit=[ended],
    )
    result = _smoke(controller)
    assert result.failure is not None
    assert verdict.disposition(result.failure) is verdict.Disposition.FAIL
    assert result.failure.klass == verdict.JOB_FAILED
    assert result.failure.evidence["evidence_unavailable"]
    assert result.failure.evidence["evidence_ref"] == {"operation_id": OPERATION}
    assert "stderr_tail" not in result.failure.evidence


def test_a_deferred_job_keeps_its_evidence_and_stays_observed() -> None:
    ended = _job(
        "failed",
        operation_id=OPERATION,
        result_evidence={"failure_kind": "agent-lease-expired"},
    )
    controller = Controller(
        evidence=(0, _bundle("lease lost\n")),
        create=[_job(None, preparation="ready")],
        submit=[ended],
    )
    result = _smoke(controller)
    assert result.failure is not None
    assert verdict.disposition(result.failure) is verdict.Disposition.OBSERVE
    assert result.failure.evidence["stderr_tail"] == "lease lost\n"
    # No typed exit code in the bundle: none is claimed, whatever the text says.
    assert "exit_code" not in result.failure.evidence


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


def _cli_error(status: int, code: str, decision: str, **fields: Any) -> dict[str, Any]:
    """The ``vonkctl --json`` document of a Controller HTTP answer (its ErrorContext)."""
    return {
        "error_type": "control_api",
        "code": code,
        "detail": "artifact input slot ids are not canonical",
        "http_status": status,
        "source": "remote_rejection",
        "retryable": status == 429 or status >= 500,
        "decision": decision,
        **fields,
    }


REFUSED = _cli_error(409, "controller.conflict", "exit")


@pytest.mark.parametrize("step", ["create", "upload"])
def test_a_final_refusal_of_the_recipes_job_request_fails_the_recipe_at_once(
    step: str,
) -> None:
    answers: dict[str, Any] = {"create": [(2, REFUSED)]}
    if step == "upload":
        answers = {
            "create": [_job(None, preparation="draft")],
            "upload": [(2, REFUSED)],
        }
    controller = Controller(**answers)
    result = _smoke(controller)
    assert result.failure is not None
    assert result.failure.klass == "recipe-data"
    assert verdict.disposition(result.failure) is verdict.Disposition.FAIL
    assert result.failure.evidence["case"] == f"job.{step}"
    assert result.failure.code == "controller.conflict"
    assert result.failure.evidence["body"]  # the Controller's reply is kept as evidence
    assert "submit" not in controller.steps


@pytest.mark.parametrize(
    "answer",
    [
        _cli_error(503, "controller.unavailable", "retry"),
        _cli_error(409, "controller.conflict", "defer", retry_after_seconds=5),
        _cli_error(404, "controller.not_found", "exit"),
        _cli_error(403, "controller.request_rejected", "exit"),
    ],
)
def test_a_job_request_the_controller_may_still_take_is_not_a_verdict(
    answer: dict[str, Any],
) -> None:
    result = _smoke(Controller(create=[(2, answer)]))
    assert result.failure is not None
    assert verdict.disposition(result.failure) is verdict.Disposition.WAIT


def test_a_refused_submit_names_no_recipe_data_and_is_not_a_verdict() -> None:
    controller = Controller(
        create=[_job(None, preparation="ready")], submit=[(2, REFUSED)]
    )
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


@pytest.mark.parametrize("state", ["needs-operator", "cancelled"])
def test_a_waiting_or_cancelled_job_ends_the_wait_at_once(state: str) -> None:
    clock = [0.0]
    controller = Controller(
        create=[_job(None, preparation="ready")],
        submit=[_job("queued")],
        detail=[_job(state)],
    )
    result = _smoke(controller, clock=clock)
    assert result.failure is not None
    assert result.failure.code == f"job.{state}"
    assert result.failure.code != "job.not_ended"
    assert clock[0] < job_budget_seconds({"timeout_seconds": 0})
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
    ]
    assert evaluate(assertions, outputs) == {}
    with pytest.raises(AssertionFailed) as missed:
        evaluate(
            [{"kind": "image-metadata", "media_type": "image/png", "width": 65}],
            outputs,
        )
    assert missed.value.kind == "image-metadata"
    with pytest.raises(AssertionFailed):
        evaluate([{"kind": "format", "format": "mp4"}], outputs)


def test_json_document_checks_required_keys_and_top_level_equals() -> None:
    assertion = {
        "kind": "json-document",
        "media_type": "application/json",
        "required_keys": ["style", "frames"],
        "equals": {"style": "walk"},
    }

    def output(content: bytes) -> list[Output]:
        return [Output("m.json", "application/json", content)]

    assert evaluate([assertion], output(b'{"style": "walk", "frames": 3}')) == {}
    for content in (
        b'{"style": "walk"}',
        b'{"style": "run", "frames": 3}',
        b'["style", "frames"]',
        b"not json",
    ):
        with pytest.raises(AssertionFailed) as missed:
            evaluate([assertion], output(content))
        assert missed.value.kind == "json-document"
    with pytest.raises(AssertionFailed):
        evaluate([assertion], [Output("m.png", "image/png", b"{}")])


@pytest.mark.parametrize(
    "payload",
    [
        b'{"style": "walk", "frames": [NaN]}',
        b'{"style": "walk", "frames": Infinity}',
        b'{"style": "walk", "frames": -Infinity}',
        b'{"style": "walk", "style": "walk", "frames": 3}',
    ],
)
def test_json_is_read_as_strictly_as_the_platform_reads_it(payload: bytes) -> None:
    """NaN, Infinity and duplicate keys fail the platform's parse, so they fail here."""
    output = [Output("m.json", "application/json", payload)]
    with pytest.raises(AssertionFailed) as missed:
        evaluate([{"kind": "json-document", "required_keys": ["style"]}], output)
    assert missed.value.kind == "json-document"
    with pytest.raises(AssertionFailed):
        evaluate([{"kind": "format", "format": "json"}], output)


def test_json_document_checks_every_selected_output() -> None:
    good = Output("a.json", "application/json", b'{"style": "walk"}')
    bad = Output("b.json", "application/json", b'{"style": "run"}')
    with pytest.raises(AssertionFailed):
        evaluate([{"kind": "json-document", "equals": {"style": "walk"}}], [good, bad])


def test_the_document_archive_manifest_is_read_strictly() -> None:
    spec = DEFINITIONS.artifact(OCR)
    assert spec is not None
    (assertion,) = [a for a in spec["assertions"] if a["kind"] == "document-archive"]
    valid = Output("d.zip", "application/zip", _archive())
    assert evaluate([assertion], [valid]) == {}
    with pytest.raises(AssertionFailed):
        evaluate(
            [assertion],
            [Output("d.zip", "application/zip", _archive(extra=float("nan")))],
        )


# -- zip-entries -----------------------------------------------------------------------

MOTION = {
    "kind": "zip-entries",
    "media_type": "application/zip",
    "exact_names": ["motion.bvh", "motion.npz"],
    "allowed_suffixes": [".bvh", ".npz"],
    "minimum_entries": 2,
    "nonempty": True,
}


def _zip(
    entries: Sequence[tuple[str, bytes]] = (
        ("motion.bvh", b"HIERARCHY"),
        ("motion.npz", b"PK"),
    ),
    *,
    compression: int = zipfile.ZIP_STORED,
    edit: Any = None,
) -> bytes:
    """An archive of ``entries``; ``edit`` may change the entries' ZipInfo before it is closed."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression) as archive:
        with warnings.catch_warnings():  # a duplicate name is the point of one case
            warnings.simplefilter("ignore", UserWarning)
            for name, data in entries:
                archive.writestr(name, data)
        if edit is not None:
            edit(archive.infolist())
    return buffer.getvalue()


def _check(content: bytes, **fields: Any) -> dict[str, str]:
    assertion = {"kind": "zip-entries", **fields}
    return evaluate([assertion], [Output("motion.zip", "application/zip", content)])


def test_a_zip_with_the_declared_entries_passes_and_is_evaluated() -> None:
    assert _check(_zip(), **{k: v for k, v in MOTION.items() if k != "kind"}) == {}
    assert (
        _check(
            _zip(compression=zipfile.ZIP_DEFLATED),
            exact_names=["motion.bvh", "motion.npz"],
        )
        == {}
    )


def test_a_zip_entries_assertion_without_fields_asks_for_one_entry() -> None:
    """Every field is optional: the defaults are any names, one entry, any suffix, empties allowed."""
    assert _check(_zip([("a.bin", b"")])) == {}
    with pytest.raises(AssertionFailed, match="no files"):
        _check(_zip([]))


@pytest.mark.parametrize(
    ("content", "fields", "message"),
    [
        (_zip([("motion.npz", b"PK"), ("motion.bvh", b"H")]), MOTION, "entry names"),
        (_zip([("motion.bvh", b"H")]), MOTION, "entry names"),
        (_zip(), {"minimum_entries": 3}, "entry count"),
        (_zip(), {"allowed_suffixes": [".bvh"]}, "entry suffix"),
        (_zip([("motion.bvh", b"")]), {"nonempty": True}, "empty entry"),
    ],
)
def test_each_zip_entries_field_fails_the_recipe_on_a_miss(
    content: bytes, fields: dict[str, Any], message: str
) -> None:
    with pytest.raises(AssertionFailed, match=message) as missed:
        _check(content, **fields)
    assert missed.value.kind == "zip-entries"


def test_only_nonempty_true_rejects_an_empty_entry() -> None:
    assert _check(_zip([("a.bin", b"")]), nonempty=False) == {}


def test_zip_entries_checks_only_the_selected_media_type() -> None:
    other = Output("m.json", "application/json", b"{}")
    zipped = Output("m.zip", "application/zip", _zip())
    assert evaluate([MOTION], [other, zipped]) == {}
    with pytest.raises(AssertionFailed, match="selected no output"):
        evaluate([MOTION], [other])


def _declare(**sizes: int) -> Any:
    def edit(infos: list[zipfile.ZipInfo]) -> None:
        for name, value in sizes.items():
            setattr(infos[0], name, value)

    return edit


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (_zip([("../motion.bvh", b"H")]), "unsafe entry"),
        (_zip([("/motion.bvh", b"H")]), "unsafe entry"),
        (_zip([("a/../../b", b"H")]), "unsafe entry"),
        (_zip([("a\\b", b"H")]), "unsafe entry"),
        (_zip([("dir/", b"")]), "unsafe entry"),
        (
            _zip(edit=lambda i: setattr(i[0], "flag_bits", i[0].flag_bits | 0x1)),
            "unsafe entry",
        ),
        (
            _zip(edit=lambda i: setattr(i[0], "external_attr", 0o120777 << 16)),
            "unsafe entry",
        ),
        (_zip([("a", b"1"), ("a", b"2")]), "duplicate entries"),
        (_zip(edit=_declare(file_size=129 * 1024**2)), "expansion is unsafe"),
        (
            _zip(edit=_declare(file_size=100_000_000, compress_size=1)),
            "expansion is unsafe",
        ),
        (_zip(edit=_declare(compress_size=0)), "expansion is unsafe"),
    ],
)
def test_an_unsafe_zip_is_a_miss_and_never_extracted(
    content: bytes, message: str
) -> None:
    with pytest.raises(AssertionFailed, match=message) as missed:
        _check(content)
    assert missed.value.kind == "zip-entries"


def _corrupt(content: bytes, at: int) -> bytes:
    return content[:at] + bytes([content[at] ^ 0xFF]) + content[at + 1 :]


def _first_data(name: str) -> int:
    """Where the first entry's data starts: a 30-byte local header and its name."""
    return 30 + len(name)


@pytest.mark.parametrize(
    "content",
    [
        b"not a zip at all",
        _zip()[:-10],
        _corrupt(_zip(), _first_data("motion.bvh")),
        _corrupt(
            _zip([("a", b"x" * 4096)], compression=zipfile.ZIP_DEFLATED),
            _first_data("a") + 1,
        ),
    ],
    ids=["not-a-zip", "truncated", "crc-mismatch", "damaged-deflate"],
)
def test_a_corrupt_zip_is_a_miss_not_a_crash(content: bytes) -> None:
    with pytest.raises(AssertionFailed) as missed:
        _check(content)
    assert missed.value.kind == "zip-entries"


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


def test_every_glb_profile_the_definitions_use_is_a_contract_profile() -> None:
    """A misspelt profile fails here, not on hardware after the job ran."""
    used = {
        assertion.get("profile", QualificationGlbProfile.TRIANGLE_MESH)
        for spec in DEFINITIONS.document["recipes"].values()
        for assertion in [*spec.get("assertions", [])]
        + [a for case in spec.get("cases", []) for a in case.get("assertions", [])]
        if assertion["kind"] == "glb-structure"
    }
    assert used, "no definition declares a glb-structure assertion"
    assert used <= set(QualificationGlbProfile), used - set(QualificationGlbProfile)


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
