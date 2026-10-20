"""The smoke of a job recipe: one artifact job against the loaded run, as its definition says.

A job recipe's run serves by holding its reservation: nothing answers on a route, so
there is no endpoint to call. Its smoke is one job, driven through ``vonkctl recipe
job`` exactly as an operator would: ``create`` (draft, inputs delivered and
finalized), ``submit``, ``detail`` until the job ends, ``download``, then the output
assertions of ``qualification/recipes/<slug>.json`` (``job_assertions``).

Request keys derive from the lane and the run, so a restarted sweep that smokes again
reconnects to the same job instead of starting a second one.

Verdicts follow ``verdict.disposition``:

* the job succeeded and its outputs hold: pass;
* the job failed on its own (the Controller names no platform cause): the recipe's
  failure, with the job's evidence: its exit code and stderr tail, read from its
  operation's failure evidence (``vonkctl fleet evidence``). A read that fails is
  recorded as ``evidence_unavailable`` and never changes the verdict;
* an assertion missed: the recipe's failure (``smoke-assertion``);
* the Controller finally refused the job's request (``create``/``upload`` carry the
  recipe's definition bound to its contract; a 409/422 with decision ``exit``): the
  recipe's failure (``recipe-data``), with the reply as evidence;
* the Controller refused otherwise or could not answer a step: a wait, smoked again later;
* the job ended for a platform reason (agent lease lost, a scope that may remain, a
  cancel, a wait for an operator) or never ended within its own timeout: uncertain,
  observed again.
"""

from __future__ import annotations

import json
import re
import tempfile
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from vonk_forge_contracts import canonical_media_types

from . import verdict
from .catalog import Recipe
from .definitions import Definitions
from .job_assertions import Output, evaluate
from .policy import Failure, classify, excerpt, redact
from .smoke import AssertionFailed, SmokeResult
from .vonkctl import Vonkctl, VonkctlError, request_key

#: The job's own ``timeout_seconds`` is the Controller's to enforce; this much longer the
#: sweep keeps looking before it calls the outcome uncertain.
JOB_GRACE_SECONDS = 600.0
DOWNLOAD_TIMEOUT_SECONDS = 1800.0
ENDED = frozenset({"succeeded", "failed", "cancelled"})
OPERATOR_WAIT = frozenset({"needs-operator", "waiting-for-operator"})
#: The steps whose request is the recipe's own data: its definition bound to its contract.
RECIPE_REQUEST_STEPS = frozenset({"create", "upload"})
#: The end of a failed job's stderr a result row keeps: what happened last.
STDERR_TAIL_BYTES = 2048


def job_budget_seconds(spec: Mapping[str, Any]) -> float:
    """How long a job smoke follows its job once submitted: the timeout and the grace."""
    return float(spec["timeout_seconds"]) + JOB_GRACE_SECONDS


class _Refused(Exception):
    def __init__(self, failure: Failure) -> None:
        super().__init__(failure.detail)
        self.failure = failure


def _text(document: Mapping[str, Any], name: str) -> str:
    value = document.get(name)
    return value if isinstance(value, str) else ""


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _call(
    vk: Vonkctl, step: str, *args: str, timeout: float | None = None
) -> Mapping[str, Any]:
    try:
        document = vk.call("recipe", "job", *args, timeout=timeout)
    except VonkctlError as error:
        evidence = {
            "case": f"job.{step}",
            "body": excerpt(error.reply.document if error.reply else ""),
        }
        if step in RECIPE_REQUEST_STEPS and verdict.request_refused(error.reply):
            failure = classify(
                "smoke",
                error.code or f"job.{step}",
                str(error),
                "recipe-data",
                evidence,
            )
        elif verdict.reply_disposition(error.reply) is verdict.Disposition.WAIT:
            failure = classify(
                "smoke", f"job.{step}", str(error), verdict.SMOKE_BUSY, evidence
            )
        else:
            failure = classify(
                "smoke", error.code or f"job.{step}", str(error), "", evidence
            )
        raise _Refused(failure) from error
    if not isinstance(document, Mapping):
        raise _Refused(
            classify(
                "smoke",
                f"job.{step}",
                "vonkctl returned no document",
                verdict.SMOKE_BUSY,
            )
        )
    return document


def _binding(spec: Mapping[str, Any], definitions: Definitions, work: Path) -> Path:
    """The ``recipe job create --file`` binding, with each fixture written beside it."""
    inputs: list[dict[str, str]] = []
    paths: dict[str, str] = {}
    (work / "inputs").mkdir()
    for item in spec.get("inputs") or []:
        fixture = definitions.fixture(str(item["fixture"]))
        path = work / "inputs" / fixture.name
        path.write_bytes(fixture.content)
        inputs.append(
            {
                "slot": str(item["slot"]),
                "name": fixture.name,
                "media_type": fixture.media_type,
            }
        )
        paths[fixture.name] = str(path)
    create = {
        "interface": spec["interface"],
        "parameters": spec.get("parameters") or {},
        "inputs": inputs,
        "output_limits": {
            **spec["output_limits"],
            "allowed_media_types": canonical_media_types(
                spec["output_limits"]["allowed_media_types"]
            ),
        },
        "timeout_seconds": spec["timeout_seconds"],
    }
    binding = work / "binding.json"
    binding.write_text(
        json.dumps({"create": create, "input_paths": paths}), encoding="utf-8"
    )
    return binding


_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_TQDM_LINE = re.compile(r".*\b\d{1,3}%\|[^|]*\|\s*\d+/\d+\s*\[[^]\r\n]*\]\s*")


def _tail(text: str, limit: int = STDERR_TAIL_BYTES) -> str:
    """The last non-progress lines, redacted and bounded by UTF-8 bytes.

    Split carriage-return redraws before filtering: progress after an exception
    must not push it out of the retained evidence. Keep progress if it is all we have.
    """
    plain = _ANSI_ESCAPE.sub("", text)
    lines = plain.splitlines()
    diagnostics = [
        line for line in lines if line.strip() and not _TQDM_LINE.fullmatch(line)
    ]
    content = "\n".join(diagnostics) if diagnostics else plain
    if diagnostics and plain.endswith(("\n", "\r")):
        content += "\n"
    encoded = redact(content).encode("utf-8")
    return encoded[-limit:].decode("utf-8", errors="ignore")


def _exit_code(bundle: Mapping[str, Any]) -> int | None:
    """The job process's exit code, as the evidence bundle types it.

    The Controller reads it from the agent's receipt; a bundle without one (no
    process receipt) claims none.
    """
    value = bundle.get("exit_code")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _job_evidence(vk: Vonkctl, job: Mapping[str, Any]) -> dict[str, Any]:
    """The cause the Controller kept for a job that ended unsuccessfully.

    Fail-open: a job without an operation, a refused or unreadable evidence read is
    recorded as ``evidence_unavailable`` with its reason; it never changes the verdict.
    """
    operation_id = _text(job, "operation_id")
    if not operation_id:
        return {"evidence_unavailable": "the job names no operation"}
    ref: dict[str, Any] = {"operation_id": operation_id}
    with tempfile.TemporaryDirectory(prefix="sweep-job-evidence-") as root:
        target = Path(root) / "evidence.json"
        reply = vk.run("fleet", "evidence", operation_id, "--output", str(target))
        if not reply.ok:
            return {"evidence_ref": ref, "evidence_unavailable": reply.error_text}
        try:
            bundle = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            return {
                "evidence_ref": ref,
                "evidence_unavailable": f"{type(error).__name__}: {error}"[:500],
            }
    bundle = _mapping(bundle)
    attempt = _mapping(bundle.get("context")).get("attempt")
    if not isinstance(attempt, int):
        attempt = _mapping(reply.document).get("attempt")
    if isinstance(attempt, int) and not isinstance(attempt, bool):
        ref["attempt"] = attempt
    diagnostics = _mapping(bundle.get("diagnostics"))
    stderr = _mapping(diagnostics.get("stderr")).get("text")
    proof: dict[str, Any] = {
        "evidence_ref": ref,
        "evidence_error_code": _text(bundle, "error_code"),
        "evidence_summary": excerpt(_text(bundle, "summary"), 512),
        "exit_code": _exit_code(bundle),
        "stderr_tail": _tail(stderr) if isinstance(stderr, str) else None,
    }
    if not diagnostics:
        proof["evidence_unavailable"] = "the evidence carries no diagnostics"
    return proof


def _ended_failure(vk: Vonkctl, job: Mapping[str, Any]) -> Failure:
    """A job that ended without success: the recipe's, unless the platform names a cause."""
    state = _text(job, "state")
    evidence = _mapping(job.get("result_evidence"))
    proof = {
        "case": "job",
        "job_id": _text(job, "id"),
        "state": state,
        "status_reason": _text(job, "status_reason"),
        "result_evidence": dict(evidence),
        **_job_evidence(vk, job),
    }
    detail = f"job {state}: {_text(job, 'status_reason') or 'no reason given'}"
    platform = (
        state != "failed"
        or evidence.get("failure_kind") is not None
        or evidence.get("active_scope_may_remain") is True
    )
    if platform:
        return classify("smoke", f"job.{state}", detail, "smoke-timeout", proof)
    return classify("smoke", "job.failed", detail, verdict.JOB_FAILED, proof)


def _outputs(job: Mapping[str, Any], download: Mapping[str, Any]) -> list[Output]:
    media = {
        _text(item, "name"): _text(item, "media_type")
        for item in job.get("output_files") or []
        if isinstance(item, Mapping)
    }
    outputs: list[Output] = []
    for item in download.get("files") or []:
        if isinstance(item, Mapping):
            name = _text(item, "name")
            content = Path(_text(item, "path")).read_bytes()
            outputs.append(Output(name, media.get(name, ""), content))
    return outputs


def smoke_job(
    recipe: Recipe,
    definitions: Definitions,
    vk: Vonkctl,
    run_id: str,
    lane_key: str,
    *,
    poll_seconds: float = 10.0,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.monotonic,
) -> SmokeResult:
    started = now()
    result = SmokeResult(ok=False, kind="job")
    spec = definitions.artifact(recipe.key)
    if spec is None:
        result.failure = classify(
            "smoke",
            "job.definition_missing",
            f"no qualification definition for job recipe {recipe.key}",
            "recipe-data",
        )
        return result
    keys = {
        step: request_key(lane_key, run_id, "job", step)
        for step in ("create", "submit")
    }
    try:
        with tempfile.TemporaryDirectory(prefix="sweep-job-") as root:
            work = Path(root)
            binding = _binding(spec, definitions, work)
            job = _call(
                vk,
                "create",
                "create",
                "--run",
                run_id,
                "--file",
                str(binding),
                "--request-key",
                keys["create"],
            )
            job_id = _text(job, "id")
            if job.get("preparation") == "draft":
                job = _call(vk, "upload", "upload", job_id, "--file", str(binding))
            if job.get("preparation") == "ready":
                job = _call(
                    vk, "submit", "submit", job_id, "--request-key", keys["submit"]
                )
            deadline = now() + job_budget_seconds(spec)
            while _text(job, "state") not in ENDED | OPERATOR_WAIT:
                if now() >= deadline:
                    result.failure = classify(
                        "smoke",
                        "job.not_ended",
                        f"job {job_id} did not end within its timeout "
                        f"({spec['timeout_seconds']}s + {JOB_GRACE_SECONDS:.0f}s)",
                        "smoke-timeout",
                        {"case": "job", "job_id": job_id, "state": _text(job, "state")},
                    )
                    return result
                sleep(poll_seconds)
                try:
                    job = _call(vk, "detail", "detail", job_id)
                except _Refused as refused:
                    if (
                        verdict.disposition(refused.failure)
                        is not verdict.Disposition.WAIT
                    ):
                        raise
            if _text(job, "state") != "succeeded":
                result.failure = _ended_failure(vk, job)
                return result
            out = work / "outputs"
            out.mkdir()
            download = _call(
                vk,
                "download",
                "download",
                job_id,
                "--output",
                str(out),
                timeout=DOWNLOAD_TIMEOUT_SECONDS,
            )
            outputs = _outputs(job, download)
            case: dict[str, Any] = {
                "case_id": "job",
                "job_id": job_id,
                "run_id": run_id,
                "outputs": [
                    {
                        "name": o.name,
                        "media_type": o.media_type,
                        "size_bytes": len(o.content),
                    }
                    for o in outputs
                ],
                "elapsed_ms": _mapping(job.get("result_evidence")).get(
                    "elapsed_milliseconds"
                ),
            }
            try:
                skipped = evaluate(list(spec.get("assertions") or []), outputs)
            except AssertionFailed as error:
                result.failure = classify(
                    "smoke",
                    "case.job",
                    str(error),
                    "smoke-assertion",
                    {
                        **case,
                        "assertion": error.kind,
                        "value": excerpt(error.value, 500),
                    },
                )
                return result
            if skipped:
                case["not_evaluated"] = skipped
            result.cases.append(case)
    except _Refused as refused:
        result.failure = refused.failure
    finally:
        result.seconds = round(now() - started, 2)
    result.ok = result.failure is None
    return result
