"""Smoke tests that match what a recipe serves.

* chat, code, reasoning, vision: a streamed completion measures time to first
  token and tokens per second, then the recipe's reviewed qualification cases
  run with their assertions (image cases carry their small PNG fixture);
* generation and other non-OpenAI recipes: readiness only (the run is healthy
  and its route published), recorded as such.

Requests go to the gateway endpoint ``vonkctl profile endpoint`` reports,
with the operator's client key.
"""

from __future__ import annotations

import json
import re
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .catalog import Recipe
from .definitions import Definitions
from .policy import Failure, classify, excerpt

PERF_PROMPT = "Count from one to twenty in words, separated by commas."
PERF_TOKENS = 64


@dataclass(frozen=True)
class HttpConfig:
    key_file: Path | None = None
    ca_file: Path | None = None
    insecure_tls: bool = False
    case_timeout: float = 180.0

    def headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.key_file is not None:
            headers["Authorization"] = (
                f"Bearer {self.key_file.read_text(encoding='utf-8').strip()}"
            )
        return headers

    def context(self) -> ssl.SSLContext | None:
        if self.insecure_tls:
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            return context
        if self.ca_file is not None:
            return ssl.create_default_context(cafile=str(self.ca_file))
        return None


@dataclass
class SmokeResult:
    ok: bool
    kind: str  # service | readiness-only
    cases: list[dict[str, Any]] = field(default_factory=list)
    perf: dict[str, Any] | None = None
    failure: Failure | None = None
    seconds: float = 0.0
    # Every quality check that ran: {case, expected, got, ok}. Information only.
    quality: list[dict[str, Any]] = field(default_factory=list)

    @property
    def quality_notes(self) -> list[dict[str, Any]]:
        """The quality checks the model missed: reported, never a failure."""
        return [item for item in self.quality if not item["ok"]]

    def record(self) -> dict[str, Any]:
        value: dict[str, Any] = {"kind": self.kind, "cases": self.cases}
        if self.perf:
            value["perf"] = self.perf
        if self.quality:
            value["quality"] = self.quality
        return value


# -- assertions (the kinds `qualification/shared.json` uses) ----------------


class AssertionFailed(Exception):
    """A case that did not hold; ``value`` is what it looked at."""

    def __init__(self, message: str, value: Any = None, kind: str | None = None):
        super().__init__(message)
        self.value = value
        self.kind = kind


def _path(value: Any, path: str) -> Any:
    current = value
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            raise AssertionFailed(f"path {path} is missing")
    return current


def _evaluate(response: Any, raw: str, assertion: dict[str, Any]) -> tuple[bool, Any]:
    """Whether one assertion holds, and the value it looked at."""
    kind = assertion.get("kind")
    if kind == "raw.not-contains":
        hit = next(
            (v for v in assertion.get("values", []) if isinstance(v, str) and v in raw),
            None,
        )
        return hit is None, hit
    path = str(assertion.get("path", ""))
    value = _path(response, path)
    expected = assertion.get("value")
    bad = False
    if kind == "path.equals":
        bad = value != expected
    elif kind == "path.regex":
        bad = not isinstance(value, str) or re.fullmatch(str(expected), value) is None
    elif kind == "path.nonempty":
        bad = not isinstance(value, (str, list, dict)) or len(value) == 0
    elif kind == "path.empty":
        bad = value not in (None, "", [], {})
    elif kind == "path.count":
        bad = not isinstance(value, (str, list, dict)) or len(value) != expected
    elif kind == "path.lte":
        bad = (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isinstance(expected, (int, float))
            or value > expected
        )
    elif kind == "path.json-equals":
        try:
            bad = (json.loads(value) if isinstance(value, str) else None) != expected
        except ValueError:
            bad = True
    elif kind == "array.path-count-equals":
        item_path = str(assertion.get("item_path", ""))
        count = 0
        for item in value if isinstance(value, list) else []:
            try:
                count += _path(item, item_path) == expected
            except AssertionFailed:
                continue
        bad = count != assertion.get("count")
    else:
        raise AssertionFailed(f"unsupported assertion kind {kind!r}")
    return not bad, value


def check_assertions(response: Any, raw: str, assertions: list[dict[str, Any]]) -> None:
    """The functional assertions: the first one that does not hold fails the case."""
    for assertion in assertions:
        holds, value = _evaluate(response, raw, assertion)
        if holds:
            continue
        if assertion.get("kind") == "raw.not-contains":
            raise AssertionFailed(
                f"raw.not-contains: response contains {value!r}",
                value,
                "raw.not-contains",
            )
        raise AssertionFailed(
            f"{assertion.get('kind')} failed at {assertion.get('path', '')}: "
            f"got {str(value)[:80]!r}",
            value,
            str(assertion.get("kind")),
        )


def check_quality(
    case_id: str, response: Any, raw: str, assertions: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """The quality assertions as ``{case, expected, got, ok}``; a miss is information."""
    notes: list[dict[str, Any]] = []
    for assertion in assertions:
        expected = assertion.get("expected")
        if not isinstance(expected, str):
            expected = str(assertion.get("value", assertion.get("values", "")))
        try:
            holds, value = _evaluate(response, raw, assertion)
        except AssertionFailed as error:
            holds, value = False, str(error)
        notes.append(
            {
                "case": case_id,
                "expected": expected[:120],
                "got": str(value)[:120],
                "ok": holds,
            }
        )
    return notes


# -- requests ---------------------------------------------------------------


def _open(
    request: urllib.request.Request, timeout: float, context: ssl.SSLContext | None
) -> Any:
    return urllib.request.urlopen(request, timeout=timeout, context=context)


def _secrets(config: HttpConfig | None) -> list[str]:
    try:
        if config is not None and config.key_file is not None:
            return [config.key_file.read_text(encoding="utf-8").strip()]
    except OSError:
        pass
    return []


def _http_body(error: urllib.error.HTTPError, limit: int = 4096) -> str:
    try:
        return error.read(limit).decode("utf-8", "replace")
    except (OSError, ValueError):
        return ""


def _transport_failure(
    error: BaseException, case: str, config: HttpConfig | None = None
) -> Failure:
    """A request that failed: the case, the status and the response body travel with it."""
    secrets = _secrets(config)
    evidence: dict[str, Any] = {"case": case, "error": type(error).__name__}
    body = getattr(error, "body", None)
    if isinstance(error, urllib.error.HTTPError):
        transient = error.code in (408, 429, 502, 503, 504)
        evidence["http_status"] = error.code
        evidence["body"] = excerpt(body or _http_body(error), secrets=secrets)
        return classify(
            "smoke",
            f"case.{case}",
            f"HTTP {error.code}",
            "smoke-timeout" if transient else "smoke-request",
            evidence,
        )
    if body is not None:  # a 200 whose body was not the JSON we expected
        evidence["http_status"] = getattr(error, "http_status", None)
        evidence["body"] = excerpt(body, secrets=secrets)
    return classify(
        "smoke",
        f"case.{case}",
        f"{type(error).__name__}: {error}",
        "smoke-timeout",
        {**evidence, "message": excerpt(error, 500, secrets)},
    )


def _assertion_failure(
    error: AssertionFailed, case: str, config: HttpConfig | None = None
) -> Failure:
    secrets = _secrets(config)
    return classify(
        "smoke",
        f"case.{case}",
        str(error),
        "smoke-assertion",
        {
            "case": case,
            "assertion": error.kind,
            "value": excerpt(error.value, 500, secrets)
            if error.value is not None
            else None,
            "http_status": getattr(error, "http_status", None),
            "body": excerpt(getattr(error, "body", ""), secrets=secrets),
        },
    )


def stream_probe(base: str, alias: str, config: HttpConfig) -> dict[str, Any]:
    """One short streamed completion: time to first token and tokens per second."""
    body = {
        "model": alias,
        "messages": [{"role": "user", "content": PERF_PROMPT}],
        "max_tokens": PERF_TOKENS,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    headers = {
        **config.headers(),
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    request = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        json.dumps(body).encode(),
        headers,
        method="POST",
    )
    started = time.monotonic()
    first = last = None
    pieces = 0
    usage_tokens: int | None = None
    with _open(request, config.case_timeout, config.context()) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            usage = chunk.get("usage")
            if isinstance(usage, dict) and isinstance(
                usage.get("completion_tokens"), int
            ):
                usage_tokens = usage["completion_tokens"]
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                text = (
                    delta.get("content")
                    or delta.get("reasoning_content")
                    or delta.get("reasoning")
                )
                if text:
                    now = time.monotonic()
                    first = first if first is not None else now
                    last = now
                    pieces += 1
    if first is None or last is None:
        raise AssertionFailed("the stream produced no tokens")
    tokens = usage_tokens if usage_tokens else pieces
    span = last - first
    return {
        "ttft_ms": round((first - started) * 1000, 1),
        "tokens": tokens,
        "tokens_per_second": round((tokens - 1) / span, 2)
        if tokens > 1 and span > 0
        else None,
        "total_ms": round((last - started) * 1000, 1),
    }


def _fallback_cases(recipe: Recipe, alias: str) -> list[dict[str, Any]]:
    """Cases from the recipe's own declared serving checks when no reviewed smoke exists."""
    cases: list[dict[str, Any]] = [
        {
            "id": "models",
            "method": "GET",
            "path": "/models",
            "body": None,
            "timeout_seconds": 60,
            "max_response_bytes": 262144,
            "assertions": [{"kind": "path.nonempty", "path": "data"}],
        }
    ]
    for check in recipe.checks:
        request = check.get("request") or {}
        if check.get("kind") == "openai.chat" and isinstance(request.get("body"), dict):
            body = {**request["body"], "model": alias, "stream": False}
            cases.append(
                {
                    "id": str(check.get("name", "chat")),
                    "method": "POST",
                    "path": "/chat/completions",
                    "body": body,
                    "timeout_seconds": 180,
                    "max_response_bytes": 262144,
                    "assertions": [{"kind": "path.nonempty", "path": "choices"}],
                }
            )
    return cases


def run_case(base: str, case: dict[str, Any], config: HttpConfig) -> dict[str, Any]:
    body = case.get("body")
    method = str(case.get("method", "GET"))
    headers = dict(config.headers())
    data = None
    if method == "POST":
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        base.rstrip("/") + str(case["path"]), data, headers, method=method
    )
    limit = int(case.get("max_response_bytes", 262144))
    timeout = min(float(case.get("timeout_seconds", 180)), config.case_timeout)
    started = time.monotonic()
    with _open(request, timeout, config.context()) as response:
        status = int(getattr(response, "status", 200))
        raw = response.read(limit + 1)
    if len(raw) > limit:
        raise AssertionFailed("response exceeds its bound")
    text = raw.decode("utf-8", "replace")
    try:
        document = json.loads(text)
        check_assertions(document, text, list(case.get("assertions", [])))
    except (AssertionFailed, ValueError) as error:
        error.body = text  # type: ignore[attr-defined]  # evidence for the result row
        error.http_status = status  # type: ignore[attr-defined]
        raise
    record: dict[str, Any] = {
        "case_id": case.get("id"),
        "method": method,
        "path": case["path"],
        "http_status": status,
        "latency_ms": round((time.monotonic() - started) * 1000, 1),
        "response_bytes": len(raw),
    }
    quality = check_quality(
        str(case.get("id")), document, text, list(case.get("quality_assertions", []))
    )
    if quality:
        record["quality"] = quality
    return record


def has_chat(recipe: Recipe, cases: list[dict[str, Any]]) -> bool:
    return any(
        str(c.get("path", "")).endswith("/chat/completions") for c in cases
    ) or any(c.get("kind") == "openai.chat" for c in recipe.checks)


def smoke_service(
    recipe: Recipe, definitions: Definitions, base: str, alias: str, config: HttpConfig
) -> SmokeResult:
    started = time.monotonic()
    cases = definitions.service_cases(recipe.key, alias) or _fallback_cases(
        recipe, alias
    )
    result = SmokeResult(ok=False, kind="service")
    try:
        if has_chat(recipe, cases):
            result.perf = stream_probe(base, alias, config)
    except AssertionFailed as error:
        result.failure = _assertion_failure(error, "stream", config)
    except (OSError, urllib.error.URLError, ValueError) as error:
        result.failure = _transport_failure(error, "stream", config)
    if result.failure is None:
        for case in cases:
            try:
                record = run_case(base, case, config)
                result.quality.extend(record.pop("quality", []))
                result.cases.append(record)
            except AssertionFailed as error:
                result.failure = _assertion_failure(error, str(case.get("id")), config)
            except (OSError, urllib.error.URLError, ValueError) as error:
                result.failure = _transport_failure(error, str(case.get("id")), config)
            if result.failure is not None:
                break
    result.ok = result.failure is None
    result.seconds = round(time.monotonic() - started, 2)
    return result


def smoke_readiness(run_id: str) -> SmokeResult:
    """Non-OpenAI recipes: the run is healthy and routed (the caller verified); nothing more is claimed."""
    return SmokeResult(
        ok=True,
        kind="readiness-only",
        cases=[{"case_id": "readiness", "run_id": run_id}],
    )
