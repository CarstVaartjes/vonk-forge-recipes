"""The smoke harness never fails a recipe for a sampling parameter its model family refuses."""

from __future__ import annotations

import email.message
import io
import json
import urllib.error

import pytest

from spark_sweep import smoke

REFUSAL = (
    "The temperature, min_p, seed, min_tokens, logit_bias, bad_words, and "
    "allowed_token_ids sampling parameters are not yet supported with diffusion models."
)


class _Reply(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def _server(monkeypatch, refuse: bool, status: int = 500):
    seen: list[dict] = []

    def fake(request, timeout, context):
        body = json.loads(request.data)
        seen.append(body)
        if refuse and any(k in body for k in smoke.SAMPLING_PARAMS):
            raise urllib.error.HTTPError(
                request.full_url,
                status,
                "err",
                email.message.Message(),
                io.BytesIO(REFUSAL.encode()),
            )
        return _Reply(b'{"choices": [{"message": {"content": "hi"}}]}')

    monkeypatch.setattr(smoke, "_open", fake)
    return seen


@pytest.mark.parametrize("status", [400, 500])
def test_a_refused_sampling_parameter_is_retried_without_them(
    monkeypatch, status
) -> None:
    seen = _server(monkeypatch, True, status)
    case = {
        "id": "stream",
        "method": "POST",
        "path": "/chat/completions",
        "body": {"model": "m", "temperature": 0, "seed": 42, "messages": []},
        "assertions": [{"kind": "path.nonempty", "path": "choices"}],
    }
    record = smoke.run_case("http://x/v1", case, smoke.HttpConfig())
    assert record["http_status"] == 200
    assert "temperature" not in seen[-1] and "seed" not in seen[-1]
    assert seen[-1]["model"] == "m" and len(seen) == 2


def test_other_errors_still_fail_and_keep_their_body(monkeypatch) -> None:
    def fake(request, timeout, context):
        raise urllib.error.HTTPError(
            request.full_url,
            500,
            "err",
            email.message.Message(),
            io.BytesIO(b"engine crashed"),
        )

    monkeypatch.setattr(smoke, "_open", fake)
    with pytest.raises(urllib.error.HTTPError) as caught:
        smoke.open_with_body("http://x", {"temperature": 0}, {}, 5, None)
    failure = smoke._transport_failure(caught.value, "stream")
    assert "engine crashed" in failure.evidence["body"]


def test_the_stream_probe_sends_no_sampling_parameters(monkeypatch) -> None:
    sent: list[dict] = []

    def fake(request, timeout, context):
        sent.append(json.loads(request.data))
        return _Reply(
            b'data: {"choices": [{"delta": {"content": "a"}}]}\n\ndata: [DONE]\n'
        )

    monkeypatch.setattr(smoke, "_open", fake)
    smoke.stream_probe("http://x/v1", "m", smoke.HttpConfig())
    assert not set(sent[0]) & smoke.SAMPLING_PARAMS
