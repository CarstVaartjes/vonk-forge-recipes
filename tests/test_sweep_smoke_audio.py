"""The sweep's audio cases call the gateway exactly as an OpenAI client does.

A local server parses the multipart upload and the speech request, so the cases
are proven on the wire: method, path, client key, form fields, file part, and
the judgement of a binary reply.
"""

from __future__ import annotations

import io
import json
import struct
import threading
import wave
from collections.abc import Iterator
from email import message_from_bytes
from email.policy import HTTP
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from qualification.case_assertions import invalid_case_assertions
from spark_sweep import smoke
from spark_sweep.definitions import Definitions

ROOT = Path(__file__).resolve().parents[1]
QUALIFICATION = ROOT / "qualification"
SHARED: dict[str, Any] = json.loads((QUALIFICATION / "shared.json").read_text("utf-8"))
JFK = (QUALIFICATION / "fixtures/jfk-inaugural-10s.wav").read_bytes()
KEY = "client-key-123"


def make_wav(seconds: float, rate: int = 24000) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(rate)
        writer.writeframes(b"\x01\x00" * int(rate * seconds))
    return out.getvalue()


class Gateway:
    """Records each request and answers like an OpenAI-compatible gateway."""

    def __init__(self) -> None:
        self.seen: list[dict[str, Any]] = []
        self.transcript = {
            "text": "Ask not what your country can do. Final judge. Land we love."
        }
        self.speech = (make_wav(2.0), "audio/wav")


@pytest.fixture
def gateway(tmp_path: Path) -> Iterator[tuple[Gateway, str, smoke.HttpConfig]]:
    state = Gateway()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: object) -> None:
            pass

        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers["Content-Length"]))
            record: dict[str, Any] = {
                "path": self.path,
                "auth": self.headers.get("Authorization"),
                "content_type": self.headers["Content-Type"],
            }
            if self.path.endswith("/audio/transcriptions"):
                message = message_from_bytes(
                    b"Content-Type: "
                    + record["content_type"].encode()
                    + b"\r\n\r\n"
                    + body,
                    policy=HTTP,
                )
                parts = list(message.iter_parts())
                record["multipart"] = message.get_content_type()
                record["fields"] = {
                    p.get_param("name", header="content-disposition"): p.get_content()
                    for p in parts
                    if p.get_filename() is None
                }
                record["order"] = [
                    p.get_param("name", header="content-disposition") for p in parts
                ]
                (upload,) = [p for p in parts if p.get_filename() is not None]
                record["file"] = {
                    "name": upload.get_filename(),
                    "type": upload.get_content_type(),
                    "bytes": upload.get_payload(decode=True),
                }
                reply, kind = json.dumps(state.transcript).encode(), "application/json"
            else:
                record["json"] = json.loads(body)
                reply, kind = state.speech
            state.seen.append(record)
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    key_file = tmp_path / "key"
    key_file.write_text(KEY + "\n", encoding="utf-8")
    try:
        yield (
            state,
            f"http://127.0.0.1:{server.server_port}/v1",
            smoke.HttpConfig(key_file=key_file),
        )
    finally:
        server.shutdown()
        server.server_close()


def cases(voice: str | None = "vivian") -> dict[str, dict[str, Any]]:
    document = {
        "service_recipes": {
            "k": {
                "alias": "alias-x",
                "smoke_cases": ["ASR_TRANSCRIPTION_JFK", "TTS_SPEECH"],
            }
        },
        "service_case_templates": SHARED["service_case_templates"],
        "fixtures": SHARED["fixtures"],
    }
    rendered = Definitions(document, QUALIFICATION).service_cases("k", "alias-x", voice)
    return {case["id"]: case for case in rendered}


def test_the_shared_audio_cases_follow_the_assertion_contract() -> None:
    assert invalid_case_assertions(SHARED) == []


def test_transcription_is_an_openai_multipart_upload(gateway) -> None:
    state, base, config = gateway
    record = smoke.run_case(base, cases()["ASR_TRANSCRIPTION_JFK"], config)
    (seen,) = state.seen
    assert seen["path"] == "/v1/audio/transcriptions"
    assert seen["auth"] == f"Bearer {KEY}"
    assert seen["multipart"] == "multipart/form-data"
    assert seen["fields"] == {
        "model": "alias-x",
        "language": "en",
        "response_format": "json",
    }
    assert seen["order"] == ["model", "language", "response_format", "file"]
    assert seen["file"] == {
        "name": "jfk-inaugural-10s.wav",
        "type": "audio/wav",
        "bytes": JFK,
    }
    assert record["http_status"] == 200
    assert [note["ok"] for note in record["quality"]] == [True, True]


def test_an_empty_transcript_fails_and_a_wrong_one_is_only_a_quality_note(
    gateway,
) -> None:
    state, base, config = gateway
    case = cases()["ASR_TRANSCRIPTION_JFK"]
    state.transcript = {"text": "  "}
    with pytest.raises(smoke.AssertionFailed):
        smoke.run_case(base, case, config)
    state.transcript = {"text": "something else"}
    record = smoke.run_case(base, case, config)
    assert [note["ok"] for note in record["quality"]] == [False, False]


def test_speech_is_an_openai_json_request_and_the_wav_is_judged(gateway) -> None:
    state, base, config = gateway
    record = smoke.run_case(base, cases()["TTS_SPEECH"], config)
    (seen,) = state.seen
    assert seen["path"] == "/v1/audio/speech"
    assert seen["auth"] == f"Bearer {KEY}"
    assert seen["content_type"] == "application/json"
    assert seen["json"] == {
        "model": "alias-x",
        "input": "The quick brown fox jumps over the lazy dog.",
        "voice": "vivian",
        "response_format": "wav",
    }
    assert record["response_bytes"] == len(state.speech[0])
    assert [note["ok"] for note in record["quality"]] == [True]


@pytest.mark.parametrize(
    ("reply", "kind"),
    [
        (make_wav(0.2), "audio/wav"),  # under the 0.5 s floor
        (b"not a wav", "audio/wav"),
        (make_wav(2.0), "application/json"),  # not an audio type
    ],
)
def test_a_reply_that_is_not_speech_fails_the_case(gateway, reply, kind) -> None:
    state, base, config = gateway
    state.speech = (reply, kind)
    with pytest.raises(smoke.AssertionFailed):
        smoke.run_case(base, cases()["TTS_SPEECH"], config)


def test_a_short_wav_that_decodes_is_only_a_quality_note(gateway) -> None:
    state, base, config = gateway
    state.speech = (make_wav(0.8), "audio/wav")
    record = smoke.run_case(base, cases()["TTS_SPEECH"], config)
    assert [note["ok"] for note in record["quality"]] == [False]


def test_a_streamed_float_wav_still_decodes() -> None:
    rate = 24000
    fmt = struct.pack("<HHIIHH", 3, 1, rate, rate * 4, 4, 32)
    data = b"\x00" * (rate * 4)  # one second
    header = b"fmt " + struct.pack("<I", 16) + fmt
    streamed = (
        b"RIFF"
        + b"\xff\xff\xff\xff"
        + b"WAVE"
        + header
        + b"data"
        + b"\xff\xff\xff\xff"
        + data
    )
    facts = smoke.wav_facts(streamed)
    assert facts is not None and facts["seconds"] == 1.0 and facts["format_tag"] == 3
    assert smoke.wav_facts(b"RIFF....WAVE") is None


def test_speech_needs_the_recipes_voice() -> None:
    with pytest.raises(ValueError, match="voices"):
        cases(voice=None)


def test_every_recipe_that_smokes_speech_states_its_voices() -> None:
    definitions = Definitions.load(QUALIFICATION)
    for key, service in definitions.document["service_recipes"].items():
        if "TTS_SPEECH" in service.get("smoke_cases", []):
            slug = key.split("/")[-1]
            api = definitions.recipe_api(slug)
            assert smoke.speech_voice(api) is not None, (
                f"{key} needs an audio_speech api"
            )
