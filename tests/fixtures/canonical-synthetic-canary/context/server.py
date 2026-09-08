#!/usr/bin/env python3
"""Deterministic OpenAI-shaped HTTP smoke service for contract tests."""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from pydantic import ValidationError
from request_contract import ChatRequest

MODEL = "canonical-synthetic-canary"
HEALTH = {
    "status": "ok",
    "service": MODEL,
    "model": MODEL,
    "healthy": True,
}
EXPECTED_RESPONSE = {
    "id": "chatcmpl-canonical-synthetic-canary",
    "object": "chat.completion",
    "created": 1735689600,
    "model": MODEL,
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "canonical synthetic ok",
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 1,
        "completion_tokens": 3,
        "total_tokens": 4,
    },
}


def matches_fixture(request: ChatRequest) -> bool:
    """Keep deterministic test content separate from request structure."""
    return (
        request.model == MODEL
        and len(request.messages) == 1
        and request.messages[0].role == "user"
        and request.messages[0].content == "ping"
        and request.max_tokens in (None, 16)
    )


class Handler(BaseHTTPRequestHandler):
    server_version = "canonical-synthetic-canary/1.0"

    def do_GET(self) -> None:
        if self.path != "/health":
            self.send_error(404)
            return
        self._write_json(200, HEALTH)

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        try:
            payload = json.loads(
                self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}"
            )
            request = ChatRequest.model_validate(payload)
        except (ValueError, ValidationError):
            self._write_json(
                400,
                {
                    "error": {
                        "message": "invalid request structure",
                        "type": "invalid_request_error",
                    }
                },
            )
            return
        if not matches_fixture(request):
            self._write_json(
                400,
                {
                    "error": {
                        "message": "unexpected request",
                        "type": "invalid_request_error",
                    }
                },
            )
            return
        if request.stream is True:
            self._write_stream()
            return
        self._write_json(200, EXPECTED_RESPONSE)

    def log_message(self, format: str, *args: object) -> None:
        return

    def _write_json(self, status: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _write_stream(self) -> None:
        chunks = [
            {
                "id": EXPECTED_RESPONSE["id"],
                "object": "chat.completion.chunk",
                "created": EXPECTED_RESPONSE["created"],
                "model": MODEL,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": "canonical synthetic ok",
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": EXPECTED_RESPONSE["id"],
                "object": "chat.completion.chunk",
                "created": EXPECTED_RESPONSE["created"],
                "model": MODEL,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            },
        ]
        body = (
            b"".join(
                (
                    "data: "
                    + json.dumps(chunk, sort_keys=True, separators=(",", ":"))
                    + "\n\n"
                ).encode()
                for chunk in chunks
            )
            + b"data: [DONE]\n\n"
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    host = os.environ.get("VONK_LISTEN_HOST", "0.0.0.0")
    port = int(os.environ.get("VONK_LISTEN_PORT", "8000"))
    ThreadingHTTPServer((host, port), Handler).serve_forever()
