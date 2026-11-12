"""MiniMax-H3-Image job: the SGLang Diffusion server and the creator's requests.

The creator serves ``sglang serve --model-path ... --backend sglang`` and sends
``POST /v1/images/generations`` (text to image) or ``POST /v1/images/edits``
(ordered reference images) with the picker's request fields. A job starts that
server on loopback, waits for its health, sends one request and writes the PNG.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from pydantic import ValidationError
from vonk_agent_protocol.job_inputs import RecipeJobInputManifest

INPUT_ROOT = Path("/inputs")
MODEL_PATH = "/models"
# The model name the picker puts in its requests; the server serves MODEL_PATH.
MODEL_NAME = "MiniMaxAI/MiniMax-H3-Image"
HOST = "127.0.0.1"
PORT = 30000
HEALTH_POLL_SECONDS = 5
STOP_GRACE_SECONDS = 30


def load_inputs() -> tuple[str, list[Path]]:
    """The prompt text and the reference images, from the declared manifest only."""
    manifest_path = INPUT_ROOT / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise SystemExit("a regular Vonk job input manifest is required")
    try:
        manifest = RecipeJobInputManifest.model_validate_json(
            manifest_path.read_bytes()
        )
    except (OSError, ValidationError) as error:
        raise SystemExit(f"invalid Vonk job input manifest: {error}") from error
    prompts = [item for item in manifest.files if item.slot == "prompt"]
    images = [item for item in manifest.files if item.slot == "image"]
    raw = (INPUT_ROOT / prompts[0].name).read_bytes()
    prompt = raw.decode("utf-8").strip()
    return prompt, [INPUT_ROOT / item.name for item in images]


def request_fields(args: argparse.Namespace, prompt: str) -> dict[str, str]:
    """The picker's request fields; strings because the edit request is a form."""
    return {
        "model": MODEL_NAME,
        "prompt": prompt,
        "size": args.size,
        "quality": args.quality,
        "seed": str(args.seed),
        "generator_device": "cpu",
        "response_format": "b64_json",
        "output_format": "png",
    }


def generation_request(fields: dict[str, str]) -> urllib.request.Request:
    body = json.dumps({**fields, "seed": int(fields["seed"])}).encode()
    return urllib.request.Request(
        f"http://{HOST}:{PORT}/v1/images/generations",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )


def edit_request(
    fields: dict[str, str], images: list[tuple[str, bytes]]
) -> urllib.request.Request:
    """``-F image=@file`` once per reference, in prompt order, then the fields."""
    boundary = uuid.uuid4().hex
    parts: list[bytes] = []
    for name, payload in images:
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="image"; '
            f'filename="{name}"\r\nContent-Type: application/octet-stream\r\n\r\n'.encode()
            + payload
            + b"\r\n"
        )
    for key, value in fields.items():
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n'.encode()
            + value.encode()
            + b"\r\n"
        )
    return urllib.request.Request(
        f"http://{HOST}:{PORT}/v1/images/edits",
        data=b"".join(parts) + f"--{boundary}--\r\n".encode(),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )


def image_bytes(response_body: bytes) -> bytes:
    try:
        data = json.loads(response_body)["data"][0]["b64_json"]
        return base64.b64decode(data, validate=True)
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise SystemExit(f"the server returned no b64_json image: {error}") from error


def serve_command() -> list[str]:
    return [
        "sglang",
        "serve",
        "--model-path",
        MODEL_PATH,
        "--backend",
        "sglang",
        "--host",
        HOST,
        "--port",
        str(PORT),
        # The pinned SGLang refuses a local text_encoder/ when model_index.json
        # names the encoder externally (fixed upstream in 6be0d75a).
        "--component-paths.text_encoder",
        f"{MODEL_PATH}/text_encoder",
    ]


def start_server(cwd: str) -> subprocess.Popen[bytes]:
    return subprocess.Popen(serve_command(), cwd=cwd, start_new_session=True)


def wait_healthy(server: subprocess.Popen[bytes], deadline: float) -> None:
    url = f"http://{HOST}:{PORT}/health"
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise SystemExit(f"the SGLang server exited with {server.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=HEALTH_POLL_SECONDS) as reply:
                if reply.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(HEALTH_POLL_SECONDS)
    raise SystemExit("the SGLang server did not become healthy before the deadline")


def stop(server: subprocess.Popen[bytes]) -> None:
    if server.poll() is not None:
        return
    for sig, wait in ((signal.SIGTERM, STOP_GRACE_SECONDS), (signal.SIGKILL, 10)):
        try:
            os.killpg(server.pid, sig)
            server.wait(timeout=wait)
            return
        except ProcessLookupError:
            return
        except subprocess.TimeoutExpired:
            continue


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-mime", choices=["image/png"], required=True)
    parser.add_argument("--size", required=True)
    parser.add_argument("--quality", choices=["lossless", "exact"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        seconds = int(os.environ["VONK_JOB_TIMEOUT_SECONDS"])
    except (KeyError, ValueError) as error:
        raise SystemExit("job must declare VONK_JOB_TIMEOUT_SECONDS") from error
    deadline = time.monotonic() + seconds
    prompt, references = load_inputs()
    fields = request_fields(args, prompt)
    if references:
        request = edit_request(fields, [(p.name, p.read_bytes()) for p in references])
    else:
        request = generation_request(fields)

    # SGLang writes outputs/ and uploads relative to its cwd; the root is read-only.
    with tempfile.TemporaryDirectory(prefix="vonk-h3-server-") as cwd:
        server = start_server(cwd)
        try:
            wait_healthy(server, deadline)
            remaining = max(1.0, deadline - time.monotonic())
            with urllib.request.urlopen(request, timeout=remaining) as reply:
                png = image_bytes(reply.read())
        finally:
            stop(server)
    if not png.startswith(b"\x89PNG\r\n\x1a\n"):
        raise SystemExit("the server image is not a PNG")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vonk-h3-image-") as scratch:
        staged = Path(scratch) / "minimax-h3-image.png"
        staged.write_bytes(png)
        os.replace(staged, args.output_dir / "minimax-h3-image.png")


if __name__ == "__main__":
    sys.exit(main())
