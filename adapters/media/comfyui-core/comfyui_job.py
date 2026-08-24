#!/usr/bin/env python3
"""Run one immutable ComfyUI API workflow as a shell-free Vonk media job."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import uuid


COMFYUI_ROOT = Path("/opt/comfyui")
WORKFLOW_ROOT = Path("/opt/vonk/source/workflows")
MODEL_MOUNT_ROOT = Path("/models")
INPUT_ROOT = Path("/inputs")
PORT = 8188
BASE_URL = f"http://127.0.0.1:{PORT}"
MIME_EXTENSIONS = {
    "image/png": (".png",),
    "image/jpeg": (".jpg", ".jpeg"),
    "video/mp4": (".mp4",),
    "video/webm": (".webm",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow", required=True)
    parser.add_argument("--workflow-sha256", required=True)
    parser.add_argument("--output-mime", required=True, choices=sorted(MIME_EXTENSIONS))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output-dir", default="/outputs")
    return parser.parse_args()


def load_workflow(relative_path: str, expected_sha256: str) -> dict[str, Any]:
    candidate = (WORKFLOW_ROOT / relative_path).resolve()
    root = WORKFLOW_ROOT.resolve()
    if root not in candidate.parents:
        raise ValueError("workflow must resolve below /opt/vonk/source/workflows")
    payload = candidate.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected_sha256:
        raise ValueError(f"workflow digest mismatch: expected {expected_sha256}, got {actual}")
    document = json.loads(payload)
    if document.get("schema_version") != 1 or not isinstance(document.get("prompt"), dict):
        raise ValueError("unsupported immutable workflow document")
    return document


def link_models(document: dict[str, Any], model_root: Path) -> None:
    categories = {"diffusion_models", "text_encoders", "vae", "loras", "diffusers"}
    for category in categories:
        (model_root / category).mkdir(parents=True, exist_ok=True)
    for model in document.get("models", []):
        artifact_id = model["artifact_id"]
        category = model["category"]
        filename = model["filename"]
        if category not in categories or Path(filename).name != filename:
            raise ValueError(f"invalid model mapping for {artifact_id}")
        source = MODEL_MOUNT_ROOT / artifact_id
        if not source.exists():
            raise FileNotFoundError(f"required model artifact is missing: {source}")
        (model_root / category / filename).symlink_to(source)
    for snapshot in document.get("diffusers_snapshots", []):
        artifact_id = snapshot["artifact_id"]
        name = snapshot["name"]
        if Path(name).name != name:
            raise ValueError(f"invalid Diffusers snapshot name: {name}")
        source = MODEL_MOUNT_ROOT if artifact_id == "weights" else MODEL_MOUNT_ROOT / artifact_id
        if not (source / "model_index.json").is_file():
            raise FileNotFoundError(f"Diffusers snapshot lacks model_index.json: {source}")
        (model_root / "diffusers" / name).symlink_to(source, target_is_directory=True)
    for model in document.get("snapshot_models", []):
        artifact_id = model["artifact_id"]
        category = model["category"]
        filename = model["filename"]
        if category not in categories or Path(filename).name != filename:
            raise ValueError(f"invalid snapshot model mapping for {artifact_id}")
        snapshot = MODEL_MOUNT_ROOT if artifact_id == "weights" else MODEL_MOUNT_ROOT / artifact_id
        destination = model_root / category / filename
        if model["mode"] == "link":
            source = snapshot / model["path"]
            if not source.is_file():
                raise FileNotFoundError(f"snapshot model file is missing: {source}")
            destination.symlink_to(source)
        elif model["mode"] == "merge-safetensors":
            sources = sorted(snapshot.glob(model["glob"]))
            if not sources:
                raise FileNotFoundError(f"snapshot model shards are missing: {model['glob']}")
            merge_safetensors(sources, destination)
        else:
            raise ValueError(f"unsupported snapshot model mode: {model['mode']}")


def safetensors_header(path: Path) -> tuple[int, dict[str, Any]]:
    with path.open("rb") as stream:
        raw_length = stream.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"invalid safetensors header: {path}")
        header_length = struct.unpack("<Q", raw_length)[0]
        header = json.loads(stream.read(header_length))
    return header_length, header


def merge_safetensors(sources: list[Path], destination: Path) -> None:
    """Merge sharded safetensors by copying tensor byte ranges without loading weights."""
    tensors: list[tuple[str, Path, int, int, str, list[int]]] = []
    metadata: dict[str, str] = {}
    for source in sources:
        header_length, header = safetensors_header(source)
        for key, value in header.get("__metadata__", {}).items():
            metadata.setdefault(key, value)
        for name, descriptor in header.items():
            if name == "__metadata__":
                continue
            start, end = descriptor["data_offsets"]
            if end < start:
                raise ValueError(f"invalid safetensors offsets in {source}: {name}")
            tensors.append((name, source, 8 + header_length + start, end - start, descriptor["dtype"], descriptor["shape"]))
    combined: dict[str, Any] = {}
    cursor = 0
    for name, _, _, length, dtype, shape in tensors:
        if name in combined:
            raise ValueError(f"duplicate tensor across safetensors shards: {name}")
        combined[name] = {"dtype": dtype, "shape": shape, "data_offsets": [cursor, cursor + length]}
        cursor += length
    if metadata:
        combined["__metadata__"] = metadata
    header = json.dumps(combined, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    header += b" " * ((8 - len(header) % 8) % 8)
    with destination.open("wb") as output:
        output.write(struct.pack("<Q", len(header)))
        output.write(header)
        for _, source, source_offset, length, _, _ in tensors:
            with source.open("rb") as input_stream:
                input_stream.seek(source_offset)
                remaining = length
                while remaining:
                    chunk = input_stream.read(min(8 * 1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError(f"truncated safetensors shard: {source}")
                    output.write(chunk)
                    remaining -= len(chunk)


def input_names(document: dict[str, Any]) -> list[str]:
    settings = document.get("inputs", {})
    files = sorted(path for path in INPUT_ROOT.iterdir() if path.is_file()) if INPUT_ROOT.exists() else []
    minimum = int(settings.get("minimum", 0))
    maximum = int(settings.get("maximum", minimum))
    if not minimum <= len(files) <= maximum:
        raise ValueError(f"workflow requires {minimum}..{maximum} input files; received {len(files)}")
    return [path.name for path in files]


def substitute(value: Any, replacements: dict[str, Any]) -> Any:
    if isinstance(value, dict):
        return {key: substitute(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [substitute(item, replacements) for item in value]
    if isinstance(value, str) and value in replacements:
        return replacements[value]
    return value


def request_json(path: str, *, body: dict[str, Any] | None = None) -> Any:
    payload = None if body is None else json.dumps(body).encode("utf-8")
    request = Request(
        f"{BASE_URL}{path}",
        data=payload,
        headers={"Content-Type": "application/json"} if payload is not None else {},
        method="POST" if payload is not None else "GET",
    )
    with urlopen(request, timeout=10) as response:
        return json.load(response)


def wait_for_server(process: subprocess.Popen[bytes], timeout: int = 180) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"ComfyUI exited during startup with status {process.returncode}")
        try:
            request_json("/system_stats")
            return
        except (HTTPError, URLError, TimeoutError):
            time.sleep(1)
    raise TimeoutError("ComfyUI did not become ready")


def wait_for_prompt(prompt_id: str, process: subprocess.Popen[bytes], timeout: int) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"ComfyUI exited while executing with status {process.returncode}")
        history = request_json(f"/history/{prompt_id}")
        item = history.get(prompt_id)
        if item:
            status = item.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(f"ComfyUI workflow failed: {status.get('messages', [])}")
            if status.get("completed"):
                return
        time.sleep(2)
    raise TimeoutError("ComfyUI workflow timed out")


def copy_output(comfy_output: Path, destination: Path, mime: str) -> Path:
    extensions = MIME_EXTENSIONS[mime]
    candidates = sorted(
        path for path in comfy_output.rglob("*") if path.is_file() and path.suffix.lower() in extensions
    )
    if not candidates:
        raise FileNotFoundError(f"workflow produced no {mime} output")
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / f"result{candidates[0].suffix.lower()}"
    shutil.copyfile(candidates[0], target)
    return target


def main() -> int:
    args = parse_args()
    document = load_workflow(args.workflow, args.workflow_sha256)
    inputs = input_names(document)
    replacements: dict[str, Any] = {
        "__VONK_PROMPT__": os.environ.get("VONK_PROMPT", ""),
        "__VONK_NEGATIVE_PROMPT__": os.environ.get("VONK_NEGATIVE_PROMPT", ""),
        "__VONK_SEED__": args.seed,
    }
    for index, name in enumerate(inputs, start=1):
        replacements[f"__VONK_INPUT_{index}__"] = name
    prompt = substitute(document["prompt"], replacements)

    with tempfile.TemporaryDirectory(prefix="vonk-comfyui-") as temp_dir:
        temporary = Path(temp_dir)
        model_root = temporary / "models"
        comfy_output = temporary / "output"
        comfy_output.mkdir()
        link_models(document, model_root)
        extra_paths = temporary / "extra_model_paths.yaml"
        extra_paths.write_text(
            "vonk:\n"
            f"  base_path: {model_root}\n"
            "  diffusion_models: diffusion_models\n"
            "  text_encoders: text_encoders\n"
            "  vae: vae\n"
            "  loras: loras\n"
            "  diffusers: diffusers\n",
            encoding="utf-8",
        )
        command = [
            sys.executable,
            str(COMFYUI_ROOT / "main.py"),
            "--listen", "127.0.0.1",
            "--port", str(PORT),
            "--disable-auto-launch",
            "--disable-metadata",
            "--output-directory", str(comfy_output),
            "--input-directory", str(INPUT_ROOT),
            "--extra-model-paths-config", str(extra_paths),
        ]
        process = subprocess.Popen(command, stdout=sys.stderr, stderr=sys.stderr)
        try:
            wait_for_server(process)
            queued = request_json("/prompt", body={"prompt": prompt, "client_id": str(uuid.uuid4())})
            prompt_id = queued.get("prompt_id")
            if not prompt_id:
                raise RuntimeError(f"ComfyUI rejected workflow: {queued}")
            wait_for_prompt(
                prompt_id,
                process,
                int(os.environ.get("VONK_COMFYUI_TIMEOUT_SECONDS", "7200")),
            )
            output = copy_output(comfy_output, Path(args.output_dir), args.output_mime)
            print(json.dumps({"output": str(output), "mime": args.output_mime}, sort_keys=True))
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
