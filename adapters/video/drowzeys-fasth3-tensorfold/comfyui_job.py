#!/usr/bin/env python3
"""Run one immutable ComfyUI API workflow as a shell-free Vonk media job."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from fractions import Fraction
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

COMFYUI_ROOT = Path("/opt/comfyui")
WORKFLOW_ROOT = Path("/opt/vonk/source/workflows")
MODEL_MOUNT_ROOT = Path("/models")
INPUT_ROOT = Path("/inputs")
INPUT_MANIFEST_NAME = "manifest.json"
MAX_INPUT_MANIFEST_BYTES = 256 * 1024
PORT = 8188
BASE_URL = f"http://127.0.0.1:{PORT}"
MIME_EXTENSIONS = {
    "image/png": (".png",),
    "image/jpeg": (".jpg", ".jpeg"),
    "video/mp4": (".mp4",),
    "video/webm": (".webm",),
}
MEDIA_EXTENSIONS = frozenset(
    extension for extensions in MIME_EXTENSIONS.values() for extension in extensions
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow", required=True)
    parser.add_argument("--workflow-sha256", required=True)
    parser.add_argument("--output-mime", required=True, choices=sorted(MIME_EXTENSIONS))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output-dir", default="/outputs")
    parser.add_argument("--num-frames", type=int, default=124)
    args = parser.parse_args()
    if not 120 <= args.num_frames <= 344 or args.num_frames % 4:
        raise ValueError("num-frames must be a multiple of 4 between 120 and 344")
    return args


def load_workflow(relative_path: str, expected_sha256: str) -> dict[str, Any]:
    candidate = (WORKFLOW_ROOT / relative_path).resolve()
    root = WORKFLOW_ROOT.resolve()
    if root not in candidate.parents:
        raise ValueError("workflow must resolve below /opt/vonk/source/workflows")
    payload = candidate.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected_sha256:
        raise ValueError(
            f"workflow digest mismatch: expected {expected_sha256}, got {actual}"
        )
    document = json.loads(payload)
    if document.get("schema_version") != 1 or not isinstance(
        document.get("prompt"), dict
    ):
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
        # Selected artifacts may mount a snapshot directory preserving the
        # upstream path, rather than a file at the mount root. ComfyUI must see
        # the checkpoint itself under the workflow's exact basename.
        if source.is_dir():
            # The workflow may alias the selected checkpoint's basename (for
            # example the FP4 Flux variant); its mount still holds one file.
            matches = [
                path for path in source.rglob(f"*{Path(filename).suffix}")
                if path.is_file()
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"model artifact {artifact_id} must contain exactly one "
                    f"{filename}; found {len(matches)}"
                )
            source = matches[0]
        link = model_root / category / filename
        if link.is_symlink() and link.readlink() == source:
            continue
        if link.exists() or link.is_symlink():
            raise ValueError(f"cached model view differs from its immutable identity: {link}")
        link.symlink_to(source)
    for snapshot in document.get("diffusers_snapshots", []):
        artifact_id = snapshot["artifact_id"]
        name = snapshot["name"]
        if Path(name).name != name:
            raise ValueError(f"invalid Diffusers snapshot name: {name}")
        source = (
            MODEL_MOUNT_ROOT
            if artifact_id == "weights"
            else MODEL_MOUNT_ROOT / artifact_id
        )
        if not (source / "model_index.json").is_file():
            raise FileNotFoundError(
                f"Diffusers snapshot lacks model_index.json: {source}"
            )
        (model_root / "diffusers" / name).symlink_to(source, target_is_directory=True)


def load_input_manifest() -> dict[str, Any]:
    from vonk_agent_protocol.job_inputs import RecipeJobInputManifest
    path = INPUT_ROOT / INPUT_MANIFEST_NAME
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_INPUT_MANIFEST_BYTES:
        raise ValueError("a bounded regular input manifest is required")
    document = RecipeJobInputManifest.model_validate_json(path.read_bytes())
    if len(document.files) != 1 or document.files[0].slot != "prompt":
        raise ValueError("exactly one prompt file is required")
    item = document.files[0]
    source = INPUT_ROOT / item.name
    if source.is_symlink() or not source.is_file() or source.suffix.lower() != ".txt" or item.media_type != "text/plain":
        raise ValueError("prompt must be a regular text file")
    if not 1 <= item.size_bytes <= 16384 or source.stat().st_size != item.size_bytes:
        raise ValueError("prompt exceeds its declared size")
    data = source.read_bytes()
    if len(data) != item.size_bytes or hashlib.sha256(data).hexdigest() != item.sha256:
        raise ValueError("prompt digest mismatch")
    if not data.decode("utf-8").strip() or b"\x00" in data:
        raise ValueError("prompt must be nonblank UTF-8 without NUL")
    if {p.name for p in INPUT_ROOT.iterdir()} != {INPUT_MANIFEST_NAME, item.name}:
        raise ValueError("undeclared input file")
    return document.model_dump(mode="json")


def _manifest_files(manifest: dict[str, Any], slot: str) -> list[str]:
    return [item["name"] for item in manifest["files"] if item["slot"] == slot]


def input_names(
    document: dict[str, Any], manifest: dict[str, Any] | None = None
) -> list[str]:
    settings = document.get("inputs", {})
    minimum = int(settings.get("minimum", 0))
    maximum = int(settings.get("maximum", minimum))
    slot = settings.get("slot")
    if manifest is not None and slot is not None:
        if not isinstance(slot, str) or not slot:
            raise ValueError("workflow file input slot is invalid")
        names = _manifest_files(manifest, slot)
    else:
        names = (
            sorted(
                path.name
                for path in INPUT_ROOT.iterdir()
                if path.is_file()
                and not path.is_symlink()
                and path.name != INPUT_MANIFEST_NAME
                and path.suffix.lower() != ".txt"
            )
            if INPUT_ROOT.exists()
            else []
        )
    if not minimum <= len(names) <= maximum:
        raise ValueError(
            f"workflow requires {minimum}..{maximum} input files; received {len(names)}"
        )
    return names


def substitute(value: Any, replacements: dict[str, Any]) -> Any:
    if isinstance(value, dict):
        return {key: substitute(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [substitute(item, replacements) for item in value]
    if isinstance(value, str) and value in replacements:
        return replacements[value]
    return value


def workflow_replacements(
    document: dict[str, Any],
    input_files: list[str],
    seed: int,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    text_values = {
        "prompt": os.environ.get("VONK_PROMPT", ""),
        "negative_prompt": os.environ.get("VONK_NEGATIVE_PROMPT", ""),
    }
    text_inputs = document.get("text_inputs", {})
    if not isinstance(text_inputs, dict) or not set(text_inputs).issubset(text_values):
        raise ValueError("workflow text_inputs contract is invalid")
    for name, settings in text_inputs.items():
        if not isinstance(settings, dict):
            raise TypeError(f"workflow text input contract is invalid: {name}")
        slot = settings.get("slot")
        if manifest is not None and slot is not None:
            if not isinstance(slot, str) or not slot:
                raise ValueError(f"workflow text input slot is invalid: {name}")
            matches = _manifest_files(manifest, slot)
            if len(matches) > 1:
                raise ValueError(
                    f"workflow accepts at most one {name.replace('_', ' ')} file"
                )
            value = ""
            if matches:
                path = INPUT_ROOT / matches[0]
                maximum_bytes = settings.get("maximum_bytes", 16384)
                if type(maximum_bytes) is not int or not 1 <= maximum_bytes <= 1048576:
                    raise ValueError(
                        f"workflow text input byte limit is invalid: {name}"
                    )
                if path.stat().st_size > maximum_bytes:
                    raise ValueError(
                        f"workflow {name.replace('_', ' ')} exceeds {maximum_bytes} bytes"
                    )
                try:
                    value = path.read_text(encoding="utf-8")
                except UnicodeDecodeError as error:
                    raise ValueError(
                        f"workflow {name.replace('_', ' ')} must be UTF-8 text"
                    ) from error
            text_values[name] = value
        else:
            value = text_values[name]
        if settings.get("required", False) and not value.strip():
            raise ValueError(f"workflow requires a non-empty {name.replace('_', ' ')}")
        maximum = settings.get("maximum_characters", 4096)
        if type(maximum) is not int or not 1 <= maximum <= 65536:
            raise ValueError(f"workflow text input limit is invalid: {name}")
        if len(value) > maximum:
            raise ValueError(
                f"workflow {name.replace('_', ' ')} exceeds {maximum} characters"
            )

    replacements: dict[str, Any] = {
        "__VONK_PROMPT__": text_values["prompt"],
        "__VONK_NEGATIVE_PROMPT__": text_values["negative_prompt"],
        "__VONK_SEED__": seed,
    }
    for index, name in enumerate(input_files, start=1):
        replacements[f"__VONK_INPUT_{index}__"] = name
    placeholder_count = document.get("inputs", {}).get(
        "placeholder_count", len(input_files)
    )
    if type(placeholder_count) is not int or not 0 <= placeholder_count <= 32:
        raise ValueError("workflow input placeholder count is invalid")
    if input_files:
        for index in range(len(input_files) + 1, placeholder_count + 1):
            replacements[f"__VONK_INPUT_{index}__"] = input_files[-1]
    return replacements


def request_json(path: str, *, body: dict[str, Any] | None = None) -> Any:
    payload = None if body is None else json.dumps(body).encode("utf-8")
    request = Request(
        f"{BASE_URL}{path}",
        data=payload,
        headers={"Content-Type": "application/json"} if payload is not None else {},
        method="POST" if payload is not None else "GET",
    )
    try:
        with urlopen(request, timeout=10) as response:
            return json.load(response)
    except HTTPError as error:
        # Preserve HTTPError for startup retry handling, and include the local
        # server's node_errors in the uncaught traceback collected as evidence.
        maximum = 64 * 1024
        with error:
            payload = error.read(maximum + 1)
        detail = payload[:maximum].decode("utf-8", errors="replace")
        if len(payload) > maximum:
            detail += " [response body truncated]"
        error.add_note(f"ComfyUI {path} HTTP {error.code} response body: {detail}")
        raise


def wait_for_server(process: subprocess.Popen[bytes], timeout: int = 180) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"ComfyUI exited during startup with status {process.returncode}"
            )
        try:
            request_json("/system_stats")
            return
        except (HTTPError, URLError, TimeoutError):
            time.sleep(1)
    raise TimeoutError("ComfyUI did not become ready")


def wait_for_prompt(
    prompt_id: str, process: subprocess.Popen[bytes], timeout: int
) -> None:
    deadline = time.monotonic() + timeout
    failures = 0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"ComfyUI exited while executing with status {process.returncode}"
            )
        try:
            history = request_json(f"/history/{prompt_id}")
        except (HTTPError, URLError, TimeoutError, ConnectionError) as error:
            if isinstance(error, HTTPError) and error.code not in (408, 429, 500, 502, 503, 504):
                raise
            # Only observe the accepted prompt again; never resubmit it. The owner
            # budget and the process-liveness check remain in force across retries.
            failures += 1
            delay = min(2 ** min(failures, 5), 30, max(0, deadline - time.monotonic()))
            print(f"ComfyUI history {prompt_id} unavailable: {error}; next check in {delay}s", flush=True)
            time.sleep(delay)
            continue
        failures = 0
        item = history.get(prompt_id)
        if item:
            status = item.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(
                    f"ComfyUI workflow failed: {status.get('messages', [])}"
                )
            if status.get("completed"):
                return
        time.sleep(2)
    raise TimeoutError("ComfyUI workflow timed out")


def _positive_float(value: object, field: str) -> float:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"MP4 output does not expose {field}") from error
    if result <= 0:
        raise RuntimeError(f"MP4 output does not expose {field}")
    return result


def validate_mp4(path: Path, contract: dict[str, Any]) -> None:
    video_contract = contract.get("video")
    if not isinstance(video_contract, dict):
        raise TypeError("MP4 result contract must declare video metadata")
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    try:
        document = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("ffprobe returned invalid JSON for MP4 output") from error
    media_format = document.get("format")
    format_names = (
        set(str(media_format.get("format_name", "")).split(","))
        if isinstance(media_format, dict)
        else set()
    )
    if "mp4" not in format_names:
        raise RuntimeError("workflow output must use the MP4 container")
    streams = document.get("streams")
    videos = (
        [stream for stream in streams if stream.get("codec_type") == "video"]
        if isinstance(streams, list)
        else []
    )
    if len(videos) != 1 or not isinstance(videos[0], dict):
        raise RuntimeError("workflow output must contain exactly one video stream")
    video = videos[0]

    codec = video_contract.get("codec")
    if not isinstance(codec, str) or not codec:
        raise ValueError("MP4 result contract must declare a video codec")
    if video.get("codec_name") != codec:
        raise RuntimeError(f"workflow output video codec must be {codec}")
    for field in ("width", "height"):
        expected = video_contract.get(field)
        if type(expected) is not int or expected <= 0:
            raise ValueError(f"MP4 result contract must declare a positive {field}")
        if video.get(field) != expected:
            raise RuntimeError(f"workflow output video {field} must be {expected}")

    expected_fps = _positive_float(video_contract.get("fps"), "declared frame rate")
    try:
        actual_fps = float(Fraction(str(video.get("avg_frame_rate"))))
    except (ValueError, ZeroDivisionError) as error:
        raise RuntimeError("MP4 output does not expose a valid frame rate") from error
    if abs(actual_fps - expected_fps) > 0.01:
        raise RuntimeError(
            f"workflow output video frame rate must be {expected_fps:g} fps"
        )

    expected_frames = video_contract.get("frames")
    if type(expected_frames) is not int or expected_frames <= 0:
        raise ValueError("MP4 result contract must declare a positive frame count")
    raw_frames = video.get("nb_read_frames", video.get("nb_frames"))
    try:
        actual_frames = int(str(raw_frames))
    except (TypeError, ValueError) as error:
        raise RuntimeError("MP4 output does not expose a frame count") from error
    if actual_frames != expected_frames:
        raise RuntimeError(
            f"workflow output video must contain exactly {expected_frames} frames"
        )

    if contract.get("audio") and len([s for s in streams if s.get("codec_type") == "audio"]) != 1:
        raise RuntimeError("workflow output must contain exactly one audio stream")
    raw_duration = video.get("duration")
    if raw_duration is None and isinstance(media_format, dict):
        raw_duration = media_format.get("duration")
    actual_duration = _positive_float(raw_duration, "a duration")
    expected_duration = expected_frames / expected_fps
    tolerance = max(0.05, 1.0 / expected_fps)
    if abs(actual_duration - expected_duration) > tolerance:
        raise RuntimeError(
            "workflow output video duration must match "
            f"{expected_frames} frames at {expected_fps:g} fps"
        )


def copy_output(
    comfy_output: Path,
    destination: Path,
    mime: str,
    contract: dict[str, Any] | None = None,
) -> Path:
    if contract is not None and (
        contract.get("mime") != mime or contract.get("count") != 1
    ):
        raise ValueError("workflow result contract does not match the requested output")
    extensions = MIME_EXTENSIONS[mime]
    media_files = sorted(
        path
        for path in comfy_output.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and path.suffix.lower() in MEDIA_EXTENSIONS
    )
    candidates = [path for path in media_files if path.suffix.lower() in extensions]
    unexpected = [path for path in media_files if path.suffix.lower() not in extensions]
    if not candidates:
        raise FileNotFoundError(f"workflow produced no {mime} output")
    if len(candidates) != 1:
        raise RuntimeError(
            f"workflow produced {len(candidates)} {mime} outputs; expected exactly one"
        )
    if unexpected:
        raise RuntimeError(
            f"workflow produced {len(unexpected)} unexpected media outputs alongside {mime}"
        )
    source = candidates[0]
    if source.stat().st_size == 0:
        raise RuntimeError(f"workflow produced an empty {mime} output")
    if mime == "video/mp4" and contract is not None:
        validate_mp4(source, contract)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / f"result{source.suffix.lower()}"
    partial = destination / f".{target.name}.partial"
    shutil.copyfile(source, partial)
    os.replace(partial, target)
    return target


def comfyui_command(
    temporary: Path, comfy_output: Path, extra_paths: Path
) -> list[str]:
    """ComfyUI's launch. Every directory it writes at startup (its user
    directory and database, its temp directory, its outputs) lies in the
    job's own temporary directory: the image's ComfyUI tree is read-only."""

    user = temporary / "user"
    user.mkdir(exist_ok=True)
    return [
        sys.executable,
        str(COMFYUI_ROOT / "main.py"),
        "--listen",
        "127.0.0.1",
        "--port",
        str(PORT),
        "--disable-auto-launch",
        "--disable-metadata",
        "--output-directory",
        str(comfy_output),
        "--input-directory",
        str(INPUT_ROOT),
        "--user-directory",
        str(user),
        "--temp-directory",
        str(temporary),
        "--extra-model-paths-config",
        str(extra_paths),
    ]


def run_job() -> int:
    args = parse_args()
    document = load_workflow(args.workflow, args.workflow_sha256)
    document["prompt"]["6"]["inputs"]["length"] = args.num_frames
    document["result"]["video"]["frames"] = args.num_frames
    manifest = load_input_manifest()
    inputs = input_names(document, manifest)
    replacements = workflow_replacements(document, inputs, args.seed, manifest)
    prompt = substitute(document["prompt"], replacements)

    with tempfile.TemporaryDirectory(prefix="vonk-comfyui-") as temp_dir:
        temporary = Path(temp_dir)
        model_root = Path("/outputs/cache/fasth3/ec1e3aa374a91c57b0b94a1623b7e657c0498cf2/1f7ecc75ded5df6da151d4d5eeee87b35161e3c2/models")
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
        command = comfyui_command(temporary, comfy_output, extra_paths)
        process = subprocess.Popen(command, stdout=sys.stderr, stderr=sys.stderr)
        try:
            wait_for_server(process)
            queued = request_json(
                "/prompt", body={"prompt": prompt, "client_id": str(uuid.uuid4())}
            )
            prompt_id = queued.get("prompt_id")
            if not prompt_id:
                raise RuntimeError(f"ComfyUI rejected workflow: {queued}")
            wait_for_prompt(
                prompt_id,
                process,
                int(os.environ.get("VONK_COMFYUI_TIMEOUT_SECONDS", "7200")),
            )
            output = copy_output(
                comfy_output,
                Path(args.output_dir),
                args.output_mime,
                document.get("result"),
            )
            print(
                json.dumps(
                    {"output": str(output), "mime": args.output_mime}, sort_keys=True
                )
            )
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    return 0


def main() -> int:
    cache = Path("/outputs/cache/fasth3/ec1e3aa374a91c57b0b94a1623b7e657c0498cf2/1f7ecc75ded5df6da151d4d5eeee87b35161e3c2")
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / ".job.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run_job()


if __name__ == "__main__":
    raise SystemExit(main())
