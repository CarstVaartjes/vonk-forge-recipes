"""Bounded, offline MOSS-VL-Realtime session replay adapter.

The input manifest is an ordered event API rather than a conventional batch
prompt. It drives OpenMOSS's native ``create_realtime_session`` implementation
while generation remains active, preserving timestamped frames, interrupting
prompts, proactive-silence tokens, and later corrective output chunks.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any, NoReturn

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from input_contract import resolve_moss_inputs

INPUTS = Path("/inputs")
MODEL = Path("/models")
ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
def fail(message: str) -> NoReturn:
    raise SystemExit(message)


def bounded_number(
    document: dict[str, Any], name: str, default: float, minimum: float, maximum: float
) -> float:
    value = document.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum or result > maximum:
        fail(f"{name} must be between {minimum} and {maximum}")
    return result


def bounded_integer(
    document: dict[str, Any], name: str, default: int, minimum: int, maximum: int
) -> int:
    value = document.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum or value > maximum:
        fail(f"{name} must be an integer between {minimum} and {maximum}")
    return value


def bounded_text(
    document: dict[str, Any], name: str, default: str = "", *, required: bool = False
) -> str:
    value = document.get(name, default)
    if not isinstance(value, str) or len(value) > 4096 or (required and not value.strip()):
        fail(f"{name} must be {'a non-empty ' if required else 'a '}string of at most 4096 characters")
    return value


def safe_frame(name: Any, authenticated_frames: frozenset[str]) -> Path:
    if not isinstance(name, str) or Path(name).name != name or len(name) > 132:
        fail("frame path must be a plain filename")
    if name not in authenticated_frames:
        fail(f"frame is not present in the authenticated frames slot: {name}")
    frame = INPUTS / name
    if (
        frame.suffix.lower() not in ALLOWED_IMAGE_SUFFIXES
        or not frame.is_file()
        or frame.is_symlink()
    ):
        fail(f"invalid frame input: {name}")
    if frame.stat().st_size > 8 * 1024 * 1024:
        fail(f"frame exceeds 8 MiB: {name}")
    try:
        with Image.open(frame) as image:
            image.verify()
        with Image.open(frame) as image:
            if image.width * image.height > 1_048_576:
                fail(f"frame exceeds the one-megapixel bound: {name}")
    except Exception as exc:
        fail(f"frame is not a valid supported image: {name}: {exc}")
    return frame


def load_manifest() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        manifest, authenticated_frames = resolve_moss_inputs(INPUTS)
    except ValueError as exc:
        fail(str(exc))
    if not manifest.is_file() or manifest.is_symlink() or manifest.stat().st_size > 1024 * 1024:
        fail("the session slot must contain a regular manifest no larger than 1 MiB")
    try:
        document = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        fail(f"invalid session.json: {exc}")
    if not isinstance(document, dict):
        fail("session.json must contain one object")
    allowed = {
        "schema_version",
        "system_prompt",
        "initial_prompt",
        "frame_queue_size",
        "max_tokens_per_second",
        "max_new_tokens",
        "playback_speed",
        "settle_seconds",
        "do_sample",
        "temperature",
        "top_k",
        "top_p",
        "repetition_penalty",
        "events",
    }
    if document.get("schema_version") != 1 or set(document) - allowed:
        fail("session.json must use schema_version 1 and contain only documented fields")
    events = document.get("events")
    if not isinstance(events, list) or not 1 <= len(events) <= 256:
        fail("events must contain between 1 and 256 entries")

    previous_at = -1.0
    previous_timestamp = -1.0
    first_timestamp: float | None = None
    frame_count = 0
    normalized: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            fail(f"event {index} must be an object")
        event_type = event.get("type")
        if event_type == "frame":
            frame_count += 1
            if frame_count > 128:
                fail("a session may contain at most 128 frames")
            if set(event) - {"type", "path", "timestamp", "at_seconds", "prompt"}:
                fail(f"frame event {index} contains unsupported fields")
            frame = safe_frame(event.get("path"), authenticated_frames)
            timestamp = bounded_number(event, "timestamp", -1, 0, 86400)
            prompt = bounded_text(event, "prompt", "", required="prompt" in event)
            if timestamp < previous_timestamp:
                fail("frame timestamps must be non-decreasing")
            if first_timestamp is None:
                first_timestamp = timestamp
            elif timestamp - first_timestamp > 600:
                fail("a bounded session may span at most 600 media seconds")
            previous_timestamp = timestamp
            item = {"type": "frame", "path": frame, "timestamp": timestamp}
            if prompt:
                item["prompt"] = prompt
        elif event_type == "prompt":
            if set(event) != {"type", "text", "at_seconds"}:
                fail(f"prompt event {index} must contain only type, text, and at_seconds")
            item = {"type": "prompt", "text": bounded_text(event, "text", required=True)}
        else:
            fail(f"event {index} type must be frame or prompt")
        at_seconds = bounded_number(event, "at_seconds", -1, 0, 86400)
        if at_seconds < previous_at:
            fail("event at_seconds values must be non-decreasing")
        previous_at = at_seconds
        item["at_seconds"] = at_seconds
        normalized.append(item)
    if frame_count == 0:
        fail("a realtime video session requires at least one frame event")
    return document, normalized


def output_kind(text: str) -> str:
    if "<|silence|>" in text:
        return "silence"
    if "<|round_start|>" in text:
        return "round-start"
    if "<|round_end|>" in text:
        return "round-end"
    if "<|response|>" in text:
        return "response-start"
    return "text"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entrypoint", required=True)
    parser.add_argument("--output-mime", required=True)
    parser.add_argument("--timeout-seconds", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.entrypoint != "/opt/vonk/source/run.py":
        fail("unexpected signed adapter entrypoint")
    if args.output_mime != "video/mp4":
        fail("MOSS-VL realtime sessions emit a video replay with a JSONL transcript sidecar")
    if not 1 <= args.timeout_seconds <= 3600 or not 0 <= args.seed < 2**63:
        fail("invalid bounded harness arguments")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        fail("offline model loading must be enforced")

    document, events = load_manifest()
    system_prompt = bounded_text(document, "system_prompt", "") or None
    initial_prompt = bounded_text(document, "initial_prompt", "")
    prompt_characters = len(system_prompt or "") + len(initial_prompt)
    prompt_characters += sum(
        len(event.get("prompt", event.get("text", ""))) for event in events
    )
    if prompt_characters > 32768:
        fail("all session prompts together may contain at most 32768 characters")
    frame_queue_size = bounded_integer(document, "frame_queue_size", 32, 1, 256)
    token_rate = bounded_integer(document, "max_tokens_per_second", 12, 1, 64)
    max_new_tokens = bounded_integer(document, "max_new_tokens", 512, 1, 4096)
    playback_speed = bounded_number(document, "playback_speed", 1.0, 0.25, 4.0)
    settle_seconds = bounded_number(document, "settle_seconds", 10.0, 0, 60)
    session_duration = events[-1]["at_seconds"] / playback_speed + settle_seconds
    if session_duration > 600:
        fail("bounded session playback plus settling may not exceed 600 seconds")
    do_sample = document.get("do_sample", False)
    if not isinstance(do_sample, bool):
        fail("do_sample must be boolean")

    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "repetition_penalty": bounded_number(document, "repetition_penalty", 1.0, 0.5, 2.0),
    }
    if do_sample:
        generate_kwargs.update(
            temperature=bounded_number(document, "temperature", 0.7, 0.000001, 2.0),
            top_k=bounded_integer(document, "top_k", 50, 1, 1000),
            top_p=bounded_number(document, "top_p", 0.9, 0.000001, 1.0),
        )

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    processor = AutoProcessor.from_pretrained(
        str(MODEL),
        trust_remote_code=True,
        local_files_only=True,
        frame_extract_num_threads=1,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL),
        trust_remote_code=True,
        local_files_only=True,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    model.eval()
    if not callable(getattr(model, "create_realtime_session", None)):
        fail("pinned checkpoint does not expose create_realtime_session")

    transcript: list[dict[str, Any]] = []
    start = time.monotonic()

    def record(record_type: str, **values: Any) -> None:
        transcript.append(
            {
                "sequence": len(transcript),
                "elapsed_seconds": round(time.monotonic() - start, 6),
                "type": record_type,
                **values,
            }
        )

    session = model.create_realtime_session(
        processor,
        initial_prompt=initial_prompt,
        system_prompt=system_prompt,
        frame_queue_size=frame_queue_size,
        max_tokens_per_turn=token_rate,
        **generate_kwargs,
    )

    def drain() -> None:
        while True:
            chunk = session.poll_output(timeout=0.0)
            if chunk is None:
                return
            record("output", kind=output_kind(chunk), text=chunk)

    try:
        session.start()
        record("session-start", model_revision="06b067617677661194cf837970fe3a10f1a0e56d")
        for event_index, event in enumerate(events):
            due = event["at_seconds"] / playback_speed
            while True:
                remaining = due - (time.monotonic() - start)
                if remaining <= 0:
                    break
                chunk = session.poll_output(timeout=min(0.05, remaining))
                if chunk is not None:
                    record("output", kind=output_kind(chunk), text=chunk)

            if event["type"] == "frame":
                with Image.open(event["path"]) as opened:
                    image = opened.convert("RGB")
                if "prompt" in event:
                    dropped = session.push_prompt_frame(
                        event["prompt"], image, timestamp=event["timestamp"]
                    )
                    record(
                        "frame-prompt-ack",
                        event_index=event_index,
                        timestamp=event["timestamp"],
                        dropped_oldest=bool(dropped),
                    )
                else:
                    dropped = session.push_frame(image, timestamp=event["timestamp"])
                    record(
                        "frame-ack",
                        event_index=event_index,
                        timestamp=event["timestamp"],
                        dropped_oldest=bool(dropped),
                    )
            else:
                session.push_prompt(event["text"])
                record("prompt-ack", event_index=event_index)
            drain()

        settle_deadline = time.monotonic() + settle_seconds
        while True:
            remaining = settle_deadline - time.monotonic()
            if remaining <= 0:
                break
            chunk = session.poll_output(timeout=min(0.1, remaining))
            if chunk is not None:
                record("output", kind=output_kind(chunk), text=chunk)
        record("session-stop")
    finally:
        session.close(timeout=30.0)
        drain()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    transcript_output = args.output_dir / "transcript.jsonl"
    with transcript_output.open("w", encoding="utf-8") as stream:
        for item in transcript:
            stream.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
    if not transcript_output.stat().st_size:
        fail("MOSS-VL session produced an empty transcript")

    frames = [event for event in events if event["type"] == "frame"]
    concat = args.output_dir / "frames.ffconcat"
    with concat.open("w", encoding="utf-8") as stream:
        stream.write("ffconcat version 1.0\n")
        for index, event in enumerate(frames):
            stream.write(f"file '{event['path']}'\n")
            if index + 1 < len(frames):
                duration = max(0.04, frames[index + 1]["timestamp"] - event["timestamp"])
            else:
                duration = 1.0
            stream.write(f"duration {duration:.6f}\n")
        stream.write(f"file '{frames[-1]['path']}'\n")
    video_output = args.output_dir / "output.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat),
            "-vf",
            "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2:black,setsar=1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-y",
            str(video_output),
        ],
        check=True,
        timeout=min(600, args.timeout_seconds),
    )
    concat.unlink()
    with video_output.open("rb") as stream:
        header = stream.read(8)
    if len(header) != 8 or header[4:] != b"ftyp":
        fail("failed to produce a valid MP4 session replay")


if __name__ == "__main__":
    main()
