"""Offline HunyuanOCR 1.5 document-image job with official DFlash inference."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


INPUTS = Path("/inputs")
MODEL = Path("/models")
SOURCE = Path("/opt/hunyuanocr")
PORT = 18080
SERVED_NAME = "tencent/HunyuanOCR"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
OUTPUT_SUFFIXES = {
    "doc_parse": ".md",
    "structured_parse": ".txt",
    "spotting_json": ".json",
    "spotting_hunyuan": ".txt",
    "layout": ".txt",
    "layout_parse": ".md",
    "chart_parse": ".md",
    "formula": ".tex",
    "table": ".html",
    "doc_trans_en2zh": ".md",
    "trans_other2en": ".md",
    "trans_other2zh": ".md",
}


def safe_input(name: object) -> Path:
    if not isinstance(name, str) or Path(name).name != name:
        raise SystemExit("job input paths must be plain filenames")
    candidate = INPUTS / name
    if (
        not candidate.is_file()
        or candidate.is_symlink()
        or candidate.suffix.lower() not in IMAGE_SUFFIXES
    ):
        raise SystemExit(f"unsupported or missing document image: {name}")
    if candidate.stat().st_size > 32 * 1024 * 1024:
        raise SystemExit(f"document image exceeds 32 MiB: {name}")
    return candidate


def manifest_inputs() -> tuple[list[Path], Path | None]:
    manifest_path = INPUTS / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SystemExit("missing or invalid signed input manifest") from error
    files = manifest.get("files") if isinstance(manifest, dict) else None
    if not isinstance(files, list):
        raise SystemExit("signed input manifest files are invalid")
    documents: list[Path] = []
    configs: list[Path] = []
    for item in files:
        if not isinstance(item, dict):
            raise SystemExit("signed input manifest entry is invalid")
        name = item.get("name")
        slot = item.get("slot")
        if not isinstance(name, str) or Path(name).name != name:
            raise SystemExit("signed input manifest contains an unsafe name")
        candidate = INPUTS / name
        if candidate.is_symlink() or not candidate.is_file():
            raise SystemExit("signed input file is unavailable")
        if slot == "document":
            documents.append(safe_input(name))
        elif slot == "config" and candidate.suffix.lower() == ".json":
            configs.append(candidate)
        else:
            raise SystemExit("signed input manifest contains an unexpected slot")
    if not 1 <= len(documents) <= 31 or len(configs) > 1:
        raise SystemExit("OCR requires 1..31 documents and at most one config")
    return sorted(documents), configs[0] if configs else None


def read_job() -> tuple[list[Path], str, int]:
    manifest_images, config_path = manifest_inputs()
    if config_path is not None:
        try:
            document = json.loads(config_path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise SystemExit("OCR config must be UTF-8 JSON") from error
        if not isinstance(document, dict) or set(document) - {
            "images",
            "task_type",
            "max_tokens",
        }:
            raise SystemExit("OCR config has unsupported fields")
        names = document.get("images")
        if (
            not isinstance(names, list)
            or not 1 <= len(names) <= 31
            or len(set(map(str, names))) != len(names)
        ):
            raise SystemExit("OCR config images must contain 1..31 unique filenames")
        images = [safe_input(name) for name in names]
        if {item.name for item in images} != {item.name for item in manifest_images}:
            raise SystemExit("OCR config images must match the document slot")
        task_type = document.get("task_type", "doc_parse")
        max_tokens = document.get("max_tokens", 32768)
    else:
        images = manifest_images
        task_type = "doc_parse"
        max_tokens = 32768
    if task_type not in OUTPUT_SUFFIXES:
        raise SystemExit(f"unsupported official task_type: {task_type}")
    if type(max_tokens) is not int or not 1 <= max_tokens <= 32768:
        raise SystemExit("max_tokens must be an integer between 1 and 32768")
    return images, task_type, max_tokens


def launch_server(log_path: Path) -> tuple[subprocess.Popen[bytes], object]:
    required = [
        MODEL / "config.json",
        MODEL / "model.safetensors",
        MODEL / "dflash" / "config.json",
        MODEL / "dflash" / "dflash.py",
        MODEL / "dflash" / "model.safetensors",
    ]
    missing = [str(item.relative_to(MODEL)) for item in required if not item.is_file()]
    if missing:
        raise SystemExit("incomplete HunyuanOCR/DFlash snapshot: " + ", ".join(missing))
    log = log_path.open("wb")
    command = [
        "vllm",
        "serve",
        str(MODEL),
        "--served-model-name",
        SERVED_NAME,
        "--tensor-parallel-size",
        "1",
        "--no-enable-prefix-caching",
        "--mm-processor-cache-gb",
        "0",
        "--allowed-local-media-path",
        str(INPUTS),
        "--limit-mm-per-prompt",
        '{"image":4,"video":0}',
        "--trust-remote-code",
        "--host",
        "127.0.0.1",
        "--port",
        str(PORT),
        "--gpu-memory-utilization",
        "0.85",
        "--max-model-len",
        "131072",
        "--max-num-batched-tokens",
        "131072",
        "--speculative-config",
        json.dumps(
            {
                "method": "dflash",
                "model": str(MODEL / "dflash"),
                "num_speculative_tokens": 15,
            },
            separators=(",", ":"),
        ),
    ]
    environment = dict(os.environ)
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "CUDA_VISIBLE_DEVICES": "0",
        }
    )
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=environment,
    )
    return process, log


def wait_ready(process: subprocess.Popen[bytes], timeout: float, log_path: Path) -> None:
    deadline = time.monotonic() + timeout
    endpoint = f"http://127.0.0.1:{PORT}/v1/models"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            detail = log_path.read_text(encoding="utf-8", errors="replace")[-8000:]
            raise SystemExit(f"HunyuanOCR vLLM exited during startup:\n{detail}")
        try:
            with urllib.request.urlopen(endpoint, timeout=2) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(1)
    raise SystemExit("HunyuanOCR vLLM readiness timed out")


def infer(image: Path, task_type: str, max_tokens: int, timeout: float) -> tuple[str, bool]:
    sys.path.insert(0, str(SOURCE / "inference"))
    from utils.output_utils import (  # type: ignore[import-not-found]
        clean_repeated_substrings,
        encode_image_as_data_url,
        has_tail_repetition,
        normalize_doc_parse_markdown,
    )
    from utils.tasks import get_prompt  # type: ignore[import-not-found]

    body = {
        "model": SERVED_NAME,
        "messages": [
            {"role": "system", "content": ""},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": encode_image_as_data_url(str(image))},
                    },
                    {"type": "text", "text": get_prompt(task_type)},
                ],
            },
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": True,
        "top_k": -1,
        "repetition_penalty": 1.08,
        "skip_special_tokens": True,
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST",
    )
    parts: list[str] = []
    length = 0
    next_check = 4000
    early_stopped = False
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            choices = event.get("choices", [])
            piece = choices[0].get("delta", {}).get("content") if choices else None
            if not isinstance(piece, str) or not piece:
                continue
            parts.append(piece)
            length += len(piece)
            if length >= next_check:
                next_check = length + 1000
                if has_tail_repetition("".join(parts)[-8000:], min_repeats=8):
                    early_stopped = True
                    break
    text = clean_repeated_substrings("".join(parts))
    if task_type == "doc_parse":
        text, _ = normalize_doc_parse_markdown(text)
    return text, early_stopped


def stop_server(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def write_bundle(
    output_dir: Path,
    results: list[tuple[Path, str, bool]],
    task_type: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "output.zip"
    manifest: dict[str, object] = {
        "schema_version": 1,
        "model": SERVED_NAME,
        "model_revision": "449e7d471a8a1ef5bd5d652e4881183d7252cbc7",
        "runtime_source_revision": "c55965d3da1e6f41987abec8068f2e70851318bc",
        "inference": "vllm-dflash",
        "task_type": task_type,
        "sampling": {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "repetition_penalty": 1.08,
        },
        "documents": [],
    }
    used: set[str] = set()
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index, (source, text, early_stopped) in enumerate(results, start=1):
            stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in source.stem)[:80]
            name = f"documents/{index:03d}-{stem}{OUTPUT_SUFFIXES[task_type]}"
            if name in used:
                name = f"documents/{index:03d}-document{OUTPUT_SUFFIXES[task_type]}"
            used.add(name)
            archive.writestr(name, text.encode("utf-8"))
            manifest["documents"].append(
                {
                    "input": source.name,
                    "output": name,
                    "early_stopped_tail_repetition": early_stopped,
                    "characters": len(text),
                }
            )
        archive.writestr(
            "manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
        )
    with target.open("rb") as created:
        signature = created.read(4)
    if signature != b"PK\x03\x04":
        raise SystemExit("failed to create OCR artifact bundle")
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entrypoint", required=True)
    parser.add_argument("--output-mime", required=True)
    parser.add_argument("--timeout-seconds", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.entrypoint != "/opt/vonk/source/run.py":
        raise SystemExit("unexpected signed adapter entrypoint")
    if args.output_mime != "application/zip":
        raise SystemExit("HunyuanOCR emits an application/zip bundle")
    if args.seed != 0:
        raise SystemExit("HunyuanOCR uses deterministic greedy decoding; seed must be zero")

    images, task_type, max_tokens = read_job()
    log_path = Path("/tmp/hunyuanocr-vllm.log")
    deadline = time.monotonic() + args.timeout_seconds
    process, log = launch_server(log_path)
    try:
        wait_ready(
            process,
            min(900, max(1, deadline - time.monotonic())),
            log_path,
        )
        results: list[tuple[Path, str, bool]] = []
        for image in images:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise SystemExit("HunyuanOCR job timed out")
            text, early_stopped = infer(image, task_type, max_tokens, remaining)
            results.append((image, text, early_stopped))
        write_bundle(args.output_dir, results, task_type)
    finally:
        stop_server(process)
        log.close()


if __name__ == "__main__":
    main()
