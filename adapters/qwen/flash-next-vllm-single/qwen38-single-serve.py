#!/usr/bin/env python3
"""Serve Qwen3.8 Flash Next NVFP4 on one Spark with the PLE table offloaded.

Reproduces what MiaAI-Lab's single-Spark start.sh derives at launch, inside the
platform's model, cache and argument contract:

* builds the memory-mapped packed PLE table (about 27 GiB, one-time, from the
  declared read-only model files) into the platform cache and reuses it;
* when the requested context exceeds the native 262144 tokens, adds the YaRN
  rope override with upstream's factor rule and its validated 524288 ceiling;
* captures a decode CUDA graph for every verify width (1 + K) * s that the
  scheduler can build (upstream CUDAGRAPH_CAPTURE_SIZES=auto), unless the
  compilation config already lists capture sizes.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

SOURCE = Path("/models")
CACHE_ROOT = Path("/outputs/cache/qwen38-single")
BUILDER = Path("/opt/vonk/build/build_ple_packed_table.py")
NATIVE_CONTEXT = 262144
YARN_CEILING = 524288


def value(arguments: list[str], option: str) -> str | None:
    if option not in arguments:
        return None
    index = arguments.index(option)
    if index + 1 >= len(arguments):
        raise SystemExit(f"{option} requires a value")
    return arguments[index + 1]


def set_option(arguments: list[str], option: str, option_value: str) -> None:
    if option in arguments:
        arguments[arguments.index(option) + 1] = option_value
    else:
        arguments.extend((option, option_value))


def fingerprint() -> str:
    digest = hashlib.sha256()
    for name in ("model.safetensors.index.json", "config.json"):
        digest.update(name.encode())
        digest.update((SOURCE / name).read_bytes())
    return digest.hexdigest()[:24]


def prepare_ple_table() -> Path:
    """Return the directory of the packed PLE table, building it once."""
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    table = CACHE_ROOT / f"ple-{fingerprint()}"
    marker = table / ".complete"
    with (CACHE_ROOT / ".ple.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not marker.is_file():
            print(f"building the packed PLE table in {table}", flush=True)
            subprocess.run(
                [sys.executable, "-u", str(BUILDER), str(SOURCE), str(table)],
                check=True,
            )
            if not any(table.glob("*.packed_u8")):
                raise SystemExit("the PLE table build produced no packed tables")
            marker.write_text("complete\n", encoding="utf-8")
        fcntl.flock(lock, fcntl.LOCK_UN)
    return table


def yarn_override(arguments: list[str]) -> None:
    max_len = value(arguments, "--max-model-len")
    if max_len is None or not max_len.isascii() or not max_len.isdigit():
        raise SystemExit("--max-model-len must be a positive integer")
    length = int(max_len)
    if length <= NATIVE_CONTEXT or "--hf-overrides" in arguments:
        return
    if length > YARN_CEILING:
        raise SystemExit(
            f"--max-model-len {length} exceeds the {YARN_CEILING} tokens upstream "
            "validated on one Spark (the KV cache would not fit)"
        )
    factor = round(math.ceil(length / NATIVE_CONTEXT * 10000) / 10000, 4)
    overrides = {
        "text_config": {
            "rope_parameters": {
                "rope_type": "yarn",
                "factor": factor,
                "original_max_position_embeddings": NATIVE_CONTEXT,
            }
        }
    }
    arguments.extend(("--hf-overrides", json.dumps(overrides, separators=(",", ":"))))


def capture_sizes(arguments: list[str]) -> None:
    raw = value(arguments, "--compilation-config")
    config = json.loads(raw) if raw else {}
    if not isinstance(config, dict):
        raise SystemExit("--compilation-config must be a JSON object")
    if "cudagraph_capture_sizes" in config:
        return
    seqs = value(arguments, "--max-num-seqs")
    if seqs is None or not seqs.isdigit() or int(seqs) < 1:
        raise SystemExit("--max-num-seqs must be a positive integer")
    speculative = value(arguments, "--speculative-config")
    depth = 0
    if speculative:
        depth = int(json.loads(speculative).get("num_speculative_tokens", 0))
    config["cudagraph_capture_sizes"] = sorted(
        {(1 + depth) * s for s in range(1, int(seqs) + 1)}
    )
    set_option(
        arguments, "--compilation-config", json.dumps(config, separators=(",", ":"))
    )


def main() -> None:
    arguments = sys.argv[1:]
    table = prepare_ple_table()
    yarn_override(arguments)
    capture_sizes(arguments)
    os.environ["VLLM_PLE_PACKED_TABLE_DIR"] = str(table)
    vllm = next(
        (
            candidate
            for candidate in ("/usr/local/bin/vllm", "/opt/vllm/.venv/bin/vllm")
            if Path(candidate).is_file()
        ),
        None,
    )
    if vllm is None:
        raise SystemExit("the pinned Qwen vLLM executable is missing")
    os.execv(vllm, (vllm, "serve", str(SOURCE), *arguments))


if __name__ == "__main__":
    main()
