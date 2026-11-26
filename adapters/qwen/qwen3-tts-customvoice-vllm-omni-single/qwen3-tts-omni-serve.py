#!/usr/bin/env python3
"""Serve Qwen3-TTS with vLLM-Omni inside the platform's argument contract.

The creator's launch is ``vllm-omni serve <model> --deploy-config qwen3_tts.yaml
--omni`` (vLLM-Omni ``examples/online_serving/text_to_speech/qwen3_tts/run_server.sh``).
The deploy config has two stages: the talker (stage 0, ``max_model_len`` 4096) and
the codec decoder (stage 1, ``max_model_len`` 65536, which must hold the whole
accumulated codec stream). A global ``--max-model-len`` would reach both stages, so
the recipe's context setting is a stage-0 override instead: ``--max-model-len N``
becomes ``--stage-overrides '{"0":{"max_model_len":N}}'``, and
``--gpu-memory-utilization F`` is scoped to both stages the same way (vLLM-Omni
resolves a per-stage override above the deploy config and any global flag). Every other
argument passes through unchanged.
"""

from __future__ import annotations

import json
import os
import sys

OMNI = "/usr/local/bin/vllm-omni"
TALKER_STAGE = "0"
STAGES = ("0", "1")


def _take(arguments: list[str], flag: str) -> tuple[list[str], str | None]:
    """Remove ``flag`` (space or ``=`` form) and return the rest plus its value."""
    result: list[str] = []
    value: str | None = None
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == flag:
            if index + 1 >= len(arguments):
                raise SystemExit(f"{flag} requires a value")
            value = arguments[index + 1]
            index += 2
            continue
        if argument.startswith(flag + "="):
            value = argument.partition("=")[2]
            index += 1
            continue
        result.append(argument)
        index += 1
    return result, value


def stage_scoped(arguments: list[str]) -> list[str]:
    """Return the arguments with stage-owned flags moved into ``--stage-overrides``."""
    if any(
        argument == "--stage-overrides" or argument.startswith("--stage-overrides=")
        for argument in arguments
    ):
        raise SystemExit("--stage-overrides is set by the launcher")
    result, context = _take(arguments, "--max-model-len")
    result, memory = _take(result, "--gpu-memory-utilization")
    overrides: dict[str, dict[str, int | float]] = {}
    if context is not None:
        if not context.isascii() or not context.isdigit() or int(context) < 1:
            raise SystemExit("--max-model-len must be a positive integer")
        overrides[TALKER_STAGE] = {"max_model_len": int(context)}
    if memory is not None:
        try:
            fraction = float(memory)
        except ValueError:
            raise SystemExit("--gpu-memory-utilization must be a number") from None
        for stage in STAGES:
            overrides.setdefault(stage, {})["gpu_memory_utilization"] = fraction
    if overrides:
        result.extend(("--stage-overrides", json.dumps(overrides, separators=(",", ":"))))
    return result


def main() -> None:
    arguments = sys.argv[1:]
    if not arguments or arguments[0] != "serve":
        raise SystemExit("usage: vllm serve <model> [options]")
    os.execv(OMNI, [OMNI, *stage_scoped(arguments)])


if __name__ == "__main__":
    main()
