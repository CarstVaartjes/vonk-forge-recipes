#!/usr/bin/env python3
"""Serve Qwen3-TTS with vLLM-Omni inside the platform's argument contract.

The creator's launch is ``vllm-omni serve <model> --deploy-config qwen3_tts.yaml
--omni`` (vLLM-Omni ``examples/online_serving/text_to_speech/qwen3_tts/run_server.sh``).
The deploy config has two stages: the talker (stage 0, ``max_model_len`` 4096) and
the codec decoder (stage 1, ``max_model_len`` 65536, which must hold the whole
accumulated codec stream). A global ``--max-model-len`` would reach both stages, so
the recipe's context setting is a stage-0 override instead: ``--max-model-len N``
becomes ``--stage-overrides '{"0":{"max_model_len":N}}'`` (vLLM-Omni resolves a
per-stage override above the deploy config and any global flag). Every other
argument passes through unchanged.
"""

from __future__ import annotations

import json
import os
import sys

OMNI = "/usr/local/bin/vllm-omni"
TALKER_STAGE = "0"


def talker_context(arguments: list[str]) -> list[str]:
    """Return the arguments with ``--max-model-len`` scoped to the talker stage."""
    if any(
        argument == "--stage-overrides" or argument.startswith("--stage-overrides=")
        for argument in arguments
    ):
        raise SystemExit("--stage-overrides is set by the launcher from --max-model-len")
    result: list[str] = []
    context: str | None = None
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == "--max-model-len":
            if index + 1 >= len(arguments):
                raise SystemExit("--max-model-len requires a value")
            context = arguments[index + 1]
            index += 2
            continue
        if argument.startswith("--max-model-len="):
            context = argument.partition("=")[2]
            index += 1
            continue
        result.append(argument)
        index += 1
    if context is None:
        return result
    if not context.isascii() or not context.isdigit() or int(context) < 1:
        raise SystemExit("--max-model-len must be a positive integer")
    overrides = {TALKER_STAGE: {"max_model_len": int(context)}}
    result.extend(("--stage-overrides", json.dumps(overrides, separators=(",", ":"))))
    return result


def main() -> None:
    arguments = sys.argv[1:]
    if not arguments or arguments[0] != "serve":
        raise SystemExit("usage: vllm serve <model> [options]")
    os.execv(OMNI, [OMNI, *talker_context(arguments)])


if __name__ == "__main__":
    main()
