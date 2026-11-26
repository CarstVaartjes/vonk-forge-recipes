#!/usr/bin/env python3
"""Serve Fish Audio Drama 3 with vLLM-Omni inside the platform's argument contract.

The creator's launch is ``vllm serve fishaudio/drama-3 --omni`` (vLLM-Omni
``recipes/fishaudio/Drama-3.md``), which loads ``vllm_omni/deploy/fish_omni.yaml``.
That deploy config has two stages on one GPU: the backbone and codebook tower
(stage 0, ``max_model_len`` 8192, ``gpu_memory_utilization`` 0.78) and the DAC
codec (stage 1, ``max_model_len`` 65536, ``gpu_memory_utilization`` 0.1). A global
flag would give both stages the same value, so the recipe's flags become per-stage
overrides (vLLM-Omni resolves ``--stage-overrides`` above the deploy config):

* ``--max-model-len N`` and ``--gpu-memory-utilization F`` go to stage 0;
* ``--codec-gpu-memory-utilization F`` goes to stage 1.

Every other argument passes through unchanged.
"""

from __future__ import annotations

import json
import os
import sys

OMNI = "/usr/local/bin/vllm-omni"
BACKBONE_STAGE = "0"
CODEC_STAGE = "1"


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


def _fraction(flag: str, value: str) -> float:
    try:
        return float(value)
    except ValueError:
        raise SystemExit(f"{flag} must be a number") from None


def stage_scoped(arguments: list[str]) -> list[str]:
    """Return the arguments with stage-owned flags moved into ``--stage-overrides``."""
    if any(
        argument == "--stage-overrides" or argument.startswith("--stage-overrides=")
        for argument in arguments
    ):
        raise SystemExit("--stage-overrides is set by the launcher")
    result, context = _take(arguments, "--max-model-len")
    result, backbone = _take(result, "--gpu-memory-utilization")
    result, codec = _take(result, "--codec-gpu-memory-utilization")
    overrides: dict[str, dict[str, int | float]] = {}
    if context is not None:
        overrides.setdefault(BACKBONE_STAGE, {})["max_model_len"] = int(context)
    if backbone is not None:
        overrides.setdefault(BACKBONE_STAGE, {})["gpu_memory_utilization"] = _fraction(
            "--gpu-memory-utilization", backbone
        )
    if codec is not None:
        overrides.setdefault(CODEC_STAGE, {})["gpu_memory_utilization"] = _fraction(
            "--codec-gpu-memory-utilization", codec
        )
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
