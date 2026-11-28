"""Wait until a Ray cluster has registered every GPU the engine will place on.

Stdlib only. Every Ray launcher COPYs this file byte-identical to
``/opt/vonk/lib/ray_ready.py`` (a build context cannot share files; a parity
test keeps the copies equal). Rank 0 starts the Ray head, then calls
``wait_for_gpus`` before it execs vLLM: the engine's placement group needs
``tensor-parallel-size * pipeline-parallel-size`` GPUs, and a node can be alive
before its GPU has registered.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable


def _option(arguments: list[str], name: str) -> int:
    for index, argument in enumerate(arguments):
        if argument == name and index + 1 < len(arguments):
            return int(arguments[index + 1])
        if argument.startswith(f"{name}="):
            return int(argument.split("=", 1)[1])
    return 1


def required_gpus(arguments: list[str]) -> int:
    """The GPU count vLLM places: tensor-parallel size times pipeline-parallel size."""
    try:
        return _option(arguments, "--tensor-parallel-size") * _option(
            arguments, "--pipeline-parallel-size"
        )
    except ValueError:
        raise SystemExit("the parallel sizes must be integers") from None


def wait_for_gpus(
    probe: Callable[[], int],
    required: int,
    deadline_s: float = 900,
    alive: Callable[[], bool] | None = None,
) -> None:
    """Poll ``probe`` (the cluster's GPU total) every 2 s until it reaches ``required``.

    Exits non-zero when ``alive`` reports false (a daemon the launch depends on
    died) or when ``deadline_s`` passes, the backstop under the platform's own
    stall window.
    """
    deadline = time.monotonic() + deadline_s
    while probe() < required:
        if alive is not None and not alive():
            raise SystemExit("the cluster daemon exited")
        if time.monotonic() >= deadline:
            raise SystemExit(f"the Ray cluster did not register {required} GPUs")
        time.sleep(2)
    print(f"Ray cluster has {required} GPUs", file=sys.stderr, flush=True)
