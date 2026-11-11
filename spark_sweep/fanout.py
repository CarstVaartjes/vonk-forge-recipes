"""Bounded concurrent fan-out for the independent read-only observations of one pass."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

#: Few enough that the Controller never sees a burst, enough that a pass costs its slowest read.
READ_WORKERS = 4


@dataclass(frozen=True)
class Outcome:
    """A read's value or the exception it raised; the consumer decides at its own call site."""

    value: Any = None
    error: BaseException | None = None

    def result(self) -> Any:
        if self.error is not None:
            raise self.error
        return self.value


def _capture(call: Callable[[], Any]) -> Outcome:
    try:
        return Outcome(value=call())
    except Exception as error:  # noqa: BLE001 - re-raised at the consumer, in its own order
        return Outcome(error=error)


def gather(
    calls: Mapping[str, Callable[[], Any]], workers: int = READ_WORKERS
) -> dict[str, Outcome]:
    """Run independent reads at once; one worker (or one read) runs them inline in order."""
    if workers <= 1 or len(calls) <= 1:
        return {name: _capture(call) for name, call in calls.items()}
    with ThreadPoolExecutor(max_workers=min(workers, len(calls))) as pool:
        futures = {name: pool.submit(_capture, call) for name, call in calls.items()}
    return {name: future.result() for name, future in futures.items()}
