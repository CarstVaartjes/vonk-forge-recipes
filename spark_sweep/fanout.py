"""Concurrent fan-out for the independent read-only observations of one pass."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Protocol


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


class Executor(Protocol):
    def submit(self, fn: Callable[..., Any], *args: Any) -> Future[Any]: ...


def gather(
    calls: Mapping[str, Callable[[], Any]], executor: Executor
) -> dict[str, Outcome]:
    """Run the pass's few independent reads at once (one thread each, on the sweep's daemon
    executor, so an interrupted sweep never waits on an in-flight read)."""
    futures = {name: executor.submit(_capture, call) for name, call in calls.items()}
    return {name: future.result() for name, future in futures.items()}
