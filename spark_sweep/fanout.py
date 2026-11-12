"""Concurrent fan-out for the independent read-only observations of one pass."""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Protocol

# The most reads of one fan-out in flight at once; the rest wait for a free slot.
READ_LIMIT = 8


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


class DaemonExecutor:
    """Smoke tests on daemon threads: a request that hangs never keeps an interrupted sweep alive."""

    def submit(self, fn: Callable[..., Any], *args: Any) -> Future[Any]:
        future: Future[Any] = Future()

        def work() -> None:
            if not future.set_running_or_notify_cancel():
                return
            try:
                future.set_result(fn(*args))
            except BaseException as error:  # noqa: BLE001 - delivered through the future
                future.set_exception(error)

        threading.Thread(target=work, daemon=True, name="smoke").start()
        return future

    def shutdown(self, **_: Any) -> None:
        return None


def gather(
    calls: Mapping[str, Callable[[], Any]],
    executor: Executor,
    limit: int = READ_LIMIT,
) -> dict[str, Outcome]:
    """Run independent reads at once (one thread each, on the sweep's daemon executor, so an
    interrupted sweep never waits on an in-flight read), at most ``limit`` working together."""
    slots = threading.BoundedSemaphore(limit)

    def bounded(call: Callable[[], Any]) -> Outcome:
        with slots:
            return _capture(call)

    futures = {name: executor.submit(bounded, call) for name, call in calls.items()}
    return {name: future.result() for name, future in futures.items()}
