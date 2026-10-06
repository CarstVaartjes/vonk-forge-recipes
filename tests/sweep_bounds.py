"""Bound fake scenarios so recovery regressions cannot consume the local test lane forever."""

from __future__ import annotations

import pytest
from sweep_fakes import FakeClock


@pytest.fixture
def bounded_clock(monkeypatch):
    original = FakeClock.sleep

    def sleep(clock, seconds):
        assert clock.sleeps < 1000, (
            "fake scenario exceeded its bounded observation steps"
        )
        original(clock, seconds)

    monkeypatch.setattr(FakeClock, "sleep", sleep)
