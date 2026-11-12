"""How long the sweep waits before it looks again while the Controller keeps not answering.

A stuck sweep must not hammer a Controller that is down or restarting, and a fleet of sweeps
must not retry in step. The wait grows exponentially from the normal poll, is capped, and
carries jitter ("equal jitter": between half and all of the grown wait, never below the
poll). The first answer resets it, so a healthy sweep keeps its normal cadence.
"""

from __future__ import annotations

import random
from collections.abc import Callable


class RecheckBackoff:
    def __init__(
        self,
        base: float,
        cap: float,
        uniform: Callable[[float, float], float] = random.uniform,
    ) -> None:
        self.base = base
        self.cap = cap
        self.uniform = uniform
        self.failures = 0

    def next_wait(self) -> float:
        """Count one more failed look and return how long to wait before the next."""
        grown = min(self.cap, self.base * 2**self.failures)
        self.failures = min(self.failures + 1, 32)
        return max(self.base, grown * self.uniform(0.5, 1.0))

    def reset(self) -> None:
        self.failures = 0
