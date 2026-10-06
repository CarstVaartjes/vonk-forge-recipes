"""Memory envelopes describe demand; exceeding idle capacity is a warning.

The Controller attempts the recipe even when the declared envelope exceeds the
idle Spark estimate. The declaration must still be internally consistent; this
producer check must not turn an informational warning into an admission ban.
"""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PHYSICAL_MEMORY_BYTES = 130_663_231_488
ADMISSIBLE_PEAK_BYTES = int(
    os.environ.get("VONK_SPARK_ADMISSIBLE_PEAK_BYTES", "124000000000")
)
EXCEPTIONS = json.loads((ROOT / "memory-envelope-exceptions.json").read_text())


def envelopes() -> dict[str, tuple[int, int, int]]:
    """Slug to the largest (peak, peak, reserve) of its roles."""
    result: dict[str, tuple[int, int, int]] = {}
    for path in sorted((ROOT / "recipes").glob("*.json")):
        document = json.loads(path.read_text())
        roles = document.get("topology", {}).get("roles", [])
        values = [
            (
                role["resources"]["memory"]["peak_bytes"],
                role["resources"]["memory"]["peak_bytes"],
                role["resources"]["memory"]["reserve_bytes"],
            )
            for role in roles
        ]
        if values:
            result[path.stem] = max(values)
    return result


class MemoryEnvelopeTests(unittest.TestCase):
    def test_every_envelope_is_consistent(self) -> None:
        for slug, (_total, peak, reserve) in envelopes().items():
            with self.subTest(recipe=slug):
                self.assertGreater(peak, 0)
                self.assertGreaterEqual(reserve, 0)
                self.assertLessEqual(reserve, peak)

    def test_exceptions_are_current(self) -> None:
        known = envelopes()
        stale = [
            slug
            for slug in EXCEPTIONS
            if slug not in known or known[slug][0] <= ADMISSIBLE_PEAK_BYTES
        ]
        self.assertEqual(
            stale,
            [],
            "memory-envelope-exceptions.json lists a recipe that is gone or now "
            "fits the Spark; remove it",
        )


if __name__ == "__main__":
    unittest.main()
