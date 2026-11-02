"""Memory envelopes describe demand; exceeding idle capacity is a warning.

The Controller attempts the recipe even when the declared envelope exceeds the
idle Spark estimate; this producer check must not turn an informational
warning into an admission ban.
"""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

from qualification.memory_need import SPARK_MEMORY_BYTES

ROOT = Path(__file__).resolve().parents[1]
PHYSICAL_MEMORY_BYTES = SPARK_MEMORY_BYTES
ADMISSIBLE_PEAK_BYTES = int(
    os.environ.get("VONK_SPARK_ADMISSIBLE_PEAK_BYTES", "124000000000")
)
EXCEPTIONS = json.loads((ROOT / "memory-envelope-exceptions.json").read_text())


def peaks() -> dict[str, int]:
    """Slug to the largest declared peak of its roles."""
    result: dict[str, int] = {}
    for path in sorted((ROOT / "recipes").glob("*.json")):
        document = json.loads(path.read_text())
        roles = document.get("topology", {}).get("roles", [])
        values = [role["resources"]["memory"]["peak_bytes"] for role in roles]
        if values:
            result[path.stem] = max(values)
    return result


class MemoryEnvelopeTests(unittest.TestCase):
    def test_exceptions_are_current(self) -> None:
        known = peaks()
        stale = [
            slug
            for slug in EXCEPTIONS
            if slug not in known or known[slug] <= ADMISSIBLE_PEAK_BYTES
        ]
        self.assertEqual(
            stale,
            [],
            "memory-envelope-exceptions.json lists a recipe that is gone or now "
            "fits the Spark; remove it",
        )


if __name__ == "__main__":
    unittest.main()
