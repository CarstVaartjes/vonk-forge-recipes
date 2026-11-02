"""Memory envelopes describe demand; exceeding idle capacity is a warning.

The Controller attempts the recipe even when the declared envelope exceeds the
idle Spark estimate; this producer check must not turn an informational
warning into an admission ban.
"""

from __future__ import annotations

import json
import os
import unittest
import warnings
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
    def test_exceptions_name_existing_recipes(self) -> None:
        known = peaks()
        self.assertEqual(
            [slug for slug in EXCEPTIONS if slug not in known],
            [],
            "memory-envelope-exceptions.json lists a recipe that is gone; remove it",
        )

    def test_an_exception_that_now_fits_only_warns(self) -> None:
        # A measured peak below the estimate is the direction we want: report it,
        # never fail on it.
        known = peaks()
        fits = [
            slug
            for slug in EXCEPTIONS
            if slug in known and known[slug] <= ADMISSIBLE_PEAK_BYTES
        ]
        if fits:
            warnings.warn(
                f"memory-envelope-exceptions.json lists recipes that now fit: {fits}",
                stacklevel=1,
            )


if __name__ == "__main__":
    unittest.main()
