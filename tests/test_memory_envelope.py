"""Every recipe's declared peak memory must be admissible on an idle Spark.

The Controller admits a role when its ``peak_bytes`` is at most the Spark's observed
available memory minus the platform floor (2 GB). ``reserve_bytes`` is informational
and is not added. An idle DGX Spark reports about 126.0 GB available (130.66 GB
physical minus the operating system), so the idle-admissible peak is 124 GB. A recipe
above it can never be admitted; above the physical memory the Controller refuses it
with ``resource.envelope_exceeds_capacity``. Override the limit with
``VONK_SPARK_ADMISSIBLE_PEAK_BYTES``.

Recipes above the limit are listed with a kit-sourced reason in
``memory-envelope-exceptions.json``; a listed recipe that now fits must be removed.
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
    def test_every_envelope_fits_the_spark(self) -> None:
        offenders = [
            f"{slug}: declared peak {peak} bytes exceeds the {ADMISSIBLE_PEAK_BYTES}-byte "
            f"idle-admissible peak by {peak - ADMISSIBLE_PEAK_BYTES} "
            f"(Spark physical memory {PHYSICAL_MEMORY_BYTES})"
            for slug, (_total, peak, _reserve) in envelopes().items()
            if peak > ADMISSIBLE_PEAK_BYTES and slug not in EXCEPTIONS
        ]
        self.assertEqual(
            offenders,
            [],
            "declared peak exceeds what an idle Spark can admit; derive peak_bytes "
            "from the kit (gpu-memory-utilization, its measured memory and its "
            "startup overhead)",
        )

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
