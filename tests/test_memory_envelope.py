"""Every recipe's declared memory envelope must fit the Spark that will run it.

A role's ``peak_bytes`` plus ``reserve_bytes`` is what the Controller must find free
on one Spark. A recipe whose envelope alone exceeds the Spark's physical memory can
never be admitted: the Controller refuses it with ``resource.envelope_exceeds_capacity``
whatever is running. Recipes that exceed it today are listed with their reason in
``memory-envelope-exceptions.json``; a listed recipe that now fits must be removed.

The Spark memory is the physical total of a DGX Spark (121.69 GiB). Override it with
``VONK_SPARK_MEMORY_BYTES`` to check against another configuration.
"""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPARK_MEMORY_BYTES = int(os.environ.get("VONK_SPARK_MEMORY_BYTES", "130663231488"))
EXCEPTIONS = json.loads((ROOT / "memory-envelope-exceptions.json").read_text())


def envelopes() -> dict[str, tuple[int, int, int]]:
    """Slug to the largest (peak plus reserve, peak, reserve) of its roles."""
    result: dict[str, tuple[int, int, int]] = {}
    for path in sorted((ROOT / "recipes").glob("*.json")):
        document = json.loads(path.read_text())
        roles = document.get("topology", {}).get("roles", [])
        values = [
            (
                role["resources"]["memory"]["peak_bytes"]
                + role["resources"]["memory"]["reserve_bytes"],
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
            f"{slug}: peak {peak} + reserve {reserve} = {total} bytes exceeds the "
            f"{SPARK_MEMORY_BYTES}-byte Spark memory by {total - SPARK_MEMORY_BYTES}"
            for slug, (total, peak, reserve) in envelopes().items()
            if total > SPARK_MEMORY_BYTES and slug not in EXCEPTIONS
        ]
        self.assertEqual(
            offenders,
            [],
            "declared memory envelope exceeds the Spark and can never be admitted; "
            "derive peak_bytes from the kit (gpu-memory-utilization and the kit's "
            "own startup overhead) and keep reserve_bytes at the platform floor",
        )

    def test_exceptions_are_current(self) -> None:
        known = envelopes()
        stale = [
            slug
            for slug in EXCEPTIONS
            if slug not in known or known[slug][0] <= SPARK_MEMORY_BYTES
        ]
        self.assertEqual(
            stale,
            [],
            "memory-envelope-exceptions.json lists a recipe that is gone or now "
            "fits the Spark; remove it",
        )


if __name__ == "__main__":
    unittest.main()
