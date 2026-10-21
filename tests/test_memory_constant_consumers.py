"""Pinned planning constants feed estimates; observed fleet capacity stays authoritative."""

from __future__ import annotations

from pathlib import Path

from qualification import memory_need, vllm_memory_fit
from spark_sweep import cli, memory, run
from spark_sweep.catalog import parse_fleet


def test_memory_consumers_import_one_pinned_policy() -> None:
    assert (
        memory.OS_FLOOR_BYTES
        == vllm_memory_fit.OS_FLOOR_BYTES
        == memory_need.OS_FLOOR_BYTES
    )
    assert (
        run.SweepConfig(Path("unused"), "test").default_spark_memory
        == memory_need.SPARK_MEMORY_BYTES
    )
    assert (
        cli.SPARK_MEMORY_BYTES
        == vllm_memory_fit.SPARK_MEMORY_BYTES
        == memory_need.SPARK_MEMORY_BYTES
    )
    assert memory_need.LIMIT_BYTES == (
        memory_need.SPARK_MEMORY_BYTES
        - memory_need.GUARD_BYTES
        - memory_need.IDLE_USE_BYTES
    )


def test_quality_note_applies_the_shared_os_floor_once() -> None:
    declared = 100
    sample = memory.observe(
        {
            "nodes": [
                {
                    "id": "fake",
                    "loaded": [{"alias": "test"}],
                    "telemetry": {
                        "sample": {
                            "memory_total_bytes": declared
                            + memory_need.OS_FLOOR_BYTES
                            + 1,
                            "memory_available_bytes": 1,
                        }
                    },
                }
            ]
        },
        ["fake"],
        "test",
        declared,
    )
    assert sample is not None
    assert not memory.notes(sample)
    sample["nodes"][0]["used_bytes"] += 1
    assert len(memory.notes(sample)) == 1


def test_observed_capacity_overrides_only_the_historical_fallback() -> None:
    observed = memory_need.SPARK_MEMORY_BYTES // 2
    fleet = parse_fleet(
        {
            "nodes": [
                {"id": "observed", "inventory": {"host_memory_total_bytes": observed}},
                {"id": "unknown", "inventory": {}},
            ]
        },
        memory_need.SPARK_MEMORY_BYTES,
    )
    assert fleet.sparks[0].memory_total == observed
    assert fleet.sparks[1].memory_total == memory_need.SPARK_MEMORY_BYTES
