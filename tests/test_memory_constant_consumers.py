"""Pinned planning constants feed estimates; observed fleet capacity stays authoritative."""

from __future__ import annotations

from pathlib import Path

import documents

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


def test_budget_margins_have_one_owner() -> None:
    """Margins beside an engine budget come only from budget_margin_bytes (audit 2026-11-01).

    The loading floor, inside-budget check and SGLang static need keep OVERHEAD_BYTES.
    """
    source = Path(memory_need.__file__).read_text()
    assert source.count("def budget_margin_bytes(") == 1
    assert memory_need.OVERHEAD_BYTES == 8_000_000_000
    cases = {
        ("deepseek-v4-flash-vllm-mia-dual"): 10_400_000_000,
        ("qwen3-8-27b-nvfp4-dflash2-eugr-vllm-single"): 6_200_000_000,
        ("glm-5-3-flash-exl3-a0-2-05bpw-0xsero-sglang-single"): 8_500_000_000,
        ("inkling-small-nvfp4-sglang-dual"): 8_000_000_000,
    }
    for slug, margin in cases.items():
        assert memory_need.budget_margin_bytes(documents.recipe(slug)) == margin, slug
    assert vllm_memory_fit.budget_margin_bytes is memory_need.budget_margin_bytes
    # The producer guard reads the owner; it restates no margin.
    guard = Path(vllm_memory_fit.__file__).read_text()
    assert "OVERHEAD_BYTES" not in guard.replace("INCIDENT_MEASURED_OVERHEAD_BYTES", "")


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
