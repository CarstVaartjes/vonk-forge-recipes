"""Pinned planning constants feed estimates; observed fleet capacity stays authoritative."""

from __future__ import annotations

from pathlib import Path

import documents

from recipe_tooling import catalog_index
from recipe_tooling.qualification import vllm_memory_fit
from recipe_tooling.qualification.memory_need import constants as mn_constants
from recipe_tooling.qualification.memory_need import evidence as mn_evidence
from spark_sweep import cli, memory, sweep_support
from spark_sweep.catalog import parse_fleet


def test_memory_consumers_import_one_pinned_policy() -> None:
    assert (
        memory.OS_FLOOR_BYTES
        == vllm_memory_fit.OS_FLOOR_BYTES
        == mn_constants.OS_FLOOR_BYTES
    )
    assert (
        sweep_support.SweepConfig(Path("unused"), "test").default_spark_memory
        == mn_constants.SPARK_MEMORY_BYTES
    )
    assert (
        cli.SPARK_MEMORY_BYTES
        == vllm_memory_fit.SPARK_MEMORY_BYTES
        == mn_constants.SPARK_MEMORY_BYTES
    )
    assert mn_constants.LIMIT_BYTES == (
        mn_constants.SPARK_MEMORY_BYTES
        - mn_constants.GUARD_BYTES
        - mn_constants.IDLE_USE_BYTES
    )


def test_budget_margins_have_one_owner() -> None:
    """Margins beside an engine budget come only from budget_margin_bytes (audit 2026-11-01).

    The loading floor, inside-budget check and SGLang static need keep OVERHEAD_BYTES.
    """
    source = Path(mn_constants.__file__).read_text()
    assert source.count("def budget_margin_bytes(") == 1
    assert mn_constants.OVERHEAD_BYTES == 8_000_000_000
    cases = {
        ("deepseek-v4-flash-mtp-tonyd2wild-vllm-dual"): 10_400_000_000,
        ("qwen3-8-27b-nvfp4-dflash2-eugr-vllm-single"): 6_200_000_000,
        ("glm-5-3-flash-exl3-a0-2-05bpw-0xsero-sglang-single"): 8_500_000_000,
        ("inkling-small-nvfp4-sglang-dual"): 8_000_000_000,
    }
    for slug, margin in cases.items():
        assert mn_constants.budget_margin_bytes(documents.recipe(slug)) == margin, slug
    assert vllm_memory_fit.budget_margin_bytes is mn_constants.budget_margin_bytes
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
                            + mn_constants.OS_FLOOR_BYTES
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
    observed = mn_constants.SPARK_MEMORY_BYTES // 2
    fleet = parse_fleet(
        {
            "nodes": [
                {"id": "observed", "inventory": {"host_memory_total_bytes": observed}},
                {"id": "unknown", "inventory": {}},
            ]
        },
        mn_constants.SPARK_MEMORY_BYTES,
    )
    assert fleet.sparks[0].memory_total == observed
    assert fleet.sparks[1].memory_total == mn_constants.SPARK_MEMORY_BYTES


def test_measured_peak_conversion_and_publish_rule_have_one_owner() -> None:
    """The Spark-use -> role-peak conversion and the publish rule live in memory_need only."""
    source = Path(mn_evidence.__file__).read_text()
    assert source.count("def measured_peak_bytes(") == 1
    assert source.count("def published_measured_peaks(") == 1
    builder = Path(catalog_index.__file__).read_text()
    assert "published_measured_peaks(" in builder
    assert "OS_FLOOR_BYTES" not in builder
    assert "ADMISSION_IDLE_USE_BYTES" not in builder
    # The sweep states the floor in its note; it never converts a use itself.
    sweep = Path(memory.__file__).read_text()
    assert "- OS_FLOOR_BYTES" not in sweep
    assert "OS_FLOOR_BYTES +" not in sweep
