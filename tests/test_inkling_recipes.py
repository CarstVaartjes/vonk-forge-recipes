from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


class InklingSmallRecipeTests(unittest.TestCase):
    def test_complete_model_inventory_and_two_spark_profile(self) -> None:
        model, recipe = (
            load("models/inkling-small-nvfp4.json"),
            load("recipes/inkling-small-nvfp4-sglang-dual.json"),
        )
        self.assertEqual(
            model["source"]["revision"], "b6a99534467840620d411e4cd4ad5819b2610d9c"
        )
        self.assertEqual(
            len({item["id"] for item in model["files"]}), len(model["files"])
        )
        self.assertEqual(recipe["topology"]["node_count"], 2)
        self.assertEqual(recipe["topology"]["parallelism"]["tensor"], 2)
        self.assertEqual(recipe["topology"]["parallelism"]["backend"], "native")

    def test_runtime_is_offline_and_multimodal_api_is_named(self) -> None:
        recipe = load("recipes/inkling-small-nvfp4-sglang-dual.json")
        interface = recipe["interfaces"][0]
        self.assertEqual(interface["adapter"], "openai")
        self.assertEqual(interface["model_aliases"], ["inkling-small"])
        self.assertEqual(recipe["runtime"]["engine"], "sglang")
        self.assertTrue(recipe["models"])

    def test_wrapper_resolves_only_the_compiler_rendezvous_sentinel(self) -> None:
        wrapper = ROOT / "adapters/inkling/small-dual/sglang-serve"
        with tempfile.TemporaryDirectory() as directory:
            launcher = Path(directory) / "sglang/launch_server.py"
            launcher.parent.mkdir()
            launcher.write_text(
                "import json, os, sys\nopen(os.environ['CAPTURE'], 'w').write(json.dumps(sys.argv[1:]))"
            )
            capture = Path(directory) / "arguments.json"
            env = {
                **os.environ,
                "PYTHONPATH": directory,
                "CAPTURE": str(capture),
                "VONK_MASTER_ADDR": "192.0.2.10",
                "VONK_MASTER_PORT": "29500",
                "VONK_LOCAL_ADDR": "192.0.2.11",
                "NCCL_SOCKET_IFNAME": "eth0",
                "NCCL_IB_HCA": "roce0",
                "NCCL_IB_GID_INDEX": "3",
                "TP_SOCKET_IFNAME": "eth0",
                "GLOO_SOCKET_IFNAME": "eth0",
            }
            result = subprocess.run(
                [
                    sys.executable,
                    str(wrapper),
                    "--nnodes",
                    "2",
                    "--node-rank",
                    "1",
                    "--dist-init-addr",
                    "VONK_MASTER_ADDR:VONK_MASTER_PORT",
                ],
                env=env,
                check=False,
            )
            self.assertEqual(result.returncode, 0)
            self.assertEqual(json.loads(capture.read_text())[-1], "192.0.2.10:29500")


if __name__ == "__main__":
    unittest.main()


DUAL = "recipes/inkling-small-nvfp4-sglang-dual.json"
# Sweep 2026-10-24, operation 1d9c19ee, content 6010d4a3: both TP ranks raised
# "Not enough GPU memory for hybrid ... max_mamba_cache_size=-247
# (total_rest_memory=-6.49 GB, mamba_cache_per_req=2.46 MB)".
FAILED_FRACTION = "0.696475667681"
FAILED_REST_GIB = 6.49


def _sglang_sizer(available_bytes: list[int], schedule: object) -> Any:
    """The pinned image's sizing methods with the Inkling TP2 runtime around them."""
    import gc
    import logging
    import math
    import runpy
    from types import SimpleNamespace

    from qualification.memory_need import load_inputs

    def override(_reason: str, **values: int) -> None:
        for name, value in values.items():
            setattr(schedule, name, value)

    namespace = runpy.run_path(
        str(ROOT / "tests/fixtures/inkling-sglang-cache-sizing.py.txt"),
        init_globals={
            "gc": gc,
            "math": math,
            "logging": logging,
            "logger": logging.getLogger("sglang"),
            "envs": SimpleNamespace(
                SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=SimpleNamespace(get=lambda: False)
            ),
            "get_schedule": lambda: schedule,
            "get_available_gpu_memory": lambda *a, **kw: available_bytes[0] / (1 << 30),
            "get_world_group": lambda: SimpleNamespace(world_size=2, cpu_group=None),
            "get_mm": lambda: SimpleNamespace(mm_feature_transport=None),
            # The failure's residual already includes the multimodal reservation.
            "mm_runtime_reservation_gb": lambda **kw: 0.0,
            "get_memory": lambda: SimpleNamespace(disable_radix_cache=False),
            "get_exec": lambda: SimpleNamespace(
                mamba=SimpleNamespace(
                    enable_linear_replayssm_spec=False,
                    enable_mamba_extra_buffer=True,
                    enable_mamba_extra_buffer_lazy=False,
                )
            ),
            "get_context": lambda: SimpleNamespace(override=override),
            "get_parallel": lambda: SimpleNamespace(pp_size=1),
            "kimi_linear_config": lambda _config: None,
        },
    )
    geometry = load_inputs().geometries["inkling-small-nvfp4"]
    sizer = namespace["KVCacheConfigurator"]()
    sizer.device, sizer.gpu_id = "cuda", 0
    sizer.post_capture_kv_active = False
    sizer.hybrid_gdn_config = None
    sizer.model_config = SimpleNamespace(is_multimodal=True, context_len=65536)
    sizer.ps = SimpleNamespace(pp_size=1, attn_dp_size=1)
    sizer.spec_algorithm = SimpleNamespace(is_none=lambda: True)
    sizer.mambaish_config = SimpleNamespace(
        mamba2_cache_params=SimpleNamespace(
            layers=list(range(geometry.sglang_swa_layers)),
            mamba_cache_per_req=geometry.sglang_state_bytes_per_slot,
        )
    )
    return sizer


def test_pinned_sglang_sizing_rejects_failed_fraction_and_holds_the_caps() -> None:
    """Replay sglang 94602c9c's pool profiling with the hardware failure.

    _profile_available_bytes leaves f * pre - R, where R is everything resident
    when it runs (weights, their expansion, the multimodal reservation). The
    failure fixes R = 0.696475667681 * pre + 6.49 GiB (SGLang prints GiB) for
    the pre-load free memory of that start, taken as the one owned
    SGLANG_PRELOAD_FREE_BYTES; the geometry's expansion is derived from it.
    """
    from decimal import Decimal
    from types import SimpleNamespace

    import pytest

    from qualification.memory_need import (
        SGLANG_PRELOAD_FREE_BYTES,
        load_inputs,
    )

    gib = 1 << 30
    recipe = load(DUAL)
    arguments = {item["name"]: item["value"] for item in recipe["runtime"]["arguments"]}
    settings = recipe["settings"]
    geometry = load_inputs().geometries["inkling-small-nvfp4"]
    model = load("models/inkling-small-nvfp4.json")
    assert any(
        item["sha256"] == geometry.configuration_sha256 for item in model["files"]
    )
    resident = (
        Decimal(FAILED_FRACTION) * SGLANG_PRELOAD_FREE_BYTES
        + Decimal(FAILED_REST_GIB) * gib
    )
    available = [int(SGLANG_PRELOAD_FREE_BYTES - resident)]
    failed = SimpleNamespace(
        mem_fraction_static=float(FAILED_FRACTION),
        max_total_tokens=None,
        max_mamba_cache_size=None,
        max_running_requests=None,
        mamba_full_memory_ratio=float(arguments["mamba-full-memory-ratio"]),
        disable_overlap_schedule=False,
    )
    sizer = _sglang_sizer(available, failed)
    with pytest.raises(
        RuntimeError,
        match=r"Not enough GPU memory for hybrid.*max_mamba_cache_size=-247 "
        r"\(total_rest_memory=-6.49 GB, mamba_cache_per_req=2.46 MB\)",
    ):
        sizer._profile_available_bytes(SGLANG_PRELOAD_FREE_BYTES / gib)

    # Next start: the printed residual rounded up, on the lowest idle Spark.
    resident = (
        Decimal(FAILED_FRACTION) * SGLANG_PRELOAD_FREE_BYTES + Decimal("6.495") * gib
    ).to_integral_value()
    available[0] = int(SGLANG_PRELOAD_FREE_BYTES - resident)
    tokens = arguments["max-total-tokens"]
    concurrency = settings["concurrency"]["value"]
    schedule = SimpleNamespace(
        mem_fraction_static=float(arguments["mem-fraction-static"]),
        max_total_tokens=tokens,
        max_mamba_cache_size=arguments["max-mamba-cache-size"],
        max_running_requests=concurrency,
        mamba_full_memory_ratio=float(arguments["mamba-full-memory-ratio"]),
        disable_overlap_schedule=False,
    )
    sizer = _sglang_sizer(available, schedule)
    # HybridSWAPoolConfigurator cell: full layers plus ratio * SWA layers,
    # BF16 K and V for this rank's share of the KV heads.
    heads = geometry.kv_heads // recipe["topology"]["parallelism"]["tensor"]
    per_layer = heads * (geometry.key_dim + geometry.value_dim) * 2
    cell = per_layer * (
        geometry.layers
        + float(arguments["swa-full-tokens-ratio"]) * geometry.sglang_swa_layers
    )
    budget = sizer._profile_available_bytes(SGLANG_PRELOAD_FREE_BYTES / gib)
    assert int(budget // cell) > tokens
    assert sizer._apply_token_constraints(int(budget // cell)) == tokens
    assert tokens == settings["context_tokens"]["value"] * concurrency
    assert sizer._calculate_mamba_ratio() == 5
    assert sizer.resolve_max_num_reqs(tokens) == concurrency

    # Well below the proven fraction the profiled pool no longer holds the cap.
    schedule.mem_fraction_static = float(arguments["mem-fraction-static"]) - 0.04
    budget = sizer._profile_available_bytes(SGLANG_PRELOAD_FREE_BYTES / gib)
    assert sizer._apply_token_constraints(int(budget // cell)) < tokens
    # The failed fraction cannot profile even with the caps.
    schedule.mem_fraction_static = float(FAILED_FRACTION)
    with pytest.raises(ValueError, match="Loaded weights leave no GPU memory"):
        sizer._profile_available_bytes(SGLANG_PRELOAD_FREE_BYTES / gib)


def _rank_budget_error(path: str = DUAL, **changes: object) -> str | None:
    """Run the changed-document guard; the empty digest skips the waiver."""
    from catalog_documents import catalog_models
    from vonk_forge_contracts import RecipeDefinition

    from qualification.memory_need import load_inputs, require_memory_need

    document = load(path)
    arguments = document["runtime"]["arguments"]
    for name, value in changes.items():
        if name == "concurrency":
            document["settings"]["concurrency"] = value
        elif name == "peak_bytes":
            for role in document["topology"]["roles"]:
                role["resources"]["memory"]["peak_bytes"] = value
        elif value is None:
            arguments[:] = [item for item in arguments if item["name"] != name]
        else:
            next(item for item in arguments if item["name"] == name)["value"] = value
    try:
        require_memory_need(
            RecipeDefinition.model_validate(document),
            catalog_models(),
            load_inputs(),
            document_digest="",
        )
    except (TypeError, ValueError) as error:
        return str(error)
    return None


def test_rank_budget_guard_accepts_the_new_settings() -> None:
    from decimal import Decimal

    from qualification.memory_need import OVERHEAD_BYTES, TOTAL_BYTES

    assert _rank_budget_error() is None
    recipe = load(DUAL)
    fraction = next(
        item["value"]
        for item in recipe["runtime"]["arguments"]
        if item["name"] == "mem-fraction-static"
    )
    for role in recipe["topology"]["roles"]:
        assert role["resources"]["memory"]["peak_bytes"] >= (
            Decimal(fraction) * TOTAL_BYTES + OVERHEAD_BYTES
        )


def test_rank_budget_guard_rejects_the_failed_settings() -> None:
    # Content 6010d4a3's launch: uncapped pools at the failed fraction.
    failed = _rank_budget_error(
        **{
            "mem-fraction-static": FAILED_FRACTION,
            "max-running-requests": None,
            "max-total-tokens": None,
            "max-mamba-cache-size": None,
            "concurrency": None,
            "peak_bytes": 99_003_761_392,
        }
    )
    assert failed is not None and "SGLang needs to start" in failed
    cases = {
        "static budget": {"mem-fraction-static": FAILED_FRACTION},
        "max-mamba-cache-size": {"max-mamba-cache-size": 4},
    }
    for message, changes in cases.items():
        error = _rank_budget_error(**changes)
        assert error is not None and message in error, (changes, error)


KV_1M = "recipes/inkling-small-nvfp4-kv-1m-drowzeys-sglang-dual.json"
# Sweep 2026-10-28, operation c812b815 (rank 0): the target's "mem usage=89.10
# GB" and the DSparkDraftModel's "mem usage=1.30 GB" leave "avail mem=22.74 GB"
# of 113.11 GB (SGLang prints GiB), then at mem-fraction-static
# 0.800230326246 "Computed max_mamba_cache_size=-2 (total_rest_memory=0.12 GB,
# mamba_cache_per_req=2.46 MB)".
KV_1M_FAILED_FRACTION = "0.800230326246"
KV_1M_PRE_LOAD_GIB = 24.01 + 89.10
KV_1M_AFTER_DRAFT_GIB = 22.74


def _dspark_sizer(fraction: float, available_gib: float) -> Any:
    """sglang b7252cc6's sizing methods around the kvquant image's TP2 start."""
    import runpy
    from types import SimpleNamespace

    from qualification.memory_need import load_inputs

    arguments = {
        item["name"]: item["value"] for item in load(KV_1M)["runtime"]["arguments"]
    }
    schedule = SimpleNamespace(
        mem_fraction_static=fraction,
        max_mamba_cache_size=None,
        max_running_requests=16,
        speculative_num_draft_tokens=int(arguments["speculative-dspark-block-size"])
        + 1,
        mamba_full_memory_ratio=float(arguments["mamba-full-memory-ratio"]),
        disable_overlap_schedule=False,
        disable_radix_cache=False,
        enable_mamba_extra_buffer=lambda: True,
        enable_mamba_extra_buffer_lazy=lambda: False,
        pre_capture_activation_reserve_mb=lambda _capacity: 0,
    )

    def override(_reason: str, **values: int) -> None:
        for name, value in values.items():
            setattr(schedule, name, value)

    namespace = runpy.run_path(
        str(ROOT / "tests/fixtures/sglang-b7252cc6-cache-sizing.py.txt"),
        init_globals={
            "TYPE_CHECKING": False,
            "math": __import__("math"),
            "envs": SimpleNamespace(
                SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=SimpleNamespace(get=lambda: False)
            ),
            "get_available_gpu_memory": lambda *a, **kw: available_gib,
            "get_world_group": lambda: SimpleNamespace(world_size=2, cpu_group=None),
        },
    )
    geometry = next(
        item
        for key, item in load_inputs().geometries.items()
        if key.startswith("inkling-small-nvfp4@")
    )
    sizer = namespace["KVCacheConfigurator"]()
    sizer.device, sizer.gpu_id = "cuda", 0
    sizer.post_capture_kv_active = False
    sizer.server_args = schedule
    sizer.ps = SimpleNamespace(pp_size=1, attn_dp_size=1)
    sizer.spec_algorithm = SimpleNamespace(is_none=lambda: False)
    sizer.mambaish_config = SimpleNamespace(
        mamba2_cache_params=SimpleNamespace(
            mamba_cache_per_req=geometry.sglang_state_bytes_per_slot
        )
    )
    sizer._override = override
    schedule.override = override
    return sizer, schedule


def test_dspark_sizing_replay_matches_the_10_28_failure_and_the_fix() -> None:
    import pytest

    pre = KV_1M_PRE_LOAD_GIB
    # The printed 0.12 GB residual fixes the available figure at the failed fraction.
    failed = float(KV_1M_FAILED_FRACTION)
    sizer, _ = _dspark_sizer(failed, pre * (1 - failed) + 0.12)
    with pytest.raises(
        RuntimeError,
        match=r"max_mamba_cache_size=-2 \(total_rest_memory=0.12 GB, "
        r"mamba_cache_per_req=2.46 MB\)",
    ):
        sizer._profile_available_bytes(pre)
    # 0.85 with the same loaded weights: ~5.75 GiB rest, about 80 slots.
    sizer, schedule = _dspark_sizer(0.85, KV_1M_AFTER_DRAFT_GIB)
    assert sizer._profile_available_bytes(pre) > 0
    assert 75 <= schedule.max_mamba_cache_size <= 85
    assert sizer._calculate_mamba_ratio() == 5
    assert schedule.max_mamba_cache_size // 5 >= 16


def test_uncapped_startup_room_refuses_the_10_28_fraction_and_accepts_the_fix() -> None:
    from decimal import Decimal

    from qualification.memory_need import OVERHEAD_BYTES, TOTAL_BYTES

    def peak(fraction: str) -> int:
        return int(Decimal(fraction) * TOTAL_BYTES + OVERHEAD_BYTES) + 1

    failed = _rank_budget_error(
        KV_1M,
        **{
            "mem-fraction-static": KV_1M_FAILED_FRACTION,
            "peak_bytes": peak(KV_1M_FAILED_FRACTION),
        },
    )
    assert failed is not None and "SGLang needs to start" in failed, failed
    assert (
        _rank_budget_error(
            KV_1M, **{"mem-fraction-static": "0.85", "peak_bytes": peak("0.85")}
        )
        is None
    )


def test_fp4_mx_block16_costs_what_nvfp4_costs() -> None:
    from catalog_documents import catalog_models  # noqa: F401
    from vonk_forge_contracts import RecipeDefinition

    from qualification.memory_need import attention_token_bytes, load_inputs

    inputs = load_inputs()
    geometry = next(
        item
        for key, item in inputs.geometries.items()
        if key.startswith("inkling-small-nvfp4@")
    )
    document = load(KV_1M)
    results = []
    for dtype in ("fp4_mx_block16", "nvfp4"):
        for item in document["runtime"]["arguments"]:
            if item["name"] == "kv-cache-dtype":
                item["value"] = dtype
        recipe = RecipeDefinition.model_validate(document)
        results.append(attention_token_bytes(recipe, geometry))
    assert results[0] == results[1]
    layers, token_bytes = results[0]
    # 0.5625 bytes per element: packed e2m1 plus one scale byte per 16 elements.
    elements = (
        layers * (geometry.kv_heads // 2) * (geometry.key_dim + geometry.value_dim)
    )
    assert token_bytes == elements * 9 // 16


def test_unproven_unchanged_document_reports_its_startup_room() -> None:
    from catalog_documents import catalog_models
    from vonk_forge_contracts import RecipeDefinition, document_sha256

    from qualification.memory_need import load_inputs, memory_warnings

    document = load(KV_1M)
    warnings = memory_warnings(
        RecipeDefinition.model_validate(document),
        catalog_models(),
        load_inputs(),
        document_sha256(document),
    )
    assert any("startup room not met" in item for item in warnings)
    assert not any("WARNING" in item[len("WARNING") :] for item in warnings)
