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
    failure fixes R = 0.696475667681 * pre + 6.49 GiB for the unknown pre-load
    free memory of that start. R is largest when pre is the Spark total, and the
    next start's budget is smallest on the lowest idle Spark, so the new fraction
    is checked against that pairing of owned constants.
    """
    from decimal import Decimal
    from types import SimpleNamespace

    import pytest

    from qualification.memory_need import (
        IDLE_AVAILABLE_BYTES,
        TOTAL_BYTES,
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
    resident = Decimal(FAILED_FRACTION) * TOTAL_BYTES + Decimal(FAILED_REST_GIB) * gib
    available = [int(TOTAL_BYTES - resident)]
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
        sizer._profile_available_bytes(TOTAL_BYTES / gib)

    # Next start: the printed residual rounded up, on the lowest idle Spark.
    resident = (
        Decimal(FAILED_FRACTION) * TOTAL_BYTES + Decimal("6.495") * gib
    ).to_integral_value()
    available[0] = int(IDLE_AVAILABLE_BYTES - resident)
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
    budget = sizer._profile_available_bytes(IDLE_AVAILABLE_BYTES / gib)
    assert int(budget // cell) > tokens
    assert sizer._apply_token_constraints(int(budget // cell)) == tokens
    assert tokens == settings["context_tokens"]["value"] * concurrency
    assert sizer._calculate_mamba_ratio() == 5
    assert sizer.resolve_max_num_reqs(tokens) == concurrency

    # One step lower, the profiled pool no longer holds the token cap.
    schedule.mem_fraction_static = float(arguments["mem-fraction-static"]) - 0.001
    budget = sizer._profile_available_bytes(IDLE_AVAILABLE_BYTES / gib)
    assert sizer._apply_token_constraints(int(budget // cell)) < tokens
    # The failed fraction cannot profile even with the caps.
    schedule.mem_fraction_static = float(FAILED_FRACTION)
    with pytest.raises(ValueError, match="Loaded weights leave no GPU memory"):
        sizer._profile_available_bytes(IDLE_AVAILABLE_BYTES / gib)


def _rank_budget_error(**changes: object) -> str | None:
    """Run the changed-document guard; the empty digest skips the waiver."""
    from catalog_documents import catalog_models
    from vonk_forge_contracts import RecipeDefinition

    from qualification.memory_need import load_inputs, require_memory_need

    document = load(DUAL)
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
    assert failed is not None and "max-total-tokens" in failed
    cases = {
        "static budget": {"mem-fraction-static": FAILED_FRACTION},
        "max-total-tokens": {"max-total-tokens": None},
        "max-mamba-cache-size": {"max-mamba-cache-size": 4},
    }
    for message, changes in cases.items():
        error = _rank_budget_error(**changes)
        assert error is not None and message in error, (changes, error)
    lower = _rank_budget_error(**{"mem-fraction-static": "0.825"})
    assert lower is not None and "static budget" in lower


def _fraction(recipe: dict) -> Decimal:
    from decimal import Decimal

    return next(
        Decimal(str(item["value"]))
        for item in recipe["runtime"]["arguments"]
        if item["name"] == "mem-fraction-static"
    )


def _creator_default(path: str, pattern: str) -> Decimal:
    import re
    from decimal import Decimal

    match = re.search(pattern, (ROOT / path).read_text(encoding="utf-8"))
    assert match, (path, pattern)
    return Decimal(match.group(1))


def test_kv_1m_keeps_the_creators_fraction_that_leaves_pool_room() -> None:
    """The 0.80 edit left 0.12 GB for state pools; the creator's 0.85 leaves room.

    Hardware, 2026-10-28 (op c812b815): at mem-fraction-static 0.800230326246
    SGLang printed avail mem 24.01 GB after loading 89.10 GB of weights and
    total_rest_memory=0.12 GB (mamba_cache_per_req=2.46 MB), then refused to
    start. Each 0.01 of fraction adds 0.01 x pre-load free memory to the rest.
    """
    from decimal import Decimal

    from qualification.memory_need import OVERHEAD_BYTES, TOTAL_BYTES

    recipe = load("recipes/inkling-small-nvfp4-kv-1m-drowzeys-sglang-dual.json")
    fraction = _fraction(recipe)
    assert fraction == _creator_default(
        "adapters/inkling/drowzeys-kvquant-dual/scripts/nvfp4-kv-boot.sh",
        r'MEMFRAC="\$\{MEMFRAC:-([0-9.]+)\}"',
    )
    pre_load_free = (24.01 + 89.10) * 1e9
    rest = 0.12e9 + float(fraction - Decimal("0.800230326246")) * pre_load_free
    assert rest > 5e9  # 0.12 GB failed; one request needs 5 slots x 2.46 MB
    for role in recipe["topology"]["roles"]:
        assert role["resources"]["memory"]["peak_bytes"] >= (
            fraction * TOTAL_BYTES + OVERHEAD_BYTES
        )


def test_mia_dual_keeps_the_creators_fraction() -> None:
    from qualification.memory_need import OVERHEAD_BYTES, TOTAL_BYTES

    recipe = load("recipes/inkling-small-nvfp4-sglang-mia-dual.json")
    fraction = _fraction(recipe)
    assert fraction == _creator_default(
        "adapters/inkling/mia-dual/start_sglang.sh",
        r'MEM_FRACTION_STATIC="\$\{MEM_FRACTION_STATIC:-([0-9.]+)\}"',
    )
    for role in recipe["topology"]["roles"]:
        assert role["resources"]["memory"]["peak_bytes"] >= (
            fraction * TOTAL_BYTES + OVERHEAD_BYTES
        )


def test_975b_eight_keeps_its_authored_fraction() -> None:
    """No creator launch script exists for the eight-Spark recipe; 0.80 is the
    fraction it was added with (a9c33133), before #1647 lowered it to 0.634."""
    from decimal import Decimal

    from qualification.memory_need import OVERHEAD_BYTES, TOTAL_BYTES

    recipe = load("recipes/inkling-975b-a41b-nvfp4-sglang-eight.json")
    fraction = _fraction(recipe)
    assert fraction == Decimal("0.80")
    for role in recipe["topology"]["roles"]:
        assert role["resources"]["memory"]["peak_bytes"] >= (
            fraction * TOTAL_BYTES + OVERHEAD_BYTES
        )
