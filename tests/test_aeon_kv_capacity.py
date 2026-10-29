"""Default context must fit the measured KV pool at the creator's utilization."""

from decimal import Decimal

from documents import recipe as load_recipe

from qualification.memory_need import TOTAL_BYTES


def test_aeon_default_context_fits_observed_kv_capacity() -> None:
    recipe = load_recipe("qwen3-8-27b-aeon-abliterated-dflash2-drowzeys-vllm-single")
    assert recipe.settings.kind == "generation"
    context = recipe.settings.context_tokens.value
    assert context is not None
    # vLLM's actual startup refusal, sweep row 26 (2026-10-18T20:32:53Z): at
    # utilization 0.6 vLLM left 43.35 GiB of KV against 51.31 GiB for 1,048,576
    # tokens (linear KV storage at unchanged dtype/topology). Each further
    # 0.1 of unified memory adds its share of KV room.
    args = {item.name: item.value for item in recipe.runtime.arguments}
    utilization = Decimal(str(args["gpu-memory-utilization"]))
    required_gib = Decimal(context) * Decimal("51.31") / 1048576
    room_gib = Decimal("43.35") + (utilization - Decimal("0.6")) * TOTAL_BYTES / 2**30
    assert required_gib < room_gib
    assert args["kv-cache-dtype"] == "fp8_e4m3"
