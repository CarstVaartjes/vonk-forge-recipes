"""Default context must fit the measured KV pool without raising utilization."""

from decimal import Decimal
from pathlib import Path

from vonk_forge_contracts import RecipeDefinition

ROOT = Path(__file__).resolve().parents[1]


def test_aeon_default_context_fits_observed_kv_capacity() -> None:
    recipe = RecipeDefinition.model_validate_json(
        (
            ROOT
            / "recipes/qwen3-8-27b-aeon-abliterated-dflash2-drowzeys-vllm-single.json"
        ).read_text()
    )
    assert recipe.settings.kind == "generation"
    context = recipe.settings.context_tokens.value
    assert context is not None
    # vLLM's actual startup refusal, sweep row 26 (2026-10-18T20:32:53Z).
    # Linear KV storage at unchanged dtype/topology: 51.31 GiB / 1048576 tokens.
    required_gib = Decimal(context) * Decimal("51.31") / 1048576
    assert required_gib < Decimal("43.35")
    args = {item.name: item.value for item in recipe.runtime.arguments}
    assert Decimal(str(args["gpu-memory-utilization"])) <= Decimal("0.6")
    assert args["kv-cache-dtype"] == "fp8_e4m3"
