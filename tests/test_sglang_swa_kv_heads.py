"""SGLang pools sliding layers with their own KV head count when the config sets one."""

import pytest
from pydantic import ValidationError

from qualification import documents
from qualification.memory_need import CacheGeometry, load_inputs, sglang_token_pools

INKLING = "inkling-975b-a41b-nvfp4-sglang-eight"  # TP8: 1 full, 2 sliding heads/rank


def test_sliding_pool_uses_the_sliding_kv_heads() -> None:
    recipe = documents.recipe(INKLING)
    geometry = load_inputs().geometries["inkling-975b-a41b-nvfp4"]
    assert geometry.sglang_swa_kv_heads == 16

    def pools(**update: object) -> int:
        return sglang_token_pools(
            recipe, geometry.model_copy(update=update), 1_000_000, INKLING
        )

    full = pools(sglang_swa_layers=0, sglang_swa_kv_heads=None)
    same_heads = pools(sglang_swa_kv_heads=None) - full
    assert pools() - full == 2 * same_heads


def test_sliding_heads_need_sliding_layers() -> None:
    entry = load_inputs().geometries["inkling-975b-a41b-nvfp4"].model_dump()
    entry["sglang_swa_layers"] = 0
    with pytest.raises(ValidationError, match="sglang_swa_kv_heads needs"):
        CacheGeometry.model_validate(entry)
