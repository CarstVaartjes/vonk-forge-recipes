from __future__ import annotations

import json
from pathlib import Path


def test_oci_runtime_metadata_and_entrypoint_are_declared() -> None:
    adapter = Path(__file__).parent
    dockerfile = (adapter / "Dockerfile").read_text()
    assert 'ai.vonkforge.runtime-interface="v1"' in dockerfile
    recipe = json.loads(
        (
            adapter.parents[2]
            / "recipes/qwen3-8-flash-next-nvfp4-sfxnz-vllm-dual.json"
        ).read_text()
    )
    assert recipe["runtime"]["entrypoint"] == ["/opt/vonk/bin/qwen38-vllm-serve"]


def test_upstream_overlays_are_baked_at_the_vllm_paths() -> None:
    dockerfile = (Path(__file__).parent / "Dockerfile").read_text()
    assert "COPY docker/ple_layer.py " in dockerfile
    assert "COPY docker/modelopt.py " in dockerfile
