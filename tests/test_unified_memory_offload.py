"""Bound resident smoke planning by the same GB10 limit as the catalog guard."""

import json
from pathlib import Path

import pytest
from catalog_documents import catalog_models
from vonk_forge_contracts import RecipeDefinition

from qualification.memory_need import MemoryNeed

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "slug,activation_bytes,loading_bytes",
    [
        ("mova-360p-diffusers-single", 2_000_000_000, 10_000_000_000),
        ("mova-720p-diffusers-single", 8_000_000_000, 10_000_000_000),
        ("hunyuan-video-15-distilled-diffusers-single", 16_000_000_000, 6_000_000_000),
        ("hunyuan-video-15-t2v-diffusers-single", 16_000_000_000, 6_000_000_000),
        (
            "hunyuan-video-15-i2v-step-distilled-diffusers-single",
            16_000_000_000,
            6_000_000_000,
        ),
        ("wan-dancer-14b-pytorch-single", 16_000_000_000, 12_000_000_000),
    ],
)
def test_resident_smoke_includes_companions_loading_and_activations(
    slug: str, activation_bytes: int, loading_bytes: int
) -> None:
    recipe = RecipeDefinition.model_validate_json(
        (ROOT / "recipes" / f"{slug}.json").read_text()
    )
    models = catalog_models()
    weights = 0
    for selection in recipe.models:
        selected = {item.file_id for item in selection.files}
        for item in models[selection.model.content_sha256].files:
            # Companion roles are not always "weights": Wan's T5/CLIP/VAE
            # .pth files are metadata; LTX's Gemma shards are text-encoder.
            if item.id in selected and item.path.endswith(
                (".safetensors", ".pth", ".bin", ".pt")
            ):
                if (
                    slug.startswith("wan-dancer")
                    and item.path == "local_model.safetensors"
                ):
                    continue  # The two stages run in separate subprocesses.
                weights += item.size_bytes
    need = MemoryNeed(weights, 0, 0, activation_bytes + loading_bytes)
    assert need.total_bytes < need.limit_bytes
    if slug.startswith("mova"):
        assert weights == 77_672_622_000  # BOTH DiTs and all companion models.
        # Pinned WanModel: 40 heads, width 5120, FFN 13824, stride 16.
        # Two FP32 attention-score buffers plus a generous live-vector/FFN
        # count and 0.5 GB VAE/audio slack. No gradients; CFG is sequential.
        height, width = (352, 640) if "360p" in slug else (720, 1280)
        tokens = height * width // 256
        workspace = (
            2 * 40 * tokens**2 * 4
            + 16 * tokens * 5120 * 4
            + 2 * tokens * 13824 * 4
            + 500_000_000
        )
        assert workspace < activation_bytes
        spec = json.loads(
            (ROOT / "qualification/recipes" / f"{slug}.json").read_text()
        )["recipes"]
        # No assumed speedup: two observed 325.32 s steps scaled by tokens.
        estimate_seconds = 2 * 325.32 * tokens / 880 + 540
        assert estimate_seconds < spec["timeout_seconds"] <= 3600
