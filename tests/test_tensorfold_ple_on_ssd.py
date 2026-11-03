"""TensorFold accepts --ple-on-ssd only for an MLX checkpoint.

The engine reads an MLX checkpoint's n-gram shards from disk with that flag; an
EXL3 pack maps its own table from its file and an NVFP4 checkpoint keeps its
tables memory-mapped, so `serve` exits 1 before loading (observed on a Spark
for the EXL3 6.05bpw recipe: "an EXL3 pack maps its own table from its file,
so drop --ple-on-ssd").  A recipe may pass it, in its arguments or an option
choice, only when every Model it runs is an MLX checkpoint.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FLAG = "ple-on-ssd"


def _passes_flag(recipe: dict) -> bool:
    arguments = list(recipe["runtime"].get("arguments", []))
    for option in recipe.get("options", []):
        for choice in option["choices"]:
            arguments.extend(choice.get("args", []))
    return any(argument["name"] == FLAG for argument in arguments)


def _quantization(slug: str) -> str:
    model = json.loads((ROOT / "models" / f"{slug}.json").read_text())
    return model["format"]["quantization"]


def test_ple_on_ssd_runs_only_mlx_checkpoints() -> None:
    offenders = []
    for path in sorted((ROOT / "recipes").glob("*.json")):
        recipe = json.loads(path.read_text())
        if recipe["runtime"].get("engine") != "tensorfold" or not _passes_flag(recipe):
            continue
        for reference in recipe["models"]:
            slug = reference["model"]["slug"]
            quantization = _quantization(slug)
            if not quantization.startswith("mlx"):
                offenders.append(f"{path.name}: {slug} ({quantization})")
    assert not offenders, offenders
