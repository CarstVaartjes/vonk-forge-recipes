"""TensorFold accepts --ple-on-ssd only for an MLX checkpoint.

The engine reads an MLX checkpoint's n-gram shards from disk with that flag; an
EXL3 pack maps its own table from its file and an NVFP4 checkpoint keeps its
tables memory-mapped, so `serve` exits 1 before loading (observed on a Spark
for the EXL3 6.05bpw recipe: "an EXL3 pack maps its own table from its file,
so drop --ple-on-ssd").  A recipe may pass it, in its arguments or an option
choice, only when every Model it runs is an MLX checkpoint.
"""

from pathlib import Path

from documents import model, recipe
from vonk_forge_contracts.recipe import RecipeDefinition

ROOT = Path(__file__).resolve().parents[1]
FLAG = "ple-on-ssd"


def _passes_flag(definition: RecipeDefinition) -> bool:
    names = [argument.name for argument in definition.runtime.arguments]
    for option in definition.options:
        for choice in option.choices:
            names.extend(argument.name for argument in choice.args)
    return FLAG in names


def test_ple_on_ssd_runs_only_mlx_checkpoints() -> None:
    offenders = []
    for path in sorted((ROOT / "recipes").glob("*.json")):
        definition = recipe(path.stem)
        if definition.runtime.engine != "tensorfold" or not _passes_flag(definition):
            continue
        for reference in definition.models:
            slug = reference.model.slug
            quantization = model(slug).format.quantization
            if not quantization.startswith("mlx"):
                offenders.append(f"{path.name}: {slug} ({quantization})")
    assert not offenders, offenders
