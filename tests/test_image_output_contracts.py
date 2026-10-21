"""Exercise image adapter writes against the qualification and sweep contract."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from vonk_forge_contracts.recipe import RecipeDefinition

from qualification.definitions_loader import load_definitions
from spark_sweep.job_assertions import Output, evaluate

Image: Any = importlib.import_module("PIL.Image")

ROOT = Path(__file__).resolve().parents[1]
RECIPES = [
    (path, recipe)
    for path in sorted((ROOT / "recipes").glob("*.json"))
    if (
        recipe := RecipeDefinition.model_validate_json(path.read_bytes())
    ).execution.build.context.path.startswith("adapters/image/")
]


@pytest.mark.parametrize("path,recipe", RECIPES, ids=[p.stem for p, _ in RECIPES])
def test_image_adapter_outputs_satisfy_sweep(
    path: Path,
    recipe: RecipeDefinition,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scripts = list((ROOT / recipe.execution.build.context.path).glob("*.py"))
    assert len(scripts) == 1, "cover each image job's executable adapter"
    spec = importlib.util.spec_from_file_location("image_output_contract", scripts[0])
    assert spec and spec.loader
    adapter: Any = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)

    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "prompt.txt").write_text("A red bicycle beside a canal.")
    source = inputs / "reference.png"
    Image.new("RGB", (1024, 1024)).save(source)
    arguments = {arg.name: arg.value for arg in recipe.runtime.arguments}
    if arguments["pipeline"] == "text-to-image":
        source.unlink()
    if hasattr(adapter, "_INPUT_DIR"):
        monkeypatch.setattr(adapter, "_INPUT_DIR", inputs)
    if hasattr(adapter, "_image_input"):
        monkeypatch.setattr(adapter, "_image_input", lambda: source)

    # Substitute inference only. Run main's argument parsing, input handling,
    # output naming and PNG writes, then the sweep's actual evaluator.
    resolution = int(str(arguments.get("resolution", 1024)))
    size = (
        int(str(arguments.get("width", resolution))),
        int(str(arguments.get("height", resolution))),
    )
    layered = arguments["pipeline"] == "image-to-layers"
    image = Image.new("RGBA" if layered else "RGB", size)
    output = [image] * int(str(arguments.get("layers", 1))) if layered else image
    pipe = Mock(return_value=SimpleNamespace(images=[output]))
    pipe.to.return_value = pipe
    pipeline_class = Mock()
    pipeline_class.from_pretrained.return_value = pipe
    diffusers = Mock(
        QwenImagePipeline=pipeline_class,
        QwenImageEditPlusPipeline=pipeline_class,
        QwenImageLayeredPipeline=pipeline_class,
    )
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    monkeypatch.setitem(sys.modules, "torch", Mock())
    if hasattr(adapter, "_pipeline"):
        monkeypatch.setattr(adapter, "_pipeline", lambda _: pipe)
    outputs = tmp_path / "outputs"
    argv = [str(scripts[0]), "--output-dir", str(outputs)]
    for argument in recipe.runtime.arguments:
        assert argument.setting is None
        argv.extend([f"--{argument.name}", str(argument.value)])
    monkeypatch.setattr(sys, "argv", argv)
    adapter.main()

    definition = load_definitions(ROOT / "qualification")["recipes"][
        f"{recipe.identity.publisher}/{path.stem}"
    ]
    written = [
        Output(file.name, "image/png", file.read_bytes())
        for file in sorted(outputs.iterdir())
    ]
    assert evaluate(definition["assertions"], written) == {}


COMFY_RECIPES = [
    (path, recipe)
    for path in sorted((ROOT / "recipes").glob("*.json"))
    if (
        recipe := RecipeDefinition.model_validate_json(path.read_bytes())
    ).runtime.engine
    == "comfyui"
    and recipe.validation.serving.interface == "image-job"
]


@pytest.mark.parametrize(
    "path,recipe", COMFY_RECIPES, ids=[p.stem for p, _ in COMFY_RECIPES]
)
def test_comfy_image_output_names_satisfy_sweep(
    path: Path, recipe: RecipeDefinition, tmp_path: Path
) -> None:
    script = ROOT / recipe.execution.build.context.path / "comfyui_job.py"
    spec = importlib.util.spec_from_file_location("comfy_output_contract", script)
    assert spec and spec.loader
    adapter: Any = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    rendered = tmp_path / "rendered"
    rendered.mkdir()
    Image.new("RGB", (8, 8)).save(rendered / "arbitrary-workflow-name_00001_.png")
    output = adapter.copy_output(rendered, tmp_path / "outputs", "image/png")
    definition = load_definitions(ROOT / "qualification")["recipes"][
        f"{recipe.identity.publisher}/{path.stem}"
    ]
    assertions = [a for a in definition["assertions"] if a["kind"] == "file-names"]
    assert assertions
    assert (
        evaluate(assertions, [Output(output.name, "image/png", output.read_bytes())])
        == {}
    )
