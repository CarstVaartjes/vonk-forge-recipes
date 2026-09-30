from __future__ import annotations

import json
from pathlib import Path

from vonk_forge_contracts import RecipeDefinition, read_recipe

ROOT = Path(__file__).resolve().parents[1]


def _recipe(slug: str) -> RecipeDefinition:
    path = ROOT / "recipes" / f"{slug}.json"
    return read_recipe(json.loads(path.read_text(encoding="utf-8")))


def _arguments(recipe: RecipeDefinition) -> dict[str, object]:
    return {item.name: item.value for item in recipe.runtime.arguments}


def test_qwen_image_2512_options_keep_square_default_and_resolve_supported_canvases() -> (
    None
):
    recipe = _recipe("qwen-image-2512-diffusers-single")
    default = recipe.with_option_choices({})
    assert (_arguments(default)["width"], _arguments(default)["height"]) == (1328, 1328)

    expected = {
        "landscape": (1664, 928),
        "portrait": (928, 1664),
        "landscape-4-3": (1472, 1104),
        "portrait-3-4": (1104, 1472),
        "landscape-3-2": (1584, 1056),
        "portrait-2-3": (1056, 1584),
    }
    for choice, size in expected.items():
        arguments = _arguments(recipe.with_option_choices({"aspect-ratio": choice}))
        assert (arguments["width"], arguments["height"]) == size
        assert size[0] % 16 == size[1] % 16 == 0


def test_hunyuan_i2v_option_resolves_to_checkpoint_supported_native_steps() -> None:
    recipe = _recipe("hunyuan-video-15-i2v-step-distilled-diffusers-single")
    from test_hunyuan_single_recipes import video_module

    module = video_module()
    default = _arguments(recipe.with_option_choices({}))
    twelve = _arguments(recipe.with_option_choices({"inference-steps": "twelve-step"}))
    assert default["num-inference-steps"] == 8
    assert twelve["num-inference-steps"] == 12
    for steps in (default["num-inference-steps"], twelve["num-inference-steps"]):
        assert module._variant("image-to-image", 480, steps, 1.0) == (
            "HunyuanVideo15ImageToVideoPipeline"
        )


def test_qwen_image_option_arguments_reach_the_native_pipeline(
    tmp_path, monkeypatch
) -> None:
    import runpy
    import sys
    from types import SimpleNamespace

    recipe = _recipe("qwen-image-2512-diffusers-single")
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "prompt.txt").write_text("A red fox", encoding="utf-8")

    class Image:
        def save(self, path, *, format):
            Path(path).write_bytes(format.encode())

    class Pipeline:
        def __init__(self):
            self.call: dict[str, object] | None = None

        def to(self, device):
            assert device == "cuda"
            return self

        def __call__(self, **kwargs):
            self.call = kwargs
            return SimpleNamespace(images=[Image()])

    pipeline = Pipeline()
    torch = SimpleNamespace(
        bfloat16=object(),
        Generator=lambda *, device: SimpleNamespace(manual_seed=lambda seed: seed),
    )
    diffusers = SimpleNamespace(
        QwenImagePipeline=SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: pipeline
        )
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    adapter = runpy.run_path(
        str(ROOT / "adapters/image/qwen-image-2512-diffusers/qwen_image.py")
    )
    adapter["_prompt"].__globals__["_INPUT_DIR"] = inputs

    for choice, size in ((None, (1328, 1328)), ("landscape", (1664, 928))):
        selected = recipe.with_option_choices(
            {} if choice is None else {"aspect-ratio": choice}
        )
        argv = ["qwen_image.py"]
        for argument in selected.runtime.arguments:
            argv.extend((f"--{argument.name}", str(argument.value)))
        output = tmp_path / (choice or "square")
        argv.extend(("--output-dir", str(output)))
        monkeypatch.setattr(sys, "argv", argv)
        adapter["main"]()
        assert pipeline.call is not None
        assert (pipeline.call["width"], pipeline.call["height"]) == size
        assert (output / "qwen-image.png").is_file()
