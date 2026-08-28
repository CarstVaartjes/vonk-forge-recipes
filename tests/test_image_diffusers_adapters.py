from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    script = ROOT / relative
    loader = importlib.machinery.SourceFileLoader(name, str(script))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


flash = _load(
    "qwen_image_flash_adapter",
    "adapters/image/nvidia-qwen-image-flash-diffusers/qwen_image_flash.py",
)
generation = _load(
    "qwen_image_2512_adapter",
    "adapters/image/qwen-image-2512-diffusers/qwen_image.py",
)
edit = _load(
    "qwen_image_edit_2511_adapter",
    "adapters/image/qwen-image-edit-2511-diffusers/qwen_image_edit.py",
)
layered = _load(
    "qwen_image_layered_adapter",
    "adapters/image/qwen-image-layered-diffusers/qwen_image_layered.py",
)


class _Layer:
    def __init__(self) -> None:
        self.saved: list[tuple[Path, str]] = []

    def save(self, path: Path, *, format: str) -> None:
        self.saved.append((path, format))
        path.write_bytes(b"png")


class ImageDiffusersAdapterTests(unittest.TestCase):
    def test_generation_adapters_require_one_bounded_utf8_prompt_file(self) -> None:
        for adapter in (flash, generation):
            with (
                self.subTest(adapter=adapter.__name__),
                tempfile.TemporaryDirectory() as temporary,
            ):
                inputs = Path(temporary)
                previous = adapter._INPUT_DIR
                adapter._INPUT_DIR = inputs
                try:
                    (inputs / "prompt.txt").write_text(
                        "  draw a red fox  \n", encoding="utf-8"
                    )
                    self.assertEqual(adapter._prompt(), "draw a red fox")

                    (inputs / "second.text").write_text("another", encoding="utf-8")
                    with self.assertRaisesRegex(SystemExit, "exactly one"):
                        adapter._prompt()
                finally:
                    adapter._INPUT_DIR = previous

    def test_generation_prompt_rejects_invalid_or_oversized_utf8(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            inputs = Path(temporary)
            previous = generation._INPUT_DIR
            generation._INPUT_DIR = inputs
            try:
                prompt = inputs / "prompt.txt"
                prompt.write_bytes(b"\xff")
                with self.assertRaisesRegex(SystemExit, "valid UTF-8"):
                    generation._prompt()
                prompt.write_bytes(b"x" * (generation._MAX_PROMPT_BYTES + 1))
                with self.assertRaisesRegex(SystemExit, "1..16384"):
                    generation._prompt()
            finally:
                generation._INPUT_DIR = previous

    def test_edit_adapter_separates_prompt_from_one_or_two_images(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            inputs = Path(temporary)
            previous = edit._INPUT_DIR
            edit._INPUT_DIR = inputs
            try:
                (inputs / "prompt.txt").write_text(
                    "replace the scarf", encoding="utf-8"
                )
                (inputs / "reference-1.png").write_bytes(b"png")
                self.assertEqual(edit._prompt(), "replace the scarf")
                self.assertEqual(
                    [path.name for path in edit._image_inputs()],
                    ["reference-1.png"],
                )

                (inputs / "second.text").write_text("another", encoding="utf-8")
                with self.assertRaisesRegex(SystemExit, "exactly one"):
                    edit._prompt()
                (inputs / "second.text").unlink()

                (inputs / "reference-2.webp").write_bytes(b"webp")
                self.assertEqual(len(edit._image_inputs()), 2)
                (inputs / "reference-3.jpg").write_bytes(b"jpg")
                with self.assertRaisesRegex(SystemExit, "one or two"):
                    edit._image_inputs()
            finally:
                edit._INPUT_DIR = previous

    def test_layered_adapter_requires_one_image_and_saves_every_layer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            inputs = Path(temporary) / "inputs"
            outputs = Path(temporary) / "outputs"
            inputs.mkdir()
            image = inputs / "composition.png"
            image.write_bytes(b"png")
            self.assertEqual(layered._image_input(inputs), image)

            layers = [_Layer() for _ in range(4)]
            paths = layered._save_layers(layers, outputs, 4)
            self.assertEqual(
                [path.name for path in paths],
                ["layer-00.png", "layer-01.png", "layer-02.png", "layer-03.png"],
            )
            self.assertTrue(all(path.is_file() for path in paths))
            self.assertTrue(all(layer.saved[0][1] == "PNG" for layer in layers))
            with self.assertRaisesRegex(SystemExit, "expected 3"):
                layered._save_layers(layers, outputs, 3)

    def test_recipes_declare_matching_prompt_and_memory_contracts(self) -> None:
        generation_slugs = (
            "nvidia-qwen-image-flash-diffusers-single",
            "qwen-image-2512-diffusers-single",
        )
        for slug in generation_slugs:
            recipe = json.loads((ROOT / "recipes" / f"{slug}.json").read_text())
            interface = recipe["interfaces"][0]
            self.assertEqual(interface["input"]["media_types"], ["text/plain"])
            self.assertEqual(interface["input"]["max_bytes"], 16 * 1024)
            self.assertEqual(
                interface["input"]["slots"],
                [
                    {
                        "id": "prompt",
                        "label": "Prompt",
                        "description": "UTF-8 text prompt for image generation.",
                        "media_types": ["text/plain"],
                        "extensions": [".txt"],
                        "min_files": 1,
                        "max_files": 1,
                        "max_file_bytes": 16 * 1024,
                        "max_total_bytes": 16 * 1024,
                    }
                ],
            )
            self.assertEqual(
                interface["output"]["slots"][0]["media_types"], ["image/png"]
            )
            self.assertEqual(interface["output"]["slots"][0]["min_files"], 1)
            self.assertEqual(interface["output"]["slots"][0]["max_files"], 1)
            self.assertIn(
                {"source": "inputs", "target": "/inputs", "read_only": True},
                recipe["runtime"]["security"]["mounts"],
            )

        flash_memory = json.loads(
            (ROOT / "recipes/nvidia-qwen-image-flash-diffusers-single.json").read_text()
        )["topology"]["roles"][0]["resources"]["memory"]
        self.assertEqual(flash_memory["startup_peak_bytes"], 118_000_000_000)
        self.assertEqual(flash_memory["steady_state_bytes"], 98_000_000_000)

        edit_recipe = json.loads(
            (ROOT / "recipes/qwen-image-edit-2511-diffusers-single.json").read_text()
        )
        self.assertEqual(
            edit_recipe["interfaces"][0]["input"]["media_types"],
            ["text/plain", "image/jpeg", "image/png", "image/webp"],
        )
        edit_slots = {
            slot["id"]: slot for slot in edit_recipe["interfaces"][0]["input"]["slots"]
        }
        self.assertEqual(
            (edit_slots["prompt"]["min_files"], edit_slots["prompt"]["max_files"]),
            (1, 1),
        )
        self.assertEqual(
            (edit_slots["image"]["min_files"], edit_slots["image"]["max_files"]), (1, 2)
        )

        layered_recipe = json.loads(
            (ROOT / "recipes/qwen-image-layered-diffusers-single.json").read_text()
        )
        layered_output = layered_recipe["interfaces"][0]["output"]["slots"][0]
        self.assertEqual(
            (layered_output["min_files"], layered_output["max_files"]), (4, 4)
        )


if __name__ == "__main__":
    unittest.main()
