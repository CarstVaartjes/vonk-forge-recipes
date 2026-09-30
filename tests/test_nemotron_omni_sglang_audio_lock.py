from __future__ import annotations

import runpy
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
VERIFIER = (
    ROOT
    / "adapters/nvidia/nemotron-3-nano-omni-30b-a3b-bf16-nvidia-sglang-single/verify-runtime.py"
)


class NemotronOmniSglangAudioLockTests(unittest.TestCase):
    def test_image_verifier_rejects_an_unusable_audio_dependency(self) -> None:
        for dependency in ("numpy", "librosa"):
            with self.subTest(dependency=dependency):

                def import_dependency(name: str, broken: str = dependency) -> object:
                    if name == broken:
                        raise ImportError(f"broken {name}")
                    return object()

                with (
                    patch("os.access", return_value=True),
                    patch("importlib.util.find_spec", return_value=object()),
                    patch("importlib.import_module", side_effect=import_dependency),
                    self.assertRaisesRegex(SystemExit, f"broken {dependency}"),
                ):
                    runpy.run_path(str(VERIFIER), run_name="__main__")

    def test_image_verifier_accepts_the_importable_audio_stack(self) -> None:
        with (
            patch("os.access", return_value=True),
            patch("importlib.util.find_spec", return_value=object()),
            patch("importlib.import_module", return_value=object()),
        ):
            runpy.run_path(str(VERIFIER), run_name="__main__")


if __name__ == "__main__":
    unittest.main()
