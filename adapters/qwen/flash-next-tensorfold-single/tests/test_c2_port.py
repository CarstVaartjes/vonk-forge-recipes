"""Focused CPU regressions for the bounded Flash Next image and draft settings."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from tensorfold.families.qwen4_exp.cuda.weights import draft_token_ids, draft_vocab_setting
from tensorfold.vision.images import ImageInputError, ImageLimits, split_images
from tensorfold.vision.qwen_cuda import image_setting
from tensorfold.vision.qwen_processing import visual_tokens_per_image


class ImageEnvelopeTests(unittest.TestCase):
    def test_image_options_refuse_values_outside_the_reviewed_envelope(self):
        with patch.dict(os.environ, {"TENSORFOLD_MAX_IMAGES": "4", "TENSORFOLD_IMAGE_TOKENS": "4096"}):
            self.assertEqual(image_setting("TENSORFOLD_MAX_IMAGES", 4, 1, 4), 4)
            self.assertEqual(image_setting("TENSORFOLD_IMAGE_TOKENS", 4096, 4096, 4096), 4096)
        for name, value, low, high in (
            ("TENSORFOLD_MAX_IMAGES", "5", 1, 4),
            ("TENSORFOLD_IMAGE_TOKENS", "8192", 4096, 4096),
            ("TENSORFOLD_MAX_IMAGES", "4.0", 1, 4),
        ):
            with self.subTest(name=name, value=value), patch.dict(os.environ, {name: value}):
                with self.assertRaises(ValueError):
                    image_setting(name, low, low, high)

    def test_visual_tokens_are_shared_and_each_image_is_capped(self):
        self.assertEqual(visual_tokens_per_image(4096, 4, 4096), 1024)
        self.assertEqual(visual_tokens_per_image(8192, 2, 4096), 4096)
        self.assertEqual(visual_tokens_per_image(8192, 8, 4096), 1024)
        with self.assertRaises(ValueError):
            visual_tokens_per_image(4, 8)

    def test_request_image_count_uses_the_selected_limit(self):
        limits = ImageLimits(max_images=4, max_total_encoded_bytes=64 * 1024**2,
                             max_total_pixels=128 * 1024**2)
        part = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}
        prompt, sources = split_images([{"role": "user", "content": [part.copy() for _ in range(4)]}],
                                       limits=limits)
        self.assertEqual(len(sources), 4)
        self.assertEqual(len(prompt[0]["content"]), 4)
        with self.assertRaises(ImageInputError):
            split_images([{"role": "user", "content": [part.copy() for _ in range(5)]}], limits=limits)


class DraftVocabularyTests(unittest.TestCase):
    def test_only_reviewed_language_choices_are_accepted(self):
        for value in ("", "default", "zh", "ja", "zh,ja"):
            with self.subTest(value=value), patch.dict(os.environ, {"TENSORFOLD_DRAFT_VOCAB": value}):
                self.assertEqual(draft_vocab_setting(), value or "default")
        for value in ("fr", "zh,fr", "zh,zh", "full", "/tmp/tokens.txt"):
            with self.subTest(value=value), patch.dict(os.environ, {"TENSORFOLD_DRAFT_VOCAB": value}):
                with self.assertRaises(ValueError):
                    draft_vocab_setting()

    def test_language_assets_extend_the_default_without_invalid_ids(self):
        default = draft_token_ids("default")
        for language in ("zh", "ja"):
            with self.subTest(language=language):
                selected = draft_token_ids(language)
                self.assertGreater(len(selected), len(default))
                self.assertTrue(set(default).issubset(set(selected)))
                self.assertTrue((selected[1:] > selected[:-1]).all())
                self.assertGreaterEqual(int(selected[0]), 0)
                self.assertLess(int(selected[-1]), 300_000)


if __name__ == "__main__":
    unittest.main()
