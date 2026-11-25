from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import pytest
from documents import model_display, recipe_display
from repo_root import ROOT

from recipe_tooling import catalog_index

ADAPTER_ROOT = ROOT / "adapters/video/minimax-h3-modular-diffusers"
ADAPTER_PATH = ADAPTER_ROOT / "minimax_h3.py"
RECIPE_PATH = ROOT / "recipes/minimax-h3-diffusers-single.json"
MODEL_PATH = ROOT / "models/minimax-h3.json"
DIFFUSERS_REVISION = "efabd60d61c2b7aabf9f182bee6b5b6058980304"
MODEL_REVISION = "42ed227ee7df40d41602854ae760620d6eb651fe"


def _canonical_digest(path: Path) -> str:
    document = json.loads(path.read_text(encoding="utf-8"))
    content = json.dumps(
        document,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def _adapter_module():
    module = types.ModuleType("minimax_h3_adapter")
    module.__file__ = str(ADAPTER_PATH)
    exec(  # noqa: S102 - load the adapter source without importing heavy dependencies.
        compile(ADAPTER_PATH.read_text(encoding="utf-8"), str(ADAPTER_PATH), "exec"),
        module.__dict__,
    )
    return module


class MiniMaxH3AuthorityTests(unittest.TestCase):
    def test_recipe_resolves_exact_local_authorities(self) -> None:
        recipe = json.loads(RECIPE_PATH.read_text(encoding="utf-8"))
        model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
        assert recipe["models"][0]["model"]["content_sha256"] == _canonical_digest(
            MODEL_PATH
        )
        assert model["source"]["revision"] == MODEL_REVISION
        assert {item["name"] for item in recipe["runtime"]["environment"]} == {
            "HF_HUB_OFFLINE",
            "TRANSFORMERS_OFFLINE",
        }

        tags = set(recipe_display(recipe["identity"]["slug"]).tags)
        assert {"video", "audio", "multimodal", "modular-diffusers"} <= tags
        assert recipe["execution"]

    def test_license_records_excluded_territories(self) -> None:
        restrictions = model_display("minimax-h3").license.territorial_restrictions
        assert restrictions is not None
        assert {"EU", "GB", "KR", "US"} <= set(restrictions.denied_jurisdictions)

    def test_signed_source_bundle_matches_recipe(self) -> None:
        recipe = json.loads(RECIPE_PATH.read_text(encoding="utf-8"))
        source_bundle = vars(catalog_index)["source_bundle"]
        archive, _, digest = source_bundle(ADAPTER_ROOT)
        context = recipe["execution"]["build"]["context"]
        assert context["path"] == "adapters/video/minimax-h3-modular-diffusers"
        assert digest
        assert archive

    def test_container_and_adapter_forbid_hidden_runtime_downloads(self) -> None:
        dockerfile = (ADAPTER_ROOT / "Dockerfile").read_text(encoding="utf-8")
        source = ADAPTER_PATH.read_text(encoding="utf-8")
        assert f"diffusers.git@{DIFFUSERS_REVISION}" in dockerfile
        assert f'org.opencontainers.image.revision="{MODEL_REVISION}"' in dockerfile
        assert source.count("local_files_only=True") >= 4
        assert "pretrained_model_name_or_path=str(MODEL_ROOT)" in source
        assert (
            'transformer_name = "transformer_ref" if workflow == "ref2va" else "transformer"'
            in source
        )
        assert "**{transformer_name: transformer}" in source
        assert "huggingface.co" not in source
        assert "MiniMaxAI/MiniMax-H3" not in source

    def test_export_is_atomic_and_joint_audio_video_is_verified(self) -> None:
        source = ADAPTER_PATH.read_text(encoding="utf-8")
        for required in (
            '"videos"',
            '"audio"',
            '"sampling_rate"',
            "encode_video(",
            "_verify_joint_av(",
            "frame_count=args.num_frames",
            "audio.sample_rate != 32000",
            'audio.layout.name != "stereo"',
            "os.replace(temporary, destination)",
        ):
            assert required in source

    def test_output_media_contract_is_exact(self) -> None:
        module = _adapter_module()
        video_stream = types.SimpleNamespace(
            width=960,
            height=544,
            average_rate=24,
            codec_context=types.SimpleNamespace(name="h264"),
        )
        audio_stream = types.SimpleNamespace(
            codec_context=types.SimpleNamespace(
                name="aac",
                sample_rate=32_000,
                layout=types.SimpleNamespace(name="stereo"),
            )
        )

        class Container:
            streams = types.SimpleNamespace(video=[video_stream], audio=[audio_stream])

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def decode(self, *, video=None, audio=None):
                if video == 0:
                    return iter([object()] * 124)
                if audio == 0:
                    return iter([types.SimpleNamespace(samples=82_667)] * 2)
                return iter(())

        with mock.patch.dict(
            sys.modules,
            {"av": types.SimpleNamespace(open=lambda *_a, **_k: Container())},
        ):
            module._verify_joint_av(
                Path("unused.mp4"), width=960, height=544, frame_count=124
            )

        video_stream.codec_context.name = "hevc"
        with (
            mock.patch.dict(
                sys.modules,
                {"av": types.SimpleNamespace(open=lambda *_a, **_k: Container())},
            ),
            pytest.raises(RuntimeError, match="H.264 video and AAC audio"),
        ):
            module._verify_joint_av(
                Path("unused.mp4"), width=960, height=544, frame_count=124
            )

    def test_recipe_binds_each_option_to_a_typed_knob(self) -> None:
        recipe = json.loads(RECIPE_PATH.read_text(encoding="utf-8"))
        arguments = {
            argument["name"]: argument["setting"]
            for argument in recipe["runtime"]["arguments"]
        }
        knobs = recipe["settings"]["knobs"]
        for name in ("num-frames", "width", "height", "num-inference-steps"):
            assert arguments[name] == name
        assert "profile" not in knobs
        assert knobs["num-inference-steps"]["value"] == 31

    def test_recipe_declares_truthful_typed_input_slots(self) -> None:
        recipe = json.loads(RECIPE_PATH.read_text(encoding="utf-8"))
        input_contract = recipe["interfaces"][0]["input"]
        slots = {slot["id"]: slot for slot in input_contract["slots"]}
        assert set(slots) == {"prompt", "first-image", "last-image", "references"}
        assert slots["prompt"]["media_types"] == ["text/plain"]
        assert slots["prompt"]["min_files"] == 1
        assert slots["references"]["max_files"] == 12
        for slot_id in ("first-image", "last-image"):
            assert (slots[slot_id]["min_files"], slots[slot_id]["max_files"]) == (0, 1)
        assert slots["references"]["min_files"] == 0
        for slot in slots.values():
            assert slot["max_file_bytes"] <= 512 * 1024 * 1024
            assert slot["max_total_bytes"] <= 1024 * 1024 * 1024
        output = recipe["interfaces"][0]["output"]
        assert output["max_total_bytes"] == 1024**3
        assert output["slots"][0]["media_types"] == ["video/mp4"]
        assert output["slots"][0]["min_files"] == 1
        assert output["slots"][0]["max_files"] == 1


class MiniMaxH3InputContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = _adapter_module()
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.module.INPUT_ROOT = Path(self.temporary.name)  # pyright: ignore[reportAttributeAccessIssue]

    def stage(self, **slots: list[tuple[str, str]]) -> None:
        files = [
            {"slot": slot.replace("_", "-"), "name": name, "media_type": media}
            for slot, items in slots.items()
            for name, media in items
        ]
        (self.module.INPUT_ROOT / "manifest.json").write_text(
            json.dumps({"files": files}), encoding="utf-8"
        )
        for name, _media in (item for items in slots.values() for item in items):
            (self.module.INPUT_ROOT / name).write_bytes(b"x")

    def test_prompt_is_required(self) -> None:
        with pytest.raises(ValueError, match="exactly one UTF-8"):
            self.module._load_prompt()
        (self.module.INPUT_ROOT / "prompt.txt").write_text(
            "  Operator supplied synchronized scene.  ", encoding="utf-8"
        )
        assert self.module._load_prompt() == "Operator supplied synchronized scene."

    def test_keyframes_come_from_their_slots(self) -> None:
        self.stage()
        assert self.module._keyframe("first-image") is None
        self.stage(first_image=[("a.png", "image/png")])
        assert self.module._keyframe("first-image") == self.module.INPUT_ROOT / "a.png"
        assert self.module._keyframe("last-image") is None

    def stubbed_diffusers(self):
        def reference(kind: str):
            return types.SimpleNamespace(from_file=lambda path: (kind, Path(path).name))

        package = types.ModuleType("diffusers.modular_pipelines.minimax_h3")
        for name, kind in (("Image", "image"), ("Video", "video"), ("Audio", "audio")):
            setattr(package, f"MiniMaxH3{name}Reference", reference(kind))
        return mock.patch.dict(
            sys.modules,
            {
                "diffusers": types.ModuleType("diffusers"),
                "diffusers.modular_pipelines": types.ModuleType(
                    "diffusers.modular_pipelines"
                ),
                "diffusers.modular_pipelines.minimax_h3": package,
            },
        )

    def references(self, **slots: list[tuple[str, str]]) -> list[object] | None:
        self.stage(**slots)
        with self.stubbed_diffusers():
            return self.module._load_references()

    def test_references_take_their_kind_from_the_manifest_media_type(self) -> None:
        assert self.references(
            references=[
                ("3-voice.wav", "audio/wav"),
                ("1-scene.png", "image/png"),
                ("2-clip.mp4", "video/mp4"),
            ]
        ) == [
            ("image", "1-scene.png"),
            ("video", "2-clip.mp4"),
            ("audio", "3-voice.wav"),
        ]

    def test_references_refuse_a_kind_over_its_cap_and_audio_alone(self) -> None:
        images = [(f"{n}.png", "image/png") for n in range(10)]
        with pytest.raises(ValueError, match="per-modality limits"):
            self.references(references=images)
        with pytest.raises(ValueError, match="at least one image or video"):
            self.references(references=[("a.wav", "audio/wav")])

    def test_references_and_keyframes_are_exclusive(self) -> None:
        self.stage(
            references=[("r.png", "image/png")], first_image=[("a.png", "image/png")]
        )
        argv = [
            "x", "--pipeline", "text-to-video", "--output-mime", "video/mp4",
            "--num-frames", "124", "--num-inference-steps", "4", "--width", "960",
            "--height", "544", "--output-dir", self.temporary.name,
        ]  # fmt: skip
        with (
            self.stubbed_diffusers(),
            mock.patch.object(sys, "argv", argv),
            pytest.raises(ValueError, match="cannot be combined"),
        ):
            self.module.main()


if __name__ == "__main__":
    unittest.main()
