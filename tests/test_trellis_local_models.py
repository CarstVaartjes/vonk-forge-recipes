"""Exercise the pinned pipelines' local loader from an unrelated job cwd."""

from __future__ import annotations

import contextlib
import json
import runpy
from pathlib import Path
from types import ModuleType
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "adapters/three-d/trellis2-native"
PATCH = runpy.run_path(str(ADAPTER / "patches/apply-local-models.py"))

# Shared verbatim loader block from both pinned pipelines/base.py sources:
# microsoft/TRELLIS.2@75fbf018 and TencentARC/Pixal3D@f7cf3842.
LOADER = """import os
import json

class Pipeline:
    @classmethod
    def from_pretrained(cls, path):
        with open(f"{path}/pipeline.json") as stream:
            args = json.load(stream)["args"]
        _models = {}
        for k, v in args['models'].items():
            if hasattr(cls, 'model_names_to_load') and k not in cls.model_names_to_load:
                continue
            try:
                _models[k] = models.from_pretrained(f"{path}/{v}")
            except Exception as e:
                _models[k] = models.from_pretrained(v)
        return _models
"""


def pipeline(tmp_path: Path, package: str) -> ModuleType:
    root = tmp_path / package
    (root / "pipelines").mkdir(parents=True)
    source = root / "pipelines/base.py"
    source.write_text(LOADER)
    PATCH["patch_tree"](root)
    module = ModuleType("local_pipeline")
    exec(compile(source.read_text(), str(source), "exec"), module.__dict__)  # noqa: S102
    return module


@pytest.mark.parametrize("package", ["trellis2", "pixal3d"])
def test_local_sub_models_never_fall_back_to_hub(tmp_path: Path, package: str) -> None:
    module = pipeline(tmp_path, package)
    model_root = tmp_path / "mounted-model"
    model_root.mkdir()
    companion = tmp_path / "companion/decoder"
    references = {"flow": "ckpts/flow", "decoder": str(companion)}
    (model_root / "pipeline.json").write_text(
        json.dumps({"args": {"models": references}})
    )
    expected = [str(model_root / "ckpts/flow"), str(companion)]
    for prefix in expected:
        Path(prefix).parent.mkdir(parents=True, exist_ok=True)
        for suffix in (".json", ".safetensors"):
            Path(prefix + suffix).write_bytes(b"local")
    hub = mock.Mock(side_effect=AssertionError("offline loader contacted Hub"))

    def load_model(prefix: str) -> str:
        if not all(
            Path(prefix + suffix).is_file() for suffix in (".json", ".safetensors")
        ):
            return hub(prefix)
        return prefix

    loader = mock.Mock(side_effect=load_model)
    module.__dict__["models"] = mock.Mock(from_pretrained=loader)
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    with contextlib.chdir(outputs), mock.patch.dict("os.environ", HF_HUB_OFFLINE="1"):
        result = module.__dict__["Pipeline"].from_pretrained(str(model_root))
        assert list(result.values()) == expected
        assert loader.call_args_list == [mock.call(prefix) for prefix in expected]
        hub.assert_not_called()
        # A missing mounted weight must identify that file, without any Hub retry.
        Path(expected[0] + ".safetensors").unlink()
        with pytest.raises(
            FileNotFoundError, match="missing local sub-model file:.*flow.safetensors"
        ):
            module.__dict__["Pipeline"].from_pretrained(str(model_root))
        hub.assert_not_called()
        # Model/ABI failures must retain their evidence, with no second load.
        Path(expected[0] + ".safetensors").write_bytes(b"local")
        loader.reset_mock(side_effect=True)
        loader.side_effect = RuntimeError("incompatible model weights")
        with pytest.raises(RuntimeError, match="incompatible model weights"):
            module.__dict__["Pipeline"].from_pretrained(str(model_root))
        loader.assert_called_once_with(expected[0])


def test_loader_patch_rejects_upstream_drift(tmp_path: Path) -> None:
    (tmp_path / "pipelines").mkdir()
    (tmp_path / "pipelines/base.py").write_text("changed upstream loader\n")
    with pytest.raises(SystemExit, match="expected one local model loader"):
        PATCH["patch_tree"](tmp_path)


def test_image_patches_both_offline_pipelines() -> None:
    dockerfile = (ADAPTER / "Dockerfile").read_text()
    assert "HF_HUB_OFFLINE=1" in dockerfile
    command = dockerfile.split("RUN python3 /opt/vonk/source/apply-local-models.py", 1)[
        1
    ].split("&&", 1)[0]
    assert "/opt/vonk/trellis2-source/trellis2" in command
    assert "/opt/vonk/pixal3d-source/pixal3d" in command


def test_trellis_decoder_uses_manifest_relative_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stage real recipe selectors with the agent's selection/path layout."""
    import shutil

    monkeypatch.syspath_prepend(str(ADAPTER))
    adapter = runpy.run_path(str(ADAPTER / "trellis2_job.py"))
    recipe = json.loads((ROOT / "recipes/trellis-2-4b-pytorch-single.json").read_text())
    target = tmp_path / "target"
    target.mkdir()
    (target / "pipeline.json").write_text(
        json.dumps({"args": {"models": {}, "image_cond_model": {"args": {}}}})
    )
    mounts: dict[str, Path] = {}
    for selection in recipe["models"]:
        if not selection["id"].startswith("sparse-decoder"):
            continue
        model = json.loads(
            (ROOT / "models" / f"{selection['model']['slug']}.json").read_text()
        )
        files = {item["id"]: item for item in model["files"]}
        for item in selection["files"]:
            mount_id = Path(item["mount"]["target"]).name
            mount = tmp_path / mount_id
            mounts[mount_id] = mount
            staged = mount / files[item["file_id"]]["path"]
            staged.parent.mkdir(parents=True, exist_ok=True)
            staged.write_bytes(b"decoder fixture")
    assert set(mounts) == {"sparse-decoder-config", "sparse-decoder-weights"}
    local = adapter["local_model_root"](
        target, mounts["sparse-decoder-config"], mounts["sparse-decoder-weights"]
    )
    try:
        config = json.loads((local / "pipeline.json").read_text())
        decoder = config["args"]["models"]["sparse_structure_decoder"]
        for suffix in (".json", ".safetensors"):
            assert Path(decoder + suffix).read_bytes() == b"decoder fixture"
        # Removing a declared companion exposes a missing local file immediately.
        weights = (
            mounts["sparse-decoder-weights"]
            / "ckpts/ss_dec_conv3d_16l8_fp16.safetensors"
        )
        weights.unlink()
        with pytest.raises(FileNotFoundError):
            Path(decoder + ".safetensors").read_bytes()
    finally:
        shutil.rmtree(local)


def test_smoke_steps_reach_all_native_samplers(tmp_path: Path, monkeypatch):
    import sys
    from types import SimpleNamespace

    class Observed(Exception):
        pass

    def sample(image, **kwargs):
        assert kwargs["pipeline_type"] == "1024_cascade"
        for key in (
            "sparse_structure_sampler_params",
            "shape_slat_sampler_params",
            "tex_slat_sampler_params",
        ):
            assert kwargs[key] == {"steps": 2}
        raise Observed

    fake_pipeline = SimpleNamespace(cuda=lambda: None, run=sample)
    fake = SimpleNamespace(
        Trellis2ImageTo3DPipeline=SimpleNamespace(
            from_pretrained=lambda _: fake_pipeline
        ),
        rembg=SimpleNamespace(BiRefNet=None),
    )
    with mock.patch.dict(
        sys.modules,
        {
            "glb_validation": SimpleNamespace(
                normalize_glb_json_padding=lambda _: None,
                validate_mesh_glb=lambda *_: None,
            ),
            "o_voxel": SimpleNamespace(),
            "trellis2": SimpleNamespace(),
            "trellis2.pipelines": fake,
        },
    ):
        module = runpy.run_path(str(ADAPTER / "trellis2_job.py"))
        main = module["main"]
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "job",
                "--output-mime",
                "model/gltf-binary",
                "--output-dir",
                str(tmp_path),
                "--sampler-steps",
                "2",
            ],
        )
        with (
            mock.patch.dict(
                main.__globals__,
                {
                    "one_input": lambda _: tmp_path / "image.png",
                    "validated_input": lambda _: "image",
                    "local_model_root": lambda *_: tmp_path / "models",
                    "shutil": SimpleNamespace(rmtree=lambda *_, **_kwargs: None),
                },
            ),
            pytest.raises(Observed),
        ):
            main()
