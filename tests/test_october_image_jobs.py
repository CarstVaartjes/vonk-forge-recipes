"""Exercise job inputs and the actual pipeline calls without loading GPU weights."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
WHEEL = next(
    (ROOT / "adapters/october-2026/qwen-image-2-1-turbo-diffusers-single").glob("*.whl")
)
sys.path.insert(0, str(WHEEL))
RecipeJobInputManifest = importlib.import_module(
    "vonk_agent_protocol.job_inputs"
).RecipeJobInputManifest


@pytest.fixture(
    params=[
        "qwen-image-2-1-turbo-diffusers-single",
        "flux-3-image-klein-pytorch-single",
    ]
)
def job(request, tmp_path, monkeypatch):
    path = ROOT / "adapters/october-2026" / request.param / "job.py"
    spec = importlib.util.spec_from_file_location("october_job", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "INPUT_ROOT", tmp_path)
    payload = b"A red bicycle by a canal."
    (tmp_path / "PROMPT.TXT").write_bytes(payload)
    manifest = RecipeJobInputManifest.model_validate_json(
        json.dumps(
            {
                "schema_version": 1,
                "total_bytes": len(payload),
                "files": [
                    {
                        "slot": "prompt",
                        "name": "PROMPT.TXT",
                        "media_type": "text/plain",
                        "size_bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                ],
            }
        )
    )
    (tmp_path / "manifest.json").write_text(manifest.model_dump_json())
    return module, tmp_path, request.param


def test_manifest_selects_uppercase_prompt_and_checks_its_bytes(job):
    module, inputs, _ = job
    assert module._load_prompt() == "A red bicycle by a canal."
    (inputs / "PROMPT.TXT").write_bytes(b"A bad bicycle by a canal.")
    with pytest.raises(SystemExit, match="digest"):
        module._load_prompt()


def test_undeclared_inputs_are_not_silently_consumed(job):
    module, inputs, _ = job
    (inputs / "other.txt").write_text("unreviewed prompt")
    with pytest.raises(SystemExit, match="do not match"):
        module._load_prompt()


def test_pipeline_receives_offline_assets_and_checkpoint_sampling(job, monkeypatch):
    module, inputs, slug = job
    output = inputs.parent / "outputs"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "job",
            "--pipeline",
            "text-to-image",
            "--output-mime",
            "image/png",
            "--output-dir",
            str(output),
        ],
    )
    image = Mock()
    if slug.startswith("qwen"):
        pipe = Mock(return_value=SimpleNamespace(images=[image]))
        factory = Mock()
        factory.from_pretrained.return_value.to.return_value = pipe
        generator = Mock()
        monkeypatch.setitem(
            sys.modules, "torch", SimpleNamespace(bfloat16="bf16", Generator=generator)
        )
        monkeypatch.setitem(
            sys.modules, "diffusers", SimpleNamespace(QwenImage21Pipeline=factory)
        )
        module.main()
        assert factory.from_pretrained.call_args.kwargs["local_files_only"] is True
        call = pipe.call_args.kwargs
        assert call["use_kv_cache"] is True
        assert "num_inference_steps" not in call and "sigmas" not in call
    else:
        models = Mock()
        generate = Mock(return_value=(image, {}))
        monkeypatch.setitem(sys.modules, "flux3", SimpleNamespace())
        monkeypatch.setitem(
            sys.modules,
            "flux3.generate",
            SimpleNamespace(Models=models, generate=generate),
        )
        module.main()
        assert models.call_args.args[:2] == (
            {"distilled": "/models/flux-3-image-klein"},
            "/models/pup",
        )
        assert generate.call_args.kwargs["pup"] is True
        assert generate.call_args.kwargs["steps"] == 16
    image.save.assert_called_once_with(output / "image.png", format="PNG")


@pytest.mark.parametrize("mode", ["RGBA", "RGB", "L"])
def test_qwen_qualification_requires_native_rgba_png(mode: str) -> None:
    # The pinned QwenImage21 VAE decodes four channels; preserve its alpha.
    # https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/blob/fc13ba1986fcb905854dc8de2f52eee395f49b5b/vae/config.json
    import io

    Image = pytest.importorskip("PIL.Image")

    from qualification.definitions_loader import load_definitions
    from spark_sweep.job_assertions import Output, evaluate
    from spark_sweep.smoke import AssertionFailed

    assertions = load_definitions()["recipes"][
        "vonk-forge/qwen-image-2-1-turbo-diffusers-single"
    ]["assertions"]
    content = io.BytesIO()
    Image.new(mode, (1024, 1024)).save(content, format="PNG")
    output = Output("image.png", "image/png", content.getvalue())
    if mode == "RGBA":
        assert evaluate(assertions, [output]) == {}
    else:
        with pytest.raises(AssertionFailed, match="color_type"):
            evaluate(assertions, [output])
    with pytest.raises(AssertionFailed):
        evaluate(assertions, [Output("image.png", "image/png", b"broken PNG")])
