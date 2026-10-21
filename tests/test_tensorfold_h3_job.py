from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONTEXT = ROOT / "adapters/video/drowzeys-fasth3-tensorfold"
sys.path.insert(0, str(next(CONTEXT.glob("*.whl"))))
Manifest = importlib.import_module(
    "vonk_agent_protocol.job_inputs"
).RecipeJobInputManifest
SPEC = importlib.util.spec_from_file_location("h3_job", CONTEXT / "comfyui_job.py")
assert SPEC and SPEC.loader
JOB = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(JOB)


def test_typed_prompt_drives_graph_and_changed_bytes_are_rejected(
    tmp_path, monkeypatch
):
    data = b"Waves roll onto a sunny beach with gentle surf."
    (tmp_path / "PROMPT.TXT").write_bytes(data)
    manifest = Manifest.model_validate_json(
        json.dumps(
            {
                "schema_version": 1,
                "total_bytes": len(data),
                "files": [
                    {
                        "slot": "prompt",
                        "name": "PROMPT.TXT",
                        "media_type": "text/plain",
                        "size_bytes": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(),
                    }
                ],
            }
        )
    )
    (tmp_path / "manifest.json").write_text(manifest.model_dump_json())
    monkeypatch.setattr(JOB, "INPUT_ROOT", tmp_path)
    document = json.loads((CONTEXT / "workflows/fasth3.json").read_text())
    parsed = JOB.load_input_manifest()
    graph = JOB.substitute(
        document["prompt"],
        JOB.workflow_replacements(
            document, JOB.input_names(document, parsed), 42, parsed
        ),
    )
    assert graph["6"]["inputs"]["prompt"] == data.decode()
    assert graph["8"]["inputs"]["noise_seed"] == 42
    (tmp_path / "PROMPT.TXT").write_bytes(b"X" * len(data))
    with pytest.raises(ValueError, match="digest"):
        JOB.load_input_manifest()


def test_model_view_reuses_cache_and_repairs_missing_link(tmp_path, monkeypatch):
    document = json.loads((CONTEXT / "workflows/fasth3.json").read_text())
    mounts = tmp_path / "mounts"
    for item in document["models"]:
        p = mounts / item["artifact_id"] / item["filename"]
        p.parent.mkdir(parents=True)
        p.write_bytes(b"immutable checkpoint")
    monkeypatch.setattr(JOB, "MODEL_MOUNT_ROOT", mounts)
    view = tmp_path / "cache"
    JOB.link_models(document, view)
    target = view / "diffusion_models" / document["models"][0]["filename"]
    derived = target.with_name(target.name + ".tf-int8")
    derived.write_bytes(b"completed native cache")
    target.unlink()
    JOB.link_models(document, view)
    JOB.link_models(document, view)
    assert target.read_bytes() == b"immutable checkpoint"
    assert derived.read_bytes() == b"completed native cache"
    assert not list(mounts.rglob("*.tf-int8"))
    target.unlink()
    target.symlink_to(tmp_path / "unreviewed")
    with pytest.raises(ValueError, match="immutable identity"):
        JOB.link_models(document, view)


def test_video_contract_rejects_missing_audio(tmp_path, monkeypatch):
    contract = json.loads((CONTEXT / "workflows/fasth3.json").read_text())["result"]
    video = {
        "codec_type": "video",
        "codec_name": "h264",
        "width": 864,
        "height": 480,
        "avg_frame_rate": "24/1",
        "nb_read_frames": "124",
        "duration": str(124 / 24),
    }
    probe = {
        "format": {"format_name": "mov,mp4"},
        "streams": [video, {"codec_type": "audio"}],
    }
    monkeypatch.setattr(
        JOB.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=json.dumps(probe))
    )
    JOB.validate_mp4(tmp_path / "result.mp4", contract)
    probe["streams"].pop()
    with pytest.raises(RuntimeError, match="audio stream"):
        JOB.validate_mp4(tmp_path / "result.mp4", contract)


def test_smoke_frame_override_updates_graph_and_output_assertion(monkeypatch):
    document = json.loads((CONTEXT / "workflows/fasth3.json").read_text())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "job",
            "--workflow",
            "fasth3.json",
            "--workflow-sha256",
            "unused",
            "--output-mime",
            "video/mp4",
            "--seed",
            "0",
            "--num-frames",
            "120",
        ],
    )
    monkeypatch.setattr(JOB, "load_workflow", lambda *_: document)
    monkeypatch.setattr(JOB, "load_input_manifest", lambda: None)

    class Observed(Exception):
        pass

    def inspect_graph(updated, manifest):
        assert updated["prompt"]["6"]["inputs"]["length"] == 120
        assert updated["result"]["video"]["frames"] == 120
        assert updated["prompt"]["10"]["inputs"]["steps"] == 8
        raise Observed

    monkeypatch.setattr(JOB, "input_names", inspect_graph)
    with pytest.raises(Observed):
        JOB.run_job()
