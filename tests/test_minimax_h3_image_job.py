"""The MiniMax-H3-Image job: inputs, the creator's requests and a dead server."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONTEXT = ROOT / "adapters/image/minimax-h3-image-sglang"
sys.path.insert(0, str(next(CONTEXT.glob("*.whl"))))


@pytest.fixture
def job(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("h3_image_job", CONTEXT / "job.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "INPUT_ROOT", tmp_path)
    payloads = {
        "a.png": ("image", "image/png", b"first"),
        "b.png": ("image", "image/png", b"second"),
        "prompt.txt": ("prompt", "text/plain", b"Keep <Picture 1>, use <Picture 2>."),
    }
    files = []
    for name, (slot, media_type, payload) in payloads.items():
        (tmp_path / name).write_bytes(payload)
        files.append(
            {
                "slot": slot,
                "name": name,
                "media_type": media_type,
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    manifest = {
        "schema_version": 1,
        "total_bytes": sum(len(p[2]) for p in payloads.values()),
        "files": files,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return module, tmp_path


ARGS = argparse.Namespace(size="2048x2048", quality="lossless", seed=42)


def test_edit_request_sends_references_in_order_with_the_pickers_fields(job):
    module, _ = job
    prompt, images = module.load_inputs()
    assert [p.name for p in images] == ["a.png", "b.png"]
    request = module.edit_request(
        module.request_fields(ARGS, prompt), [(p.name, p.read_bytes()) for p in images]
    )
    body = request.data.decode()
    assert request.full_url.endswith("/v1/images/edits")
    assert body.index("first") < body.index("second") < body.index('name="model"')
    for field in ('name="seed"\r\n\r\n42', 'name="generator_device"\r\n\r\ncpu'):
        assert field in body


def test_generation_request_is_the_pickers_json(job):
    module, _ = job
    sent = json.loads(module.generation_request(module.request_fields(ARGS, "a")).data)
    assert sent == {
        "model": "MiniMaxAI/MiniMax-H3-Image",
        "prompt": "a",
        "size": "2048x2048",
        "quality": "lossless",
        "seed": 42,
        "generator_device": "cpu",
        "response_format": "b64_json",
        "output_format": "png",
    }


def test_a_server_that_died_fails_the_job_instead_of_waiting(job):
    module, _ = job
    dead = SimpleNamespace(poll=lambda: 1, returncode=1)
    with pytest.raises(SystemExit, match="exited with 1"):
        module.wait_healthy(dead, deadline=module.time.monotonic() + 3600)


def test_server_overrides_the_text_encoder_and_runs_in_a_scratch_cwd(job, monkeypatch):
    module, _ = job
    assert "--component-paths.text_encoder" in module.serve_command()
    seen = {}
    monkeypatch.setattr(
        module.subprocess, "Popen", lambda cmd, **kw: seen.update(kw) or object()
    )
    module.start_server("/tmp/scratch")
    assert seen["cwd"] == "/tmp/scratch"
