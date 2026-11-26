"""The Kokoro-FastAPI launcher serves the creator's app from the mounted Model."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import sys
import tempfile
import types
from typing import Any

import pytest
from repo_root import ROOT

LAUNCHER = ROOT / "adapters/audio/kokoro-fastapi/kokoro-serve.py"


def _fake_app_modules(monkeypatch: pytest.MonkeyPatch, seen: dict[str, Any]) -> None:
    """The creator's modules as the launcher uses them; settings read the environment."""
    mappings: dict[str, dict[str, str]] = {"models": {"kokoro": "kokoro-v1_0"}}

    def run(app: str, **options: Any) -> None:
        seen.update(app=app, mappings=mappings, **options)

    main = types.ModuleType("api.src.main")
    main.__dict__["app"] = "app"
    compatible = types.ModuleType("api.src.routers.openai_compatible")
    compatible.__dict__["_openai_mappings"] = mappings
    routers = types.ModuleType("api.src.routers")
    routers.__dict__["openai_compatible"] = compatible
    uvicorn = types.ModuleType("uvicorn")
    uvicorn.__dict__["run"] = run
    for name, module in {
        "api": types.ModuleType("api"),
        "api.src": types.ModuleType("api.src"),
        "api.src.main": main,
        "api.src.routers": routers,
        "api.src.routers.openai_compatible": compatible,
        "uvicorn": uvicorn,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)


def test_serves_the_app_on_the_platform_port_with_weights_from_models(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    seen: dict[str, Any] = {}
    _fake_app_modules(monkeypatch, seen)
    monkeypatch.setattr(os, "chdir", lambda path: seen.update(cwd=path))
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.delenv("MODEL_DIR", raising=False)
    monkeypatch.delenv("TEMP_FILE_DIR", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pytorch-server",
            "--served-model-name",
            "voice",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
        ],
    )
    loader = importlib.machinery.SourceFileLoader("kokoro_serve", str(LAUNCHER))
    spec = importlib.util.spec_from_loader("kokoro_serve", loader)
    assert spec
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    module.main()

    assert os.environ["MODEL_DIR"] == "/models"
    assert seen["cwd"] == "/app"
    assert seen["app"] == "app"
    assert (seen["host"], seen["port"], seen["workers"]) == ("127.0.0.1", 8000, 1)
    assert seen["mappings"]["models"]["voice"] == "kokoro-v1_0"
    assert os.environ["TEMP_FILE_DIR"].startswith(str(tmp_path))
