"""The creator-selected SGLang uses a msgspec record, not a dataclass."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
VERIFIER = ROOT / "adapters/mimo/mia-sglang-dual/verify-runtime.py"


@pytest.mark.parametrize("missing", [None, "enable_multimodal"])
def test_image_verifier_checks_the_pinned_record_fields(
    monkeypatch: pytest.MonkeyPatch, missing: str | None
) -> None:
    server = ModuleType("sglang.srt.server_args")
    server_args = SimpleNamespace()
    monkeypatch.setattr(server, "ServerArgs", server_args, raising=False)
    msgspec = ModuleType("msgspec")
    fields: set[str] = set()

    def record_fields(record: object) -> list[SimpleNamespace]:
        assert record is server_args
        return [SimpleNamespace(name=name) for name in fields]

    monkeypatch.setattr(
        msgspec, "structs", SimpleNamespace(fields=record_fields), raising=False
    )
    torch = ModuleType("torch")
    monkeypatch.setattr(torch, "version", SimpleNamespace(cuda="13.0"), raising=False)
    monkeypatch.setitem(sys.modules, "sglang.srt.server_args", server)
    monkeypatch.setitem(sys.modules, "msgspec", msgspec)
    monkeypatch.setitem(sys.modules, "torch", torch)
    verifier = runpy.run_path(str(VERIFIER))
    fields.update(verifier["REQUIRED_SERVER_FIELDS"])
    if missing:
        fields.remove(missing)
    plan = SimpleNamespace(
        stdout='{"args":["--enable-multimodal","--speculative-algorithm","EAGLE"]}'
    )
    with (
        patch("platform.machine", return_value="aarch64"),
        patch("os.path.isfile", return_value=True),
        patch("os.access", return_value=True),
        patch("shutil.which", return_value="/usr/bin/ffprobe"),
        patch("subprocess.run", return_value=plan),
        patch("importlib.import_module", return_value=object()),
    ):
        if missing:
            with pytest.raises(SystemExit, match=missing):
                verifier["main"]()
        else:
            verifier["main"]()
