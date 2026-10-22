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


def test_declared_alias_reaches_the_creator_launcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name, value in {
        "VONK_LOCAL_ADDR": "192.0.2.1",
        "VONK_MASTER_ADDR": "192.0.2.1",
        "VONK_MASTER_PORT": "29500",
        "NCCL_SOCKET_IFNAME": "eth0",
        "NCCL_IB_HCA": "mlx5_0",
        "NCCL_IB_GID_INDEX": "3",
        "GLOO_SOCKET_IFNAME": "eth0",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sglang-serve",
            "--nnodes",
            "2",
            "--node-rank",
            "0",
            "--dist-init-addr",
            "VONK_MASTER_ADDR:VONK_MASTER_PORT",
            "--served-model-name",
            "mimo-v2.6-flash",
        ],
    )
    monkeypatch.delenv("SERVED_MODEL_NAME", raising=False)
    captured: list[dict[str, str]] = []

    def capture_exec(
        path: str, arguments: list[str], environment: dict[str, str]
    ) -> None:
        captured.append(environment)
        raise InterruptedError("captured creator launch")

    monkeypatch.setattr("os.execve", capture_exec)
    with pytest.raises(InterruptedError, match="captured creator launch"):
        runpy.run_path(
            str(ROOT / "adapters/mimo/mia-sglang-dual/sglang-serve"),
            run_name="__main__",
        )
    assert captured[0]["SERVED_MODEL_NAME"] == "mimo-v2.6-flash"
    assert "--served-model-name" not in captured[0].get("EXTRA_SGLANG_ARGS", "")
