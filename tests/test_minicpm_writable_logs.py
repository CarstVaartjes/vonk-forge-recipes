"""Every TabbyAPI image must keep relative file logs off the read-only root."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _tabby_launchers() -> list[Path]:
    launchers = []
    for dockerfile in sorted((ROOT / "adapters").rglob("Dockerfile*")):
        if "/opt/tabbyAPI" not in dockerfile.read_text():
            continue
        # Every TabbyAPI build context must ship a launcher whose cwd is managed.
        candidates = [
            path
            for path in dockerfile.parent.iterdir()
            if path.name in {"exllamav3-serve", "tabby-serve"}
        ]
        assert len(candidates) == 1, f"uncovered TabbyAPI image: {dockerfile}"
        launchers.extend(candidates)
    assert launchers
    return launchers


@pytest.mark.parametrize(
    "launcher", _tabby_launchers(), ids=lambda path: path.parent.name
)
def test_tabby_startup_can_write_logs_without_writing_the_image(
    tmp_path: Path, launcher: Path
) -> None:
    script = """
import json, os, runpy, sys, types
from pathlib import Path
from unittest.mock import patch
common = types.ModuleType("common")
common.config_from_env = lambda env, model: {"network": {}}
common.dump_yaml = json.dumps
sys.modules["common"] = common
launcher = sys.argv[1]
sys.argv = [launcher]
def start(program, args):
    from pathlib import Path
    log = Path("logs/startup.log")
    log.parent.mkdir(exist_ok=True)
    log.write_text("startup succeeded")
    print(json.dumps({"cwd": str(Path.cwd()), "args": args}))
os.execv = start
read_text = Path.read_text
def read_template(path, *args, **kwargs):
    if path == Path("/opt/vonk/tabby-config.yml"):
        return read_text(Path(launcher).parent / "tabby-config.yml")
    return read_text(path, *args, **kwargs)
with patch.object(Path, "read_text", read_template):
    runpy.run_path(launcher, run_name="__main__")
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(launcher)],
        env={**os.environ, "XDG_CACHE_HOME": str(tmp_path)},
        capture_output=True,
        text=True,
        check=True,
    )
    launch = json.loads(result.stdout)
    state = Path(launch["cwd"])
    assert state.is_relative_to(tmp_path)
    assert (state / "logs/startup.log").read_text() == "startup succeeded"
    assert "/opt/tabbyAPI/main.py" in launch["args"]
    links = list((state / "models").iterdir())
    assert len(links) == 1
    assert links[0].readlink() == Path("/models/target")


@pytest.mark.parametrize(
    "launcher", _tabby_launchers(), ids=lambda path: path.parent.name
)
def test_tabby_launchers_select_managed_working_directory(launcher: Path) -> None:
    source = launcher.read_text()
    assert "os.chdir(state)" in source
    assert "os.chdir(TABBY)" not in source
    assert (
        'os.environ["XDG_CACHE_HOME"]' in source
        or 'os.environ.get("XDG_CACHE_HOME", "/tmp")' in source
    )
