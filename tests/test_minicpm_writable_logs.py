"""TabbyAPI must create its relative logs under the managed writable cache."""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_tabby_startup_can_write_logs_without_writing_the_image(tmp_path: Path) -> None:
    launcher = ROOT / "adapters/minicpm/vcruz305-exl3-single/exllamav3-serve"
    script = """
import json, os, runpy, sys, types
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
runpy.run_path(launcher, run_name="__main__")
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(launcher)],
        env={**os.environ, "XDG_CACHE_HOME": str(tmp_path)},
        capture_output=True,
        text=True,
        check=True,
    )
    state = tmp_path / "minicpm-tabby"
    assert (state / "logs/startup.log").read_text() == "startup succeeded"
    launch = json.loads(result.stdout)
    assert launch["cwd"] == str(state)
    assert "/opt/tabbyAPI/main.py" in launch["args"]
    assert (state / "models/MiniCPM-V-4.7").readlink() == Path("/models/target")
