"""A new Controller call in a sweep loop fails unless it is explained; old lines pass."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
from pathlib import Path

from added_diff import added_diff

ROOT = Path(__file__).resolve().parents[1]
LOADER = importlib.machinery.SourceFileLoader(
    "check_serial_reads", str(ROOT / "tools/check-serial-reads")
)
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
assert SPEC is not None
guard = importlib.util.module_from_spec(SPEC)
sys.modules[LOADER.name] = guard
LOADER.exec_module(guard)

SERIAL = "for load in loads:\n    self.vk.run('profile', 'progress')\n"
EXPLAINED = (
    "for alias in aliases:\n"
    "    # sequential: every removal edits the one shared profile\n"
    "    self.vk.run('profile', 'remove', alias)\n"
)
FANNED_OUT = "replies = gather({k: partial(self.vk.run, 'x') for k in keys}, ex)\n"


def test_an_added_serial_call_fails_and_a_reason_or_fan_out_passes() -> None:
    assert guard.scan(SERIAL, {2}) == [2]
    assert guard.scan(EXPLAINED, {3}) == []
    assert guard.scan(FANNED_OUT, {1}) == []
    assert guard.scan("self.vk.run('x')\n", {1}) == []  # not in a loop


def test_only_added_lines_of_sweep_modules_count(tmp_path: Path) -> None:
    assert guard.scan(SERIAL, {1}) == []  # the call line is old
    for name in ("spark_sweep/run.py", "tools/helper.py"):
        (tmp_path / name).parent.mkdir(exist_ok=True)
        (tmp_path / name).write_text(SERIAL)
    diff = added_diff({"spark_sweep/run.py": SERIAL, "tools/helper.py": SERIAL})
    assert [p.split(":")[0] for p in guard.check(tmp_path, diff)] == [
        "spark_sweep/run.py"
    ]
