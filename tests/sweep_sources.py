"""The source of ``Sweep``: spark_sweep/run.py and the mixin modules holding its methods."""

from __future__ import annotations

from pathlib import Path

from repo_root import ROOT

SWEEP = ROOT / "spark_sweep"
RUN_FILES = (
    "run.py",
    "sweep_adopt.py",
    "sweep_apply.py",
    "sweep_cleanup.py",
    "sweep_exit.py",
    "sweep_failure.py",
    "sweep_fleet.py",
    "sweep_loading.py",
    "sweep_platform.py",
    "sweep_reconcile.py",
    "sweep_schedule.py",
    "sweep_smoke.py",
    "sweep_takeover.py",
    "sweep_tick.py",
)


def run_paths() -> list[Path]:
    return [SWEEP / name for name in RUN_FILES]


def run_source() -> str:
    """The Sweep modules as one parseable text (one ``from __future__`` line is enough)."""
    return "\n".join(
        path.read_text(encoding="utf-8").replace(
            "from __future__ import annotations\n", ""
        )
        for path in run_paths()
    )
