"""The Python gates must stay configured, deterministic and wired to CI.

Every finding here changes which diagnostics exist, so a dropped setting makes
the reviewed baseline disagree with the checker without any source change. That
already happened once: the repository shipped the pyright baseline but no
`[tool.pyright]` section, so CI resolved pyright's defaults instead.
"""

from __future__ import annotations

import re
import runpy
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
WORKFLOWS = tuple(sorted((ROOT / ".github" / "workflows").glob("*.yml")))


def test_pyright_configuration_is_pinned() -> None:
    pyright = PYPROJECT.get("tool", {}).get("pyright")
    assert isinstance(pyright, dict), "pyproject.toml has no [tool.pyright] section"
    # Both settings change the diagnostic set, so the baseline is only
    # meaningful while they are pinned.
    assert pyright.get("pythonVersion") == "3.14"
    assert pyright.get("typeCheckingMode") == "basic"
    extra_paths = pyright.get("extraPaths")
    assert isinstance(extra_paths, list) and "contracts/src" in extra_paths


def test_ruff_configuration_is_pinned_and_documented() -> None:
    ruff = PYPROJECT.get("tool", {}).get("ruff")
    assert isinstance(ruff, dict), "pyproject.toml has no [tool.ruff] section"
    assert ruff.get("required-version") == "==0.16.1"
    # Adapter sources are executed by the 3.12 interpreter inside the pinned
    # upstream images, so the formatter must keep emitting 3.12 syntax.
    assert ruff.get("target-version") == "py312"


def test_vendored_trees_stay_excluded_from_both_gates() -> None:
    ruff = PYPROJECT["tool"]["ruff"]
    excluded = set(ruff.get("exclude", []))
    assert {
        "adapters/deepseek/mia-vllm/encoding/encoding_dsv4.py",
        "adapters/deepseek/mia-vllm-vision/encoding/encoding_dsv4.py",
        "adapters/llm/ui-mate-vllm/agents",
    } <= excluded
    # The exclusion list must not swallow the rest of the adapter tree.
    assert not [entry for entry in excluded if entry in {"adapters", "adapters/**"}]


def test_every_python_gate_runs_in_ci() -> None:
    # Every gate is named through its wrapper. A bare `ruff check .` or
    # `ruff format --check .` resolves files by extension and silently skips
    # the extensionless executables, so requiring the wrapper is the point.
    checks = {
        "lint": re.compile(r"tools/check-python-lint"),
        "format": re.compile(r"tools/check-python-format"),
        "types": re.compile(r"scripts/check-python-types"),
    }
    for name, pattern in checks.items():
        assert any(
            pattern.search(path.read_text(encoding="utf-8")) for path in WORKFLOWS
        ), f"no workflow runs the {name} gate"


def _entry_points() -> set[str]:
    """Return the tracked extensionless Python executables, from the shebang."""

    listed = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    entry_points: set[str] = set()
    for name in listed.stdout.splitlines():
        if not name or "." in name:
            continue
        path = ROOT / name
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as handle:
            first_line = handle.readline()
        if first_line.startswith("#!") and "python" in first_line:
            entry_points.add(name)
    return entry_points


def test_extensionless_entry_points_stay_inside_every_gate() -> None:
    """A gate that regresses to a bare `.` scan must fail, not pass quietly."""

    entry_points = _entry_points()
    assert "tools/build-catalog-index" in entry_points

    # The type gate is importable, so its discovery is checked directly rather
    # than by matching its source text.
    types_tool = runpy.run_path(str(ROOT / "scripts/check-python-types"))
    assert set(types_tool["_entry_points"]()) == entry_points

    # A shell wrapper cannot be imported, so its discovery contract is asserted
    # on the two commands that implement it.
    for name in ("tools/check-python-lint", "tools/check-python-format"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert "git ls-files" in text, f"{name} no longer discovers entry points"
        assert "'#!'*python*" in text, f"{name} no longer matches the shebang"
