"""The Python gates must stay configured, deterministic and wired to CI.

Every finding here changes which diagnostics exist, so a dropped setting makes
the reviewed baseline disagree with the checker without any source change. That
already happened once: the repository shipped the pyright baseline but no
`[tool.pyright]` section, so CI resolved pyright's defaults instead.
"""

from __future__ import annotations

import re
import tomllib

from repo_root import ROOT

PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
WORKFLOWS = tuple(sorted((ROOT / ".github" / "workflows").glob("*.yml")))


def test_pyright_configuration_is_pinned() -> None:
    pyright = PYPROJECT.get("tool", {}).get("pyright")
    assert isinstance(pyright, dict), "pyproject.toml has no [tool.pyright] section"
    # Both settings change the diagnostic set, so the baseline is only
    # meaningful while they are pinned.
    assert pyright.get("pythonVersion") == "3.14"
    assert pyright.get("typeCheckingMode") == "standard"
    extra_paths = pyright.get("extraPaths")
    assert isinstance(extra_paths, list)
    assert "contracts/src" in extra_paths


def test_ruff_configuration_is_pinned_and_documented() -> None:
    ruff = PYPROJECT.get("tool", {}).get("ruff")
    assert isinstance(ruff, dict), "pyproject.toml has no [tool.ruff] section"
    # The version has one owner, uv.lock; a second pin here would drift from it.
    assert "required-version" not in ruff
    # CI runs tools/ launchers with the runner's python3 (3.12), so the formatter must keep
    # emitting 3.12 syntax (no `except A, B:`).
    assert ruff.get("target-version") == "py312"


def test_adapters_are_excluded_from_both_gates() -> None:
    """Adapter code follows its upstream recipe, not our style."""

    assert "adapters" in PYPROJECT["tool"]["ruff"].get("exclude", [])
    assert PYPROJECT["tool"]["ruff"].get("force-exclude") is True
    assert "adapters" in PYPROJECT["tool"]["pyright"].get("exclude", [])


def test_every_python_gate_runs_in_ci() -> None:
    checks = {
        "lint": re.compile(r"ruff check --force-exclude -- \."),
        "format": re.compile(r"ruff format --check --force-exclude -- \."),
        "types": re.compile(r"scripts/check-python-types"),
    }
    for name, pattern in checks.items():
        assert any(
            pattern.search(path.read_text(encoding="utf-8")) for path in WORKFLOWS
        ), f"no workflow runs the {name} gate"


def test_extensionless_entry_points_stay_inside_every_gate() -> None:
    """Ruff's own file selection, not a shebang scan, owns the entry points."""

    ruff = PYPROJECT["tool"]["ruff"]
    assert "tools/*" in ruff["extend-include"]
    assert "scripts/check-python-types" in ruff["extend-include"]
    assert "tools/*.*" in ruff["extend-exclude"]
    assert (ROOT / "tools" / "build-catalog-index").is_file()


def test_producer_test_environment_has_one_owner() -> None:
    """No workflow pins a producer test dependency itself.

    The refresh workflow once installed its own shorter list, so the full-suite step
    died at collection on `import PIL` for every refresh. The one list is the
    `producer-tests` dependency group in `pyproject.toml` (locked in `uv.lock`); a
    workflow installs it with `uv sync --frozen` (default groups).
    """

    group = " ".join(PYPROJECT["dependency-groups"]["producer-tests"])
    for package in ("jsonschema", "numpy", "trimesh", "pytest-xdist"):
        assert package in group
    for path in WORKFLOWS:
        text = path.read_text(encoding="utf-8")
        for copied in ("pytest", "jsonschema", "Pillow", "trimesh", "numpy"):
            pin = re.search(
                rf"['\" ]{copied}[\w-]*\s*(==|>=|<=|~=|<|>)", text, re.IGNORECASE
            )
            assert pin is None, f"{path.name} pins {pin and pin.group(0)} itself"
    for name in ("validate.yml", "refresh-upstream.yml", "publish.yml"):
        text = (ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")
        assert "uv sync --frozen" in text, name
