"""One version owner reaches source-tree and installed consumers alike.

The committed ``_version.py`` holds the major and its placeholder; publication
stamps the computed version, and the stamped version reaches the built wheel.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from zipfile import ZipFile

from repo_root import ROOT

from recipe_tooling.contract_version import stamp
from vonk_forge_contracts import CONTRACT_MAJOR, CONTRACT_VERSION, __version__

OWNER = "src/vonk_forge_contracts/_version.py"


def test_contract_exports_the_owned_version() -> None:
    assert CONTRACT_VERSION == __version__ == f"{CONTRACT_MAJOR}.0.0"
    project = tomllib.loads((ROOT / "contracts/pyproject.toml").read_text())
    assert "version" not in project["project"]
    assert project["tool"]["hatch"]["version"]["path"] == OWNER


def test_source_tree_sweep_starts_without_installed_contracts(tmp_path: Path) -> None:
    # The sweep runs the contracts from this tree with no package installed;
    # an empty working directory keeps uv from layering a local virtualenv.
    env = {key: value for key, value in os.environ.items() if key != "VIRTUAL_ENV"}
    result = subprocess.run(
        [str(ROOT / "tools/sweep-recipes"), "--help"],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-2000:]


def test_stamped_version_reaches_built_wheel_and_external_consumer(
    tmp_path: Path,
) -> None:
    source = tmp_path / "contracts"
    shutil.copytree(ROOT / "contracts", source)
    owner = source / OWNER
    bumped = f"{CONTRACT_MAJOR}.73.1"
    owner.write_text(stamp(owner.read_text(), bumped))
    subprocess.run(
        [
            "uv",
            "build",
            "--project",
            str(source),
            "--wheel",
            "--out-dir",
            str(tmp_path / "dist"),
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )
    (wheel,) = (tmp_path / "dist").glob("*.whl")
    with ZipFile(wheel) as archive:
        (package_metadata,) = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        assert f"Version: {bumped}\n" in archive.read(package_metadata).decode()
    env = os.environ | {"PYTHONPATH": str(wheel)}
    consumer = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from vonk_forge_contracts import CONTRACT_VERSION, CONTRACT_MAJOR, __version__; "
                "print(CONTRACT_VERSION, CONTRACT_MAJOR, __version__)"
            ),
        ],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert consumer.stdout.strip() == f"{bumped} {CONTRACT_MAJOR} {bumped}"
