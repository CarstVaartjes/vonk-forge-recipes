"""A package metadata bump must reach installed consumers without source edits."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from zipfile import ZipFile

from vonk_forge_contracts import CONTRACT_MAJOR, CONTRACT_VERSION, __version__

ROOT = Path(__file__).resolve().parents[1]


def test_contract_exports_installed_package_version() -> None:
    from importlib.metadata import version

    expected = version("vonk-forge-public-contracts")
    assert CONTRACT_VERSION == __version__ == expected
    assert CONTRACT_MAJOR == int(expected.split(".", 1)[0])
    assert (
        expected
        == tomllib.loads((ROOT / "contracts/pyproject.toml").read_text())["project"][
            "version"
        ]
    )


def test_metadata_only_bump_reaches_built_wheel_and_external_consumer(
    tmp_path: Path,
) -> None:
    source = tmp_path / "contracts"
    shutil.copytree(ROOT / "contracts", source)
    metadata = source / "pyproject.toml"
    current = tomllib.loads(metadata.read_text())["project"]["version"]
    bumped = "73.2.1"
    metadata.write_text(
        metadata.read_text().replace(f'version = "{current}"', f'version = "{bumped}"')
    )
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
            "from vonk_forge_contracts import CONTRACT_VERSION, CONTRACT_MAJOR, __version__; "
            "print(CONTRACT_VERSION, CONTRACT_MAJOR, __version__)",
        ],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert consumer.stdout.strip() == f"{bumped} 73 {bumped}"
