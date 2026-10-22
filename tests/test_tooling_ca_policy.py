"""One uv owner and current trust for CI's TLS clients, outside recipe identity."""

from __future__ import annotations

import re
import subprocess
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SETUP = ".github/actions/setup-tooling/action.yml"
UV_PIN = re.compile(
    r"\buv(?:x|[_-]version)?[\"' \t]*(?:==|=|@|:)[\"' \t]*v?\d+\.\d+\.\d+"
    r"|\buv(?:/v?|/releases/download/v?)\d+\.\d+\.\d+",
    re.IGNORECASE,
)


CI_PIN = re.compile(r'^        version: "(\d+\.\d+\.\d+)"$', re.MULTILINE)


def uv_pin_problems(sources: list[tuple[str, str]], version: str) -> list[str]:
    problems = []
    for name, text in sources:
        if name == "pyproject.toml":
            continue
        if name == SETUP:
            # The setup action owns CI's exact uv; nothing else may name it.
            if CI_PIN.findall(text) != [version] or UV_PIN.search(text):
                problems.append(name)
            continue
        if version in text or UV_PIN.search(text):
            problems.append(name)
        if "uses: astral-sh/setup-uv@" in text:
            problems.append(name)
    return sorted(set(problems))


def test_uv_has_one_exact_owner() -> None:
    # CI installs exactly the setup action's uv; local tools accept that
    # release or newer, so a newer developer uv never blocks a gate.
    version = CI_PIN.findall((ROOT / SETUP).read_text())
    assert len(version) == 1
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert project["tool"]["uv"]["required-version"] == f">={version[0]}"
    pin = "==" + version[0]
    paths = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()
    sources = [
        (name, (ROOT / name).read_text())
        for name in paths
        if name.startswith((".github/", "tools/", "scripts/", "contracts/"))
        and (ROOT / name).is_file()
        and (ROOT / name).suffix not in {".png", ".whl", ".pyc"}
    ]
    assert not uv_pin_problems(sources, pin.removeprefix("=="))
    assert (ROOT / SETUP).is_file()
    for path in (ROOT / ".github/workflows").glob("*.yml"):
        assert "astral-sh/setup-uv@" not in path.read_text()


@pytest.mark.parametrize(
    "name,text",
    [
        (
            ".github/workflows/test.yml",
            'uses: astral-sh/setup-uv@sha\n  version: "9.8.7"',
        ),
        ("tools/bootstrap", "uv==9.8.7"),
        ("tools/bootstrap", 'UV_VERSION="9.8.7"'),
        ("tools/versions.json", '{"uv": "9.8.7"}'),
        ("tools/Dockerfile", "FROM ghcr.io/astral-sh/uv:9.8.7"),
        (
            "tools/install",
            "https://github.com/astral-sh/uv/releases/download/9.8.7/uv.tar.gz",
        ),
        (
            SETUP,
            'uses: astral-sh/setup-uv@sha\n        version: "9.8.7"',
        ),
        (
            SETUP,
            'uses: astral-sh/setup-uv@sha\n        version: "1.2.3"\n'
            '        version: "1.2.3"',
        ),
    ],
)
def test_guard_rejects_independent_uv_pins(name: str, text: str) -> None:
    assert uv_pin_problems([(name, text)], "1.2.3") == [name]


def test_ci_refreshes_trust_before_installing_tls_tools() -> None:
    action = (ROOT / SETUP).read_text()
    assert action.index("apt-get update") < action.index("apt-get install")
    assert "--only-upgrade ca-certificates" in action
    assert action.index("update-ca-certificates") < action.index(
        "uses: astral-sh/setup-uv@"
    )
    assert "-r .github/requirements-tooling.txt" in action
    for variable in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
        assert f'echo "{variable}=$bundle" >> "$GITHUB_ENV"' in action
    pins = (ROOT / ".github/requirements-tooling.txt").read_text().splitlines()
    assert {
        line.split("==")[0] for line in pins if line and not line.startswith(("#", " "))
    } == {
        "certifi",
        "pip",
    }
    assert all(
        "==" in line for line in pins if line and not line.startswith(("#", " "))
    )
    assert "--require-hashes" in action


def test_remote_smoke_uses_ci_bundle_without_changing_recipe_image() -> None:
    workflow = (ROOT / ".github/workflows/recipe-images.yml").read_text()
    assert (
        '--with-requirements "$GITHUB_WORKSPACE/.github/requirements-platform-tooling.txt"'
        in workflow
    )
    assert 'cp "$SSL_CERT_FILE" "$RUNNER_TEMP/upload/ca-bundle.crt"' in workflow
    assert "--env SSL_CERT_FILE=/tmp/ca-bundle.crt" in workflow
    assert "--env REQUESTS_CA_BUNDLE=/tmp/ca-bundle.crt" in workflow
    assert (
        '--volume "$RUNNER_TEMP/plan/ca-bundle.crt:/tmp/ca-bundle.crt:ro"' in workflow
    )
    for job in ("copy", "build", "resolve"):
        section = re.split(r"\n  [a-z][a-z-]*:", workflow.split(f"\n  {job}:\n", 1)[1])[
            0
        ]
        assert "uses: ./.github/actions/setup-tooling" in section
