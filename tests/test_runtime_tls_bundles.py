"""Runtime HTTPS adapters must refresh and verify both Python trust stores."""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
ADAPTERS = (
    "deepseek/mia-vllm-vision",
    "deepseek/sfxnz-vllm-vision-dual",
    "qwen/drowzeys-aday777-single",
)
BUILD_HOSTS = {
    "archive.ubuntu.com",
    "security.ubuntu.com",
    "ports.ubuntu.com",
    "pypi.org",
    "files.pythonhosted.org",
}


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_runtime_https_image_refreshes_and_checks_both_stores(adapter: str) -> None:
    context = ROOT / "adapters" / adapter
    dockerfile = (context / "Dockerfile").read_text()
    assert "--reinstall ca-certificates" in dockerfile
    assert dockerfile.index("apt-get update") < dockerfile.index(
        "--reinstall ca-certificates"
    )
    assert "&& update-ca-certificates" in dockerfile
    assert (
        "--no-deps --require-hashes -r /tmp/runtime-tls/requirements-tls.lock"
        in dockerfile
    )
    assert (
        "COPY requirements-tls.lock verify-runtime-tls.py /tmp/runtime-tls/"
        in dockerfile
    )
    # Verify after all upstream installers and before dropping privileges.
    verify = dockerfile.index("python3 /tmp/runtime-tls/verify-runtime-tls.py")
    assert dockerfile.rindex("pip install") < verify
    assert verify < dockerfile.index("USER 10001:10001")
    lock = (context / "requirements-tls.lock").read_text()
    assert "certifi==2026.10.19 \\\n" in lock
    assert lock.count("--hash=sha256:") == 2
    assert "certifi==2026.10.19" in (context / "requirements-tls.in").read_text()
    ours = json.loads((context / "vendored-upstream.json").read_text())["ours"]
    assert {
        "requirements-tls.in",
        "requirements-tls.lock",
        "verify-runtime-tls.py",
    } <= set(ours)
    recipes = [
        json.loads(path.read_text()) for path in (ROOT / "recipes").glob("*.json")
    ]
    builds = [
        recipe["execution"]["build"]
        for recipe in recipes
        if recipe["execution"]["build"]["context"]["path"] == f"adapters/{adapter}"
    ]
    assert builds
    assert all(BUILD_HOSTS <= set(build["network"]["hosts"]) for build in builds)


def run_guard(
    *,
    version: str = "2026.10.19",
    system_root: str = "unaffected root",
    certifi_root: str = "unaffected root",
    empty: bool = False,
) -> None:
    def context(*, cafile: str | None = None) -> SimpleNamespace:
        name = certifi_root if cafile else system_root
        roots = [] if empty else [{"subject": ((("commonName", name),),)}]
        return SimpleNamespace(get_ca_certs=lambda: roots)

    with patch.dict(
        sys.modules,
        {
            "certifi": SimpleNamespace(
                __version__=version, where=lambda: "certifi.pem"
            ),
            "ssl": SimpleNamespace(create_default_context=context),
        },
    ):
        runpy.run_path(str(ROOT / "adapters" / ADAPTERS[0] / "verify-runtime-tls.py"))


def test_guard_copies_agree_and_accept_unaffected_roots() -> None:
    assert (
        len(
            {
                (ROOT / "adapters" / adapter / "verify-runtime-tls.py").read_bytes()
                for adapter in ADAPTERS
            }
        )
        == 1
    )
    run_guard(system_root="emSign ECC Root CA - C3", certifi_root="emSign Root CA - G1")


def test_guard_rejects_old_certifi() -> None:
    with pytest.raises(AssertionError, match="2026.10.18"):
        run_guard(version="2026.10.18")


@pytest.mark.parametrize("store", ("system_root", "certifi_root"))
def test_guard_rejects_compromised_root_in_either_store(store: str) -> None:
    with pytest.raises(AssertionError, match="compromised emSign root"):
        run_guard(
            system_root="emSign ECC Root CA - G3"
            if store == "system_root"
            else "unaffected root",
            certifi_root="emSign ECC Root CA - G3"
            if store == "certifi_root"
            else "unaffected root",
        )


def test_guard_rejects_empty_store() -> None:
    with pytest.raises(AssertionError, match="empty runtime CA store"):
        run_guard(empty=True)
