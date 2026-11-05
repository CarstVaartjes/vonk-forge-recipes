from __future__ import annotations

import json
import re
import shlex
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


INSTALL = re.compile(r"(?:\bpip3?|-m pip|\buv pip)\s+install\b(?P<args>[^&;|]*)")
ARGUMENT_FLAGS = {"-r", "-c", "-f", "--requirement", "--constraint", "--find-links"}


def fetching_installs(dockerfile_text: str) -> list[str]:
    """pip/uv installs that may reach an index: no --no-index, and not only local paths.

    An install whose every target is a local path with --no-deps fetches nothing.
    """
    joined = re.sub(r"\\\n", " ", dockerfile_text)
    found: list[str] = []
    for match in INSTALL.finditer(joined):
        args = shlex.split(match.group("args"), posix=True)
        if "--no-index" in args:
            continue
        targets: list[str] = []
        flags = iter(args)
        requirement_file = False
        for arg in flags:
            if arg in ARGUMENT_FLAGS:
                next(flags, None)
                requirement_file = requirement_file or arg in {"-r", "--requirement"}
            elif not arg.startswith("-"):
                targets.append(arg)
        local_only = (
            "--no-deps" in args
            and not requirement_file
            and bool(targets)
            and all(t.startswith(("/", ".")) for t in targets)
        )
        if not local_only:
            found.append(match.group(0).strip())
    return found


class BuildNetworkHostMetadataTests(unittest.TestCase):
    def test_declared_build_hosts_are_used_by_source_builds(self) -> None:
        for path in sorted((ROOT / "recipes").glob("*.json")):
            recipe = load(path)
            network = recipe["execution"]["build"]["network"]
            self.assertEqual(
                len(network["hosts"]), len(set(network["hosts"])), path.name
            )
            dockerfile = ROOT / recipe["execution"]["build"]["dockerfile"]
            self.assertTrue(dockerfile.is_file(), path.name)

    def test_package_installs_declare_an_index_host(self) -> None:
        """The Spark builder gives an empty host list --network=none: a pip install would fail."""
        for path in sorted((ROOT / "recipes").glob("*.json")):
            build = load(path)["execution"]["build"]
            if build["network"]["hosts"]:
                continue
            text = (ROOT / build["dockerfile"]).read_text(encoding="utf-8")
            self.assertEqual([], fetching_installs(text), path.name)

    def test_fetching_install_detection(self) -> None:
        self.assertTrue(fetching_installs("RUN pip install foo"))
        self.assertTrue(fetching_installs("RUN python -m pip install -r /tmp/a.txt"))
        self.assertFalse(fetching_installs("RUN pip install --no-index /tmp/a.whl"))
        self.assertFalse(
            fetching_installs(
                "RUN python -m pip install --no-deps /tmp/src && pip show x"
            )
        )

    def test_nvcr_is_only_a_base_image_registry(self) -> None:
        for path in sorted((ROOT / "recipes").glob("*.json")):
            recipe = load(path)
            build = recipe["execution"]["build"]
            hosts = build["network"]["hosts"]
            if "nvcr.io" not in hosts:
                continue
            dockerfile = ROOT / build["dockerfile"]
            steps = "\n".join(
                line
                for line in dockerfile.read_text().splitlines()
                if not line.lstrip().upper().startswith("FROM ")
            )
            self.assertIn("https://nvcr.io", steps, path.name)

    def test_security_and_redirect_hosts_remain_explicit(self) -> None:
        for slug in (
            "ltx-2-19b-dev-fp4-pytorch-single",
            "step1x-3d-geometry-pytorch-single",
            "step1x-3d-label-geometry-pytorch-single",
            "step1x-3d-texture-pytorch-single",
        ):
            recipe = load(ROOT / "recipes" / f"{slug}.json")
            hosts = recipe["execution"]["build"]["network"]["hosts"]
            self.assertTrue(hosts, slug)
            if slug.startswith("step1x"):
                self.assertIn("security.ubuntu.com", hosts)

    def test_pytorch_index_object_host_is_explicit(self) -> None:
        for path in sorted((ROOT / "recipes").glob("*.json")):
            recipe = load(path)
            build = recipe["execution"]["build"]
            dockerfile = ROOT / build["dockerfile"]
            source = dockerfile.read_text(encoding="utf-8")
            if "https://download.pytorch.org/whl/" not in source:
                continue
            hosts = build["network"]["hosts"]
            self.assertIn("download.pytorch.org", hosts, path.name)
            self.assertIn("download-r2.pytorch.org", hosts, path.name)
            if any(
                f"https://download.pytorch.org/whl/{flavor}" in source
                for flavor in ("cu130", "cu132")
            ):
                self.assertIn("pypi.nvidia.com", hosts, path.name)


if __name__ == "__main__":
    unittest.main()
